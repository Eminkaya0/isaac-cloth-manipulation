"""
Franka Panda picks up a particle cloth from a workbench and flips it 180 degrees.

State machine phases:
  0 HOVER       - move end-effector above the cloth
  1 DESCEND     - lower onto the cloth surface
  2 ATTACH      - pin cloth particles to gripper finger via CreatePhysicsAttachment
  3 CLOSE_GRIP  - cosmetic gripper close (attachment does the holding)
  4 LIFT        - raise the cloth off the table
  5 FLIP        - rotate panda_joint7 by +pi (wrist roll)
  6 LOWER       - place flipped cloth back near the table
  7 RELEASE     - delete attachment + open gripper, cloth settles
  8 RETREAT     - return to a safe pose

Particle cloth requires GPU (PhysX GPU dynamics) so the World is created with
device="cuda".  The high-level Franka helper class is *not* used — its
ParallelGripper.initialize() crashes when joint positions live on cuda — so we
load the Franka USD manually and wrap it in a SingleArticulation, which is
device-safe.

Run from /home/emin/isaac-sim/ with:
    ./python.sh standalone_examples/api/isaacsim.core.api/cloth_pick_franka.py
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "extra_args": ["--/app/useFabricSceneDelegate=0"]})

import sys
import time

import carb
import numpy as np
import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

import omni.kit.app
import omni.kit.commands
import omni.timeline
# Enable the omni.physx.commands extension so CreatePhysicsAttachment / DeletePrims
# are registered (otherwise omni.kit.commands.execute raises "not registered").
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.physx.commands", True)
import omni.physxcommands  # noqa: F401  registers CreatePhysicsAttachment etc.
from omni.physx.scripts import deformableUtils, particleUtils, physicsUtils

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.examples.franka.kinematics_solver import KinematicsSolver


ASSETS_ROOT = get_assets_root_path()
if ASSETS_ROOT is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()


FRANKA_USD = ASSETS_ROOT + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
FRANKA_PATH = "/World/Franka"
GRIPPER_PRIM = FRANKA_PATH + "/panda_rightfinger"
EE_PRIM = FRANKA_PATH + "/panda_hand"

HUMANOID_USD = "/home/emin/isaac-sim/robots/g1_brainco.usd"
HUMANOID_CONTAINER = "/World/HumanoidContainer"
HUMANOID_PATH = HUMANOID_CONTAINER + "/humanoid"
# G1 pelvis sits at ~0.79 m above its USD origin when the robot is in the
# default standing pose — adjust this if the model floats / sinks.
HUMANOID_STAND_Z = 0.79

CLOTH_ROOT = "/World/Cloth"

TABLE_TOP_Z = 0.40

# All cloths on the table. The demo automatically targets whichever one is
# closest to NEAREST_REF below (i.e. "pick the nearest cloth").
ALL_CLOTHS = [
    {"name": "cloth_red",   "pos": (0.45,  0.18, TABLE_TOP_Z + 0.02), "color": (0.85, 0.15, 0.15)},
    {"name": "cloth_green", "pos": (0.45,  0.00, TABLE_TOP_Z + 0.02), "color": (0.20, 0.75, 0.30)},
    {"name": "cloth_blue",  "pos": (0.45, -0.18, TABLE_TOP_Z + 0.02), "color": (0.15, 0.25, 0.85)},
]

# Where "near" is measured from. Franka home end-effector pose is roughly
# (0.39, 0.00, 0.47), so the green cloth ends up as the nearest one — change
# this point and the demo will pick a different cloth.
NEAREST_REF = np.array([0.39, 0.0, 0.47])

RETREAT_POS = np.array([0.30, 0.0, TABLE_TOP_Z + 0.40])

DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # gripper pointing down (w,x,y,z)

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0

PHASE_NAMES = [
    "HOVER", "DESCEND", "ATTACH", "CLOSE_GRIP",
    "LIFT", "FLIP", "LOWER", "RELEASE", "RETREAT",
]
PHASE_DURATIONS = [80, 60, 1, 20, 80, 100, 60, 1, 80]


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _patch_kinematics_to_numpy(kin):
    """Lula's compute_inverse_kinematics requires a numpy warm_start; the joints
    view will return a cuda tensor when the World runs on cuda. Wrap it."""
    orig = kin._joints_view.get_joint_positions

    def _np_get():
        result = orig()
        return _to_numpy(result)

    kin._joints_view.get_joint_positions = _np_get


class ClothPickDemo:
    def __init__(self):
        # Particle cloth physics is GPU-only and is gated by the physics
        # context being initialized in GPU pipeline mode, which only happens
        # when the World device is cuda.  We keep cuda here for that reason and
        # later patch the Lula IK joints_view to hand back numpy arrays.
        self.world = World(stage_units_in_meters=1.0, backend="torch", device="cuda")
        self.stage = simulation_app.context.get_stage()
        self.world.scene.add_default_ground_plane()

        self._build_table()
        self._build_franka()
        self._build_humanoid()
        self._build_cloth()
        self._select_nearest_cloth()

        self.kin = None
        self.finger_dof_indices = None
        self.phase = 0
        self.phase_step = 0
        self.attached = False
        self.wrist_target_after_flip = None

    def _select_nearest_cloth(self):
        """Pick the cloth whose centre is closest to NEAREST_REF and lock all
        pick / hover / lift / place positions to that cloth."""
        best = min(
            ALL_CLOTHS,
            key=lambda spec: float(np.linalg.norm(np.array(spec["pos"]) - NEAREST_REF)),
        )
        center = np.array(best["pos"])
        self.target_cloth_path = CLOTH_ROOT + "/" + best["name"]
        self.target_attach_path = self.target_cloth_path + "/clothAttachment"
        self.target_cloth_center = center
        self.target_hover_pos = np.array([center[0], center[1], TABLE_TOP_Z + 0.25])
        self.target_grasp_pos = np.array([center[0], center[1], TABLE_TOP_Z + 0.03])
        self.target_lift_pos = np.array([center[0], center[1], TABLE_TOP_Z + 0.30])
        self.target_place_pos = np.array([center[0], center[1], TABLE_TOP_Z + 0.08])
        print(
            f"[demo] nearest cloth to {NEAREST_REF.tolist()} -> {best['name']} "
            f"at pos={center.tolist()}",
            flush=True,
        )

    def _build_table(self):
        table_path = "/World/Table"
        xform = UsdGeom.Xform.Define(self.stage, table_path)
        physicsUtils.set_or_add_translate_op(xform, Gf.Vec3f(0.45, 0.0, TABLE_TOP_Z / 2.0))
        physicsUtils.set_or_add_scale_op(xform, Gf.Vec3f(0.35, 0.45, TABLE_TOP_Z / 2.0))
        cube = UsdGeom.Cube.Define(self.stage, table_path + "/geom")
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        self._bind_preview_surface(cube.GetPrim(), table_path + "/wood_mat",
                                    diffuse=(0.45, 0.27, 0.12), roughness=0.7)

    def _bind_preview_surface(self, target_prim, mat_path, diffuse, roughness=0.5):
        material = UsdShade.Material.Define(self.stage, mat_path)
        shader = UsdShade.Shader.Define(self.stage, mat_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(target_prim).Bind(material)

    def _build_franka(self):
        add_reference_to_stage(usd_path=FRANKA_USD, prim_path=FRANKA_PATH)
        self.franka = self.world.scene.add(
            SingleArticulation(prim_path=FRANKA_PATH, name="franka", position=np.array([0.0, 0.0, 0.0]))
        )
        self.end_effector = SingleRigidPrim(prim_path=EE_PRIM, name="franka_ee")

    def _build_humanoid(self):
        # Decorative humanoid behind the workbench. We want the visuals only;
        # any leftover physics (rigid bodies, articulation, joints) corrupts
        # the Franka articulation state.  Strip everything physics-related.
        #
        # The robot USD has its own xformOpOrder which conflicts with anything
        # we set directly on the reference root.  So we create an *outer*
        # container Xform (our own, clean) and add the robot reference as a
        # CHILD of it.  Our outer transform then applies cleanly on top of
        # whatever internal ops exist.
        container = UsdGeom.Xform.Define(self.stage, HUMANOID_CONTAINER)
        xform_api = UsdGeom.XformCommonAPI(container.GetPrim())
        xform_api.SetTranslate(Gf.Vec3d(1.05, 0.0, HUMANOID_STAND_Z))
        xform_api.SetRotate(Gf.Vec3f(0.0, 0.0, 180.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

        add_reference_to_stage(usd_path=HUMANOID_USD, prim_path=HUMANOID_PATH)
        humanoid_prim = self.stage.GetPrimAtPath(HUMANOID_PATH)

        joint_count = 0
        body_count = 0
        for prim in Usd.PrimRange(humanoid_prim):
            type_name = prim.GetTypeName()
            # Any physics joint subtype (Revolute, Fixed, Prismatic, Spherical,
            # Distance, ...) — deactivate so PhysX doesn't warn about
            # static-body joints.
            if "Joint" in type_name and type_name.startswith("Physics"):
                prim.SetActive(False)
                joint_count += 1
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                body_count += 1
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
        print(
            f"[humanoid] stripped {body_count} rigid bodies, deactivated {joint_count} joints",
            flush=True,
        )

    def _build_cloth(self):
        UsdGeom.Xform.Define(self.stage, CLOTH_ROOT)

        # One shared particle system / material for every cloth.
        radius = 0.5 * (0.20 / 10.0)
        rest_offset = radius
        contact_offset = rest_offset * 1.5
        particle_system_path = CLOTH_ROOT + "/particleSystem"
        particleUtils.add_physx_particle_system(
            stage=self.stage,
            particle_system_path=Sdf.Path(particle_system_path),
            simulation_owner=self.world.get_physics_context().prim_path,
            contact_offset=contact_offset,
            rest_offset=rest_offset,
            particle_contact_offset=contact_offset,
            solid_rest_offset=rest_offset,
            fluid_rest_offset=rest_offset,
        )
        particle_material_path = CLOTH_ROOT + "/particleMaterial"
        particleUtils.add_pbd_particle_material(
            stage=self.stage,
            path=Sdf.Path(particle_material_path),
            friction=0.6, damping=0.1, drag=0.1, lift=0.3,
        )
        physicsUtils.add_physics_material_to_prim(
            self.stage,
            self.stage.GetPrimAtPath(particle_system_path),
            Sdf.Path(particle_material_path),
        )

        for i, spec in enumerate(ALL_CLOTHS):
            self._spawn_cloth(
                path=CLOTH_ROOT + "/" + spec["name"],
                position=np.array(spec["pos"]),
                color=spec["color"],
                particle_group=i,
                particle_system_path=particle_system_path,
            )

    def _spawn_cloth(self, path, position, color, particle_group, particle_system_path):
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        points, indices = deformableUtils.create_triangle_mesh_square(dimx=10, dimy=10, scale=0.15)
        mesh.GetPointsAttr().Set(points)
        mesh.GetFaceVertexIndicesAttr().Set(indices)
        mesh.GetFaceVertexCountsAttr().Set([3] * (len(indices) // 3))
        physicsUtils.setup_transform_as_scale_orient_translate(mesh)
        physicsUtils.set_or_add_translate_op(
            mesh, Gf.Vec3f(float(position[0]), float(position[1]), float(position[2]))
        )
        mesh.CreateDisplayColorPrimvar().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        particleUtils.add_physx_particle_cloth(
            stage=self.stage,
            path=Sdf.Path(path),
            dynamic_mesh_path=None,
            particle_system_path=Sdf.Path(particle_system_path),
            self_collision=True,
            self_collision_filter=True,
            particle_group=particle_group,
        )

    def _post_init(self):
        if self.finger_dof_indices is not None:
            return
        self.end_effector.initialize()
        dof_names = list(self.franka.dof_names)
        self.finger_dof_indices = [dof_names.index("panda_finger_joint1"), dof_names.index("panda_finger_joint2")]
        print(f"[init] dof_names={dof_names}", flush=True)
        print(f"[init] finger dof indices={self.finger_dof_indices}", flush=True)
        self.kin = KinematicsSolver(self.franka)
        _patch_kinematics_to_numpy(self.kin)

    def _apply_ee_pose(self, position, orientation=DOWN_QUAT):
        action, ok = self.kin.compute_inverse_kinematics(
            target_position=np.array(position, dtype=np.float32),
            target_orientation=np.array(orientation, dtype=np.float32),
        )
        if ok:
            self.franka.get_articulation_controller().apply_action(action)
        else:
            carb.log_warn(f"[phase {PHASE_NAMES[self.phase]}] IK failed for target {position}")

    def _set_gripper(self, finger_position):
        joints = _to_numpy(self.franka.get_joint_positions()).astype(np.float32).copy()
        for idx in self.finger_dof_indices:
            joints[idx] = finger_position
        self.franka.get_articulation_controller().apply_action(ArticulationAction(joint_positions=joints))

    def _apply_wrist(self, target_radians):
        joints = _to_numpy(self.franka.get_joint_positions()).astype(np.float32).copy()
        joints[6] = float(target_radians)
        self.franka.get_articulation_controller().apply_action(ArticulationAction(joint_positions=joints))

    def _attach_cloth(self):
        omni.kit.commands.execute(
            "CreatePhysicsAttachment",
            target_attachment_path=Sdf.Path(self.target_attach_path),
            actor0_path=Sdf.Path(self.target_cloth_path),
            actor1_path=Sdf.Path(GRIPPER_PRIM),
        )
        self.attached = True
        print(f"[ATTACH] {self.target_cloth_path} pinned to panda_rightfinger")

    def _detach_cloth(self):
        if self.attached:
            omni.kit.commands.execute("DeletePrims", paths=[self.target_attach_path])
            self.attached = False
            print(f"[RELEASE] {self.target_attach_path} deleted")

    def _advance(self):
        self.phase_step += 1
        if self.phase_step >= PHASE_DURATIONS[self.phase]:
            print(f"[phase] FINISHED {PHASE_NAMES[self.phase]} after {self.phase_step} steps", flush=True)
            self.phase += 1
            self.phase_step = 0

    def step(self):
        self._post_init()
        if self.phase >= len(PHASE_NAMES):
            return False

        if self.phase_step == 0:
            ee_pos, _ = self.end_effector.get_world_pose()
            print(f"[phase {self.phase}] START {PHASE_NAMES[self.phase]}  ee_pos={_to_numpy(ee_pos)}", flush=True)

        p = self.phase
        if p == 0:
            self._apply_ee_pose(self.target_hover_pos)
        elif p == 1:
            self._apply_ee_pose(self.target_grasp_pos)
        elif p == 2:
            self._attach_cloth()
        elif p == 3:
            self._set_gripper(GRIPPER_CLOSED)
        elif p == 4:
            self._apply_ee_pose(self.target_lift_pos)
        elif p == 5:
            if self.wrist_target_after_flip is None:
                current_val = float(_to_numpy(self.franka.get_joint_positions())[6])
                self.wrist_target_after_flip = current_val + np.pi
                print(f"[FLIP] wrist {current_val:.3f} -> {self.wrist_target_after_flip:.3f}")
            self._apply_wrist(self.wrist_target_after_flip)
        elif p == 6:
            self._apply_ee_pose(self.target_place_pos)
        elif p == 7:
            self._detach_cloth()
            self._set_gripper(GRIPPER_OPEN)
        elif p == 8:
            self._apply_ee_pose(RETREAT_POS)

        self._advance()
        return True


def main():
    print("[demo] building scene...", flush=True)
    demo = ClothPickDemo()
    print("[demo] scene built, calling world.reset()", flush=True)
    demo.world.reset(soft=False)
    omni.timeline.get_timeline_interface().play()
    print("[demo] timeline play, entering main loop", flush=True)

    sim_steps = 0
    slow_step = 0.02  # ~50 fps, so the GUI run is visible to a human watching
    while simulation_app.is_running():
        demo.world.step(render=True)
        if not demo.world.is_playing():
            continue
        sim_steps += 1
        time.sleep(slow_step)
        if sim_steps < 30:
            continue
        if not demo.step():
            print("[demo] phases done — holding GUI open", flush=True)
            for _ in range(300):
                demo.world.step(render=True)
                time.sleep(0.02)
            break

    print("[demo] Cloth pick-and-flip demo complete.", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
