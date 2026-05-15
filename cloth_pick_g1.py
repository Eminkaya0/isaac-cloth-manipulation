"""
Unitree G1 (29-DOF + BrainCo dexterous hands) picks a particle cloth from a
workbench using its real right arm + right hand.

No Franka in this scene.  G1 is loaded as an active articulation (its URDF was
imported with fix_base=True so the pelvis is welded to the world; that gives
us a stable manipulator without needing a balance policy).  We don't have a
Lula IK config for G1, so the right arm follows scripted joint-space
waypoints — fast to set up, no IK plumbing.

State machine phases:
    0 REST       - hold the default joint positions
    1 REACH      - move right arm into a pose above the cloth corner
    2 LOWER      - drop the hand to the cloth surface
    3 ATTACH     - pin cloth particles to the right palm via CreatePhysicsAttachment
    4 GRIP       - curl the right-hand finger proximal joints (cosmetic)
    5 LIFT       - raise the arm so the cloth comes off the table
    6 RELEASE    - delete the attachment + open the fingers
    7 RETREAT    - return to the rest pose

Run from /home/emin/isaac-sim/ with:
    ./python.sh standalone_examples/api/isaacsim.core.api/cloth_pick_g1.py
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys
import time

import carb
import numpy as np
import torch
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

import omni.kit.app
import omni.kit.commands
import omni.timeline
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.physx.commands", True)
import omni.physxcommands  # noqa: F401
from omni.physx.scripts import deformableUtils, particleUtils, physicsUtils

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction


HUMANOID_USD = "/home/emin/isaac-sim/robots/g1_brainco.usd"
HUMANOID_CONTAINER = "/World/HumanoidContainer"
HUMANOID_PATH = HUMANOID_CONTAINER + "/humanoid"

# After URDF import the right palm sits at this link inside the converted USD.
# We'll resolve the full prim path lazily once we know how the reference is
# composed (it's logged at startup so you can confirm).
RIGHT_PALM_LINK = "right_base_link"

CLOTH_ROOT = "/World/Cloth"
# Raise the table to where the G1 right palm can reach (palm bottoms out
# around z≈0.55-0.65 with the current arm waypoints and waist bend).
TABLE_TOP_Z = 0.60

CLOTHS = [
    {"name": "cloth_red",   "pos": (0.45,  0.18, TABLE_TOP_Z + 0.02), "color": (0.85, 0.15, 0.15)},
    {"name": "cloth_green", "pos": (0.45,  0.00, TABLE_TOP_Z + 0.02), "color": (0.20, 0.75, 0.30)},
    {"name": "cloth_blue",  "pos": (0.45, -0.18, TABLE_TOP_Z + 0.02), "color": (0.15, 0.25, 0.85)},
]
TARGET_CLOTH_NAME = "cloth_green"

# Right arm joint sequence (URDF order).
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Bend the torso slightly forward so the arm has more reach toward the table.
WAIST_PITCH_FORWARD = 0.35  # mild forward lean, much less crouchy
# Drop the pelvis anchor 25 cm so the arm has reach to the table-top cloth.
PELVIS_ANCHOR_Z = 0.55

# Per-phase target joint positions for the seven right-arm DOFs above.
# G1 joint conventions (discovered by trial):
#   shoulder_pitch < 0 -> arm rotates FORWARD (toward cloth in front)
#   shoulder_pitch > 0 -> arm goes UP/BACK toward the torso
#   shoulder_roll  > 0 -> arm sweeps IN  (toward body centre, world -Y)
#   shoulder_roll  < 0 -> arm sweeps OUT (away from body, world +Y)
#   elbow          < 0 -> elbow flexes
# Target cloth in world: roughly (0.45, 0.0, 0.42).  G1 pelvis at (1.05, 0, 0.79),
# facing -X.  Arm needs to swing forward + slightly inward + down.
REST_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Negative pitch + positive elbow rotates upper arm forward, forearm flexes
# down — confirmed by palm trajectory. Shoulder roll +0.2 brings hand in
# toward body-centre (palm currently at +y=0.08, cloth at y=0).
REACH_POSE = np.array([-1.3, 0.10, 0.0, 1.6, 0.0, 0.0, 0.0])
GRASP_POSE = np.array([-1.7, 0.10, 0.0, 1.8, 0.0, 0.0, 0.0])
LIFT_POSE = np.array([-0.7, 0.10, 0.0, 1.5, 0.0, 0.0, 0.0])

CLOTH_TARGET_WORLD = np.array([0.45, 0.00, 0.62])

# Right-hand finger proximal joints — curling these counts as "grip".
RIGHT_FINGER_JOINTS = [
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
]
FINGERS_OPEN = 0.0
FINGERS_CLOSED = 1.0  # curl ~57° at each proximal

PHASE_NAMES = ["REST", "REACH", "LOWER", "ATTACH", "GRIP", "LIFT", "RELEASE", "RETREAT"]
PHASE_DURATIONS = [60, 120, 80, 1, 30, 120, 1, 120]  # in sim steps


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class ClothPickG1:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0, backend="torch", device="cuda")
        self.stage = simulation_app.context.get_stage()
        self.world.scene.add_default_ground_plane()

        self._build_table()
        self._build_humanoid()
        self._build_cloth()

        self.dof_index = None      # dict joint_name -> dof index
        self.target_cloth_path = CLOTH_ROOT + "/" + TARGET_CLOTH_NAME
        self.target_attach_path = self.target_cloth_path + "/clothAttachment"
        self.palm_prim_path = None  # resolved at runtime
        self.phase = 0
        self.phase_step = 0
        self.attached = False

    def _build_table(self):
        table_path = "/World/Table"
        xform = UsdGeom.Xform.Define(self.stage, table_path)
        physicsUtils.set_or_add_translate_op(xform, Gf.Vec3f(0.45, 0.0, TABLE_TOP_Z / 2.0))
        physicsUtils.set_or_add_scale_op(xform, Gf.Vec3f(0.35, 0.45, TABLE_TOP_Z / 2.0))
        cube = UsdGeom.Cube.Define(self.stage, table_path + "/geom")
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        material = UsdShade.Material.Define(self.stage, table_path + "/wood_mat")
        shader = UsdShade.Shader.Define(self.stage, table_path + "/wood_mat/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.45, 0.27, 0.12))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)

    def _build_humanoid(self):
        # G1 USD is imported without fix_base — we add our own UsdPhysics
        # FixedJoint linking the world to its pelvis, so the manipulator has
        # a stable base without needing a balance policy.
        container = UsdGeom.Xform.Define(self.stage, HUMANOID_CONTAINER)
        xform_api = UsdGeom.XformCommonAPI(container.GetPrim())
        xform_api.SetTranslate(Gf.Vec3d(1.05, 0.0, 0.79))
        xform_api.SetRotate(Gf.Vec3f(0.0, 0.0, 180.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        add_reference_to_stage(usd_path=HUMANOID_USD, prim_path=HUMANOID_PATH)
        self.humanoid = self.world.scene.add(
            SingleArticulation(prim_path=HUMANOID_PATH, name="g1")
        )

        # URDF import creates an outer `pelvis` group whose child `pelvis`
        # is the actual rigid body link.  Weld the inner pelvis to the world
        # via a FixedJoint, anchoring it at the container's desired world
        # pose (otherwise the joint snaps the pelvis to origin and G1
        # vanishes below the floor).
        pelvis_path = HUMANOID_PATH + "/pelvis/pelvis"
        joint_path = "/World/PelvisAnchor"
        fixed_joint = UsdPhysics.FixedJoint.Define(self.stage, joint_path)
        fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(pelvis_path)])
        # body0 == world; tell the joint where the world-side anchor lives so
        # the pelvis stays at the container's nominal world pose.
        fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(1.05, 0.0, PELVIS_ANCHOR_Z))
        fixed_joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), 180.0).GetQuat())
        )
        print(f"[g1] anchored {pelvis_path} via FixedJoint at {joint_path}", flush=True)

    def _build_cloth(self):
        UsdGeom.Xform.Define(self.stage, CLOTH_ROOT)
        radius = 0.5 * (0.20 / 10.0)
        rest_offset = radius
        contact_offset = rest_offset * 1.5
        particle_system_path = CLOTH_ROOT + "/particleSystem"
        particleUtils.add_physx_particle_system(
            stage=self.stage,
            particle_system_path=Sdf.Path(particle_system_path),
            simulation_owner=self.world.get_physics_context().prim_path,
            contact_offset=contact_offset, rest_offset=rest_offset,
            particle_contact_offset=contact_offset,
            solid_rest_offset=rest_offset, fluid_rest_offset=rest_offset,
        )
        particle_material_path = CLOTH_ROOT + "/particleMaterial"
        particleUtils.add_pbd_particle_material(
            stage=self.stage, path=Sdf.Path(particle_material_path),
            friction=0.6, damping=0.1, drag=0.1, lift=0.3,
        )
        physicsUtils.add_physics_material_to_prim(
            self.stage,
            self.stage.GetPrimAtPath(particle_system_path),
            Sdf.Path(particle_material_path),
        )
        for i, spec in enumerate(CLOTHS):
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
            stage=self.stage, path=Sdf.Path(path), dynamic_mesh_path=None,
            particle_system_path=Sdf.Path(particle_system_path),
            self_collision=True, self_collision_filter=True,
            particle_group=particle_group,
        )

    def _post_init(self):
        if self.dof_index is not None:
            return
        dof_names = list(self.humanoid.dof_names)
        self.dof_index = {name: i for i, name in enumerate(dof_names)}
        print(f"[g1] {len(dof_names)} dofs", flush=True)
        missing = [n for n in RIGHT_ARM_JOINTS + RIGHT_FINGER_JOINTS if n not in self.dof_index]
        if missing:
            print(f"[g1] MISSING dof names: {missing}", flush=True)
        from pxr import Usd
        root_prim = self.stage.GetPrimAtPath(HUMANOID_PATH)
        for prim in Usd.PrimRange(root_prim):
            if prim.GetName() == RIGHT_PALM_LINK:
                self.palm_prim_path = str(prim.GetPath())
                break
        print(f"[g1] right palm prim -> {self.palm_prim_path}", flush=True)

        # Ensure every revolute joint actually has a position drive — without
        # this the articulation silently ignores joint targets and looks like
        # a frozen statue.
        applied = 0
        for prim in Usd.PrimRange(root_prim):
            if prim.GetTypeName() == "PhysicsRevoluteJoint":
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateTypeAttr().Set("force")
                drive.CreateStiffnessAttr().Set(2000.0)
                drive.CreateDampingAttr().Set(150.0)
                drive.CreateMaxForceAttr().Set(1e6)
                applied += 1
        print(f"[g1] applied position drive (stiffness=2000) to {applied} revolute joints", flush=True)
        # Pin every joint at zero, then add a forward waist bend + leg squat
        # so the humanoid visually crouches while the (anchored) pelvis sits
        # lower for the arm to reach the table.
        zero_targets = torch.zeros(len(dof_names), dtype=torch.float32, device="cuda")
        pose_defaults = {"waist_pitch_joint": WAIST_PITCH_FORWARD}
        for name, value in pose_defaults.items():
            if name in self.dof_index:
                zero_targets[self.dof_index[name]] = value
        all_indices = torch.arange(len(dof_names), dtype=torch.long, device="cuda")
        self.humanoid.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=zero_targets, joint_indices=all_indices)
        )
        print(f"[g1] joint targets initialized (waist_pitch={WAIST_PITCH_FORWARD:.2f})",
              flush=True)
        print(f"[g1] initial joint pos[shoulder_pitch]="
              f"{_to_numpy(self.humanoid.get_joint_positions())[self.dof_index['right_shoulder_pitch_joint']]:.3f}",
              flush=True)

    def _apply_arm_pose(self, pose_array):
        indices = torch.tensor([self.dof_index[n] for n in RIGHT_ARM_JOINTS],
                                dtype=torch.long, device="cuda")
        positions = torch.tensor(pose_array, dtype=torch.float32, device="cuda")
        self.humanoid.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=positions, joint_indices=indices)
        )

    def _apply_fingers(self, value):
        names = [n for n in RIGHT_FINGER_JOINTS if n in self.dof_index]
        indices = torch.tensor([self.dof_index[n] for n in names],
                                dtype=torch.long, device="cuda")
        positions = torch.full((len(names),), float(value),
                                 dtype=torch.float32, device="cuda")
        self.humanoid.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=positions, joint_indices=indices)
        )

    def _attach_cloth(self):
        if self.palm_prim_path is None:
            print("[g1] cannot attach — palm prim not resolved", flush=True)
            return
        omni.kit.commands.execute(
            "CreatePhysicsAttachment",
            target_attachment_path=Sdf.Path(self.target_attach_path),
            actor0_path=Sdf.Path(self.target_cloth_path),
            actor1_path=Sdf.Path(self.palm_prim_path),
        )
        self.attached = True
        print(f"[ATTACH] {self.target_cloth_path} -> {self.palm_prim_path}", flush=True)

    def _detach_cloth(self):
        if self.attached:
            omni.kit.commands.execute("DeletePrims", paths=[self.target_attach_path])
            self.attached = False
            print(f"[RELEASE] {self.target_attach_path} deleted", flush=True)

    def _advance(self):
        self.phase_step += 1
        if self.phase_step >= PHASE_DURATIONS[self.phase]:
            print(f"[phase] FINISHED {PHASE_NAMES[self.phase]} after {self.phase_step} steps",
                  flush=True)
            self.phase += 1
            self.phase_step = 0

    def step(self):
        self._post_init()
        if self.phase >= len(PHASE_NAMES):
            return False
        if self.phase_step == 0:
            sp = _to_numpy(self.humanoid.get_joint_positions())[self.dof_index["right_shoulder_pitch_joint"]]
            try:
                palm = SingleRigidPrim(prim_path=self.palm_prim_path, name="_palm_dbg")
                palm.initialize()
                ppos, _ = palm.get_world_pose()
                ppos_np = _to_numpy(ppos)
                delta = ppos_np - CLOTH_TARGET_WORLD
                dist = float(np.linalg.norm(delta))
                print(f"[phase {self.phase}] START {PHASE_NAMES[self.phase]}  "
                      f"shoulder_pitch={sp:.3f}  palm={ppos_np}  delta={delta}  dist={dist:.3f}",
                      flush=True)
            except Exception as exc:
                print(f"[phase {self.phase}] START {PHASE_NAMES[self.phase]}  err={exc}", flush=True)

        p = PHASE_NAMES[self.phase]
        if p == "REST":
            self._apply_arm_pose(REST_POSE)
            self._apply_fingers(FINGERS_OPEN)
        elif p == "REACH":
            self._apply_arm_pose(REACH_POSE)
        elif p == "LOWER":
            self._apply_arm_pose(GRASP_POSE)
        elif p == "ATTACH":
            self._attach_cloth()
        elif p == "GRIP":
            self._apply_fingers(FINGERS_CLOSED)
        elif p == "LIFT":
            self._apply_arm_pose(LIFT_POSE)
        elif p == "RELEASE":
            self._detach_cloth()
            self._apply_fingers(FINGERS_OPEN)
        elif p == "RETREAT":
            self._apply_arm_pose(REST_POSE)

        self._advance()
        return True


def main():
    print("[demo] building scene...", flush=True)
    demo = ClothPickG1()
    print("[demo] scene built, calling world.reset()", flush=True)
    demo.world.reset(soft=False)
    omni.timeline.get_timeline_interface().play()
    print("[demo] timeline play, entering main loop", flush=True)

    sim_steps = 0
    while simulation_app.is_running():
        demo.world.step(render=True)
        if not demo.world.is_playing():
            continue
        sim_steps += 1
        time.sleep(0.02)
        if sim_steps < 30:
            continue
        if not demo.step():
            for _ in range(300):
                demo.world.step(render=True)
                time.sleep(0.02)
            break
    print("[demo] G1 cloth pick demo complete.", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
