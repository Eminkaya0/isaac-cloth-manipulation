"""
Keyboard teleoperation variant of cloth_pick_franka.py.

Same scene (workbench, Franka, H1 humanoid prop, three coloured particle
cloths) but you drive the Franka end-effector with keyboard keys and toggle
the cloth grasp manually.

Controls (focus must be on the Isaac Sim viewport window). The mapping
deliberately avoids WASD (viewport camera) and SPACE (timeline play/pause)
so it doesn't fight Isaac Sim's own shortcuts:
    Up Arrow / Down Arrow       move EE forward / backward       (+X / -X)
    Left Arrow / Right Arrow    move EE left / right             (+Y / -Y)
    PageUp / PageDown           move EE up / down                (+Z / -Z)
    G                           toggle grasp:
                                    - not holding: pin nearest cloth (within 15 cm) to gripper
                                    - holding:     release the cloth
    H                           reset EE target to the home pose above the table
    ESC                         quit

Run from /home/emin/isaac-sim/ with:
    ./python.sh standalone_examples/api/isaacsim.core.api/cloth_teleop_franka.py
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "extra_args": ["--/app/useFabricSceneDelegate=0"]})

import sys
import time

import carb
import carb.input
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

import omni.appwindow
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

H1_USD = ASSETS_ROOT + "/Isaac/Robots/Unitree/H1/h1.usd"
H1_CONTAINER = "/World/HumanoidContainer"
H1_PATH = H1_CONTAINER + "/h1"

CLOTH_ROOT = "/World/Cloth"

TABLE_TOP_Z = 0.40

CLOTHS = [
    {"name": "cloth_green", "pos": (0.45,  0.00, TABLE_TOP_Z + 0.02), "color": (0.20, 0.75, 0.30)},
    {"name": "cloth_red",   "pos": (0.45,  0.18, TABLE_TOP_Z + 0.02), "color": (0.85, 0.15, 0.15)},
    {"name": "cloth_blue",  "pos": (0.45, -0.18, TABLE_TOP_Z + 0.02), "color": (0.15, 0.25, 0.85)},
]

HOME_TARGET = np.array([0.40, 0.0, TABLE_TOP_Z + 0.30])

DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0

STEP_LINEAR = 0.02  # 2 cm per keypress
GRASP_RADIUS = 0.15  # closest-cloth threshold for SPACE


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _patch_kinematics_to_numpy(kin):
    orig = kin._joints_view.get_joint_positions

    def _np_get():
        return _to_numpy(orig())

    kin._joints_view.get_joint_positions = _np_get


class ClothTeleop:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0, backend="torch", device="cuda")
        self.stage = simulation_app.context.get_stage()
        self.world.scene.add_default_ground_plane()

        self._build_table()
        self._build_franka()
        self._build_humanoid()
        self._build_cloths()

        self.kin = None
        self.finger_dof_indices = None

        self.target_pos = HOME_TARGET.copy()
        self.attached_cloth_path = None
        self.gripper_target = GRIPPER_OPEN

        # keyboard hookup
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_kb = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)
        self._quit_requested = False

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
        container = UsdGeom.Xform.Define(self.stage, H1_CONTAINER)
        xform_api = UsdGeom.XformCommonAPI(container.GetPrim())
        xform_api.SetTranslate(Gf.Vec3d(1.05, 0.0, 1.05))
        xform_api.SetRotate(Gf.Vec3f(0.0, 0.0, 180.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

        add_reference_to_stage(usd_path=H1_USD, prim_path=H1_PATH)
        humanoid_prim = self.stage.GetPrimAtPath(H1_PATH)

        for prim in Usd.PrimRange(humanoid_prim):
            type_name = prim.GetTypeName()
            if "Joint" in type_name and type_name.startswith("Physics"):
                prim.SetActive(False)
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.RemoveAPI(UsdPhysics.CollisionAPI)

    def _build_cloths(self):
        UsdGeom.Xform.Define(self.stage, CLOTH_ROOT)
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

        self.cloth_paths = []
        for i, spec in enumerate(CLOTHS):
            path = CLOTH_ROOT + "/" + spec["name"]
            self._spawn_cloth(path, np.array(spec["pos"]), spec["color"], i, particle_system_path)
            self.cloth_paths.append(path)

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
        self.kin = KinematicsSolver(self.franka)
        _patch_kinematics_to_numpy(self.kin)
        print("[teleop] ready — focus the viewport. Arrows=XY, PgUp/PgDn=Z, G=grasp, H=home, ESC=quit", flush=True)
        ee_pos, _ = self.end_effector.get_world_pose()
        self.target_pos = _to_numpy(ee_pos).astype(np.float32).copy()
        print(f"[teleop] initial target_pos={self.target_pos}", flush=True)

    def _on_key(self, event, *args, **kwargs):
        if event.type not in (carb.input.KeyboardEventType.KEY_PRESS,
                               carb.input.KeyboardEventType.KEY_REPEAT):
            return True
        k = event.input
        K = carb.input.KeyboardInput

        # Arrow keys + PageUp/Down for translation (no conflict with viewport
        # WASD camera nav).  G for grasp (no conflict with SPACE play/pause).
        if k == K.UP:
            self.target_pos[0] += STEP_LINEAR
        elif k == K.DOWN:
            self.target_pos[0] -= STEP_LINEAR
        elif k == K.LEFT:
            self.target_pos[1] += STEP_LINEAR
        elif k == K.RIGHT:
            self.target_pos[1] -= STEP_LINEAR
        elif k == K.PAGE_UP:
            self.target_pos[2] += STEP_LINEAR
        elif k == K.PAGE_DOWN:
            self.target_pos[2] -= STEP_LINEAR
        elif k == K.G and event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._toggle_grasp()
        elif k == K.H and event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.target_pos = HOME_TARGET.astype(np.float32).copy()
            print(f"[teleop] reset target_pos={self.target_pos}", flush=True)
        elif k == K.ESCAPE and event.type == carb.input.KeyboardEventType.KEY_PRESS:
            print("[teleop] ESC -> quitting", flush=True)
            self._quit_requested = True
        return True

    def _log_target(self):
        # don't spam — only print every few keystrokes
        pass  # could be wired up if you want chattier output

    def _toggle_grasp(self):
        if self.attached_cloth_path is not None:
            attach_prim_path = self.attached_cloth_path + "/teleopAttachment"
            omni.kit.commands.execute("DeletePrims", paths=[attach_prim_path])
            print(f"[teleop] RELEASED {self.attached_cloth_path}", flush=True)
            self.attached_cloth_path = None
            self.gripper_target = GRIPPER_OPEN
            return

        # find nearest cloth within GRASP_RADIUS
        ee_pos = self.target_pos
        best_path = None
        best_dist = GRASP_RADIUS
        for spec in CLOTHS:
            cloth_pos = np.array(spec["pos"])
            d = float(np.linalg.norm(cloth_pos - ee_pos))
            path = CLOTH_ROOT + "/" + spec["name"]
            if d < best_dist:
                best_dist = d
                best_path = path
        if best_path is None:
            print(f"[teleop] no cloth within {GRASP_RADIUS:.2f} m of target_pos={ee_pos}", flush=True)
            return

        attach_prim_path = best_path + "/teleopAttachment"
        omni.kit.commands.execute(
            "CreatePhysicsAttachment",
            target_attachment_path=Sdf.Path(attach_prim_path),
            actor0_path=Sdf.Path(best_path),
            actor1_path=Sdf.Path(GRIPPER_PRIM),
        )
        self.attached_cloth_path = best_path
        self.gripper_target = GRIPPER_CLOSED
        print(f"[teleop] GRABBED {best_path} (distance {best_dist:.3f} m)", flush=True)

    def _apply_ee_pose(self, position, orientation=DOWN_QUAT):
        action, ok = self.kin.compute_inverse_kinematics(
            target_position=np.array(position, dtype=np.float32),
            target_orientation=np.array(orientation, dtype=np.float32),
        )
        if ok:
            self.franka.get_articulation_controller().apply_action(action)

    def _apply_gripper(self):
        joints = _to_numpy(self.franka.get_joint_positions()).astype(np.float32).copy()
        for idx in self.finger_dof_indices:
            joints[idx] = self.gripper_target
        self.franka.get_articulation_controller().apply_action(ArticulationAction(joint_positions=joints))

    def step(self):
        self._post_init()
        self._apply_ee_pose(self.target_pos)
        self._apply_gripper()


def main():
    print("[teleop] building scene...", flush=True)
    demo = ClothTeleop()
    print("[teleop] scene built, calling world.reset()", flush=True)
    demo.world.reset(soft=False)
    omni.timeline.get_timeline_interface().play()
    print("[teleop] timeline play — focus the viewport window for keyboard input", flush=True)

    sim_steps = 0
    while simulation_app.is_running() and not demo._quit_requested:
        demo.world.step(render=True)
        if not demo.world.is_playing():
            continue
        sim_steps += 1
        time.sleep(0.02)
        if sim_steps < 30:
            continue
        demo.step()

    print("[teleop] shutting down", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
