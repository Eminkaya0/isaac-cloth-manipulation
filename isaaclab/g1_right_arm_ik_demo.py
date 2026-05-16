"""
G1 + Inspire 5-finger hand right-arm IK demo (Isaac Lab).

Cycles the right-hand end-effector through a small list of Cartesian goals,
relying on Isaac Lab's DifferentialIKController to compute the joint motion.
Nothing about the body sign convention or drive stiffness has to be guessed —
the prebuilt G1_INSPIRE_FTP_CFG already has gravity disabled, the root link
welded, and tuned PD gains for grasping.

Run:
    cd /home/emin/IsaacLab
    ./isaaclab.sh -p g1_right_arm_ik_demo.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 right-arm IK demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--ee_body",
    type=str,
    default="right_elbow_roll_link",
    help="Body name to track with the IK controller (right-hand frame).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import G1_MINIMAL_CFG  # uses g1_minimal.usd (no broken joints)


@configclass
class G1ArmSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.90)),
    )
    robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]

    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # End-effector + arm joint subset.  The Inspire-hand variant has joints
    # named right_shoulder_{pitch,roll,yaw}_joint, right_elbow_joint, and
    # right_wrist_{roll,pitch,yaw}_joint.
    # G1_MINIMAL_CFG joints: shoulder_{pitch,roll,yaw} + elbow_{pitch,roll}.
    # No separate wrist joints in the minimal variant — the elbow_roll plays
    # that role.
    right_arm_cfg = SceneEntityCfg(
        "robot",
        joint_names=[
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_elbow_roll_joint",
        ],
        body_names=[args_cli.ee_body],
    )
    right_arm_cfg.resolve(scene)

    if robot.is_fixed_base:
        ee_jacobi_idx = right_arm_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = right_arm_cfg.body_ids[0]

    # Markers for the EE pose and the current goal.
    fm_cfg = FRAME_MARKER_CFG.copy()
    fm_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    ee_marker = VisualizationMarkers(fm_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(fm_cfg.replace(prim_path="/Visuals/ee_goal"))

    # A small tour of goal poses for the right hand.  Position is in the
    # robot's root frame; quaternion in (w, x, y, z) order.  All poses keep
    # the palm roughly horizontal facing down.
    PALM_DOWN = [0.0, 1.0, 0.0, 0.0]
    ee_goals = [
        [ 0.30, -0.20, 0.30, *PALM_DOWN],   # forward, slightly right & down
        [ 0.40, -0.10, 0.10, *PALM_DOWN],   # further forward, lower
        [ 0.20, -0.30, 0.20, *PALM_DOWN],   # pull back & out
        [ 0.30, -0.20, 0.50, *PALM_DOWN],   # raise up
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)

    ik_commands = torch.zeros(scene.num_envs, diff_ik.action_dim, device=robot.device)
    current_goal_idx = 0
    ik_commands[:] = ee_goals[current_goal_idx]
    diff_ik.reset()
    diff_ik.set_command(ik_commands)

    sim_dt = sim.get_physics_dt()
    count = 0

    print(f"[g1-ik] tracking body '{args_cli.ee_body}' (id={right_arm_cfg.body_ids[0]}) "
          f"with {len(right_arm_cfg.joint_ids)} arm joints", flush=True)
    while simulation_app.is_running():
        if count % 200 == 0:
            count = 0
            # Reset robot to default joints + zero velocity.
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            ik_commands[:] = ee_goals[current_goal_idx]
            diff_ik.reset()
            diff_ik.set_command(ik_commands)
            joint_pos_des = joint_pos[:, right_arm_cfg.joint_ids].clone()
            print(f"[g1-ik] goal #{current_goal_idx}: {ee_goals[current_goal_idx].tolist()}", flush=True)
        else:
            jacobian = robot.root_physx_view.get_jacobians()[
                :, ee_jacobi_idx, :, right_arm_cfg.joint_ids
            ]
            ee_pose_w = robot.data.body_pose_w[:, right_arm_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, right_arm_cfg.joint_ids]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            joint_pos_des = diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        robot.set_joint_position_target(joint_pos_des, joint_ids=right_arm_cfg.joint_ids)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        # Update visualization markers.
        ee_pose_w = robot.data.body_state_w[:, right_arm_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 1.0])

    # Workaround for Isaac Lab × Isaac Sim 5.1 — InteractiveScene does not get
    # a usable PhysicsScene auto-created; we add one explicitly before the
    # scene + reset.
    import isaacsim.core.utils.stage as stage_utils
    from pxr import UsdPhysics, Sdf
    UsdPhysics.Scene.Define(stage_utils.get_current_stage(), Sdf.Path("/physicsScene"))

    scene = InteractiveScene(G1ArmSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5))
    sim.reset()
    print("[g1-ik] setup complete", flush=True)
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
