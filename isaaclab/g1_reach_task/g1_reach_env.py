"""
DirectRLEnv: Unitree G1 (29-DOF, fix-rooted) learning to reach a randomly
sampled 3D point with its right palm.

Run training (skrl PPO):
    cd /home/emin/IsaacLab
    ./isaaclab.sh -p g1_reach_task/train.py --task G1-Reach-v0 --num_envs 1024 --headless

Run a trained policy:
    ./isaaclab.sh -p g1_reach_task/train.py --task G1-Reach-v0 --num_envs 16 --play
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from isaaclab_assets import G1_CFG


# Joint subset we control (5-DOF G1 right arm in the Lab-shipped USD).
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]

# End-effector body in the current G1 USD (after merging fingers / no
# explicit palm).  right_elbow_roll_link is the last arm link.
EE_BODY = "right_elbow_roll_link"


@configclass
class G1ReachEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 6.0
    decimation = 2
    # 5 arm joints — policy outputs delta on each, scaled
    action_space = 5
    # obs: arm joint pos (5) + arm joint vel (5) + EE pos (3) + target pos (3)
    observation_space = 16
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=2.5, replicate_physics=True
    )

    # Force the robot's root link to be fixed so G1 stands without policy.
    robot: ArticulationCfg = G1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    robot.spawn.articulation_props.fix_root_link = True

    # Table in front of the humanoid (workbench dimensions in metres).
    table_size = (0.50, 0.80, 0.05)   # x_depth, y_width, z_thickness
    table_pos = (0.45, 0.0, 0.85)     # centre, in env-local frame
    table_top_z = table_pos[2] + table_size[2] / 2.0   # ≈ 0.875

    # Reach target sampling region — XY on the table top, Z just above it.
    target_xyz_min = (0.20, -0.30, table_top_z + 0.01)
    target_xyz_max = (0.45,  0.30, table_top_z + 0.05)

    # Reward shaping
    # Absolute joint-pos action: target = default + action * scale.
    # Action ∈ [-1, 1] (clamped), so joints stay within default ± 1.5 rad
    # (~86°) — wide enough to reach the table, narrow enough to not blow up.
    action_scale = 1.5
    reach_threshold = 0.07
    reach_bonus = 30.0
    action_cost_scale = 0.005


class G1ReachEnv(DirectRLEnv):
    cfg: G1ReachEnvCfg

    def __init__(self, cfg: G1ReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_joint_ids, _ = self.robot.find_joints(RIGHT_ARM_JOINTS, preserve_order=True)
        self._ee_body_id = self.robot.find_bodies([EE_BODY])[0][0]
        self._action_scale = self.cfg.action_scale

        # Targets per env, in env-local frame.
        self._target_pos_e = torch.zeros(self.num_envs, 3, device=self.device)

        # Cached default arm joint positions
        self._arm_default = self.robot.data.default_joint_pos[:, self._arm_joint_ids].clone()
        # Running joint-target buffer (cumulative delta from default).
        self._arm_joint_target = self._arm_default.clone()

        # Visualization marker for the target
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.10, 0.10, 0.10)
        self._target_marker = VisualizationMarkers(
            marker_cfg.replace(prim_path="/Visuals/reach_target")
        )

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        # Workbench removed for the basic reach stage — the visual cube was
        # still casting a collision shape that the arm bumped into.  Add the
        # table back (with explicit no-collision) once we wire the cloth task.

        self.scene.clone_environments(copy_from_source=False)
        spawn_ground = sim_utils.GroundPlaneCfg()
        spawn_ground.func("/World/defaultGroundPlane", spawn_ground)
        spawn_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.90))
        spawn_light.func("/World/Light", spawn_light)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # Absolute target = default + action * scale.  Stays in a bounded
        # region around the rest pose so the policy can't run away.
        target = self._arm_default + self._actions * self._action_scale
        self.robot.set_joint_position_target(target, joint_ids=self._arm_joint_ids)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        arm_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        arm_vel = self.robot.data.joint_vel[:, self._arm_joint_ids]
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_id]
        # convert to env-local frame (same frame as self._target_pos_e)
        ee_pos_e = ee_pos_w - self.scene.env_origins

        obs = torch.cat([arm_pos, arm_vel, ee_pos_e, self._target_pos_e], dim=-1)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_id]
        ee_pos_e = ee_pos_w - self.scene.env_origins
        dist = torch.norm(ee_pos_e - self._target_pos_e, dim=-1)

        reach_bonus = (dist < self.cfg.reach_threshold).float() * self.cfg.reach_bonus
        action_cost = torch.sum(self._actions ** 2, dim=-1) * self.cfg.action_cost_scale

        reward = -dist + reach_bonus - action_cost
        return reward

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset robot to default state
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # Reset the running arm target back to default on reset
        self._arm_joint_target[env_ids] = self._arm_default[env_ids]

        # Resample target in env-local frame
        mins = torch.tensor(self.cfg.target_xyz_min, device=self.device)
        maxs = torch.tensor(self.cfg.target_xyz_max, device=self.device)
        self._target_pos_e[env_ids] = sample_uniform(mins, maxs, (len(env_ids), 3), self.device)

        # Visualize: marker world pos = env origin + local target
        env_origins = self.scene.env_origins[env_ids]
        self._target_marker.visualize(
            translations=env_origins + self._target_pos_e[env_ids],
        )

        # one-shot debug: print first env's palm pose at reset, plus target.
        if not getattr(self, "_debug_printed", False) and 0 in env_ids:
            ee_pos_w = self.robot.data.body_pos_w[0, self._ee_body_id]
            env_origin = self.scene.env_origins[0]
            ee_local = ee_pos_w - env_origin
            print(f"[g1-reach][DEBUG] env0 default state:", flush=True)
            print(f"[g1-reach][DEBUG]   env_origin = {env_origin.tolist()}", flush=True)
            print(f"[g1-reach][DEBUG]   palm_w   = {ee_pos_w.tolist()}", flush=True)
            print(f"[g1-reach][DEBUG]   palm_e   = {ee_local.tolist()}", flush=True)
            print(f"[g1-reach][DEBUG]   target_e = {self._target_pos_e[0].tolist()}", flush=True)
            print(f"[g1-reach][DEBUG]   dist     = {torch.norm(ee_local - self._target_pos_e[0]).item():.3f}", flush=True)
            self._debug_printed = True


# Registration handled in package __init__.py
