"""Headless eval — rolls out the trained G1-Reach policy for a configurable
number of full episodes and logs:
    - mean / median end-of-episode palm→target distance
    - reach success rate (final distance < threshold)
    - per-step mean distance trajectory (first env, sampled)

Usage:
    cd /home/emin/IsaacLab
    ./isaaclab.sh -p g1_reach_task/eval_distance.py \
        --checkpoint /home/emin/IsaacLab/logs/skrl/g1_reach_direct/<run>/checkpoints/best_agent.pt \
        --num_envs 64 --episodes 10
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval G1-Reach policy distance metrics.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=10,
                    help="Number of full episodes per env to roll out.")
parser.add_argument("--threshold", type=float, default=0.08,
                    help="Distance under which an episode is counted as reached.")
parser.add_argument("--task", type=str, default="Isaac-G1-Reach-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# force headless
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- after sim ----
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401  triggers task registry incl. G1-Reach
from isaaclab_tasks.utils import parse_env_cfg

from skrl.utils.runner.torch import Runner
import yaml


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    # Underlying DirectRLEnv (needed for direct access to .robot etc.)
    env = gym_env.unwrapped

    # Wrap the SAME env for skrl Runner consumption.
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    wrapped = SkrlVecEnvWrapper(gym_env)

    cfg_yaml_path = os.path.join(
        os.path.dirname(__file__),
        "..", "source", "isaaclab_tasks", "isaaclab_tasks",
        "direct", "g1_reach", "agents", "skrl_ppo_cfg.yaml",
    )
    with open(cfg_yaml_path) as f:
        agent_cfg = yaml.safe_load(f)
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    runner = Runner(wrapped, agent_cfg)
    runner.agent.load(os.path.abspath(args_cli.checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)
    policy = runner.agent

    obs, _ = wrapped.reset()

    ep_count = 0
    final_distances: list[float] = []
    ep_steps = 0
    max_ep_steps = env.max_episode_length

    print(f"[eval] num_envs={args_cli.num_envs} target_episodes_per_env={args_cli.episodes}",
          flush=True)

    debug_step = 0
    while ep_count < args_cli.episodes * args_cli.num_envs:
        with torch.inference_mode():
            outputs = policy.act(obs, states=None, timestep=0, timesteps=0)
            # deterministic mean action for eval
            actions = outputs[-1].get("mean_actions", outputs[0])
            if debug_step < 5:
                print(f"[debug] step {debug_step} obs[0]={obs[0,:5].cpu().tolist()}  "
                      f"actions[0]={actions[0].cpu().tolist()}", flush=True)
                debug_step += 1
        obs, _rewards, terminated, truncated, _info = wrapped.step(actions)
        ep_steps += 1

        if torch.any(terminated | truncated):
            ee_pos = env.robot.data.body_pos_w[:, env._ee_body_id] - env.robot.data.root_pos_w
            dist = torch.norm(ee_pos - env._target_pos_e, dim=-1).cpu().tolist()
            done_mask = (terminated | truncated).cpu().tolist()
            for d, done in zip(dist, done_mask):
                if done:
                    final_distances.append(float(d))
                    ep_count += 1

    fd = torch.tensor(final_distances)
    success = (fd < args_cli.threshold).float().mean().item()
    print(f"[eval] episodes={len(final_distances)}", flush=True)
    print(f"[eval]   mean   = {fd.mean().item():.4f} m", flush=True)
    print(f"[eval]   median = {fd.median().item():.4f} m", flush=True)
    print(f"[eval]   p90    = {fd.quantile(0.90).item():.4f} m", flush=True)
    print(f"[eval]   min    = {fd.min().item():.4f} m", flush=True)
    print(f"[eval]   max    = {fd.max().item():.4f} m", flush=True)
    print(f"[eval]   success rate (<{args_cli.threshold}m) = {success * 100:.1f}%",
          flush=True)


main()
simulation_app.close()
