"""Minimal training/play launcher for the G1-Reach DirectRLEnv via skrl PPO.

Usage:
    cd /home/emin/IsaacLab
    ./isaaclab.sh -p g1_reach_task/train.py --num_envs 1024 --headless          # train
    ./isaaclab.sh -p g1_reach_task/train.py --num_envs 16 --play                 # eval
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1-Reach training driver.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--play", action="store_true", help="Play a (rolled-out) policy instead of training.")
parser.add_argument("--video", action="store_true", help="Record a small mp4 of the env.")
parser.add_argument("--total_timesteps", type=int, default=200_000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- after sim is up ---
import os
import sys
import torch

# Make the parent dir importable so `import g1_reach_task` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import g1_reach_task  # registers env
import gymnasium as gym
from g1_reach_task.g1_reach_env import G1ReachEnvCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo import PPO_CFG as PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer


class Policy(GaussianMixin, Model):
    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        GaussianMixin.__init__(self)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 128), torch.nn.ELU(),
            torch.nn.Linear(128, self.num_actions),
        )
        self.log_std_parameter = torch.nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        DeterministicMixin.__init__(self)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def main():
    # Build env
    env_cfg = G1ReachEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("G1-Reach-v0", cfg=env_cfg)
    env = SkrlVecEnvWrapper(env)
    env = wrap_env(env, wrapper="isaaclab")  # ensures correct interface

    device = env.device

    models = {
        "policy": Policy(env.observation_space, env.action_space, device),
        "value":  Value(env.observation_space, env.action_space, device),
    }

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 16
    cfg["learning_epochs"] = 8
    cfg["mini_batches"] = 4
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 3e-4
    cfg["learning_rate_scheduler"] = KLAdaptiveLR
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["entropy_loss_scale"] = 0.01
    cfg["experiment"]["directory"] = os.path.join(os.path.dirname(__file__), "runs")
    cfg["experiment"]["experiment_name"] = "g1_reach_ppo"

    memory = RandomMemory(memory_size=cfg["rollouts"], num_envs=env.num_envs, device=device)
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    trainer_cfg = {
        "timesteps": args_cli.total_timesteps,
        "headless": True,
    }
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    if args_cli.play:
        # find latest checkpoint
        runs_dir = cfg["experiment"]["directory"]
        if os.path.isdir(runs_dir):
            ckpts = sorted(
                [p for r, _, fs in os.walk(runs_dir) for p in
                 [os.path.join(r, f) for f in fs if f.endswith(".pt")]],
                key=os.path.getmtime,
            )
            if ckpts:
                agent.load(ckpts[-1])
                print(f"[g1-reach] loaded {ckpts[-1]}", flush=True)
        trainer.eval()
    else:
        trainer.train()


main()
simulation_app.close()
