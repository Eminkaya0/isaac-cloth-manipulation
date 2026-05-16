"""G1 right-arm reach DirectRLEnv (custom)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-G1-Reach-v0",
    entry_point=f"{__name__}.g1_reach_env:G1ReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_reach_env:G1ReachEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
