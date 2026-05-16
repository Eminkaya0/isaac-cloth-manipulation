# Isaac Lab port (WIP)

Second-stage scaffolding for the cloth-manipulation project, now built on top
of **NVIDIA Isaac Lab** (`main` branch, paired with Isaac Sim 5.1.0).

## What's in this folder

```
isaaclab/
├── g1_right_arm_ik_demo.py        # standalone IK demo: G1 + DifferentialIKController
└── g1_reach_task/                  # DirectRLEnv + skrl PPO training task
    ├── g1_reach_env.py             # the env class + cfg
    ├── __init__direct.py           # gym.register(...) — copy to direct/g1_reach/__init__.py
    ├── eval_distance.py            # headless rollout, mean/median/success metrics
    ├── train.py                    # legacy custom train driver (kept for reference)
    └── agents/
        └── skrl_ppo_cfg.yaml       # skrl PPO config (matches Lab idioms)
```

## Where these files live in a real Isaac Lab checkout

Files that belong inside Isaac Lab's task tree (so `Isaac-G1-Reach-v0` shows
up in `list_envs.py`):

```
$IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/g1_reach/
├── __init__.py                     # (this repo: isaaclab/g1_reach_task/__init__direct.py)
├── g1_reach_env.py
└── agents/
    ├── __init__.py                 # empty file
    └── skrl_ppo_cfg.yaml
```

The companion driver/eval scripts (`eval_distance.py`, etc.) sit anywhere on
`PYTHONPATH`. We dropped them at `$IsaacLab/g1_reach_task/`.

## Setup

```bash
# 1. Clone & install Isaac Lab against an existing Isaac Sim 5.1 install
git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
ln -sf /path/to/isaac-sim ~/IsaacLab/_isaac_sim
cd ~/IsaacLab && ./isaaclab.sh -i

# 2. Copy task files into the Lab tree
mkdir -p ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/g1_reach/agents
cp isaaclab/g1_reach_task/g1_reach_env.py \
   ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/g1_reach/
cp isaaclab/g1_reach_task/__init__direct.py \
   ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/g1_reach/__init__.py
cp isaaclab/g1_reach_task/agents/skrl_ppo_cfg.yaml \
   ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/g1_reach/agents/
touch ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/g1_reach/agents/__init__.py

# 3. Copy eval helper to a path Lab's python can import
mkdir -p ~/IsaacLab/g1_reach_task
cp isaaclab/g1_reach_task/eval_distance.py isaaclab/g1_reach_task/__init__.py \
   ~/IsaacLab/g1_reach_task/
```

## Run

```bash
cd ~/IsaacLab

# Train
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-G1-Reach-v0 --num_envs 1024 --headless --max_iterations 1500

# Headless eval (mean / median / success rate over N episodes)
./isaaclab.sh -p g1_reach_task/eval_distance.py \
    --checkpoint logs/skrl/g1_reach_direct/<run>/checkpoints/best_agent.pt \
    --num_envs 32 --episodes 5 --headless

# Visualise in GUI
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
    --task Isaac-G1-Reach-v0 --num_envs 9 \
    --checkpoint logs/skrl/g1_reach_direct/<run>/checkpoints/best_agent.pt
```

The standalone IK demo `g1_right_arm_ik_demo.py` runs without RL — drop it
directly into `~/IsaacLab/` and:

```bash
./isaaclab.sh -p g1_right_arm_ik_demo.py
```

## Status

Infrastructure works end-to-end:

- ✅ Isaac Lab + Isaac Sim 5.1 pinned via `_isaac_sim` symlink
- ✅ `Isaac-G1-Reach-v0` DirectRLEnv registered; G1 spawns with welded root,
  gravity off, ImplicitActuator PD gains from `G1_CFG`
- ✅ skrl PPO trains the 5-DOF right-arm joints (shoulder pitch/roll/yaw +
  elbow pitch/roll); ~5 min for 1500 iterations × 1024 envs on a single
  RTX 5070
- ✅ Headless `eval_distance.py` rolls the policy and reports mean / median /
  p90 / success-rate
- ✅ Known Isaac Sim 5.1 PhysX workaround applied in
  `g1_right_arm_ik_demo.py`: manual `UsdPhysics.Scene.Define()` before
  `sim.reset()` (Lab issue #2827)

**What does not work yet:**

The PPO reward shape we hand-rolled (`-dist + reach_bonus*[dist<7cm] -
action_cost`) does not converge to actual reaching — across multiple action
parameterisations (absolute, cumulative, anchored-default-delta) the policy
collapses to a saturated extreme arm pose and stays there.  Eval reports
mean palm-target distance ≈ 1.0 m and 0 % success.

The fix for next session is to lift the reward shaping from Isaac Lab's
stock `manager_based/manipulation/reach/` Franka env (proven to train
reaching policies in under an hour) and just port it to G1's joint set.
The action-space wiring + skrl Runner setup is already wired through, so
that's the only piece that needs rework.

## File map at the time of commit

- `g1_reach_env.py` ends with absolute-pos action: `target = default +
  action * 1.5`.
- Reward = `-dist + 30 * (dist < 0.07) - 0.005 * |action|²`.
- Episode 6 s, target sampled at table-top height (0.88–0.93 m) in env-local
  frame in front of the humanoid.
- The visual workbench was removed at the last iteration to isolate
  reaching from collision-avoidance; the comment in `_setup_scene` shows
  where to re-add it.
