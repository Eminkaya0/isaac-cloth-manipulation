# Isaac Sim Humanoid Cloth Manipulation

Demo scripts for NVIDIA **Isaac Sim 5.1.0** showing a Franka Panda (and the
Unitree G1 humanoid) picking up and flipping a particle cloth on a workbench.
Built incrementally, with three different demos plus a one-shot URDF→USD
converter for the Unitree G1.

## What you get

| File | What it does |
| --- | --- |
| `cloth_pick_franka.py` | Wood-toned workbench + three coloured particle cloths + a Franka Panda. The state machine automatically picks the cloth closest to the Franka's home pose, lifts it, flips the wrist 180°, and releases. A static Unitree G1 stands behind the table for scale/scene richness. |
| `cloth_teleop_franka.py` | Same scene, but you drive the Franka end-effector with the keyboard: arrow keys for XY, PageUp/PageDown for Z, **G** to grasp the nearest cloth, **H** to reset to home, **Esc** to quit. |
| `cloth_pick_g1.py` | Removes the Franka; uses the Unitree G1's *actual* right arm + BrainCo dexterous hand to reach the cloth. Pelvis is welded to the world via `UsdPhysics.FixedJoint`, joints are driven via scripted waypoints. Reach is approximate — see "Known Limitations" below. |
| `g1_urdf_import.py` | One-shot script that converts the Unitree G1 (29-DOF + BrainCo hand) URDF into a USD using Isaac Sim's URDF importer. Run this once before `cloth_pick_g1.py`. |

## Requirements

- Isaac Sim 5.1.0 (tested with `5.1.0-rc.19`)
- Linux with NVIDIA GPU (PhysX GPU dynamics is mandatory for particle cloth)
- The Unitree G1 URDF (for the G1 demo only):
  ```bash
  mkdir -p /home/<you>/isaac-sim/robots
  cd /home/<you>/isaac-sim/robots
  git clone --depth 1 https://github.com/unitreerobotics/unitree_ros.git
  ```

## Setup

1. Drop the four `.py` files into Isaac Sim's example tree:
   ```
   cp *.py /path/to/isaac-sim/standalone_examples/api/isaacsim.core.api/
   ```
2. For the G1 demo, convert the URDF to USD once:
   ```
   cd /path/to/isaac-sim
   ./python.sh standalone_examples/api/isaacsim.core.api/g1_urdf_import.py
   ```
   This writes `/home/<you>/isaac-sim/robots/g1_brainco.usd` (plus a
   `configuration/` sub-folder of USD layers).

## Run

```bash
cd /path/to/isaac-sim
./python.sh standalone_examples/api/isaacsim.core.api/cloth_pick_franka.py
# or
./python.sh standalone_examples/api/isaacsim.core.api/cloth_teleop_franka.py
# or
./python.sh standalone_examples/api/isaacsim.core.api/cloth_pick_g1.py
```

The first start of any of them spends ~15–30 s compiling shaders before the
viewport shows the scene.

## Architecture Notes

A few non-obvious things that took time to figure out — written down here so
the next person doesn't repeat the pain.

### Particle cloth requires `World(device="cuda")`

`SingleClothPrim` / `SingleParticleSystem` only initialise when the simulation
view runs on cuda. Setting `World(backend="numpy")` and trying to flip
`enable_gpu_dynamics(True)` afterwards does **not** work — the simulation view
is decided at World init based on the device.

### The Lula IK helper needs numpy warm-starts

With `device="cuda"` every `get_joint_positions()` returns a cuda tensor.
`KinematicsSolver` (Lula) crashes when it tries to pass that into its C++
bindings. The Franka demos monkey-patch the joints view to convert to numpy:

```python
def _patch_kinematics_to_numpy(kin):
    orig = kin._joints_view.get_joint_positions
    def _np_get(): return _to_numpy(orig())
    kin._joints_view.get_joint_positions = _np_get
```

### The old `Franka` helper class breaks on cuda

`isaacsim.robot.manipulators.examples.franka.Franka` instantiates a
`ParallelGripper` whose `initialize()` does
`np.array([t[idx] for idx in ...])` against the cuda joint-position tensor and
dies. We side-step the helper entirely:

```python
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation

add_reference_to_stage(usd_path=FRANKA_USD, prim_path="/World/Franka")
franka = world.scene.add(SingleArticulation(prim_path="/World/Franka", name="franka"))
```

…and drive the finger joints by index in our own little gripper helpers.

### Cloth attachment

`omni.kit.commands.execute("CreatePhysicsAttachment", ...)` is the official
mechanism for pinning particle cloth to a rigid body. The command lives in the
`omni.physx.commands` extension, which has to be enabled explicitly at the top
of a standalone script:

```python
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "omni.physx.commands", True
)
import omni.physxcommands  # noqa: F401  registers the commands
```

Releasing the cloth is `omni.kit.commands.execute("DeletePrims", paths=[...])`.

### `useFabricSceneDelegate=0` vs. URDF-imported visuals

The cloth demos originally launched with
`extra_args=["--/app/useFabricSceneDelegate=0"]` (copied from the stock
`cloth.py` example). With FSD off, the URDF-imported G1 articulation's joint
positions change in PhysX but the Hydra renderer never sees the updates — so
the humanoid looks frozen even though it's "moving". Dropping that flag on the
G1 script is what makes the arm visually animate.

### G1 anchor

`fix_base=True` in the URDF importer connects the pelvis to a world-link
*inside* the imported USD. Once that USD gets referenced into another scene
the constraint silently breaks. The reliable pattern is to import with
`fix_base=False` and add our own `UsdPhysics.FixedJoint` at scene build time
between world (`body0` empty) and the inner `pelvis/pelvis` link with
`localPos0` set to the desired anchor pose.

## Known Limitations

- **G1 arm IK is *not* solved.** `cloth_pick_g1.py` uses hand-tuned joint
  waypoints, so the right palm approaches the cloth but does not land exactly
  on it. A proper fix is either a Lula `robot_description.yaml` for the right
  arm chain or a Jacobian-based IK solver. Moving the project to **Isaac Lab**
  (which has G1 environments + IK preconfigured) is probably the better path.
- The G1 in `cloth_pick_franka.py` is a **static prop** (all rigid bodies +
  collision + articulation API removed at scene build time). It does not
  participate in physics; it's there for visual context.
- Particle cloth is deprecated in PhysX but still works in Isaac Sim 5.1.0.
  The newer Deformable (FEM) cloth path requires a different rig — not
  attempted here.

## Credits

- Unitree G1 URDF + meshes: [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros)
- Isaac Sim: NVIDIA
- Built collaboratively, with lots of trial-and-error joint tuning.
