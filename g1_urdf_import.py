"""
One-shot URDF -> USD converter for the Unitree G1 (29-DOF, BrainCo hand
variant).  Writes /home/emin/isaac-sim/robots/g1_brainco.usd which can then be
referenced into other scenes.

Run from /home/emin/isaac-sim/ with:
    ./python.sh standalone_examples/api/isaacsim.core.api/g1_urdf_import.py
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import os

import omni.kit.commands


URDF_PATH = "/home/emin/isaac-sim/robots/unitree_ros/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf"
DEST_USD = "/home/emin/isaac-sim/robots/g1_brainco.usd"


def main():
    assert os.path.exists(URDF_PATH), f"URDF not found: {URDF_PATH}"
    print(f"[urdf] importing {URDF_PATH}", flush=True)

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    assert status, "URDFCreateImportConfig failed"
    import_config.merge_fixed_joints = True
    # Don't fix_base inside the USD — when this USD is re-referenced into
    # another scene the world-link target gets ambiguous and the articulation
    # silently fails to simulate.  We add a UsdPhysics.FixedJoint at scene
    # build time instead.
    import_config.fix_base = False
    import_config.import_inertia_tensor = False
    import_config.distance_scale = 1.0
    import_config.convex_decomp = False
    import_config.self_collision = False
    import_config.make_default_prim = True
    # Drive parameters — without these joints behave like floppy ropes.
    # 1 = position drive, strength = stiffness, damping = damping.
    import_config.set_default_drive_type(1)
    import_config.set_default_drive_strength(400.0)
    import_config.set_default_position_drive_damping(40.0)

    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=URDF_PATH,
        import_config=import_config,
        dest_path=DEST_USD,
        get_articulation_root=True,
    )
    print(f"[urdf] status={status} prim_path={prim_path}", flush=True)
    print(f"[urdf] saved -> {DEST_USD}", flush=True)
    print(f"[urdf] size  -> {os.path.getsize(DEST_USD)} bytes", flush=True)


main()
simulation_app.close()
