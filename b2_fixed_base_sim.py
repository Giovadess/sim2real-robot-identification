"""Run Unitree's B2 SDK2 MuJoCo bridge with the base clamped for SysID tests.

This test-only wrapper uses DDS domain 1 on loopback. It overrides the free
base state after every physics step; it is not a physical rig model.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import types
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


GAMMA_SIM = (Path.home() / "gamma_zero_tienkgun/deployment/unitree_b2/sdk2/unitree_mujoco_b2.py")
BRIDGE_DIR = Path.home() / "gamma_zero_tienkgun/third_party/unitree_mujoco/simulate_python"
GAMMA_SCENE = (Path.home() / "gamma_zero_tienkgun/third_party/unitree_mujoco/unitree_robots/b2/scene.xml")
GAMMA_ROBOT_XML = GAMMA_SCENE.with_name("b2.xml")
MAIN_BOX_SIZE = (0.385, 0.175, 0.235)  # half sizes, 77 x 35 x 47 cm
BELLY_BLOCK_SIZE = (0.15, 0.10, 0.075)  # half sizes, 30 x 20 x 15 cm
MAIN_BOX_TOP = 2.0 * MAIN_BOX_SIZE[2]
BELLY_BLOCK_TOP = MAIN_BOX_TOP + 2.0 * BELLY_BLOCK_SIZE[2]
BASE_HEIGHT = BELLY_BLOCK_TOP + 0.075  # trunk collision half-height; rests on the spacer


def make_rig_scene(temp_dir: Path) -> Path:
    tree = ET.parse(GAMMA_SCENE)
    root = tree.getroot()
    include = root.find("include")
    if include is None:
        raise RuntimeError("Unitree B2 scene has no robot include")
    robot_tree = ET.parse(GAMMA_ROBOT_XML)
    compiler = robot_tree.getroot().find("compiler")
    if compiler is None:
        raise RuntimeError("Unitree B2 model has no compiler element")
    compiler.set("meshdir", str(temp_dir))
    for asset in (GAMMA_SCENE.parent / "assets").iterdir():
        if asset.is_file():
            (temp_dir / asset.name).symlink_to(asset)
    robot_path = temp_dir / "b2_sysid_rig_robot.xml"
    robot_tree.write(robot_path, encoding="utf-8", xml_declaration=True)
    include.set("file", str(robot_path))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Unitree B2 scene has no worldbody")
    for geom in list(worldbody.findall("geom")):
        if geom.get("type") == "box":
            worldbody.remove(geom)
    ET.SubElement(worldbody, "geom", {
        "name": "rig_main_box", "type": "box", "pos": f"0 0 {MAIN_BOX_SIZE[2]}",
        "size": " ".join(map(str, MAIN_BOX_SIZE)), "rgba": "0.20 0.22 0.25 1",
        "friction": "0.8 0.005 0.0001",
    })
    ET.SubElement(worldbody, "geom", {
        "name": "rig_belly_block", "type": "box", "pos": f"0 0 {MAIN_BOX_TOP + BELLY_BLOCK_SIZE[2]}",
        "size": " ".join(map(str, BELLY_BLOCK_SIZE)), "rgba": "0.85 0.85 0.80 1",
        "friction": "0.8 0.005 0.0001",
    })
    path = temp_dir / "b2_sysid_rig_scene.xml"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--viewer", action="store_true", help="Show the MuJoCo simulation")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    config = types.ModuleType("config")
    config.ROBOT = "b2"
    sys.modules["config"] = config
    sys.path.insert(0, str(BRIDGE_DIR))
    from unitree_sdk2py_bridge import UnitreeSdk2Bridge

    received_command = False
    original_handler = UnitreeSdk2Bridge.LowCmdHandler

    def low_cmd_handler(bridge, msg):
        nonlocal received_command
        received_command = True
        original_handler(bridge, msg)

    UnitreeSdk2Bridge.LowCmdHandler = low_cmd_handler
    step = mujoco.mj_step
    home_q = None
    reported_contacts = set()

    def step_with_fixed_base(model, data):
        nonlocal home_q
        if home_q is None:
            home_q = np.array(model.key("home").qpos[7:19], copy=True)
        if not received_command:
            data.qpos[7:19] = home_q
            data.qvel[6:18] = 0.0
        data.qpos[:7] = (0.0, 0.0, BASE_HEIGHT, 1.0, 0.0, 0.0, 0.0)
        data.qvel[:6] = 0.0
        step(model, data)
        for i in range(data.ncon):
            contact = data.contact[i]
            geom_ids = (contact.geom1, contact.geom2)
            names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                          or f"geom#{g}" for g in geom_ids)
            if any(name.startswith("rig_") for name in names) and names not in reported_contacts:
                bodies = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
                               or f"body#{int(model.geom_bodyid[g])}" for g in geom_ids)
                reported_contacts.add(names)
                print(f"RIG CONTACT: {names[0]} [{bodies[0]}] <-> {names[1]} [{bodies[1]}], "
                      f"penetration={max(0.0, -contact.dist):.4f} m at t={data.time:.2f}s "
                      f"pos={np.round(contact.pos, 3).tolist()}", flush=True)
        if not received_command:
            data.qpos[7:19] = home_q
            data.qvel[6:18] = 0.0
        data.qpos[:7] = (0.0, 0.0, BASE_HEIGHT, 1.0, 0.0, 0.0, 0.0)
        data.qvel[:6] = 0.0
        mujoco.mj_forward(model, data)

    mujoco.mj_step = step_with_fixed_base
    rig_temp_dir = tempfile.TemporaryDirectory(prefix="b2_sysid_rig_")
    rig_scene = make_rig_scene(Path(rig_temp_dir.name))
    source = GAMMA_SIM.read_text()
    scene_assignment = 'config.ROBOT_SCENE = str(UNITREE_MUJOCO / "unitree_robots" / "b2" / "scene.xml")'
    if scene_assignment not in source:
        raise RuntimeError("Could not redirect the Gamma simulator to the modeled B2 rig scene")
    source = source.replace(scene_assignment, f"config.ROBOT_SCENE = {str(rig_scene)!r}", 1)
    sys.argv = [str(GAMMA_SIM), "--seconds", str(args.seconds)]
    if args.viewer:
        sys.argv.append("--viewer")
    namespace = {"__name__": "__main__", "__file__": str(GAMMA_SIM)}
    exec(compile(source, str(GAMMA_SIM), "exec"), namespace)


if __name__ == "__main__":
    main()
