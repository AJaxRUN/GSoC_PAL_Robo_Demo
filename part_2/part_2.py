import mujoco
import numpy as np
from mujoco import viewer
import xml.etree.ElementTree as ET
from copy import deepcopy
import time
from utils import rename_elements

# Parameters
num_envs = 6
env_separation = 2.0
envs_per_row = 3

def replicate_ant_model(xml_path, num_envs, env_separation, envs_per_row):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    ant_body = worldbody.find(".//body[@name='torso']")

    torso_cameras = ant_body.findall(".//camera")
    track_camera = None
    for cam in torso_cameras:
        if cam.get("name") == "track":
            track_camera = cam
            ant_body.remove(cam)
            worldbody.append(cam)
            break

    worldbody_cameras = worldbody.findall(".//camera")
    track_cameras = [cam for cam in worldbody_cameras if cam.get("name") == "track"]
    if len(track_cameras) > 1:
        for cam in track_cameras[1:]:
            worldbody.remove(cam)
    elif len(track_cameras) == 0 and track_camera is None:
        ET.SubElement(worldbody, "camera", name="track", mode="trackcom", pos="0 -3 3", xyaxes="1 0 0 0 1 0")

    for i in range(1, num_envs):
        new_ant = deepcopy(ant_body)
        new_ant.set("name", f"ant_{i}")
        row = i // envs_per_row
        col = i % envs_per_row
        x = col * env_separation
        y = row * env_separation
        new_ant.set("pos", f"{x} {y} 0.4")
        rename_elements(new_ant, i)
        worldbody.append(new_ant)

    actuator = root.find("actuator")
    original_motors = actuator.findall("motor")
    for i in range(1, num_envs):
        for j, motor in enumerate(original_motors):
            new_motor = deepcopy(motor)
            orig_name = motor.get("name")
            if orig_name:
                new_motor.set("name", f"{orig_name}_{i}")
            else:
                joint_name = motor.get("joint")
                if joint_name:
                    new_motor.set("name", f"{joint_name}_motor_{i}")
            new_motor.set("joint", f"{motor.get('joint')}_{i}")
            actuator.append(new_motor)

    new_xml_path = "replicated_ant.xml"
    tree.write(new_xml_path)
    return new_xml_path


new_xml_path = replicate_ant_model("ant.xml", num_envs, env_separation, envs_per_row)

model = mujoco.MjModel.from_xml_path(new_xml_path)
data = mujoco.MjData(model)

timestep = model.opt.timestep
print(f"Simulation timestep: {timestep} seconds")

with viewer.launch_passive(model, data) as v:
    start_time = time.time()
    sim_time = 0.0
    for _ in range(10000):
        action = np.random.uniform(-1, 1, model.nu)
        data.ctrl[:] = action

        mujoco.mj_step(model, data)
        sim_time += timestep

        elapsed_time = time.time() - start_time
        sleep_duration = sim_time - elapsed_time
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        v.sync()