import mujoco
import numpy as np
from mujoco import viewer
import jax.numpy as jp
import jax
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
import os
from brax.io import model
import xml.etree.ElementTree as ET
from part_2.utils import rename_elements
from copy import deepcopy
import time


env = envs.create("ant", backend="positional")
make_inference_fn, params, _  = ppo.train(
    environment=env,
    num_timesteps=1000,          
    num_evals=1,
    reward_scaling=10,
    episode_length=100,
    normalize_observations=True,
    action_repeat=1,
    batch_size=32,
    num_minibatches=4,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    discounting=0.99,
    unroll_length=5,
    progress_fn=lambda step, metrics: print(f"Step {step}: {metrics}"),
    seed=0,
)
model.save_params('ant_policy', params)
print("Training complete. Policy saved.")

# Step 2: Test the Policy in MuJoCo with Replicated Model
def replicate_ant_model(xml_path, num_envs, env_separation, ens_per_row):
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
        row = i // ens_per_row
        col = i % ens_per_row
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


num_envs = 6
env_separation = 2.0
ens_per_row = 3

new_xml_path = replicate_ant_model("ant.xml", num_envs, env_separation, ens_per_row)

# params = model.load_params('ant_policy')
inference_fn = make_inference_fn(params)
key = jax.random.PRNGKey(0)

mj_model = mujoco.MjModel.from_xml_path(new_xml_path)
data = mujoco.MjData(mj_model)
timestep = mj_model.opt.timestep

def get_obs(data):
    qpos = data.qpos[7:22]
    qvel = data.qvel[6:18]
    obs = np.concatenate([qpos, qvel])
    return obs

with viewer.launch_passive(mj_model, data) as v:
    start_time = time.time()
    sim_time = 0.0
    for _ in range(10000):
        obs = get_obs(data)
        action = inference_fn(jp.array([obs]), key)[0]
        print("Action:", action)
        full_action = np.tile(action, num_envs)
        data.ctrl[:] = full_action
        
        mujoco.mj_step(mj_model, data)
        sim_time += timestep
        
        elapsed_time = time.time() - start_time
        sleep_duration = sim_time - elapsed_time
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        
        v.sync()
        
        if data.time > 10:
            mujoco.mj_resetData(mj_model, data)
            sim_time = 0.0
            start_time = time.time()