import mujoco
import numpy as np
from mujoco import viewer
import jax.numpy as jp
import jax
from brax import envs
from brax.training.agents.ppo import train as ppo_train
from brax.training.agents.ppo import networks as ppo_networks
from flax.training import orbax_utils
import orbax.checkpoint as ocp
import os
import xml.etree.ElementTree as ET
from copy import deepcopy
import time

env = envs.create("ant", backend="generalized")

ckpt_path = os.path.abspath("ant_policy_ckpt")
os.makedirs(ckpt_path, exist_ok=True)

def save_policy(step, params, metrics):
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = os.path.join(ckpt_path, f"step_{step}")
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)

env = envs.create("ant", backend="generalized")
make_policy, network_params, metrics = ppo_train.train(
    environment=env,
    num_timesteps=10_000_000,
    episode_length=1000,
    action_repeat=1,
    num_envs=8,
    num_eval_envs=4,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    discounting=0.99,
    seed=0,
    unroll_length=10,
    batch_size=128,
    num_minibatches=8,
    num_updates_per_batch=4,
    num_evals=10,
    num_resets_per_eval=0,
    normalize_observations=True,
    reward_scaling=10.0,
    clipping_epsilon=0.2,
    gae_lambda=0.95,
    deterministic_eval=True,
    network_factory=ppo_networks.make_ppo_networks,
    progress_fn=lambda step, metrics: print(f"Step {step}: {metrics}"),
    policy_params_fn=lambda step, make_policy, params: save_policy(step, params, None),
)
print("Training complete. Policy saved.")

inference_fn = make_policy(network_params, deterministic=True)
key = jax.random.PRNGKey(0)

# Step 2: Test the Policy in MuJoCo with Replicated Model
def replicate_ant_model(xml_path, num_envs, env_separation, envs_per_row):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    ant_body = worldbody.find(".//body[@name='torso']")
    if ant_body is None:
        raise ValueError("No 'torso' body found in XML")

    torso_cameras = ant_body.findall(".//camera")
    for cam in torso_cameras:
        if cam.get("name") == "track":
            ant_body.remove(cam)
            worldbody.append(cam)
            break

    worldbody_cameras = worldbody.findall(".//camera")
    track_cameras = [cam for cam in worldbody_cameras if cam.get("name") == "track"]
    if len(track_cameras) > 1:
        for cam in track_cameras[1:]:
            worldbody.remove(cam)
    elif len(track_cameras) == 0:
        ET.SubElement(worldbody, "camera", name="track", mode="trackcom", pos="0 -3 3", xyaxes="1 0 0 0 1 0")

    def rename_elements(element, suffix):
        if element.tag == "body" and element.get("name"):
            element.set("name", f"{element.get('name')}_{suffix}")
        if element.tag == "joint" and element.get("name"):
            element.set("name", f"{element.get('name')}_{suffix}")
        if element.tag == "geom" and element.get("name"):
            element.set("name", f"{element.get('name')}_{suffix}")
        if element.tag == "site" and element.get("name"):
            element.set("name", f"{element.get('name')}_{suffix}")
        if element.tag == "camera":
            element.getparent().remove(element)
        for child in element:
            rename_elements(child, suffix)

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
        for motor in original_motors:
            new_motor = deepcopy(motor)
            orig_name = motor.get("name")
            if orig_name:
                new_motor.set("name", f"{orig_name}_{i}")
            new_motor.set("joint", f"{motor.get('joint')}_{i}")
            actuator.append(new_motor)

    new_xml_path = "replicated_ant.xml"
    tree.write(new_xml_path)
    return new_xml_path

num_envs = 6
env_separation = 2.0
envs_per_row = 3
new_xml_path = replicate_ant_model("ant.xml", num_envs, env_separation, envs_per_row)

model = mujoco.MjModel.from_xml_path(new_xml_path)
data = mujoco.MjData(model)
timestep = model.opt.timestep

def get_obs(data):
    # Brax "ant" observation: 15 qpos (excluding root x, y, z, quat) + 12 qvel
    qpos = data.qpos[7:22]  # 15 joint positions for the first ant (torso + 4 legs)
    qvel = data.qvel[6:18]  # 12 joint velocities for the first ant
    obs = np.concatenate([qpos, qvel])  # Shape: (27,)
    return obs

with viewer.launch_passive(model, data) as v:
    start_time = time.time()
    sim_time = 0.0
    for _ in range(10000):
        obs = get_obs(data)
        # Call inference_fn with just observations since deterministic=True
        action = inference_fn(jp.array([obs]), key)[0]
        full_action = np.tile(action, num_envs)
        data.ctrl[:] = full_action
        
        mujoco.mj_step(model, data)
        sim_time += timestep
        
        elapsed_time = time.time() - start_time
        sleep_duration = sim_time - elapsed_time
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        
        v.sync()
        
        if data.time > 10:
            mujoco.mj_resetData(model, data)
            sim_time = 0.0
            start_time = time.time()