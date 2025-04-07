import mujoco
import numpy as np
from mujoco import viewer
import jax
import jax.numpy as jp
from brax import envs
from brax.training.agents.ppo import train as ppo
from flax.training import orbax_utils
import orbax.checkpoint as ocp
import os
import xml.etree.ElementTree as ET
from copy import deepcopy


# Step 1: Train PPO Policy with Brax
env = envs.create("ant", backend="generalized")  # Use Brax Ant environment
train_fn = ppo.make_train(
    environment=env,
    num_timesteps=10_000_000,  # Adjust as needed
    num_evals=10,
    reward_scaling=1.0,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    batch_size=256,
    num_minibatches=32,
    ppo_entropy_cost=1e-2,
)

# Callback to save policy
ckpt_path = "ant_policy_ckpt"
os.makedirs(ckpt_path, exist_ok=True)

def save_policy(step, make_policy, params):
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = os.path.join(ckpt_path, f"step_{step}")
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)

# Train the policy
train_output = train_fn(
    policy_params_fn=save_policy,
    random_key=jax.random.PRNGKey(0),
)
print("Training complete. Policy saved.")

# Extract the inference function and parameters
make_policy, params, metrics = train_output
inference_fn = make_policy(params["normalizer"], params["policy"])

# Step 2: Test the Policy in MuJoCo with Replicated Model
def replicate_ant_model(xml_path, num_envs, env_separation, envs_per_row):
    # Same function as in Part 2
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    ant_body = next(body for body in worldbody.findall("body") if body.find("geom") is not None)
    
    for i in range(1, num_envs):
        new_ant = deepcopy(ant_body)
        new_ant.set("name", f"ant_{i}")
        row = i // envs_per_row
        col = i % envs_per_row
        x = col * env_separation
        y = row * env_separation
        new_ant.set("pos", f"{x} {y} 0.4")
        for joint in new_ant.findall(".//joint"):
            joint.set("name", f"{joint.get('name')}_{i}")
        for geom in new_ant.findall(".//geom"):
            geom.set("name", f"{geom.get('name')}_{i}")
        worldbody.append(new_ant)
    
    actuator = root.find("actuator")
    original_motors = actuator.findall("motor")
    for i in range(1, num_envs):
        for motor in original_motors:
            new_motor = deepcopy(motor)
            new_motor.set("name", f"{motor.get('name')}_{i}")
            new_motor.set("joint", f"{motor.get('joint')}_{i}")
            actuator.append(new_motor)
    
    new_xml_path = "replicated_ant.xml"
    tree.write(new_xml_path)
    return new_xml_path

# Replicate the model
num_envs = 6
env_separation = 2.0
envs_per_row = 3
new_xml_path = replicate_ant_model("ant.xml", num_envs, env_separation, envs_per_row)

# Load the replicated model
model = mujoco.MjModel.from_xml_path(new_xml_path)
data = mujoco.MjData(model)

# Observation function compatible with Brax
def get_obs(data):
    qpos = data.qpos[7:]  # Skip root joint (free joint: 7 DoF)
    qvel = data.qvel[6:]  # Skip root joint velocities
    return np.concatenate([qpos, qvel])

# Simulation with trained policy
with viewer.launch_passive(model, data) as v:
    for _ in range(10000):
        # Get observation for each ant (simplified: apply to first ant's state)
        obs = get_obs(data)
        
        # Apply policy (Brax expects batched input, so add batch dimension)
        action = inference_fn(jp.array([obs]))[0]  # Remove batch dimension
        
        # Repeat action for all ants (8 controls per ant)
        full_action = np.tile(action, num_envs)
        data.ctrl[:] = full_action
        
        mujoco.mj_step(model, data)
        v.sync()
        
        if data.time > 10:
            mujoco.mj_resetData(model, data)