# GSoC_PAL_Robo_Demo
**Brax Training Viewer for Real-Time Policy Visualization**
- ``PART 1``: Using MuJoCo Python bindings, create a simulation with the Ant robot and apply random controls to its joints. Show the simulation using
*mujoco.viewer*.

- ``PART 2``: Create a Python function that gets as input parameters num_envs, env_separation, and ens_per_row; it also receives the MuJoCo XML model and replicates the robot as many times as needed in the same model. Simulate the new replicated model and control all the robots by applying random actions.

- ``PART 3``: Use the Ant Brax environment and train a control policy using Brax’ PPO implementation.
Save the policy and test it in the MuJoCo, updating the code you did in the first bullet point.