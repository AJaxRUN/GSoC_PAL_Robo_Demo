import mujoco
import numpy as np
from mujoco import viewer
import time

model = mujoco.MjModel.from_xml_path("ant.xml")
data = mujoco.MjData(model)

timestep = model.opt.timestep
print(f"Simulation timestep: {timestep} seconds")

with viewer.launch_passive(model, data) as v:
    start_time = time.time()
    sim_time = 0.0
    for _ in range(10000):
        action = np.random.uniform(-1, 1, model.nu)
        data.ctrl[:] = action

        sim_time += timestep
        mujoco.mj_step(model, data)

        elapsed_time = time.time() - start_time
        sleep_duration = sim_time - elapsed_time
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        v.sync()