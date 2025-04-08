import time
import numpy as np

import robosuite as suite
from robosuite.robots import MobileRobot

MAX_FR = 25  # max frame rate for running simulation

if __name__ == "__main__":

    # Set environment and robot directly
    options = {
        "env_name": "Lift",
        "robots": "Jaco",  # You can also use ["Jaco"] if needed
    }

    # Initialize the environment
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    env.reset()
    env.viewer.set_camera(camera_id=0)

    for robot in env.robots:
        if isinstance(robot, MobileRobot):
            robot.enable_parts(legs=False, base=False)

    # Run visualization loop
    for i in range(10000):
        start = time.time()
        action = np.random.randn(*env.action_spec[0].shape)
        obs, reward, done, _ = env.step(action)
        env.render()

        # Limit frame rate
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)
