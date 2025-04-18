import time
import numpy as np

import robosuite as suite
from robosuite.robots import MobileRobot

MAX_FR = 25  # max frame rate for running simulation

if __name__ == "__main__":

    # Set environment and robot directly
    options = {
        "env_name": "Lift",
        "robots": "Panda",  # You can also use ["Jaco"] if needed
    }

    # Initialize the environment
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
        control_freq=20,
    )

    env.reset()
    env.viewer.set_camera(camera_id=0)

    for robot in env.robots:
        if isinstance(robot, MobileRobot):
            robot.enable_parts(legs=False, base=False)

    learning_rate = 0.1
    previous_reward = -np.inf
    last_action = np.zeros(env.action_spec[0].shape)
    exploration_noise_std = 0.01

    # Run visualization loop
    for i in range(10000):
        start = time.time()
        action = np.zeros(*env.action_spec[0].shape)
        cube_pos = env.sim.data.body_xpos[env.cube_body_id]
        gripper = env.robots[0].gripper['right']
        eef_pos = env.sim.data.get_site_xpos(gripper.important_sites["grip_site"])
        
        #env.robots[0]._hand_pos
        action[:3] = np.array(cube_pos) - np.array(eef_pos)
        #np.random.randn(*env.action_spec[0].shape) #TODO: set the actions
        obs, reward, done, _ = env.step(action)
        env.render()
        # print("Done:", done)
        #print("Done: ", done)

        if done:
            env.reset()
            previous_reward = -np.inf
            last_action = np.zeros(env.action_spec[0].shape)

        # Limit frame rate
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)