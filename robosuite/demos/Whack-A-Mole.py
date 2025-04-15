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
        # cube_pos = env.sim.data.body_xpos[env.cube_body_id]
        # eef_pos = env.robots[0]._hand_pos
        action = np.random.randn(*env.action_spec[0].shape) #TODO: set the actions
            # last_action = np.random.randn(*env.action_spec[0].shape)


            # action[:3] += np.random.normal(0, exploration_noise_std, 3)
            # #action = np.clip(action, *env.action_spec[0].low(lower bound of array??), env.action_spec[0].high(upper bound??)) #trying to change to have lower and upper bounds of the action space dimension
            # print("action attributes", dir(env.action_spec[0])) #TODO: checking attributes to change above line


            # reward_diff = reward - previous_reward
            # if reward_diff > 0:
            #     last_action[:3] += learning_rate * last_action[:3]
            # elif reward_diff <= 0:
            #     last_action[:3] -= learning_rate * last_action[:3]
            

            # last_action[3] = 0.0
            # last_action = np.clip(last_action, env.action_spec[0].low, env.action_spec[0].high)

            # previous_reward = reward`

        # eef_pos = np.array(env.robots[0]._hand_pos["right"])
        # cube_pos = env.sim.data.body_xpos[env.cube_body_id]
        #print("cube_pos:", cube_pos, type(cube_pos))
        #print("eef_pos:", eef_pos, type(eef_pos))
            # direction = eef_pos - cube_pos
            # direction = 0.05 * direction / np.linalg.norm(direction)
        # direction = eef_pos - cube_pos
        # norm_direction = np.linalg.norm(direction)
        # if norm_direction > 0:
        #     direction = 0.05 * direction / norm_direction
        # else:
        #     direction = np.zeros(3)

        # action = np.zeros(env.action_spec[0].shape)
        # action[:3] = direction
        #action[3] = 0.0  # keep gripper open



        #gripper_pos = env.sim.data.get_site_xpos(robot.gripper.important_sites["grip_site"])
        #direction = cube_pos - eef_pos
        #is this already done for us??
        #direction = 0.05 * direction / np.linalg.norm(direction) # basically square root of (x² + y² + z²) to get the norm
        #action = np.zeros(env.action_spec[0].shape) #this basically zeros it
        #action[:3] = direction  # move toward object by setting the first 3 to x,y,z directions
        #action[3] = 0.0 # i dont think we need this line casue it controls the gripper
        #initial code below
        obs, reward, done, _ = env.step(action)
        env.render()
        print("Reward:", reward)
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