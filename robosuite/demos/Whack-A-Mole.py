# import time
# import numpy as np




# import robosuite as suite
# from robosuite.robots import MobileRobot




# MAX_FR = 25  # max frame rate for running simulation




# if __name__ == "__main__":




#    # Set environment and robot directly
#    options = {
#        "env_name": "Lift",
#        "robots": "Panda",  # You can also use ["Jaco"] if needed
#    }




# # Initialize the environment
# env = suite.make(
#    **options,
#    has_renderer=True,
#    has_offscreen_renderer=False,
#    ignore_done=False,
#    use_camera_obs=False,
#    control_freq=20,
# )




# env.reset()
# env.viewer.set_camera(camera_id=0)




# for robot in env.robots:
#    if isinstance(robot, MobileRobot):
#        robot.enable_parts(legs=False, base=False)




# learning_rate = 0.1
# previous_reward = 0
# # cube_pos = env.sim.data.body_xpos[env.cube_body_id]
# # gripper = env.robots[0].gripper['right']
# # eef_pos = env.sim.data.get_site_xpos(gripper.important_sites["grip_site"])
# # cube_array = np.array(cube_pos)
# # eef_array = np.array(eef_pos)
# # last_action = np.zeros(env.action_spec[0].shape)
# # # last_action[2:3] = cube_array[2:3] - eef_array[2:3]
# # action = np.zeros(*env.action_spec[0].shape)
# # # change = 0.02
# # change = 0.1
# # random_action = np.random.randn(*env.action_spec[0].shape)# * change
# #exploration_noise_std = 0.01




# # Run visualization loop
# for i in range(10000):
#    start = time.time()
#    action = np.zeros(*env.action_spec[0].shape)
#    cube_pos = env.sim.data.body_xpos[env.cube_body_id]
#    gripper = env.robots[0].gripper['right']
#    eef_pos = env.sim.data.get_site_xpos(gripper.important_sites["grip_site"])
#    # change = 0.02
# #    random_action = np.random.randn(*env.action_spec[0].shape) * change

# #    if i % 40:
# #        env.reset()
  
#    #env.robots[0]._hand_pos
#    action[:3] = np.array(cube_pos) - np.array(eef_pos)
#        #action[:3] = last_action[:3] + (random_action[:3])
# #    action[:3] = last_action[:3] + random_action[:3]
#    #action[:3] = random_action[:3]
#    #np.random.randn(*env.action_spec[0].shape) #TODO: set the actions
  
#    obs, reward, done, _ = env.step(action)
#    print("Reward: ", reward)
#    env.render()




# #    if (reward > previous_reward):
# #        print("Prev Reward:", previous_reward)
# #        print("Reward: ", reward)
# #        last_action = action
# #        previous_reward = reward
# #        change *= 0.95
# #        change = max(change, 0.01)
#        #random_action = random_action * 0.5   
#    # else:
#    #     #action[:3] = action[:3] - (random[:3] * 0.01)
#    #     #reward = previous_reward
#    #     random_action = np.random.randn(*env.action_spec[0].shape) * change


# #    print("Last: ", last_action)
# #    print("Action: ", action)
# #    print("Random: ", random_action)




#    if done:
#        env.reset()
#     #    previous_reward = -np.inf
#     #    last_action = np.zeros(env.action_spec[0].shape)




#    # Limit frame rate
#    elapsed = time.time() - start
#    diff = 1 / MAX_FR - elapsed
#    if diff > 0:
#        time.sleep(diff)





import time
import numpy as np
import os

import robosuite as suite
from robosuite.robots import MobileRobot
from robosuite.wrappers import GymWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

MAX_FR = 25  # max frame rate for running simulation

if __name__ == "__main__":

    # Set environment and robot
    options = {
        "env_name": "Lift",
        "robots": "Panda",
    }

    # Initialize environment
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

    # Disable mobile base movement if using MobileRobot
    for robot in env.robots:
        if isinstance(robot, MobileRobot):
            robot.enable_parts(legs=False, base=False)

    # === Load trained PPO model ===
    gym_env = GymWrapper(env)
    vec_env = DummyVecEnv([lambda: gym_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    vec_env.training = False  # Disable training mode for evaluation
    vec_env.reset()

    # model_path = "home/FRI_WBC/point_model_latest_checkpoint.zip"
    model_path = "home/FRI_WBC/point_model_2025_05_05_13h_44m_08s.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model = PPO.load(model_path, env=vec_env)
    obs = vec_env.reset()

    # === Run visualization loop with trained model ===
    for i in range(10000):
        start = time.time()

        # Get model-predicted action
        action, _ = model.predict(obs, deterministic=True)

        # Take a step in the environment
        obs, reward, done, _ = vec_env.step(action)
        gym_env.render()

        print(f"Step: {i}, Reward: {reward}")

        if done:
            obs = vec_env.reset()

        # Limit frame rate
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)
