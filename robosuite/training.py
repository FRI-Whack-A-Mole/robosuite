import numpy as np
import os
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robosuite.environments.manipulation.lift import Lift
#from PointEnv import PointEnv
import datetime

#######################################################################################

# attempting to generate unique file names  including year, time, up to the seconds so it makes sense
def create_filenames_for_each_model(name_model_actual="point_model"):   # from our current model name
    # printf like in java | can make it less specific if needed
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%Hh_%Mm_%Ss") # year, month, day, hour, minute, sec
    return f"{name_model_actual}_{timestamp}.zip" 

# saved in my home to easier find them ==> subject to change as needed
checkpoint_directory = "home/FRI_WBC"  # this one location other than tensorboard log to save checkpoints for models
os.makedirs(checkpoint_directory, exist_ok=True)  # param just to ensure the directory exists

latest_model_path = os.path.join(checkpoint_directory, "point_model_latest_checkpoint.zip")

#######################################################################################

if __name__=="__main__":

    target_ee_position = np.array([0.5, 0.0, 1.0])

    env = suite.make(
        robots="Panda",  
        env_name= "Lift",
        has_renderer=False,  # Rn I unabled visual sim if its needed
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True  #  dense reward
    )
    #env.reset()
    env.set_target_position(target_ee_position)
    
    # Vectorize and Normalize Environment
    my_wrapped_env = GymWrapper(env)
    my_vec_env = DummyVecEnv([lambda: my_wrapped_env])
    my_vec_env = VecNormalize(my_vec_env, norm_obs=True, norm_reward=True)

#############################################################################################################
   
    # Set up and train the PPO model  ==> changed loading of the model ==> Updated path and inn case nothing loads
    file_path = "point_model.zip"
    if os.path.exists(latest_model_path):   # this will automatically load the latest model
        print(f"Loading the latest model from: {latest_model_path}")   # loading latest model form latest path
       # also if needed to load specific model can just modify the path and ensure the pth is included and the name of teh model
        model = PPO.load(latest_model_path)  # change to if model exists and if not create new one
        model.env = my_vec_env
    else:
        print("No existing model found. This will create a training model from scratch.")  #just in case
        model = PPO(
            policy="MlpPolicy",   # add some logic that change the names for every chabge in the model training   
            env=my_vec_env,          
            learning_rate=0.0005,   # increased to improve training
            n_steps=3000,            
            batch_size=500,          
            verbose=1,                
            tensorboard_log="./log/tb.log"  # since in same dir already have the file ==> no need for abs path
        ) 

        # tensorboard_log="/home/anastasiia/UMI-On-Legs-Reinforcement-Learning - GENERAL/log/tb.log"   # Cab change this with respect to directory location
        # tensorboard_log cformer directory use "/home/anastasiia/UMI-On-Legs-Reinforcement-Learning - GENERAL/log/tb.log"
        # "./log/tb.log" <==== or ./log/anastasiia//tb.log

################################################################################################################

    model.learn(
        total_timesteps=1000000,   #updated so it trains longer
        log_interval=1
    )

    # Save the trained model
    # model.save("point_model")   <=== I changed this one to update the saving params | see next lines

    # attempting to create unique file name for each model

    unique_model_path = os.path.join(checkpoint_directory, create_filenames_for_each_model())
    model.save(unique_model_path)

    model.save(latest_model_path)

    print(f"Model saved at: {unique_model_path}")  # ensuring its saved where it needs to be
    print(f"Latest model saved at: {latest_model_path}")  # ensuring the latest model is saved

# import numpy as np
# import os
# import robosuite as suite
# from robosuite.wrappers import GymWrapper
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from robosuite.environments.manipulation.lift import Lift
# import datetime
# import time

# #######################################################################################

# # attempting to generate unique file names  including year, time, up to the seconds so it makes sense
# def create_filenames_for_each_model(name_model_actual="point_model"):   # from our current model name
#     timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%Hh_%Mm_%Ss")
#     return f"{name_model_actual}_{timestamp}.zip" 

# # saved in my home to easier find them ==> subject to change as needed
# checkpoint_directory = "home/FRI_WBC"
# os.makedirs(checkpoint_directory, exist_ok=True)

# latest_model_path = os.path.join(checkpoint_directory, "point_model_latest_checkpoint.zip")

# #######################################################################################

# if __name__ == "__main__":

#     target_ee_position = np.array([0.5, 0.0, 1.0])

#     env = suite.make(
#         robots="Panda",  
#         env_name="Lift",
#         has_renderer=False,
#         has_offscreen_renderer=False,
#         use_camera_obs=False,
#         reward_shaping=True
#     )
#     env.set_target_position(target_ee_position)
    
#     my_wrapped_env = GymWrapper(env)
#     my_vec_env = DummyVecEnv([lambda: my_wrapped_env])
#     my_vec_env = VecNormalize(my_vec_env, norm_obs=True, norm_reward=True)

#     file_path = "point_model.zip"
#     if os.path.exists(latest_model_path):
#         print(f"Loading the latest model from: {latest_model_path}")
#         model = PPO.load(latest_model_path)
#         model.env = my_vec_env
#     else:
#         print("No existing model found. This will create a training model from scratch.")
#         model = PPO(
#             policy="MlpPolicy",
#             env=my_vec_env,          
#             learning_rate=0.0005,
#             n_steps=3000,
#             batch_size=500,
#             verbose=1,
#             tensorboard_log="./log/tb.log"
#         )

#     model.learn(
#         total_timesteps=1000000,
#         log_interval=1
#     )

#     unique_model_path = os.path.join(checkpoint_directory, create_filenames_for_each_model())
#     model.save(unique_model_path)
#     model.save(latest_model_path)

#     print(f"Model saved at: {unique_model_path}")
#     print(f"Latest model saved at: {latest_model_path}")

#     ####################################################################################
#     # PPO Evaluation with rendering (optional)
#     print("\nEvaluating trained model with rendering...\n")

#     eval_env = suite.make(
#         robots="Panda",
#         env_name="Lift",
#         has_renderer=True,
#         has_offscreen_renderer=False,
#         use_camera_obs=False,
#         reward_shaping=True
#     )
#     eval_env.set_target_position(target_ee_position)
#     eval_wrapped_env = GymWrapper(eval_env)
#     obs = eval_wrapped_env.reset()
#     eval_env.viewer.set_camera(camera_id=0)

#     for step in range(100):  # 100 steps of visual evaluation
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, _ = eval_wrapped_env.step(action)
#         eval_wrapped_env.render()

#         print(f"Step {step} | Reward: {reward:.4f}")
#         if done:
#             print("Resetting environment.")
#             obs = eval_wrapped_env.reset()
#         time.sleep(1 / 25)  # simulate 25 FPS