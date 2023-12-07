import gymnasium
import manipulator_mujoco

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from save_best_train import SaveOnBestTrainingRewardCallback
from stable_baselines3 import SAC
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results


from stable_baselines3.common.evaluation import evaluate_policy

# Create log dir
log_dir = "/home/tammer/Manipulator-Mujoco/demo/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

env = gymnasium.make('manipulator_mujoco/UR5eSBEnvPos-v0', render_mode='human')
env = Monitor(env, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=200, log_dir=log_dir)

model = SAC("MlpPolicy", env, verbose=1)

timesteps = 1e3

model.learn(total_timesteps=timesteps, log_interval=4, callback=callback)
plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "SAC UR5e")
plt.show()
model.save("sac_ur5e")
print("Done training...")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_ur5e")

#print("***********Evaluating the model... ***********")
#mean_reward, std_reward = evaluate_policy(model, 
#                                          env, 
#                                          n_eval_episodes=10, 
#                                          deterministic=True, 
#                                          render=False, 
#                                          callback=None, 
#                                          reward_threshold=None, 
#                                          return_episode_rewards=False, 
#                                          warn=True)


print("***********Playing the model forward... ***********")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()