import gymnasium
import manipulator_mujoco
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
#from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from save_best_train import SaveOnBestTrainingRewardCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results

import os

LEARN = True
MONITOR = True
PLAY = False
model_name = "sac_ur5e_6"
timesteps = 2e5

# Create log dir
log_dir = "/home/tammer/Manipulator-Mujoco/demo/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Create the environment with rendering in human mode
#env = gymnasium.make('manipulator_mujoco/UR5eSBEnvPos-v0', render_mode='human')
#env = gymnasium.make('manipulator_mujoco/UR5eSBEnvVel-v0', render_mode='human')
env = gymnasium.make('manipulator_mujoco/UR5eSBEnvVelPIH-v0', render_mode='human')

if MONITOR:
    env = Monitor(env, log_dir)
    callback = SaveOnBestTrainingRewardCallback(check_freq=200, log_dir=log_dir)

### https://stable-baselines.readthedocs.io/en/master/modules/sac.html
if LEARN:
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps, log_interval=4, callback=callback)
    if MONITOR:
        plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "SAC UR5e")
        plt.show()
    model.save(model_name)
    del model

#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

if PLAY:
    model = SAC.load(model_name)
    print("***********Playing the model forward... ***********")
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
