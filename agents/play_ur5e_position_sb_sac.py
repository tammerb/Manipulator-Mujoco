import gymnasium
import manipulator_mujoco
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

import time

# Create the environment with rendering in human mode
env = gymnasium.make('manipulator_mujoco/UR5eSBEnvPos-v0', render_mode='human')

### https://stable-baselines.readthedocs.io/en/master/modules/sac.html
model = SAC.load("sac_ur5e_4")

#print("***********Evaluating the model... ***********")
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)


print("***********Playing the model forward... ***********")

# Play the trained agent
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
