import gymnasium
import manipulator_mujoco

# Create the environment with rendering in human mode
#env = gymnasium.make('manipulator_mujoco/UR5eSBEnvPos-v0', render_mode='human')
#env = gymnasium.make('manipulator_mujoco/UR5eSBEnvVel-v0', render_mode='human')
env = gymnasium.make('manipulator_mujoco/UR5eSBEnvVelPIH-v0', render_mode='human')

# Reset the environment with a specific seed for reproducibility
observation, info = env.reset(seed=42)
reward = 0
# Run simulation for a fixed number of steps
for _ in range(10000):
#while True:
    # Choose a random action from the available action space
    action = env.action_space.sample()

    # Take a step in the environment using the chosen action
    observation, step_reward, terminated, truncated, info = env.step(action)
    reward += step_reward

    # Check if the episode is over (terminated) or max steps reached (truncated)
    if terminated or truncated:
        # If the episode ends or is truncated, reset the environment
        observation, info = env.reset()

# Close the environment when the simulation is done
print(reward)
env.close()
