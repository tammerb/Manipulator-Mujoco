# Import the registration function from Gymnasium
from gymnasium.envs.registration import register

register(
    id="manipulator_mujoco/AuboI5Env-v0",
    entry_point="manipulator_mujoco.envs:AuboI5Env",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)

register(
    id="manipulator_mujoco/UR5eEnv-v0",
    entry_point="manipulator_mujoco.envs:UR5eEnv",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)

register(
    id="manipulator_mujoco/UR5eEnvPos-v0",
    entry_point="manipulator_mujoco.envs:UR5eEnvPos",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)

register(
    id="manipulator_mujoco/UR5eSBEnvPos-v0",
    entry_point="manipulator_mujoco.envs:UR5eSBEnvPos",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)

register(
    id="manipulator_mujoco/UR5eSBEnvVel-v0",
    entry_point="manipulator_mujoco.envs:UR5eSBEnvVel",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)

register(
    id="manipulator_mujoco/UR5eSBEnvVelPIH-v0",
    entry_point="manipulator_mujoco.envs:UR5eSBEnvVelPIH",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)