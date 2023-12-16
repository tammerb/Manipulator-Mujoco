import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces

import time
import os
import numpy as np
from dm_control import mjcf
import mujoco.viewer
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm
from manipulator_mujoco.robots import TWOF85
from manipulator_mujoco.props import Hole
from manipulator_mujoco.props import Peg
from manipulator_mujoco.props import Primitive
from manipulator_mujoco.controllers import OperationalSpaceController
from manipulator_mujoco.utils.transform_utils import (
    quat_distance
)

class UR5eSBEnvVelPIH(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": None}

    def __init__(self, render_mode=None):
        super().__init__()
        # Define action and observation space
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(7, ),
            dtype=np.float64)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
        )
        self.reward = 0
        
        self.truncate_criteria = 0.005
        self.terminate_counter = 0


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode

        ############################
        # create MJCF model
        ############################
        
        # checkerboard floor
        self._arena = StandardArena()

        # ur5e arm
        self._arm = Arm(
            xml_path= os.path.join(
                os.path.dirname(__file__),'../assets/robots/ur5e/ur5e.xml',),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )
        
        # place hole in environment
        self._target = Hole()
        self._arena.attach(self._target.mjcf_model, 
                           pos=[0.5,0,0], 
                           quat=[0.7071068, 0, 0, -0.7071068],
        )
        
        # attach peg to arm
        self._peg = Peg()
        self._arm.attach_tool(self._peg.mjcf_model, pos=[0, 0, 0], quat=[0, 0, 0, 1])

        # attach arm to arena
        self._arena.attach(
            self._arm.mjcf_model, pos=[0,0,0], quat=[0.7071068, 0, 0, -0.7071068]
        )
       
        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        self.initial_eef_pose = self._arm.get_eef_pose(self._physics)
        self.goal_eef_pose = np.array([0.5,
                                       0.0,
                                       0.0,
                                       0.0,
                                       0.0,
                                       0.0,
                                       1.0]) # (x,y,z,qx,qy,qz,qw)
        
        # set up OSC controller
        self._controller = OperationalSpaceController(
            physics=self._physics,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=200,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=2.0,
        )

        # for GUI and time keeping
        self._timestep = self._physics.model.opt.timestep
        self._viewer = None
        self._step_start = None

        self.sensors = self._arm._mjcf_root.find_all('sensor')
        #self.eef_vel = self._arm.forearm.qvel
        

    def _get_euc_dist(self, dx, dy, dz):
        euc_dist = math.sqrt((dx)**2 + (dy)**2 + (dz)**2)
        return euc_dist

    def _get_obs(self) -> np.ndarray:

        # Pose difference observation
        x, y, z, qx, qy, qz, qw = self._arm.get_eef_pose(self._physics)
        goal_x, goal_y, goal_z, goal_qx, goal_qy, goal_qz, goal_qw = self.goal_eef_pose
        quat_dist = quat_distance([qx,qy,qz,qw,],
                                      [goal_qx, goal_qy, goal_qz, goal_qw])

        #print(quat_dist)
        #self._data = self._physics.bind(self.sensors).sensordata
        #force_x, force_y, force_z = self._data[6:9]
        #torqx, torqy, torqz = self._data[9:]

        return np.array([x - goal_x,
                         y - goal_y,
                         z - goal_z,
                         quat_dist[0],
                         quat_dist[1],
                         quat_dist[2]])

    def _get_info(self) -> dict:
        arm_qpos = self._physics.bind(self._arm.joints).qpos
        arm_qvel = self._physics.bind(self._arm.joints).qvel
        return {'joint_pos': arm_qpos, 'joing_vel': arm_qvel}

    def step(self, action):
        #peg_geom = self._peg.geom()
        #print(dir(self._physics.bind(peg_geom)))
        terminated = False

        self._controller.vrun(action)

        # step physics
        for _ in range(10):
            self._physics.step()
            # render frame
            if self._render_mode == "human":
                self._render_frame()
      
        observation = self._get_obs()
        euc_dist = self._get_euc_dist(observation[0], observation[1], observation[2])
        self.reward = -0.6*euc_dist
        self.reward -= 0.4*np.sum(np.absolute(observation[:3]))
        self.reward -= 0.067*np.sum(np.absolute(observation[3:]))
        
        #Set each episode to 100 timesteps
        if self.terminate_counter >= 399:
            terminated = True
        self.terminate_counter += 1

        if euc_dist < self.truncate_criteria:
            self.reward += 60
            truncated = True
        else: truncated = False

        info = self._get_info()
        reward = self.reward
        #print("Reward: {}".format(reward))
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # reset physics
        with self._physics.reset_context():
            # TODO Make starting position random
            # FOR NOW put arm in a reasonable starting position

            #random_arm_reset = np.random.uniform(low=-0.1, high = 0.1)
            random_arm_reset = 0

            self._physics.bind(self._arm.joints).qpos = [
                -0.25776116, -1.2351148,   2.14294879, -2.47863031, -1.57079633, -0.25776116
            ]
            # x_offset = 0.2
            # y_offset = 0.2
            # z_offset = 0.05
            # self.goal_eef_pose = np.array([
            #     np.random.uniform(low= 0.493 - x_offset, high = 0.493 + x_offset),
            #     np.random.uniform(low= 0.128 - y_offset, high = 0.128 + y_offset),
            #     np.random.uniform(low= 0.1 - z_offset, high = 0.1 + z_offset)
            # ])
            #low=np.array([-0.1, 0.2, 0.2]),
            #high=np.array([0.1, 0.25, 0.3]),
            
            #self._physics.bind(self._target.geom).pos = self.goal_eef_pose

            self.terminate_counter = 0
            self.reward = 0



        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def render(self) -> np.ndarray:
        """
        Renders the current frame and returns it as an RGB array if the render mode is set to "rgb_array".

        Returns:
            np.ndarray: RGB array of the current frame.
        """
        if self._render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> None:
        """
        Renders the current frame and updates the viewer if the render mode is set to "human".
        """
        if self._viewer is None and self._render_mode == "human":
            # launch viewer
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
            )
        if self._step_start is None and self._render_mode == "human":
            # initialize step timer
            self._step_start = time.time()

        if self._render_mode == "human":
            # render viewer
            self._viewer.sync()

            # TODO come up with a better frame rate keeping strategy
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            self._step_start = time.time()

        else:  # rgb_array
            return self._physics.render()

    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self._viewer is not None:
            self._viewer.close()