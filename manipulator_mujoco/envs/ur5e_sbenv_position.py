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
from manipulator_mujoco.props import Primitive
from manipulator_mujoco.controllers import OperationalSpaceController

class UR5eSBEnvPos(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": None}

    def __init__(self, render_mode=None):
        super().__init__()
        # Define action and observation space
        self.action_space = spaces.Box(
            low=np.array([ 0.493 - 0.3,
                           0.128 - 0.3,
                           0.448 - 0.3]),
            high=np.array([0.493 + 0.3,
                           0.128 + 0.3,
                           0.448 + 0.3]),
            shape=(3, ),
            dtype=np.float64)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
        )
        self.reward = 0
        # TODO Make goal eef pose random during reset
        self.goal_eef_pose = np.array([0.15, 0.15, 0.15]) # (x,y,z)
        
        self.truncate_criteria = 0.03
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
                os.path.dirname(__file__),
                '../assets/robots/ur5e/ur5e.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )
        
        # place target in environment
        self._target = Primitive(type="cylinder", size=[0.01, 0.01], pos=[0,0,0], rgba=[0, 0, 1, 1])
        self._arena.attach(self._target.mjcf_model, 
                           pos=[0,0,0]
        )
        self._gripper = Primitive(type="cylinder", size=[0.02, 0.02], pos=[0,0,0.02], rgba=[1, 0, 0, 1], friction=[1, 0.3, 0.0001])

        # attach gripper to arm
        self._arm.attach_tool(self._gripper.mjcf_model, pos=[0, 0, 0], quat=[0, 0, 0, 1])

        # attach arm to arena
        self._arena.attach(
            self._arm.mjcf_model, pos=[0,0,0], quat=[0.7071068, 0, 0, -0.7071068]
        )
       
        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

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

    def _get_euc_dist(self, dx, dy, dz):
        euc_dist = math.sqrt((dx)**2 + (dy)**2 + (dz)**2)
        return euc_dist

    def _get_obs(self) -> np.ndarray:
        # TODO come up with an observations that makes sense for your RL task
        x, y, z, xx, yy, zz, ww = self._arm.get_eef_pose(self._physics)
        tx, ty, tz = self.goal_eef_pose
        return np.array([x - tx, y - ty, z - tz])
        #return np.zeros(6)

    def _get_info(self) -> dict:
        # TODO come up with an info dict that makes sense for your RL task
        return {}

    def step(self, action):
        terminated = False

        # TODO sleep while the arm moves from current to commanded pose (speed up?)


        # Turn 3dof action into 7 dof pose for controller
        action7dof = np.append(action, [0,0,0,1])

        self._controller.run(action7dof)

        # step physics
        for _ in range(20):
            self._physics.step()
            # render frame
            if self._render_mode == "human":
                self._render_frame()
      
        # TODO come up with a reward, termination function that makes sense for your RL task
        observation = self._get_obs()
        euc_dist = self._get_euc_dist(observation[0], observation[1], observation[2])
        self.reward -= 0.6*euc_dist
        self.reward -= 0.4*np.sum(np.absolute(observation))
        
        self.terminate_counter += 1
        if self.terminate_counter > 20:
            terminated = True

        if euc_dist < self.truncate_criteria:
            self.reward += 60
            truncated = True
        else: truncated = False

        info = self._get_info()
        reward = self.reward
        print("Reward: {}".format(reward))
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # reset physics
        with self._physics.reset_context():
            # TODO Make starting position random
            # FOR NOW put arm in a reasonable starting position

            random_arm_reset = np.random.uniform(low=-0.1, high = 0.1)

            self._physics.bind(self._arm.joints).qpos = [
                0.0 + random_arm_reset,
                -1.5707 + random_arm_reset, 
                1.5707+ random_arm_reset,
                -1.5707 + random_arm_reset, 
                -1.5707 + random_arm_reset, 
                0.0 + random_arm_reset
            ]
            x_offset = 0.2
            y_offset = 0.2
            z_offset = 0.2
            self.goal_eef_pose = np.array([
                np.random.uniform(low= 0.493 - x_offset, high = 0.493 + x_offset),
                np.random.uniform(low= 0.128 - y_offset, high = 0.128 + y_offset),
                np.random.uniform(low= 0.448 - z_offset, high = 0.448 + z_offset)
            ])
            #low=np.array([-0.1, 0.2, 0.2]),
            #high=np.array([0.1, 0.25, 0.3]),
            
            self._physics.bind(self._target.geom).pos = self.goal_eef_pose

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