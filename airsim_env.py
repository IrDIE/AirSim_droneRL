import random

from loguru import logger

from gymnasium import Env
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box

import numpy as np
import math
from airsim.utils import to_eularian_angles
from baselines_wrappers.atari_wrappers import MaxAndSkipEnv, ClipRewardEnv
from baselines_wrappers.wrappers import TimeLimit

from airsim import MultirotorClient
from  matplotlib.pyplot import get_cmap
import airsim
from utils.pytorch_wrappers import TransposeImageObs
import time
from PIL import Image
import psutil
import cv2
from utils.utils import get_scale_factor
from typing import TypeVar
from random import sample
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")

RESCALE_N = 0.35
RANDOM_SEED = 42
COLLISION_REWARD = -2
clock_speed = 10 # in cfg
ACTION_DURATION = 1 / clock_speed
RESIZE_OBSERVATION = (64, 64)


class AirSimGym_env(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, client : MultirotorClient, env_type, vehicle_name, initial_positions, observation_as_depth,height_airsim_restart_positions, done_xy=None, max_yaw_or_rate = 90, action_type = 'discrete'):
        super().__init__()
        self.observation_as_depth = observation_as_depth
        self.client = client
        self.action_type = action_type # 'continuous' or 'discrete'
        self.env_type = env_type # outdoor or indoor - reward generation differ
        self.vehicle_name = vehicle_name
        self.max_yaw_or_rate = max_yaw_or_rate
        self.is_rate = True
        self.done_xy = done_xy
        self.level = 0
        if action_type == 'continuous':
            self.action_space = Box( low=-1., high=1., shape=(4,),dtype=np.float32)
        else:
            self.action_space = Discrete(6) #[0,1,2,3,4,5]

        self.noop_action = self.define_noop_action()
        self.observation_shape = self.get_observation().shape #logger.info(f'self.observation_shape = {self.observation_shape}') # (360, 640, 3)
        logger.info(f'self.observation_shape = {self.observation_shape}')
        self.observation_space = Box(
            low=0., high=1., shape=self.observation_shape, dtype=np.float32 # 3 channels
        )
        self.height_airsim_restart_positions = height_airsim_restart_positions # [-5.35,-5.4, -6.5,-7.6,-8.5, -9. ] # restart height -- depends on environment
        self.reward_range = None
        self.initial_positions = initial_positions # select random pose from inital posinionы for outdoor or indoor

    def step_discrete(self, action):
        # # FOR DEBUG
        # reset_pos = airsim.Pose(airsim.Vector3r(0, 60, -0.8339),
        #                         airsim.to_quaternion(0, 0, (0) * np.pi / 180))
        # self.client.simSetVehiclePose(reset_pos, ignore_collison=True, vehicle_name='drone0')

        #self.client.simPause(False)


        if type(action) == int and action == self.noop_action: # == if action == 0: pass
            pass
        # forward
        if action == 1: self.move_forward()

        # rotate left
        if action == 2: self.rotate_left()

        # rotate right
        if action == 3: self.rotate_right()

        # up
        if action == 4: self.move_up()

        # down
        if action == 5: self.move_down()
        #self.client.simPause(True)
        observation = self.get_observation()
        info = self._get_info()

        reward, terminated, truncated = self.compute_reward_maze()  # TODO - define rewarn calculation function
        # logger.info(f'action = {action}')
        if action == 1: reward += 0.1

        return observation, reward, terminated, truncated, info

    def compute_reward_maze(self):
        truncated = False
        terminated = False
        out_of_env = False
        reward = 0
        y_current = 0
        min_speed = 0.2
        levels = [7,17,28,45,57]


        if_collision = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).has_collided
        kinematic = self.client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated
        position = kinematic.position
        quad_vel=kinematic.linear_velocity
        vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float32)
        if if_collision:
            # logger.info(f'COLLISION')
            reward = COLLISION_REWARD
            terminated = True # if collision
            return reward, terminated, truncated

        speed_current = np.linalg.norm(vel)
        #logger.info(f'speed_current={speed_current}')


        if position.y_val > levels[self.level]:
            self.level +=1
            reward = (2.)*(1 + self.level / len(levels))
        elif speed_current < min_speed:
            reward = -0.05
        else:
            reward = float(vel[1])  # y-positive velocity


        if self.done_xy is not None:
            out_of_maze = self.check_if_out_of_env(position = position)
            floor = True if position.z_val >  1.19 else False
            out_of_env = floor or out_of_maze
            #logger.info(f'out_of_env = {out_of_env}')
        truncated = False if not out_of_env else True
        # logger.info(f'reward={reward}. position.y_val={position.y_val}')
        return reward, terminated, truncated

    def get_yaw(self):
        quaternions = self.client.getMultirotorState().kinematics_estimated.orientation
        a, b, yaw_rad = to_eularian_angles(quaternions)
        yaw_deg = math.degrees(yaw_rad)
        return yaw_deg, yaw_rad

    def move_forward(self):
        yaw_deg, yaw_rad = self.get_yaw()
        z = self.client.simGetGroundTruthKinematics().position.z_val
        # need rad
        vx = math.cos(yaw_rad) * 0.35
        vy = math.sin(yaw_rad) * 0.35
        #logger.info(f'vx={vx}, vy={vy}')
        self.client.moveByVelocityAsync(vx , vy , 0, ACTION_DURATION, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False))#.join()


    def rotate_left(self):
        yaw_deg, yaw_rad = self.get_yaw()
        z = self.client.simGetGroundTruthKinematics().position.z_val
        yaw_rad -= math.radians(10)
        self.client.moveByVelocityAsync(0, 0, z, ACTION_DURATION, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False)).join()

        self.client.moveByAngleZAsync(0,0,z,yaw_rad,1)#.join()
        #self.client.moveByVelocityAsync(0, 0, z, 1, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False))

    def rotate_right(self):
        yaw_deg, yaw_rad = self.get_yaw()
        yaw_rad += math.radians(10)
        z = self.client.simGetGroundTruthKinematics().position.z_val
        self.client.moveByVelocityAsync(0, 0, z, ACTION_DURATION, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False)).join()

        self.client.moveByAngleZAsync(0,0,z,yaw_rad,1)#.join()
        #self.client.moveByVelocityAsync(0, 0, z, 1, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False))


    def move_up(self):
        linear_velocity = self.client.simGetGroundTruthKinematics().linear_velocity
        x, y, z = linear_velocity.x_val, linear_velocity.y_val, linear_velocity.z_val
        z -= 0.007

        self.client.moveByVelocityAsync(0,0, z, ACTION_DURATION, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False))#.join()
        #self.client.moveByVelocityAsync(0, 0, 0, ACTION_DURATION, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False))#.join()
        #

    def move_down(self):
        kinematic = self.client.simGetGroundTruthKinematics()
        # logger.info(f'kinematic z = {kinematic.position.z_val}')
        linear_velocity = kinematic.linear_velocity
        x, y, z = linear_velocity.x_val, linear_velocity.y_val, linear_velocity.z_val
        z += 0.009

        self.client.moveByVelocityAsync(0, 0, z , ACTION_DURATION, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False))#.join()
        #self.client.moveByVelocityAsync(0, 0, 0, ACTION_DURATION, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False))#.join()
        #

    def step_continuous(self, action):
        """
                :param action: array with 4 floats from -1 to 1 : x,y,z velocities and yaw_or_rate
                :return:
                """
        if type(action) == int and action == self.noop_action:
            pass
        else:
            vx, vy, vz, yaw_or_rate = action
            yaw_or_rate = (yaw_or_rate) * self.max_yaw_or_rate
            start = time.time()

            self.client.moveByVelocityAsync(vx=float(vx), vy=float(vy), vz=float(vz), duration=1,
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=airsim.YawMode(is_rate=self.is_rate,
                                                                    yaw_or_rate=float(yaw_or_rate)),
                                            vehicle_name=self.vehicle_name)

            # for i in range(3):
            #     logger.info(f'\n*{i} in step **\n{self._get_info(get_kinematic=True)}')
            #     time.sleep(0.1)
            # do one step in environment that corresponds to action
        observation = self.get_observation()
        reward, done = self.compute_reward()  # TODO - define rewarn calculation function
        info = self._get_info()
        return observation, reward, done, info

    def step(self, action):

        if self.action_type == 'discrete':
            return self.step_discrete(action)

        elif self.action_type == 'continuous':
            raise NotImplementedError() #return self.step_continuous(action)
        else:
            raise KeyError(f"self.action_type = {self.action_type} is invalid. discrete or continuous are available =)")

    def check_if_out_of_env(self, position):

        x_current, y_current, z_current = position.x_val, position.y_val, position.z_val
        if float(z_current) < float(-9.5) and self.env_type == 'outdoor':
            return True
        min_x, max_x = self.done_xy[0]
        min_y, max_y= self.done_xy[1]
        if x_current > max_x or x_current < min_x or y_current > max_y or y_current < min_y:
            return True # out of environment defined in unreale4
        else:
            return False

    def compute_reward_outdoor(self):
        #raise NotImplementedError("compute_reward_indoor Not Implemented")
        kinematic = self.client.simGetGroundTruthKinematics()
        # logger.info(f'lin velocity = {kinematic.linear_velocity}')
        velocity = kinematic.linear_velocity
        x_velocity, y_velocity = velocity.x_val, velocity.y_val
        velocity = math.sqrt((x_velocity*x_velocity) + (y_velocity*y_velocity))
        position = kinematic.position

        x_current, y_current = position.x_val, position.y_val
        x_start, y_start = self.start_point
        x_delta , y_delta = (x_start - x_current), (y_start-y_current)
        distance = math.sqrt((x_delta*x_delta) + (y_delta*y_delta))
        # logger.info(f'distance = {distance}')
        # calculate % of max possible
        reward = distance * velocity / self.max_distance_xy

        # logger.info(f'\n**********\nreward = {reward}')
        # logger.info(f'velocity={velocity}')
        # logger.info(f'distance={distance}')
        # logger.info(f'self.max_distance_xy={self.max_distance_xy}')
        # logger.info(f'distance / self.max_distance_xy={distance / self.max_distance_xy}')


        return reward
    def compute_reward_indoor(self):
        raise NotImplementedError("compute_reward_indoor Not Implemented")
    def compute_reward(self):
        if_collision = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).has_collided
        out_of_env = False
        truncated = False

        if self.done_xy is not None:
            out_of_env, y_current = self.check_if_out_of_env(get_y=True)
        if if_collision:
            reward = -10
            terminated = True
            return reward, terminated, truncated
        else:
            terminated = False if not out_of_env else True
            if self.env_type == 'outdoor':
                reward = self.compute_reward_outdoor()
                return reward, terminated, truncated
            elif self.env_type == 'indoor':
                reward = self.compute_reward_indoor()
                return reward, terminated, truncated
            else:
                raise KeyError(f"self.env_type = {self.env_type} is invalid. indoor or outdoor are available =)")

    def reset(self, seed=None, options=None, level = 0):
        self.level = 0
        # self.client.confirmConnection()
        self.client.reset()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True)
        if seed is None:
            np.random.seed(RANDOM_SEED)
            self.np_random = np.random.default_rng()

        # select random pose from inital posinion and generate z
        self.client.simPause(False)
        x, y, angle = [*sample(self.initial_positions, 1)][0]
        reset_height = [*sample(self.height_airsim_restart_positions, 1)][0]
        if self.env_type == 'outdoor':
            # define some params for reward calculation
            self.start_point = (x,y)
            min_x, max_x = self.done_xy[0]
            min_y, max_y = self.done_xy[1]
            x_max_possible = max(abs( min_x - x), abs(max_x-x)) # x_max_possible distance from strart point
            y_max_possible = max(abs(min_y - y), abs(max_y - y))  # y_max_possible distance from strart point
            self.max_distance_xy = math.sqrt((x_max_possible*x_max_possible) + (y_max_possible*y_max_possible))
            # logger.info(f'self.max_distance_xy = {self.max_distance_xy}')
        reset_pos = airsim.Pose(airsim.Vector3r(x, y, reset_height),
                               airsim.to_quaternion(0, 0, (angle)*np.pi/180))

        # self.client.moveByVelocityAsync(0, 0, -1, 2 * ACTION_DURATION).join()
        self.client.moveByVelocityAsync(0, 0, 0, 2 * ACTION_DURATION).join()
        self.client.simSetVehiclePose(reset_pos, ignore_collison=True, vehicle_name=self.vehicle_name)
        self.client.simPause(True)
        self.client.hoverAsync().join()
        # logger.info(f'in reset')
        observation = self.get_observation()
        info = self._get_info()
        time.sleep(1)
        self.client.simPause(False)

        return observation, info

    def _get_info(self, get_kinematic = False):

        if get_kinematic : return self.client.simGetGroundTruthKinematics()
        else : return {}


    def get_raw_observ(self, depth = True):
        """
        AirSim observation : image from fpv-camera of drone
        :return:
        """
        if depth: observation = get_DepthImageRGB(self.client, self.vehicle_name, self.env_type)
        else : observation = get_MonocularImageRGB(self.client, self.vehicle_name)
        return observation

    def get_observation(self, rescale_n = RESCALE_N):

        raw_observation = self.get_raw_observ(depth = self.observation_as_depth)
        if self.observation_as_depth:
            cmap = get_cmap('jet')
            depth_map_heat = cmap(raw_observation)[:, :, :3] # already normalized from 0 to 1
            if RESIZE_OBSERVATION is not None:
                image_scaled = cv2.resize(depth_map_heat, RESIZE_OBSERVATION)

            else:
                image_scaled = cv2.resize(depth_map_heat,
                                            (int(depth_map_heat.shape[1] * RESCALE_N), int(depth_map_heat.shape[0] * RESCALE_N)))

        else:
            mono_image = raw_observation # already normalized from 0 to 1
            image_scaled = cv2.resize(mono_image,
                                            (mono_image.shape[1] * RESCALE_N, mono_image.shape[0] * RESCALE_N))

        return image_scaled

    def define_noop_action(self):
        return 0

    def render(self):
        """
        check 'debug' mode in PEDRA 2-0 -> how it rendered
        :return:
        """
        pass

    def close(self, env_process):
        close_env(env_process)


def close_env(env_process):
    process = psutil.Process(env_process.pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()
    logger.info("Environment closed")


def get_DepthImageRGB(client, vehicle_name, env_type):
    camera_name = 1
    max_tries = 15
    tries = 0
    correct = False
    # if env_type == 'indoor':
    #
    #     while not correct and tries < max_tries: # TODO - chech how it will work for indoor environment
    #         tries += 1
    #         responses = client.simGetImages(
    #             [airsim.ImageRequest(camera_name, airsim.ImageType.DepthVis, False, False)],
    #             vehicle_name=vehicle_name)[0]
    #         img1d = np.fromstring(responses.image_data_uint8, dtype=np.uint8)
    #         # AirSim bug: Sometimes it returns invalid depth map with a few 255 and all 0s
    #         if np.max(img1d) == 255 and np.mean(img1d) < 0.05:
    #             correct = False
    #         else:
    #             correct = True
    #
    #     depth = img1d.reshape(responses.height, responses.width, 3)[:, :, 0]
    # elif env_type == 'outdoor':
    while not correct and tries < max_tries: # TODO - chech how it will work for indoor environment
        tries += 1
        responses = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.DepthVis, True)],
                                             vehicle_name=vehicle_name)
        responses = responses[0]
        # if random.random() > 0.98:
        #     logger.info(f'manually call error cannot reshape array of size 1 into shape (0,0)')
        #     raise ValueError('cannot reshape array of size 1 into shape (0,0)')
        if responses.width == 0:
            logger.info(f'\n /// BUG ***\n')
            logger.info(f'responses =BUG= {responses}')
            logger.info(f'responses.width, responses.height = {responses.width, responses.height}')

        depth = airsim.list_to_2d_float_array(responses.image_data_float, responses.width, responses.height)
        if responses.width == 320:
            correct = True

    return depth

def get_MonocularImageRGB(client, vehicle_name):

    responses1 = client.simGetImages([
        airsim.ImageRequest('front_center', airsim.ImageType.Scene, False,
                            False)], vehicle_name=vehicle_name)  # scene vision image in uncompressed RGBA array

    response = responses1[0]
    img1d = np.fromstring(response.image_data_uint8 , dtype=np.uint8)  # get numpy array / image_data_uint8
    img_rgba = img1d.reshape(response.height, response.width, 3)
    img = Image.fromarray(img_rgba)
    img_rgb = img.convert('RGB')
    camera_image_rgb = np.asarray(img_rgb)
    camera_image = camera_image_rgb / 255

    return camera_image

def make_airsim_deepmind(airsim_env_class, max_episode_steps=None, skip=2):
    env = airsim_env_class # gym.make(env_id, new_step_api =False , render_mode=render_mode) # apply_api_compatibility=False
    # env = NoopResetEnv(env, noop_max=30) # We already set random initialisation in .reset()
    env = MaxAndSkipEnv(env, skip=skip) # TODO - WTF

    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    env = TransposeImageObs(env, axis_order=[2, 0, 1])  # Convert to torch order (C, H, W)
    return env

