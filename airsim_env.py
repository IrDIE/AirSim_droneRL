from utils.utils import generate_json
import subprocess
from configparser import ConfigParser
from loguru import logger
from utils.utils import read_cfg
from gymnasium import Env
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box
from baselines_wrappers.dummy_vec_env import DummyVecEnv
from utils.pytorch_wrappers import BatchedPytorchFrameStack
from utils.initial_positions import get_airsim_position
from utils.utils import get_distance_to_goal_3d
import numpy as np
import math
from airsim.utils import to_eularian_angles
from airsim import MultirotorClient
import airsim
import time
from PIL import Image
import psutil
from random import sample
import cv2
import math

RANDOM_SEED = 42
COLLISION_REWARD = -2
clock_speed = 2  # in cfg # TODO: read clock speed from cfg
ACTION_DURATION = 1 / clock_speed


class AirSimGym_env(Env):

    def __init__(
        self, client: MultirotorClient, cfg, action_type, starts_goals, points, done_xy
    ):
        super().__init__()
        self.observation_as_depth = cfg.getboolean(
            "environment", "observation_as_depth"
        )
        self.get_vel_obs = cfg.getboolean("environment", "get_vel_obs")
        self.client = client
        self.action_type = action_type  # 'continuous' or 'discrete'
        self.env_type = cfg.get(
            "environment", "type"
        )  # outdoor or indoor - reward generation differ

        self.done_xy = done_xy
        self.level = 0
        if action_type == "continuous":
            self.action_space = Box(low=-1.5, high=1.5, shape=(3,), dtype=np.float32)
        else:
            self.action_space = Discrete(6)  # [0,1,2,3,4,5]

        self.noop_action = self.define_noop_action()
        self.observation_shape = self.get_observation().shape  #

        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=self.observation_shape,
            dtype=np.float32,  # 3 channels
        )

        self.starts_goals = starts_goals
        self.points = points
        self.yaws = [0]

    def step_discrete(self, action):

        if action == 0:  # == if action == 0: pass
            self.not_move()
        # forward
        if action == 1:
            self.move_forward()

        # rotate left
        if action == 2:
            self.rotate_left()

        # rotate right
        if action == 3:
            self.rotate_right()

        # up
        if action == 4:
            self.move_up()

        # down
        if action == 5:
            self.move_down()

        observation = self.get_observation()
        info = self._get_info()
        reward, terminated, truncated = self.compute_reward()  # _maze()
        # TODO - define rewarn calculation function
        # logger.info(f'action = {action}')
        # logger.info(f"terminateds or truncateds = {terminated}, {truncated}")

        return observation, reward, terminated, truncated, info

    def correct_continuous_action(self, action):
        """
        здесь после clip или нет
        да

        """
        v_xy_sp = action[0] + 1
        v_z_sp = float(action[1])

        yaw_rate_sp = action[-1]

        _, yaw_rad = self.get_yaw()
        self.yaw = yaw_rad
        self.yaw_sp = self.yaw + yaw_rate_sp

        if self.yaw_sp > math.radians(180):
            self.yaw_sp -= math.pi * 2
        elif self.yaw_sp < math.radians(-180):
            self.yaw_sp += math.pi * 2

        vx_local_sp = v_xy_sp * math.cos(self.yaw_sp)
        vy_local_sp = v_xy_sp * math.sin(self.yaw_sp)

        return vx_local_sp, vy_local_sp, -v_z_sp, yaw_rate_sp

    def step_continuous(self, action):
        """
        :param action: array with 3 floats from -1 to 1 : x,y,z velocities and yaw_or_rate
        :return:
        """

        vx, vy, vz, yaw = self.correct_continuous_action(action)
        # self.client.simPause(False)
        self.client.moveByVelocityAsync(
            vx,
            vy,
            vz,
            ACTION_DURATION,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(yaw)),
        ).join()
        # self.client.simPause(True)

        # vx, vy, vz = action
        # self.client.moveByVelocityAsync(
        #     vx=float(vx),
        #     vy=float(vy),
        #     vz=float(vz),
        #     duration=ACTION_DURATION,
        # )

        # do one step in environment that corresponds to action
        observation = self.get_observation()
        info = self._get_info()
        reward, terminated, truncated = self.compute_reward()
        return observation, reward, terminated, truncated, info

    def compute_reward_maze(self):
        truncated = False
        terminated = False
        out_of_env = False
        min_speed = 0.2
        levels = [7, 17, 28, 45, 57]
        if_collision = self.client.simGetCollisionInfo().has_collided

        if if_collision:
            reward = COLLISION_REWARD
            terminated = True  # if collision
            return reward, terminated, truncated

        kinematic = self.client.getMultirotorState().kinematics_estimated
        position = kinematic.position
        quad_vel = kinematic.linear_velocity
        vel = np.array(
            [quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float32
        )
        speed_current = np.linalg.norm(vel)

        # reward = float(vel[1])
        if position.y_val > levels[self.level]:
            self.level += 1
            reward = (-COLLISION_REWARD) * (1 + self.level / len(levels))
        elif speed_current < min_speed:
            reward = -0.05  # slow
        else:
            reward = float(vel[1]) * 0.1

        if self.done_xy is not None:
            out_of_maze = self.check_if_out_of_env(position=position)
            floor = True if position.z_val > 1.165 else False
            out_of_env = floor or out_of_maze

        truncated = False if not out_of_env else True

        return reward, terminated, truncated

    def get_yaw(self):
        quaternions = self.client.getMultirotorState().kinematics_estimated.orientation
        a, b, yaw_rad = to_eularian_angles(quaternions)
        yaw_deg = math.degrees(yaw_rad)
        return yaw_deg, yaw_rad

    def not_move(self):
        yaw_deg, yaw_rad = self.get_yaw()
        z = self.client.simGetGroundTruthKinematics().position.z_val
        self.client.moveByAngleZAsync(0, 0, z, yaw_rad, ACTION_DURATION)

    def move_forward(self):
        # _, yaw_rad = self.get_yaw()
        z = self.client.simGetGroundTruthKinematics().position.z_val
        # vx, vy = math.cos(yaw_rad), math.sin(yaw_rad)
        # logger.info(f"vx = {vx}, vy  = {vy}")
        self.client.moveByVelocityZBodyFrameAsync(
            1.5,
            0,
            z - 0.002,
            ACTION_DURATION * 1.5,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            # airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(yaw_rad))
        ).join()

    def rotate_left(self):
        yaw_deg, yaw_rad = self.get_yaw()
        z = self.client.simGetGroundTruthKinematics().position.z_val
        yaw_rad -= math.radians(10)
        self.client.moveByRollPitchYawZAsync(
            0, 0, -yaw_rad, z, ACTION_DURATION
        ).join()  #  -yaw_rad if in config initial_positions.py yaw = 180

    def rotate_right(self):
        yaw_deg, yaw_rad = self.get_yaw()
        yaw_rad += math.radians(10)
        z = self.client.simGetGroundTruthKinematics().position.z_val
        self.client.moveByRollPitchYawZAsync(0, 0, -yaw_rad, z, ACTION_DURATION).join()

    def move_up(self):
        linear_velocity = self.client.simGetGroundTruthKinematics().linear_velocity
        x, y, z = linear_velocity.x_val, linear_velocity.y_val, linear_velocity.z_val
        z -= 1

        self.client.moveByVelocityAsync(
            0,
            0,
            z,
            ACTION_DURATION,
        ).join()

    def move_down(self):
        kinematic = self.client.simGetGroundTruthKinematics()
        linear_velocity = kinematic.linear_velocity
        x, y, z = linear_velocity.x_val, linear_velocity.y_val, linear_velocity.z_val
        z += 0.5
        self.client.moveByVelocityAsync(
            0,
            0,
            z,
            ACTION_DURATION,
        ).join()

    def step(self, action):
        self.client.simPause(False)

        if self.action_type == "discrete":
            observation, reward, terminated, truncated, info = self.step_discrete(
                action
            )
            logger.info(f"action={action}")
        else:
            observation, reward, terminated, truncated, info = self.step_continuous(
                action
            )

        if self.action_type == "discrete":
            self.client.simPause(True)
        # self.client.simPause(True) # TODO: after continuus action too?
        if self.get_vel_obs:
            vel = self._get_info(get_kinematic=True)
            observation = [observation, vel]
        return observation, reward, terminated, truncated, info

    def check_if_out_of_env(self, position):
        assert (
            self.env_type == "outdoor"
        ), "\nonly for `outdoor` checking out of enf must be made!\n "

        x_current, y_current, z_current = position.x_val, position.y_val, position.z_val

        min_x, max_x = self.done_xy[0]
        min_y, max_y = self.done_xy[1]
        min_z, max_z = self.done_xy[2]
        if (
            False
            or x_current > max_x
            or x_current < min_x
            or y_current > max_y
            or y_current < min_y
            or z_current > max_z
            or z_current < min_z
        ):
            return True  # out of environment defined in unreale4
        else:
            return False

    def reward_outdoor(self):
        truncated = False
        terminated = False
        out_of_env = False
        delta_d_coef = 30

        collision_reward = -5
        out_of_env_reward = -4

        if_collision = self.client.simGetCollisionInfo().has_collided

        if if_collision:
            terminated = True
            return collision_reward, terminated, truncated

        kinematic = self.client.getMultirotorState().kinematics_estimated
        position = kinematic.position

        if self.done_xy is not None:
            out_of_env = self.check_if_out_of_env(position=position)
        truncated = False if not out_of_env else True
        if out_of_env:
            return out_of_env_reward, terminated, truncated

        quad_vel = kinematic.linear_velocity
        vel = np.array([quad_vel.x_val, quad_vel.y_val], dtype=np.float32)
        speed_xy_current = np.linalg.norm(vel)
        # logger.info(f"speed = {speed_xy_current}")

        # current_distance_to_goal = self.get_distance_to_goal(position, self.goal_point)
        # delta_d = self.last_distance_to_goal - current_distance_to_goal

        # delta_d = delta_d * delta_d_coef / self.start_goal_dist
        # self.last_distance_to_goal = current_distance_to_goal

        dist_obstackle = 1 - self.min_collision_dist
        if dist_obstackle < 0.9:
            dist_obstackle = 0

        reward = speed_xy_current - dist_obstackle
        # logger.info(f"\nReward components: speed_xy_current={speed_xy_current}, dist_obstackle={dist_obstackle}\nreward={reward}\n\n")

        return reward, terminated, truncated

    def reward_outdoor_z(self):
        truncated = False
        terminated = False
        out_of_env = False
        # delta_d_coef = 50

        collision_reward = -2
        out_of_env_reward = -1

        if_collision = self.client.simGetCollisionInfo().has_collided

        if if_collision:
            terminated = True
            return collision_reward, terminated, truncated

        kinematic = self.client.getMultirotorState().kinematics_estimated
        position = kinematic.position

        if self.done_xy is not None:
            out_of_env = self.check_if_out_of_env(position=position)
        truncated = False if not out_of_env else True
        if out_of_env:
            return out_of_env_reward, terminated, truncated

        quad_vel = kinematic.linear_velocity
        vel = np.array([quad_vel.x_val, quad_vel.y_val], dtype=np.float32)
        speed_xy_current = np.linalg.norm(vel)

        # dynamic distance
        # current_distance_to_goal = self.get_distance_to_goal(position, self.goal_point)
        # delta_d = self.last_distance_to_goal - current_distance_to_goal
        # delta_d = delta_d * delta_d_coef / self.start_goal_dist
        # self.last_distance_to_goal = current_distance_to_goal

        # fast delta z reward
        current_z = self.client.getMultirotorState().kinematics_estimated.position.z_val
        delta_z = np.abs(self.last_z - current_z)
        self.last_z = current_z
        # logger.info(f'\ndelta_z = {delta_z}, z_current={current_z}')
        delta_z_reward = 0
        z_distance_reward = 0
        if delta_z > 0.3:
            delta_z_reward = -delta_z * 0.5

        if current_z < -9:  # for NH environment
            z_distance_reward = -0.1

        # dist to obstackle
        dist_obstackle = 1 - self.min_collision_dist
        if dist_obstackle < 0.9:
            dist_obstackle = 0

        reward = 0.4 * speed_xy_current - 2 * dist_obstackle + delta_z_reward + z_distance_reward
        logger.info(
            f"\nspeed_xy_current={speed_xy_current}, dist_obstackle={dist_obstackle}"
        )

        return reward, terminated, truncated

    def reward_outdoor_1(self):
        truncated = False
        terminated = False
        out_of_env = False
        delta_d_coef = 50

        collision_reward = -2
        out_of_env_reward = -1

        if_collision = self.client.simGetCollisionInfo().has_collided

        if if_collision:
            terminated = True
            return collision_reward, terminated, truncated

        kinematic = self.client.getMultirotorState().kinematics_estimated
        position = kinematic.position

        if self.done_xy is not None:
            out_of_env = self.check_if_out_of_env(position=position)
        truncated = False if not out_of_env else True
        if out_of_env:
            return out_of_env_reward, terminated, truncated

        # st_x, st_y, st_h = self.start_point # st = start
        # delta_x, delta_y = st_x - position.x_val, st_y - position.y_val
        # dist_from_start = math.sqrt(pow(delta_x, 2) + pow(delta_y, 2) )

        quad_vel = kinematic.linear_velocity
        vel = np.array([quad_vel.x_val, quad_vel.y_val], dtype=np.float32)

        speed_xy_current = np.linalg.norm(vel)
        # logger.info(f"speed = {speed_xy_current}")

        current_distance_to_goal = self.get_distance_to_goal(position, self.goal_point)
        delta_d = self.last_distance_to_goal - current_distance_to_goal

        delta_d = delta_d * delta_d_coef / self.start_goal_dist
        self.last_distance_to_goal = current_distance_to_goal

        dist_obstackle = 1 - self.min_collision_dist
        if dist_obstackle < 0.9:
            dist_obstackle = 0

        reward = delta_d * 0.1 + 0.4 * speed_xy_current - 2 * dist_obstackle
        # logger.info(
        #     f"\nReward components: delta_d={delta_d},speed_xy_current={speed_xy_current}, dist_obstackle={dist_obstackle}\nreward={reward}\n\n"
        # )

        return reward, terminated, truncated

    def reward_indoor(self):
        truncated = False
        terminated = False
        out_of_env = False
        delta_d_coef = 30
        if_collision = self.client.simGetCollisionInfo().has_collided

        if if_collision:
            reward = COLLISION_REWARD
            terminated = True
            return reward, terminated, truncated

        kinematic = self.client.getMultirotorState().kinematics_estimated
        position = kinematic.position
        quad_vel = kinematic.linear_velocity
        vel = np.array([quad_vel.x_val, quad_vel.y_val], dtype=np.float32)
        speed_xy_current = np.linalg.norm(vel)

        if self.done_xy is not None:
            out_of_env = self.check_if_out_of_env(position=position)
        truncated = False if not out_of_env else True

        current_distance_to_goal = self.get_distance_to_goal(position, self.goal_point)
        delta_d = self.last_distance_to_goal - current_distance_to_goal

        delta_d = delta_d * delta_d_coef / self.start_goal_dist
        self.last_distance_to_goal = current_distance_to_goal

        dist_obstackle = 1 - self.min_collision_dist
        if dist_obstackle < 0.9:
            dist_obstackle = 0
        # logger.info(f"\nReward components: delta_d={delta_d}, 0.5*speed_xy_current={0.5*speed_xy_current}, dist_obstackle={dist_obstackle}")
        reward = delta_d + speed_xy_current - dist_obstackle

        return reward, terminated, truncated

    def compute_reward(self):
        if self.env_type == "indoor":
            return self.reward_indoor()
        elif self.env_type == "outdoor":
            return self.reward_outdoor_z()
        else:
            return NotImplementedError()

    def get_distance_to_goal(self, position, goal):
        position_current = np.array(
            [position.x_val, position.y_val, position.z_val], dtype=np.float32
        )

        return get_distance_to_goal_3d(position_current, goal)

    def reset(self):
        self.level = 0
        logger.info(f"doing reset")
        self.client.simPause(False)
        # self.client.confirmConnection()
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        start_point_id = sample(self.starts_goals.keys(), 1)[0]
        goal_point_id = sample(self.starts_goals[start_point_id], 1)[0]
        # start/ goal point fot both indoor/outdoor
        self.goal_point = self.points[goal_point_id]
        self.start_point = self.points[start_point_id]
        self.start_goal_dist = get_distance_to_goal_3d(
            self.start_point, self.goal_point
        )

        self.last_distance_to_goal = get_distance_to_goal_3d(
            self.start_point, self.goal_point
        )
        self.last_z = (
            self.client.getMultirotorState().kinematics_estimated.position.z_val
        )

        x, y, reset_height = self.points[start_point_id]
        angle = 0
        if self.action_type == "discrete":
            angle = sample(self.yaws, 1)[0]

        reset_pos = airsim.Pose(
            airsim.Vector3r(x, y, reset_height),
            airsim.to_quaternion(0, 0, (angle) * np.pi / 180),
        )

        self.client.moveByVelocityAsync(0, 0, 0, 2 * ACTION_DURATION).join()
        self.client.simSetVehiclePose(reset_pos, ignore_collision=True)

        self.client.hoverAsync().join()
        if self.action_type == "discrete":
            self.client.simPause(True)
        observation = self.get_observation()
        info = {}
        if self.get_vel_obs:
            vel = self._get_info(get_kinematic=True)
            observation = [observation, vel]
        time.sleep(0.3)

        return observation, info

    def _get_info(self, get_kinematic=False):

        if get_kinematic:
            kinematic = self.client.getMultirotorState().kinematics_estimated
            position = kinematic.position
            quad_vel = kinematic.linear_velocity
            vel = np.array(
                [quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float32
            )
            return vel
        else:
            return {}

    def get_raw_observ(self, depth=True):
        """
        AirSim observation : image from fpv-camera of drone
        :return:
        """
        if depth:
            observation = get_DepthImageRGB(
                self.client,
            )
        else:
            observation = get_MonocularImageRGB(self.client)
        return observation

    def get_observation(self):
        raw_observation = self.get_raw_observ(depth=self.observation_as_depth)

        if self.observation_as_depth:
            # while True:
            #     cv2.imshow('raw_observation clipped not *3',np.clip(raw_observation, 0, 254.5))
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            raw_observation = np.clip(raw_observation, 0, 254.5)
            image_scaled = raw_observation / 255.0
            image_scaled = image_scaled[..., None]
            # while True:
            #     cv2.imshow('raw_observation clipped', raw_observation)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break

        else:
            image_scaled = raw_observation  # already normalized from 0 to 1
        self.min_collision_dist = np.min(image_scaled) * 1000
        return image_scaled

    def define_noop_action(self):
        return 0


def start_environment(exe_path):
    path = exe_path
    # env_process = []
    env_process = subprocess.Popen(path)
    time.sleep(5)
    logger.info("Successfully loaded environment: " + exe_path)
    return env_process


def connect_drone(ip_address="127.0.0.5", num_agents=1, client=[]):
    if client != []:
        client.reset()
    client = airsim.MultirotorClient(ip=ip_address, timeout_value=10)
    client.confirmConnection()
    time.sleep(0.1)

    for agents in range(num_agents):
        name_agent = "drone" + str(agents)
        client.enableApiControl(True, name_agent)
        client.armDisarm(True, name_agent)
        client.takeoffAsync()
        time.sleep(0.1)

    return client


def connect_exe_env(
    cfg_agent,
    exe_path,
    cfg_env_path,
    documents_path,
):
    cfg = read_cfg(config_filename="./configs/config.cfg", verbose=False)
    cfg.num_agents = 1

    cfg_env = ConfigParser()
    cfg_env.read(cfg_env_path)

    points, starts_goals, airsim_positions_raw, done_xy = get_airsim_position(
        cfg_env.get("environment", "name")
    )
    generate_json(
        cfg, initial_positions=airsim_positions_raw, documents_path=documents_path
    )

    env_process = start_environment(exe_path)
    client = connect_drone()  # first takeoff

    env_airsim = AirSimGym_env(
        client,
        cfg_env,
        cfg_agent.get("agent", "action_type"),
        starts_goals,
        points,
        done_xy=done_xy,
    )

    if cfg_env.getboolean("environment", "get_vel_obs"):
        logger.info(
            f" Created invironment with observations as [depth, velocity]. images stacking not used "
        )
        return env_airsim, env_process

    vec_env = DummyVecEnv(env_airsim)
    # set batched environment
    env = BatchedPytorchFrameStack(
        vec_env, k=cfg_env.getint("environment", "stack_last_k")
    )
    return env, env_process


def close_env(env_process):
    process = psutil.Process(env_process.pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()
    logger.info("Environment closed")


def get_DepthImageRGB(client):
    camera_name = 1
    max_tries = 5
    tries = 0
    correct = False
    while not correct and tries < max_tries:
        tries += 1
        responses = client.simGetImages(
            [airsim.ImageRequest(camera_name, airsim.ImageType.DepthVis, True)],
        )
        responses = responses[0]
        if int(responses.width) > 0:
            correct = True
        if responses.width == 0:
            logger.info(f"\n /// BUG ***\n")
            logger.info(f"responses =BUG= {responses}")
            logger.info(
                f"responses.width, responses.height = {responses.width, responses.height}"
            )

        depth = airsim.list_to_2d_float_array(
            responses.image_data_float, responses.width, responses.height
        )

    return depth


def get_MonocularImageRGB(client):

    responses1 = client.simGetImages(
        [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)],
    )  # scene vision image in uncompressed RGBA array

    response = responses1[0]
    img1d = np.fromstring(
        response.image_data_uint8, dtype=np.uint8
    )  # get numpy array / image_data_uint8
    img_rgba = img1d.reshape(response.height, response.width, 3)
    img = Image.fromarray(img_rgba)
    img_rgb = img.convert("RGB")
    camera_image_rgb = np.asarray(img_rgb)
    camera_image = camera_image_rgb / 255

    return camera_image


def check_env_connection():
    cfg_agent_path = "configs/agents_conf/dueling_double_dqn.ini"
    exe_path = "unreal_envs/AirSimNH/AirSimNH/WindowsNoEditor/AirSimNH.exe"
    cfg_env_path = "configs/env_conf/cfg_NH_ddqn.ini"
    documents_path = "../../../../../Documents"

    cfg_agent = ConfigParser()
    cfg_agent.read(cfg_agent_path)

    # configs\env_conf\cfg_building99.ini
    env, env_process = connect_exe_env(
        exe_path=exe_path,
        documents_path=documents_path,
        cfg_env_path=cfg_env_path,  # "configs/env_conf/cfg_NH.ini" "configs/env_conf/cfg_building99.ini"
        cfg_agent=cfg_agent,  #
    )
    # observation_as_depth=True (72, 128, 1) image_scaled: max=0.998039186000824 self.observation_shape = (72, 128, 1)
    # observation_as_depth=False (72, 128, 3) image_scaled: max=0.984313725490196 self.observation_shape = (72, 128, 3)

    observation = env.reset()
    # logger.info(f'env.observation_space.shape[2]={env.observation_space.shape}')

    # stacked = np.stack([lazy_frames.get_frames() for lazy_frames in observation])
    # logger.info(f'observation = {type(stacked)},{np.shape(stacked)}')
    # logger.info(f'env.observation_space.shape[2]={env.observation_space.shape[2]}')

    # -------------- testing step()

    time.sleep(4)

    for _ in range(10):
        observation, reward, terminated, truncated, info = env.step(1)

        logger.info(f"reward: {reward}")
    time.sleep(0.5)


if __name__ == "__main__":
    check_env_connection()
