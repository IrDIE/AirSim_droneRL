import cv2
import gym
import numpy as np
import nvidia_smi
import os, subprocess
from airsim_env import close_env
import math
import random
import json
import time
from utils import generate_json
import airsim
from PIL import Image
from baselines_wrappers.monitor import Monitor
from baselines_wrappers.dummy_vec_env import DummyVecEnv, VecEnv
from loguru import logger
from airsim_env import AirSimGym_env, make_airsim_deepmind
from pytorch_wrappers import BatchedPytorchFrameStack

from utils import read_cfg, visualize_observation

from unreal_envs.initial_positions import outdoor_courtyard, get_airsim_position

NUM_ENVS = 1
LOGG = True

if LOGG: logger.add(f"{os.path.dirname(os.path.realpath(__file__))}/logs/log_{time.time()}.log")

def start_environment(exe_path):
    path = exe_path
    # env_process = []
    env_process = subprocess.Popen(path)
    time.sleep(5)
    logger.info("Successfully loaded environment: " + exe_path)
    return env_process
def connect_drone(ip_address='127.0.0.5', num_agents=1, client=[]):
    if client != []:
        client.reset()
    client = airsim.MultirotorClient(ip=ip_address, timeout_value=10)
    client.confirmConnection()
    time.sleep(1)

    old_posit = {}
    for agents in range(num_agents):
        name_agent = "drone" + str(agents)
        client.enableApiControl(True, name_agent)
        client.armDisarm(True, name_agent)
        client.takeoffAsync(vehicle_name=name_agent).join()
        time.sleep(0.1)
        old_posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)
    return client

def try_reset_AirsimGymEnv():
    exe_path = "./unreal_envs/outdoor_courtyard/outdoor_courtyard.exe"
    cfg = read_cfg(config_filename='./configs/config.cfg', verbose=False)
    cfg.num_agents = 1
    restart_positions, airsim_positions_raw, done_xy = get_airsim_position(name='outdoor_courtyard')
    generate_json(cfg, initial_positions=airsim_positions_raw)
    env_process = start_environment(exe_path)
    client, _, initial_position = connect_drone()  # first takeoff
    env = AirSimGym_env(client, env_type='outdoor', vehicle_name='drone0',
                               initial_positions=restart_positions, done_xy=done_xy)


    logger.info(f'try reset with another initial position')
    time.sleep(2)

    i = 0
    while True:
        # observation_ = env.get_observation()
        # observation_scaled = cv2.resize(observation_, (observation_.shape[1] ,observation_.shape[0] ))
        # cv2.imshow('', observation_scaled)
        logger.info(f'i ==========================================  {i}')
        if i % 6 == 0 and i != 0:
            observation_ , _= env.reset()
            time.sleep(5)

            observation_, _ = env.reset()
            time.sleep(5)

            observation_, _ = env.reset()
            time.sleep(5)

        i+=1



    close_env(env_process)

def try_gym_make_batched_AirsimGymEnv():
    exe_path = "./unreal_envs/outdoor_courtyard/outdoor_courtyard.exe"

    cfg = read_cfg(config_filename='./configs/config.cfg', verbose=False)
    cfg.num_agents = 1
    restart_positions, airsim_positions_raw,done_xy = get_airsim_position(name='outdoor_courtyard')
    generate_json(cfg, initial_positions=airsim_positions_raw)
    env_process = start_environment(exe_path)
    client, _, initial_position = connect_drone()  # first takeoff
    env_airsim = AirSimGym_env(client, env_type='outdoor', vehicle_name='drone0',
                        initial_positions=restart_positions, observation_as_depth = False,done_xy=done_xy)

    make_env = lambda: Monitor(make_airsim_deepmind(env_airsim, render_mode='rgb_array', scale_values=True),
                               allow_early_resets=True)


    # set batched environment
    vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
    env = BatchedPytorchFrameStack(vec_env, k=3)
    observation = env.reset()

    logger.info(f'shape observation 0 .get_frames() = {observation[0].get_frames().shape}')

    frames3 = observation[0].get_frames()
    fr1, fr2, fr3 = frames3[:3, :, :].transpose(1,2,0), frames3[3:6, :, :].transpose(1,2,0), frames3[6:9, :, :].transpose(1,2,0)

    fr = np.concatenate((fr1, fr2, fr3), axis=1)

    while True:
        observation_scaled = cv2.resize(fr, (fr.shape[1] ,fr.shape[0] ))
        cv2.imshow('', observation_scaled)
        if cv2.waitKey(33) == ord('q'): break

    close_env(env_process)

def try_step_batched_AirsimGymEnv():
    exe_path = "./unreal_envs/outdoor_courtyard/outdoor_courtyard.exe"
    cfg = read_cfg(config_filename='./configs/config.cfg', verbose=False)
    cfg.num_agents = 1
    restart_positions, airsim_positions_raw ,done_xy = get_airsim_position(name='outdoor_courtyard')
    generate_json(cfg, initial_positions=airsim_positions_raw)
    env_process = start_environment(exe_path)
    client = connect_drone()  # first takeoff

    env_airsim = AirSimGym_env(client, env_type='outdoor', vehicle_name='drone0', action_type= 'discrete',
                        initial_positions=restart_positions, observation_as_depth = True,done_xy=done_xy)
    make_env = lambda: Monitor(make_airsim_deepmind(env_airsim, render_mode='rgb_array', scale_values=True),
                               allow_early_resets=True)
    # set batched environment
    vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
    env = BatchedPytorchFrameStack(vec_env, k=2) # TODO - WTF
    observation = env.reset()
    visualize_observation(observation)
    logger.info(f'done reset() \nstart doing .step() ')
    time.sleep(2)
    observation, reward, done, info = env.step(actions=1)
    logger.info(f'reward = {reward}')
    visualize_observation(observation)
    time.sleep(1)
    observation, reward, done, info = env.step(actions=1)
    logger.info(f'reward = {reward}')
    visualize_observation(observation)
    time.sleep(1)
    observation, reward, done, info = env.step(actions=2)
    logger.info(f'reward = {reward}')
    visualize_observation(observation)
    time.sleep(1)
    observation, reward, done, info = env.step(actions=2)
    logger.info(f'reward = {reward}')
    visualize_observation(observation)
    time.sleep(1)


    # time.sleep(2)
    # observation, reward, done, info = env.step(actions=np.asarray([1, 1, 0, -1], dtype=np.float32))
    # visualize_observation(observation)
    # time.sleep(2)
    # observation, reward, done, info = env.step(actions=np.asarray([1, -1, 0, -5], dtype=np.float32))
    # visualize_observation(observation)
    # time.sleep(2)
    # observation, reward, done, info = env.step(actions=np.asarray([1, 1, 1, -2], dtype=np.float32))
    # time.sleep(2)
    # logger.info(f'done step() \nstart doing .step() ')
    # time.sleep(2)




    # logger.info(f'shape observation 0 .get_frames() = {observation[0].get_frames().shape}')


    close_env(env_process)


def main():
    from random import sample
    initial_positions = [
        [1,2],
        [0,22],
        [1, 5],
        [0, 77],
    ]
    for _ in range(20): print(sample([+1, -1], 1))


if __name__ == "__main__":
    try_step_batched_AirsimGymEnv()

