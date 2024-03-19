import subprocess
import time
import os

import cv2

from airsim_env import close_env
from utils.utils import generate_json
import airsim
from baselines_wrappers.monitor import Monitor
from baselines_wrappers.dummy_vec_env import DummyVecEnv
from airsim_env import AirSimGym_env, make_airsim_deepmind
from utils.pytorch_wrappers import BatchedPytorchFrameStack
from utils.utils import read_cfg, generate_json_simple_maze, create_folder, load_save_logg_reward, visualize_observation
from utils.initial_positions import get_airsim_position
from agents.dqn import *
from agents.double_dqn import *
from agents.dueling_double_dqn import *

EPOCHS = 30
LOGG = True # <------- change here
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
    time.sleep(0.1)


    for agents in range(num_agents):
        name_agent = "drone" + str(agents)
        client.enableApiControl(True, name_agent)
        client.armDisarm(True, name_agent)
        client.takeoffAsync(vehicle_name=name_agent).join()
        time.sleep(0.1)

    return client


def connect_exe_env(height_airsim_restart_positions, env_type='outdoor', max_episode_steps=100,
                    documents_path = '../../../../../Documents',
                    exe_path="./unreal_envs/outdoor_courtyard/outdoor_courtyard.exe",
                    name='outdoor_courtyard'):
    cfg = read_cfg(config_filename='./configs/config.cfg', verbose=False)
    cfg.num_agents = 1
    restart_positions, airsim_positions_raw, done_xy = get_airsim_position(name=name)
    generate_json(cfg, initial_positions=airsim_positions_raw, documents_path=documents_path)

    env_process = start_environment(exe_path)
    client = connect_drone()  # first takeoff

    env_airsim = AirSimGym_env(client, env_type=env_type, vehicle_name='drone0', action_type='discrete',
                               initial_positions=restart_positions, observation_as_depth=True, done_xy=done_xy, height_airsim_restart_positions = height_airsim_restart_positions)
    make_env = lambda: Monitor(make_airsim_deepmind(env_airsim, max_episode_steps=max_episode_steps),
                               allow_early_resets=True)
    # set batched environment
    vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
    env = BatchedPytorchFrameStack(vec_env, k=2)

    return env, env_process

def connect_indoor_simple_env(exe_path="./unreal_envs/easy_maze/Blocks.exe"):
    cfg = read_cfg(config_filename='./configs/config.cfg', verbose=False)
    cfg.num_agents = 1
    restart_positions, airsim_positions_raw, done_xy = get_airsim_position(name='indoor_maze_easy')
    generate_json(cfg, initial_positions=airsim_positions_raw, documents_path = '../../../../../Documents')
    env_process = start_environment(exe_path)
    client = connect_drone()  # first takeoff

    env_airsim = AirSimGym_env(client, env_type='indoor', vehicle_name='drone0', action_type='discrete',
                               initial_positions=restart_positions, observation_as_depth=True, done_xy=done_xy,height_airsim_restart_positions=[-0.8339])
    make_env = lambda: Monitor(make_airsim_deepmind(env_airsim, max_episode_steps=100),
                               allow_early_resets=True)
    # set batched environment
    vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
    env = BatchedPytorchFrameStack(vec_env, k=2)

    return env, env_process
def inference_setup(env):
    load_path = './saved_weights/dueling_ddqn2/restart_5/'  # <------- change here
    online_net = Double_Dueling_DQN(env=env, save_path=load_path, load_path=load_path) # <------- change here
    states = env.reset()
    res = 3
    for step in itertools.count():
        # select action
        # res = visualize_observation(states)
        states_ = np.stack([lasy.get_frames() for lasy in states])
        actions = online_net.action(states_, epsilon=-1, inference=True)
        # take action
        new_states, rewards, terminated, truncated, infos = env.step(actions)
        logger.info(f'actions = {actions}, rewards = {rewards}, terminated = {terminated}, truncated = {truncated}')
        states = new_states

        if terminated[0] or truncated[0]:
            env.reset()
            time.sleep(1)
        if res == 0:
            break


def inference(height_airsim_restart_positions):
    env, env_process = connect_exe_env(height_airsim_restart_positions, exe_path="./unreal_envs/outdoor_courtyard/outdoor_courtyard.exe")
    inference_setup(env)
    close_env(env_process)


def train_outroor_DQN(logg_tb, save_path, epoch, reward_loggs, height_airsim_restart_positions=[-5.35,-5.4, -6.5,-7.6,-8.5, -9. ], load_path=None):
    env, env_process = connect_exe_env(height_airsim_restart_positions, exe_path="./unreal_envs/outdoor_courtyard/outdoor_courtyard.exe")
    res = 0
    try:
        res = training_dqn(env, logg_tb=logg_tb, epoch=epoch, save_path=save_path, load_path=load_path,
                           reward_loggs=reward_loggs)
        close_env(env_process)
    except ValueError as e:
        if str(e) == 'cannot reshape array of size 1 into shape (0,0)':
            logger.info('Recovering from AirSim error')
            close_env(env_process)
            res = -2
    except Exception as restart:
        logger.info(f'API is dead... \n{str(restart)}\nClose .exe ')
        close_env(env_process)
        res = -2

    return res


def train_outroor_DDQN(logg_tb, save_path, epoch, reward_loggs, height_airsim_restart_positions=[-5.35,-5.4, -6.5,-7.6,-8.5, -9. ], load_path=None):
    env, env_process = connect_exe_env(height_airsim_restart_positions, exe_path="./unreal_envs/outdoor_courtyard/outdoor_courtyard.exe")
    res = 0
    try:
        res = training_ddqn(env, logg_tb=logg_tb, epoch=epoch, save_path=save_path, load_path=load_path,
                            reward_loggs=reward_loggs)
        close_env(env_process)
    except ValueError as e:
        if str(e) == 'cannot reshape array of size 1 into shape (0,0)':
            logger.info('Recovering from AirSim error')
            close_env(env_process)
            res = -2
    except Exception as restart:
        logger.info(f'API is dead... \n{str(restart)}\nClose .exe ')
        close_env(env_process)
        res = -2

    return res


def train_outroor_DDDQN(logg_tb, save_path, epoch, reward_loggs, height_airsim_restart_positions = [-5.35,-5.4, -6.5,-7.6,-8.5, -9. ],load_path=None):
    env, env_process = connect_exe_env(height_airsim_restart_positions, env_type='indoor', max_episode_steps=100,
                                       exe_path="./unreal_envs/easy_maze/Blocks.exe",
                                       name='indoor_maze_easy')
    res = 0
    try:
        res = training_dddqn(env, logg_tb=logg_tb, epoch=epoch, save_path=save_path, load_path=load_path,
                             reward_loggs=reward_loggs)
        close_env(env_process)
    except ValueError as e:
        if str(e) == 'cannot reshape array of size 1 into shape (0,0)':
            logger.info('Recovering from AirSim error')
            close_env(env_process)
            res = -2
    except Exception as restart:
        logger.info(f'API is dead... \n{str(restart)}\nClose .exe ')
        close_env(env_process)
        res = -2

    return res


def main_dqn():
    for epoch in range(EPOCHS):

        LOGG_TB_DIR = f"logs/dqn_TarUpdFrqcy_maxEposodeStep/restart_exe_{epoch}/"  # <------- change here
        SAVE_PATH = f"./saved_weights/dqn_TarUpdFrqcy_maxEposodeStep/restart_{epoch}/"  # <------- change here

        csv_rewards_log = 'restart_best_rewards'
        create_folder(SAVE_PATH)
        load_path = None
        rewards_logs = load_save_logg_reward(save=False, save_path=SAVE_PATH, csv_rewards_log=csv_rewards_log)
        if len(rewards_logs) > 1:
            restart_n = rewards_logs[rewards_logs['reward'] == rewards_logs['reward'].max()]['restart_n'].values[0]
            load_path = f"./saved_weights/dqn_TarUpdFrqcy_maxEposodeStep/{restart_n}"  # <------- change here

        train_outroor_DQN(logg_tb=LOGG_TB_DIR, save_path=SAVE_PATH, epoch=epoch, load_path=load_path,
                          reward_loggs=rewards_logs)
        time.sleep(5)


def main_ddqn():
    for epoch in range(EPOCHS):

        LOGG_TB_DIR = f"logs/ddqn/restart_exe_{epoch}/"  # <------- change here
        SAVE_PATH = f"./saved_weights/ddqn/restart_{epoch}/"  # <------- change here

        csv_rewards_log = 'restart_best_rewards'
        create_folder(SAVE_PATH)
        load_path = None
        rewards_logs = load_save_logg_reward(save=False, save_path=SAVE_PATH, csv_rewards_log=csv_rewards_log)
        if len(rewards_logs) > 1:
            restart_n = rewards_logs[rewards_logs['reward'] == rewards_logs['reward'].max()]['restart_n'].values[0]
            load_path = f"./saved_weights/ddqn/{restart_n}"  # <------- change here

        train_outroor_DDQN(logg_tb=LOGG_TB_DIR, save_path=SAVE_PATH, epoch=epoch, load_path=load_path,
                           reward_loggs=rewards_logs)
        time.sleep(5)


def main_dddqn():
    for epoch in range(EPOCHS):

        LOGG_TB_DIR = f"logs/dueling_ddqn/restart_exe_{epoch}/"  # <------- change here
        SAVE_PATH = f"./saved_weights/dueling_ddqn/restart_{epoch}/"  # <------- change here

        csv_rewards_log = 'restart_best_rewards'
        create_folder(SAVE_PATH)
        load_path = None
        rewards_logs = load_save_logg_reward(save=False, save_path=SAVE_PATH, csv_rewards_log=csv_rewards_log)
        if len(rewards_logs) > 1:
            restart_n = rewards_logs[rewards_logs['reward'] == rewards_logs['reward'].max()]['restart_n'].values[0]
            load_path = f"./saved_weights/dueling_ddqn/{restart_n}"  # <------- change here

        train_outroor_DDDQN(logg_tb=LOGG_TB_DIR, save_path=SAVE_PATH, epoch=epoch, load_path=load_path,
                            reward_loggs=rewards_logs, height_airsim_restart_positions= [-0.8339] )
        time.sleep(5)


def try_maze():
    env, env_process = connect_indoor_simple_env()
    env.reset()
    time.sleep(3)
    logger.info(f'env created')
    logger.info(f'step 0')
    observation, reward, terminated, truncated, info = env.step([0])

    visualize_observation(observation)
    logger.info(f'step 1')
    observation, reward, terminated, truncated, info=env.step([1])

    visualize_observation(observation)

    logger.info(f'step 2')
    observation, reward, terminated, truncated, info=env.step([2])

    visualize_observation(observation)

    logger.info(f'step 3')
    observation, reward, terminated, truncated, info=env.step([3])

    visualize_observation(observation)

    logger.info(f'step 4')
    observation, reward, terminated, truncated, info=env.step([4])

    visualize_observation(observation)

    logger.info(f'step 5')
    observation, reward, terminated, truncated, info=env.step([5])

    visualize_observation(observation)

    close_env(env_process)


if __name__ == "__main__":

    try_maze()

    #inference()  # main()


