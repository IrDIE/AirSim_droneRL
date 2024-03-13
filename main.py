import subprocess
import time
import os
from airsim_env import close_env
from utils.utils import generate_json
import airsim
from baselines_wrappers.monitor import Monitor
from baselines_wrappers.dummy_vec_env import DummyVecEnv
from airsim_env import AirSimGym_env, make_airsim_deepmind
from utils.pytorch_wrappers import BatchedPytorchFrameStack, PytorchLazyFrames
from utils.utils import read_cfg, visualize_observation, create_folder
from unreal_envs.initial_positions import get_airsim_position
from agents.dqn import *
from agents.double_dqn import *


EPOCHS = 150
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
    time.sleep(0.1)

    old_posit = {}
    for agents in range(num_agents):
        name_agent = "drone" + str(agents)
        client.enableApiControl(True, name_agent)
        client.armDisarm(True, name_agent)
        client.takeoffAsync(vehicle_name=name_agent).join()
        time.sleep(0.1)
        old_posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)
    return client

def connect_exe_env(exe_path = "./unreal_envs/outdoor_courtyard/outdoor_courtyard.exe"):

    cfg = read_cfg(config_filename='./configs/config.cfg', verbose=False)
    cfg.num_agents = 1
    restart_positions, airsim_positions_raw, done_xy = get_airsim_position(name='outdoor_courtyard')
    generate_json(cfg, initial_positions=airsim_positions_raw)
    env_process = start_environment(exe_path)
    client = connect_drone()  # first takeoff

    env_airsim = AirSimGym_env(client, env_type='outdoor', vehicle_name='drone0', action_type='discrete',
                               initial_positions=restart_positions, observation_as_depth=True, done_xy=done_xy)
    make_env = lambda: Monitor(make_airsim_deepmind(env_airsim, max_episode_steps = 100 ),
                               allow_early_resets=True)
    # set batched environment
    vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
    env = BatchedPytorchFrameStack(vec_env, k=2)

    return env, env_process

def inference_setup(env):
    load_path = './saved_weights/dqn/restart_5/dqn_best.pt'
    online_net = DQN(env=env, save_path=load_path, load_path=load_path)
    states = env.reset()
    res = 3
    for step in itertools.count():
        # select action
        #res = visualize_observation(states)
        states_ = np.stack([lasy.get_frames() for lasy in states])
        actions = online_net.action(states_, epsilon = -1, inference=True)
        # take action
        new_states, rewards, terminated, truncated, infos = env.step(actions)
        logger.info(f'actions = {actions}, rewards = {rewards}, terminated = {terminated}, truncated = {truncated}')
        states = new_states

        if terminated[0] or truncated[0]:
            env.reset()
            time.sleep(0.01)
        if res == 0:
            break

def inference():
    env, env_process = connect_exe_env(exe_path="./unreal_envs/outdoor_courtyard/outdoor_courtyard.exe")
    inference_setup(env)
    close_env(env_process)

def train_outroor_DQN(logg_tb, save_path, epoch, reward_loggs, load_path = None):
    env, env_process = connect_exe_env(exe_path="./unreal_envs/outdoor_courtyard/outdoor_courtyard.exe")
    res=0
    try:
        res = training_dqn(env, logg_tb=logg_tb, epoch=epoch, save_path=save_path, load_path=load_path, reward_loggs=reward_loggs)
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

def train_outroor_DDQN(logg_tb, save_path, epoch, reward_loggs, load_path = None):
    env, env_process = connect_exe_env(exe_path="./unreal_envs/outdoor_courtyard/outdoor_courtyard.exe")
    res=0
    try:
        res = training_ddqn(env, logg_tb=logg_tb, epoch=epoch, save_path=save_path, load_path=load_path, reward_loggs=reward_loggs)
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

        LOGG_TB_DIR = f"logs/dqn2/restart_exe_{epoch}/" # <------- change here
        SAVE_PATH = f"./saved_weights/dqn2/restart_{epoch}/" # <------- change here

        csv_rewards_log = 'restart_best_rewards'
        create_folder(SAVE_PATH)
        load_path = None
        rewards_logs = load_save_logg_reward(save=False, save_path=SAVE_PATH, csv_rewards_log=csv_rewards_log)
        if len(rewards_logs) > 1:
            restart_n = rewards_logs[rewards_logs['reward'] == rewards_logs['reward'].max()]['restart_n'].values[0]
            load_path = f"./saved_weights/dqn2/{restart_n}" # <------- change here

        train_outroor_DQN(logg_tb = LOGG_TB_DIR, save_path = SAVE_PATH, epoch=epoch, load_path = load_path,
                          reward_loggs=rewards_logs)
        time.sleep(5)

def main_ddqn():
    for epoch in range(EPOCHS):

        LOGG_TB_DIR = f"logs/ddqn/restart_exe_{epoch}/" # <------- change here
        SAVE_PATH = f"./saved_weights/ddqn/restart_{epoch}/" # <------- change here

        csv_rewards_log = 'restart_best_rewards'
        create_folder(SAVE_PATH)
        load_path = None
        rewards_logs = load_save_logg_reward(save=False, save_path=SAVE_PATH, csv_rewards_log=csv_rewards_log)
        if len(rewards_logs) > 1:
            restart_n = rewards_logs[rewards_logs['reward'] == rewards_logs['reward'].max()]['restart_n'].values[0]
            load_path = f"./saved_weights/ddqn/{restart_n}" # <------- change here

        train_outroor_DDQN(logg_tb = LOGG_TB_DIR, save_path = SAVE_PATH, epoch=epoch, load_path = load_path,
                          reward_loggs=rewards_logs)
        time.sleep(5)


if __name__ == "__main__":
    main_dqn() #main()

