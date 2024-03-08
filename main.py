import cv2
import random
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import subprocess
from airsim_env import close_env
import itertools
from utils.utils import generate_json
import airsim
from baselines_wrappers.monitor import Monitor
from baselines_wrappers.dummy_vec_env import DummyVecEnv
from airsim_env import AirSimGym_env, make_airsim_deepmind
from pytorch_wrappers import BatchedPytorchFrameStack, PytorchLazyFrames

from utils.utils import read_cfg, visualize_observation
from unreal_envs.initial_positions import get_airsim_position
from agents.dqn import *


LOGGING_DIR = f"logs/dqn/{time.time()}"
LOGG = True
RESTART_EXE = 10

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
    env = BatchedPytorchFrameStack(vec_env, k=2)

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

def connect_exe_env(exe_path = "./unreal_envs/outdoor_courtyard/outdoor_courtyard.exe"):

    cfg = read_cfg(config_filename='./configs/config.cfg', verbose=False)
    cfg.num_agents = 1
    restart_positions, airsim_positions_raw, done_xy = get_airsim_position(name='outdoor_courtyard')
    generate_json(cfg, initial_positions=airsim_positions_raw)
    env_process = start_environment(exe_path)
    client = connect_drone()  # first takeoff

    env_airsim = AirSimGym_env(client, env_type='outdoor', vehicle_name='drone0', action_type='discrete',
                               initial_positions=restart_positions, observation_as_depth=True, done_xy=done_xy)
    make_env = lambda: Monitor(make_airsim_deepmind(env_airsim, render_mode='rgb_array', scale_values=True),
                               allow_early_resets=True)
    # set batched environment
    vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
    env = BatchedPytorchFrameStack(vec_env, k=2)

    return env, env_process

def inference_setup(env):
    online_net = DQN(env=env)
    states = env.reset()
    for step in itertools.count():
        # select action
        visualize_observation(states)
        states_ = np.stack([lasy.get_frames() for lasy in states])
        actions = online_net.action(states_, epsilon = -1, inference=True)
        # take action


        new_states, rewards, dones, infos = env.step(actions)
        logger.info(f'actions = {actions}, rewards = {rewards}')
        states = new_states

        if dones[0]:
            env.reset()


def training_setup(env):
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    info_buffer = deque(maxlen=100)
    online_net = DQN(env=env)
    target_net = DQN(env=env)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = Adam(lr=LR, params=online_net.parameters())
    tb_summary = SummaryWriter(LOGGING_DIR)

    episode_count = 0

    # init replay buffer before training
    states = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        actions = [env.action_space.sample() for _ in range(NUM_ENVS)]  # sample from env randomly
        new_states, rewards, dones, _ = env.step(actions)

        for state, action, reward, done, new_state in zip(states, actions, rewards, dones, new_states):
            transition = (state, action, reward, done, new_state)
            replay_buffer.append(transition)
        states = new_states

    # main training loop
    states = env.reset()

    last_rew = -10.
    for step in itertools.count():
        # select action
        epsilon = np.interp(step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        if isinstance(states[0], PytorchLazyFrames):
            states_ = np.stack([lasy.get_frames() for lasy in states])
            actions = online_net.action(states_, epsilon)
        else:
            actions = online_net.action(states, epsilon)  # epsilon - random policy now inside .action

        # take action

        new_states, rewards, dones, infos = env.step(actions)

        for state, action, reward, done, new_state, info in zip(states, actions, rewards, dones, new_states, infos):
            transition = (state, action, reward, done, new_state)
            replay_buffer.append(transition)

            if done:
                info_buffer.append(info['episode'])
                episode_count += 1

        states = new_states

        transition_sample = random.sample(replay_buffer, BATCH_SIZE)
        loss = online_net.compute_loss(target_net, transition_sample)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % TARGET_UPDATE_FREQ:
            target_net.load_state_dict(online_net.state_dict())

        if step % LOGGING_INTERVAL == 0:
            mean_rew = np.mean([e['r'] for e in info_buffer]) or 0
            if mean_rew > last_rew:
                logger.info(f"\n*****\nCkeckpoint for best model with reward = {mean_rew} at step {step}. Saving model weights....")
                online_net.save_best_last(best=True)
            logger.info(
                f"\nCkeckpoint for last model with reward = {mean_rew} at step {step}. Saving model weights....")
            online_net.save_best_last(best=False)

            last_rew = mean_rew
            mean_duration = np.mean([e['l'] for e in info_buffer]) or 0
            logger.info(f'Episode: {step}\nReward  == {mean_rew}\nDuration == {mean_duration}')

            tb_summary.add_scalar('mean_rew', mean_rew if mean_rew is not None else 0, global_step=step)
            tb_summary.add_scalar('mean_duration', mean_duration if mean_duration is not None else 0, global_step=step)
            tb_summary.add_scalar('episode_count', episode_count, global_step=step)

            if step > RESTART_EXE:
                logger.info(f'Episode: {step}\nRestart .exe')
                return -1

def try_train_outroor_DQN():
    env, env_process = connect_exe_env(exe_path="./unreal_envs/outdoor_courtyard/outdoor_courtyard.exe")
    res = training_setup(env)
    #inference_setup(env)

    time.sleep(2)
    close_env(env_process)
    time.sleep(5)

def main():
    for epoch in range(3):
        global LOGGING_DIR
        LOGGING_DIR = f"logs/dqn/{time.time()}"
        try_train_outroor_DQN()


if __name__ == "__main__":
    main()

