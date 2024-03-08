import subprocess
from airsim_env import close_env
from utils.utils import generate_json
import airsim
from baselines_wrappers.monitor import Monitor
from baselines_wrappers.dummy_vec_env import DummyVecEnv
from airsim_env import AirSimGym_env, make_airsim_deepmind
from utils.pytorch_wrappers import BatchedPytorchFrameStack, PytorchLazyFrames
from agents.dqn import *
from utils.utils import read_cfg, visualize_observation
from unreal_envs.initial_positions import get_airsim_position
from agents.dqn import *



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


def train_outroor_DQN():
    env, env_process = connect_exe_env(exe_path="./unreal_envs/outdoor_courtyard/outdoor_courtyard.exe")
    res = training_dqn(env)
    #inference_setup(env)
    time.sleep(2)
    close_env(env_process)
    time.sleep(5)

def main():
    for epoch in range(3):
        global LOGGING_DIR
        LOGGING_DIR = f"logs/dqn/{time.time()}"
        train_outroor_DQN()


if __name__ == "__main__":
    main()

