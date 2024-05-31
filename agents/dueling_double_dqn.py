from torch import nn
import itertools
import random
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import warnings
import numpy as np
import torch
from airsim_env import close_env, connect_exe_env
import time
from utils.utils import create_folder
from loguru import logger
from utils.utils import update_logg_reward,load_save_logg_reward
from utils.pytorch_wrappers import PytorchLazyFrames
warnings.filterwarnings('ignore')


LOAD_FROM = None # './saved_weights/dddqn_fixed/no_per/'
GAMMA = 0.99 # DISCOUNT RATE
BATCH_SIZE = 12#64 # 128 #32 # FROM REPLAY BUFFER
BUFFER_SIZE = 50_000
MIN_REPLAY_SIZE = 15 #5_000
EPSILON_START = 0.75 # E GREEDY POLICY
EPSILON_END = 0.05
EPSILON_DECAY = 20_000
LR = 1e-4
NUM_ENVS = 1
TARGET_UPDATE_FREQ = 1_000 // NUM_ENVS
LOGGING_INTERVAL =50
SAVE_WEIGHTS = './saved_weights/dddqn_fixed/no_per1/'
LOGG_TB_DIR = "./logs/dddqn_fixed/no_per1/"

def logg_hyperparams(tb_logger : SummaryWriter, info : dict={}):
    hyperparameters = {
        'LOAD_FROM': LOAD_FROM,
        'EPSILON_START':EPSILON_START,
        'EPSILON_END': EPSILON_END,
        'EPSILON_DECAY': EPSILON_DECAY,
        'BATCH_SIZE': BATCH_SIZE,
        'BUFFER_SIZE': BUFFER_SIZE,
        'MIN_REPLAY_SIZE': MIN_REPLAY_SIZE,
        'LR': LR,
        'GAMMA': GAMMA,
        'TARGET_UPDATE_FREQ': TARGET_UPDATE_FREQ,
        'SAVE_WEIGHTS': SAVE_WEIGHTS,
        'LOGG_TB_DIR': LOGG_TB_DIR,
        'LOGGING_INTERVAL' : LOGGING_INTERVAL
    }
    hyperparameters.update(info)
    hyp_str = '\n'.join(['%s =  %s\n' % (key, value) for (key, value) in hyperparameters.items()])
    tb_logger.add_text(text_string = hyp_str, tag='Hyperparameters')


class Double_Dueling_DQN(nn.Module):
    def __init__(self, env, save_path, device, load_ckpt=None,  optimizer=None):
        super(Double_Dueling_DQN, self).__init__()
        self.device = device
        self.not_calculated_flatten = True
        self.outp_size = 1024
        self.action_shape = env.action_space.n
        self.convNet = self.get_conv_net(env)

        self.dueling_state = self.get_dueling_state()
        self.dueling_action = self.get_dueling_action()

        self.save_path  = save_path



    def forward(self, x):
        enconed = self.convNet(x)
        V = self.dueling_state(enconed)
        A = self.dueling_action(enconed)
        return V,A


    def action(self, states, epsilon, inference=False):
        states = torch.tensor(states, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
        _, advantage = self.forward(states) # in online net -> get action advantage
        max_indexs = torch.argmax(advantage, dim=1)
        actions = max_indexs.detach().tolist()
        if not inference:
            for i in range(len(actions)):
                if np.random.random() <= epsilon :
                    actions[i] = np.random.randint(0, self.action_shape - 1)

        return actions

    def get_dueling_state(self):
        return nn.Sequential(
            nn.Linear(self.outp_size, self.outp_size),
            nn.ReLU(),
            nn.Linear(self.outp_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),

        )

    def get_dueling_action(self):
        return nn.Sequential(
            nn.Linear(self.outp_size, self.outp_size),
            nn.ReLU(),
            nn.Linear(self.outp_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_shape),

        )

    def get_conv_net(self, env):
        self.in_channels = list([env.observation_space.shape[2]])
        self.convNet_ = nn.Sequential(
            nn.Conv2d(self.in_channels[0], 32, kernel_size=(5, 5), stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        if self.not_calculated_flatten:
            with torch.no_grad():
                sample = env.observation_space.sample()[None]
                torch_sample = torch.as_tensor(sample).permute(0, 3, 1, 2)
                self.flatten_size = self.convNet_(torch_sample.float() ).shape[1]
            self.not_calculated_flatten = False

        return nn.Sequential(
            self.convNet_,
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.outp_size),
            nn.ReLU()
        )

    def compute_loss(self, target_net, trg):
        # trg - state, action, reward, terminated, truncated, new_state
        states_ = [t[0] for t in trg]
        actions = torch.as_tensor(np.asarray([t[1] for t in trg]), dtype=torch.int64, device=self.device).unsqueeze(-1)
        rews = torch.as_tensor(np.asarray([t[2] for t in trg]), dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones = torch.as_tensor(np.asarray([t[3] or t[4] for t in trg]), dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_states_ = [t[5] for t in trg]

        if isinstance(states_[0], PytorchLazyFrames):
            states = torch.as_tensor(np.stack([lazy_frames.get_frames() for lazy_frames in states_]), dtype=torch.float32, device=self.device)
            states = states.permute(0, 3, 1, 2)
            new_states = torch.as_tensor(np.stack([lazy_frames.get_frames() for lazy_frames in new_states_]), dtype=torch.float32, device=self.device)
            new_states = new_states.permute(0, 3, 1, 2)
        else :
            states = torch.as_tensor(states_, dtype=torch.float32, device=self.device)
            new_states = torch.as_tensor(np.asarray(new_states_), dtype=torch.float32, device=self.device)

        # for double:
        V_states, A_states = self.forward(states)

        V_new_states, A_new_states = target_net.forward(new_states)


        V_s_eval, A_s_eval = self.forward(new_states)
        q_pred = torch.add(V_states,
             (A_states - A_states.mean(dim=1, keepdim=True))) # action_q_values
        q_next = torch.add(V_new_states,
                       (A_new_states - A_new_states.mean(dim=1, keepdim=True)))
        q_eval = torch.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        best_q_index = q_eval.argmax(dim=1, keepdim=True) # max_actions
        targets_selected_q_values = torch.gather(input=q_next, dim = 1, index= best_q_index) # = q_next[indices, max_actions]
        targets = rews + GAMMA * (1 - dones) * targets_selected_q_values
        # loss
        action_q_values = torch.gather(q_pred, dim=1, index=actions)
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)
        #nn.functional.smooth_l1_loss()
        return loss

    def do_checkpoint(self, load = True, path_load=None, best = False, alias = '', optimizer=None):
        signature = 'dqn_best.pt' if best else 'dqn_last.pt' # save/load best or last
        signature = alias + signature
        if load:
            ckpt_info = torch.load(path_load + signature)
            logger.info(f'ckpt_info from path : {path_load + signature} \nloaded with keys = {ckpt_info.keys()}')
            self.load_state_dict(ckpt_info['model_state_dict'], strict=False)
            optimizer.load_state_dict(ckpt_info['optimizer_state_dict'])
        else: # save
            assert optimizer is not None
            ckpt_info = {
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(ckpt_info, self.save_path + signature)


def training_dddqn(env, logg_tb, save_path, load_ckpt=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'---------- device  =  {device} ')

    replay_buffer = deque(maxlen=BUFFER_SIZE)
    online_net = Double_Dueling_DQN(env=env, save_path=save_path, device=device,load_ckpt = LOAD_FROM).to(device)
    optimizer = Adam(lr=LR, params=online_net.parameters())
    target_net = Double_Dueling_DQN(env=env, save_path=save_path, device=device,load_ckpt=LOAD_FROM).to(device)
    if load_ckpt is not None:
        online_net.do_checkpoint(load=True, optimizer=optimizer, path_load=load_ckpt)
    tb_summary = SummaryWriter(logg_tb)
    logg_hyperparams(tb_summary)

    target_net.load_state_dict(online_net.state_dict())
    target_net.train()
    online_net.train()

    # init replay buffer before training
    global_step = 0
    while global_step < MIN_REPLAY_SIZE:
        states = env.reset()
        done = False
        global_step += 1
        if global_step % LOGGING_INTERVAL == 0: logger.info(f'COLLECTING REPLAY BUFFER at step = {global_step}')
        while not done:
            try:
                actions = [env.action_space.sample() for _ in range(NUM_ENVS)]  # sample from env randomly
                new_states, rewards, terminateds, truncateds, infos = env.step(actions)
                done = terminateds or truncateds
                for state, action, reward, terminated, truncated, new_state in zip(states, actions, rewards, terminateds,
                                                                                   truncateds, new_states):
                    transition = (state, action, reward, terminated, truncated, new_state)
                    replay_buffer.append(transition)

            except ValueError as e:
                if str(e) == 'cannot reshape array of size 1 into shape (0,0)':
                    logger.info(f'Recovering from AirSim error in replay buffer. global_step = {global_step}')
                    states = env.reset()
                    actions = [env.action_space.sample() for _ in range(NUM_ENVS)]
                    new_states, rewards, terminateds, truncateds, infos = env.step(actions)

                #logger.info(f'replay {i}, terminateds={terminateds}, truncateds={truncateds}')


    # main training loop

    logger.info(f'\n***\nFinish collect buffer. Start main training....')
    global_step = 0
    online_net_sum_loss_episodes, critic_sum_loss_episodes, sum_reward_episodes, tds_avg = 0, 0, 0, 0
    time_limit = 600

    for episode in itertools.count():
        actor_sum_loss_per_ep, critic_sum_loss_per_ep, sum_reward_per_episode, tds_per_ep = 0, 0, 0, 0
        states = env.reset()
        done = False
        step_epoch = 0
        # select action
        while not done and step_epoch < time_limit:

            epsilon = np.interp(episode * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
            # take action
            try:
                step_epoch += 1
                global_step += 1
                if isinstance(states[0], PytorchLazyFrames):
                    states_ = np.stack([lasy.get_frames() for lasy in states])
                    actions = online_net.action(states_, epsilon)
                else:
                    actions = online_net.action(states, epsilon)  # epsilon - random policy now inside .action
                new_states, rewards, terminateds, truncateds, infos = env.step(actions)

                done = terminateds or truncateds
                sum_reward_per_episode += rewards.item()

                for state, action, reward, terminated, truncated, new_state, info in zip(states, actions, rewards,
                                                                                         terminateds, truncateds,
                                                                                         new_states, infos):
                    transition = (state, action, reward, terminated, truncated, new_state)
                    replay_buffer.append(transition)


                states = new_states
                transition_sample = random.sample(replay_buffer, BATCH_SIZE)
                loss = online_net.compute_loss(target_net, transition_sample)
                actor_sum_loss_per_ep += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            except ValueError as e:
                if str(e) == 'cannot reshape array of size 1 into shape (0,0)':
                    logger.info(f'Recovering from AirSim error in main training. step_epoch={step_epoch} ; global_step= {global_step}')
                    states = env.reset()
                    if isinstance(states[0], PytorchLazyFrames):
                        states_ = np.stack([lasy.get_frames() for lasy in states])
                        actions = online_net.action(states_, epsilon)
                    else:
                        actions = online_net.action(states, epsilon)  # epsilon - random policy now inside .action

                    new_states, rewards, terminateds, truncateds, infos = env.step(actions)

            if global_step % TARGET_UPDATE_FREQ:
                target_net.load_state_dict(online_net.state_dict())


        sum_reward_episodes += sum_reward_per_episode
        tb_summary.add_scalar('sum_reward_per_episode', sum_reward_per_episode, global_step=episode)
        online_net_sum_loss_episodes += actor_sum_loss_per_ep
        avg_per_episode = sum_reward_episodes / (episode + 1)
        actor_loss_per_episode = online_net_sum_loss_episodes / (episode + 1)
        tb_summary.add_scalar('avg_reward_current', avg_per_episode, global_step=episode)
        tb_summary.add_scalar('actor_loss_per_episode', actor_loss_per_episode, global_step=episode)
        online_net.do_checkpoint(load=False, optimizer=optimizer,best=False )

def train_outroor_DDDQN(logg_tb, save_path, height_airsim_restart_positions = [-5.35,-5.4, -6.5,-7.6,-8.5, -9. ]):

    env, env_process = connect_exe_env(stack_last_k=4,\
                                       height_airsim_restart_positions = height_airsim_restart_positions, \
                                       env_type='indoor', \
                                       max_episode_steps=150,\
                                       exe_path = "./unreal_envs/easy_maze/Blocks.exe",  \
                                       documents_path = '../../../../../Documents',\
                                       action_type='discrete',\
                                       name='indoor_maze_easy')
    # try: # '../../../../../Documents'
    res = training_dddqn(env, logg_tb=logg_tb, save_path=save_path)
    close_env(env_process)
    # except Exception as restart:
    #     logger.info(f'\nAPI is dead... \n{str(restart)}\nClose .exe ')
    #     close_env(env_process)
    #     res = -2

    return res





