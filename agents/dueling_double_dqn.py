from torch import nn
import itertools
import random
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import warnings
import numpy as np
import torch

from loguru import logger
from utils.loss import smooth_l1_loss
from utils.utils import update_logg_reward,load_save_logg_reward
from utils.pytorch_wrappers import PytorchLazyFrames
warnings.filterwarnings('ignore')

GAMMA = 0.99 # DISCOUNT RATE

BATCH_SIZE = 16 # 128 #32 # FROM REPLAY BUFFER
BUFFER_SIZE = 10_000
MIN_REPLAY_SIZE = 800 #1000

EPSILON_START = 0.9 # E GREEDY POLICY
EPSILON_END = 0.05
EPSILON_DECAY = 800

LR = 5e-4

NUM_ENVS = 1
TARGET_UPDATE_FREQ = 250 // NUM_ENVS
LOGGING_INTERVAL = 50 #30 #
RESTART_EXE = 1000

class Double_Dueling_DQN(nn.Module):
    def __init__(self, env, save_path, load_path):
        super(Double_Dueling_DQN, self).__init__()
        self.not_calculated_flatten = True
        self.action_shape = env.action_space.n
        self.convNet = self.get_conv_net(env)

        self.dueling_state = self.get_dueling_state()
        self.dueling_action = self.get_dueling_action()

        self.save_path, self.load_path = save_path, load_path

        try:
            self.load_weights(self.load_path + '/dqn_best.pt')
            logger.info(f"Found {self.load_path  + '/dqn_best.pt'} weights, attempt to load ...")
        except:
            logger.info('No best.pt weights, random initialization ...')


    def forward(self, x):
        enconed = self.convNet(x)
        V = self.dueling_state(enconed)
        A = self.dueling_action(enconed)
        return V,A


    def action(self, states, epsilon, inference=False):
        states = torch.tensor(states, dtype=torch.float32)
        _, advantage = self.forward(states) # in online net -> get action advantage
        max_indexs = torch.argmax(advantage, dim=1)
        actions = max_indexs.detach().tolist()
        if not inference:
            for i in range(len(actions)):
                if np.random.random() <= epsilon :
                    actions[i] = np.random.randint(0, self.action_shape - 1)

        return actions

    def get_convBlock(self, from_ch, to_ch, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(from_ch, to_ch, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(to_ch),
            nn.ReLU()
        )

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

    def get_conv_net(self, env, depths = [64,128,64], kernel_size = [3,4,3], stride = [4,2,1], outp_size = 512*2):
        #logger.info(f'env.observation_space.shape = {env.observation_space.shape}')
        self.in_channels = list([env.observation_space.shape[0]])
        self.depth = self.in_channels + depths
        self.kernel_size = kernel_size
        self.stride = stride
        self.outp_size = outp_size

        conv_blocks = [self.get_convBlock(from_ch = inp, to_ch = out, kernel_size = kernel_size, stride = stride) \
                       for inp,out, kernel_size, stride in zip(self.depth, self.depth[1:], self.kernel_size, self.stride)
                       ]


        self.convNet_ = nn.Sequential(
            nn.Conv2d(self.in_channels[0], 64, kernel_size=(3,3), stride=1),
            # nn.MaxPool2d((2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1),
            # nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        # self.convNet_ = nn.Sequential(
        #     *conv_blocks,
        #     nn.Flatten()
        # )

        if self.not_calculated_flatten:
            with torch.no_grad():
                self.flatten_size = self.convNet_(torch.as_tensor(env.observation_space.sample()[None]).float() ).shape[1]
                #logger.info(f'self.flatten_size = {self.flatten_size}')
            self.not_calculated_flatten = False

        return nn.Sequential(
            self.convNet_,
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.outp_size),
            nn.ReLU()
        )

    def compute_loss(self, target_net, trg):
        # trg - state, action, reward, terminated, truncated, new_state
        states_ = [t[0] for t in trg]
        actions = torch.as_tensor(np.asarray([t[1] for t in trg]), dtype=torch.int64).unsqueeze(-1)
        rews = torch.as_tensor(np.asarray([t[2] for t in trg]), dtype=torch.float32).unsqueeze(-1)
        dones = torch.as_tensor(np.asarray([t[3] for t in trg]), dtype=torch.float32).unsqueeze(-1)
        new_states_ = [t[5] for t in trg]

        if isinstance(states_[0], PytorchLazyFrames):
            states = torch.as_tensor(np.stack([lazy_frames.get_frames() for lazy_frames in states_]), dtype=torch.float32)
            new_states = torch.as_tensor(np.stack([lazy_frames.get_frames() for lazy_frames in new_states_]), dtype=torch.float32)
        else :
            states = torch.as_tensor(states_, dtype=torch.float32)
            new_states = torch.as_tensor(np.asarray(new_states_), dtype=torch.float32)

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
        loss = smooth_l1_loss(action_q_values, targets)

        return loss

    def save_best_last(self, best = True):
        if best :
            torch.save(self.state_dict(), self.save_path + 'dqn_best.pt')
        else:
            torch.save(self.state_dict(), self.save_path + 'dqn_last.pt')

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))


def training_dddqn(env, logg_tb, epoch, save_path, reward_loggs, csv_rewards_log ='restart_best_rewards', collision_reward=-2, load_path = None):
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    info_buffer = deque(maxlen=200)
    online_net = Double_Dueling_DQN(env=env, save_path=save_path, load_path = load_path)
    online_net.train()
    target_net = Double_Dueling_DQN(env=env, save_path=save_path, load_path=load_path)
    target_net.train()

    target_net.load_state_dict(online_net.state_dict())
    optimizer = Adam(lr=LR, params=online_net.parameters())
    tb_summary = SummaryWriter(logg_tb)

    episode_count = 0

    # init replay buffer before training
    states = env.reset()
    for i in range(MIN_REPLAY_SIZE):

        actions = [env.action_space.sample() for _ in range(NUM_ENVS)]  # sample from env randomly
        new_states, rewards, terminateds, truncateds, infos = env.step(actions)
        #logger.info(f'replay {i}, terminateds={terminateds}, truncateds={truncateds}')
        for state, action, reward, terminated, truncated, new_state in zip(states, actions, rewards, terminateds, truncateds, new_states):
            transition = (state, action, reward, terminated, truncated, new_state)
            replay_buffer.append(transition)
        states = new_states

    # main training loop
    states = env.reset()
    logger.info(f'\n***\nFinish collect buffer. Start main training....')

    last_rew = collision_reward
    for step in itertools.count():
        # select action
        epsilon = np.interp(step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        if isinstance(states[0], PytorchLazyFrames):
            states_ = np.stack([lasy.get_frames() for lasy in states])
            actions = online_net.action(states_, epsilon)
        else:
            actions = online_net.action(states, epsilon)  # epsilon - random policy now inside .action

        # take action
        new_states, rewards, terminateds, truncateds, infos = env.step(actions)

        for state, action, reward, terminated, truncated, new_state, info in zip(states, actions, rewards, terminateds, truncateds, new_states, infos):
            transition = (state, action, reward, terminated, truncated, new_state)
            replay_buffer.append(transition)

            if terminated or truncated:
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
            mean_rew = np.mean([e['r'] for e in info_buffer if len(info_buffer) > 0 ])
            mean_rew = -10 if np.isnan(mean_rew) else mean_rew
            mean_duration = np.mean([e['l'] for e in info_buffer]) or 0
            mean_duration = 0 if np.isnan(mean_duration) else mean_duration

            reward_loggs = update_logg_reward(df = reward_loggs, restart_n = f'restart_{epoch}', reward = mean_rew , duration=mean_duration)
            load_save_logg_reward(df = reward_loggs,save=True, save_path=save_path, csv_rewards_log=csv_rewards_log)

            if mean_rew > last_rew:
                logger.info(f"\n*****\nCkeckpoint for best model with reward = {mean_rew} at step {step}. Saving model weights....")
                online_net.save_best_last(best=True)
                last_rew = mean_rew

            logger.info(
                f"\nCkeckpoint for last model with reward = {mean_rew} at step {step}. Saving model weights....")
            online_net.save_best_last(best=False)

            logger.info(f'Episode: {step}\nReward  == {mean_rew}\nDuration == {mean_duration}. Epsilon={epsilon}.')

            tb_summary.add_scalar('mean_rew', mean_rew if mean_rew is not None else 0, global_step=step)
            tb_summary.add_scalar('mean_duration', mean_duration if mean_duration is not None else 0, global_step=step)
            tb_summary.add_scalar('episode_count', episode_count, global_step=step)

            # if step > RESTART_EXE:
            #     logger.info(f'Episode: {step}\nRestart .exe')
            #     return -1

