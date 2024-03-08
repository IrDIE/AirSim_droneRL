from torch import nn
import itertools
import random
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import warnings
import numpy as np
import torch
import time
import os
from loguru import logger
from utils.loss import smooth_l1_loss
from utils.pytorch_wrappers import PytorchLazyFrames
warnings.filterwarnings('ignore')


GAMMA = 0.99 # DISCOUNT RATE
BATCH_SIZE = 2 #32 # FROM REPLAY BUFFER
BUFFER_SIZE = 50_000
MIN_REPLAY_SIZE = 10 #1_000
EPSILON_START = 1.0 # E GREEDY POLICY
EPSILON_END = 0.02
EPSILON_DECAY = 10_000

LR = 0.0005

NUM_ENVS = 1
TARGET_UPDATE_FREQ = 1000 // NUM_ENVS
SAVE_PATH, SAVE_INTERVAL = "./saved_weights/", 5 # 300

LOGGING_INTERVAL = 5
LOGGING_DIR = f"logs/dqn/{time.time()}"


RESTART_EXE = 300 # looks like airsim api with started .exe can`t last forever =) , so need to restart

class DQN(nn.Module):
    def __init__(self, env):
        super(DQN, self).__init__()
        self.action_shape = env.action_space.n
        self.convNet = self.get_conv_net(env)
        logger.info(f"self.convNet = \n{self.convNet}")
        try:
            self.load_weights( SAVE_PATH + 'dqn_best.pt')
            logger.info(f"Found { SAVE_PATH + 'dqn_best.pt'} weights, attempt to load ...")
        except:
            logger.info('No best.pt weights, random initialization ...')


    def forward(self, x, conv = True):
        if conv : return self.convNet(x)
        else :return self.net(x)

    def action(self, states, epsilon, inference=False):
        states = torch.tensor(states, dtype=torch.float32)
        q_values = self.forward(states)
        max_indexs = torch.argmax(q_values, dim=1)
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
    def get_conv_net(self, env, depths = [32,64,64], kernel_size = [8,4,3], stride = [4,2,1], outp_size = 512):
        self.in_channels = list([env.observation_space.shape[0]])
        self.depth = self.in_channels + depths
        self.kernel_size = kernel_size
        self.stride = stride

        conv_blocks = [self.get_convBlock(from_ch = inp, to_ch = out, kernel_size = kernel_size, stride = stride) \
                       for inp,out, kernel_size, stride in zip(self.depth, self.depth[1:], self.kernel_size, self.stride)
                       ]
        self.convNet = nn.Sequential(
            *conv_blocks,
            nn.Flatten()
        )

        with torch.no_grad():
            flatten_size = self.convNet(torch.as_tensor(env.observation_space.sample()[None]).float() ).shape[1]

        return nn.Sequential(
            self.convNet,
            nn.Linear(flatten_size, outp_size ),
            nn.ReLU(),
            nn.Linear(outp_size, self.action_shape)
        )

    def compute_loss(self, target_net, trg):
        states_ = [t[0] for t in trg]

        actions = torch.as_tensor(np.asarray([t[1] for t in trg]), dtype=torch.int64).unsqueeze(-1)
        rews = torch.as_tensor(np.asarray([t[2] for t in trg]), dtype=torch.float32).unsqueeze(-1)
        dones = torch.as_tensor(np.asarray([t[3] for t in trg]), dtype=torch.float32).unsqueeze(-1)
        new_states_ = [t[4] for t in trg]
        if isinstance(states_[0], PytorchLazyFrames):
            states = torch.as_tensor(np.stack([lazy_frames.get_frames() for lazy_frames in states_]), dtype=torch.float32)
            new_states = torch.as_tensor(np.stack([lazy_frames.get_frames() for lazy_frames in new_states_]), dtype=torch.float32)
        else :
            states = torch.as_tensor(states_, dtype=torch.float32)
            new_states = torch.as_tensor(np.asarray(new_states_), dtype=torch.float32)

        target_q_values = target_net(new_states)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = rews + GAMMA * (1 - dones) * max_target_q_values
        # loss
        q_values = self(states)
        action_q_values = torch.gather(q_values, dim=1, index=actions)
        loss = smooth_l1_loss(action_q_values, targets)

        return loss

    def save_best_last(self, best = True):
        if best :
            torch.save(self.convNet.state_dict(), SAVE_PATH + 'dqn_best.pt')
        else:
            torch.save(self.convNet.state_dict(), SAVE_PATH + 'dqn_last.pt')

    def load_weights(self, weights_path):

        self.convNet.load_state_dict(torch.load(weights_path))



def training_dqn(env):
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

