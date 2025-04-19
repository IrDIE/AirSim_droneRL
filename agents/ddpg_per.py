import random
import time
import itertools
import torch.nn.functional as F
import numpy as np
import torch
import os
import torch.nn as nn
from loguru import logger
from collections import deque
from utils.utils import logg_hyperparams
from torch.utils.tensorboard import SummaryWriter
from airsim_env import close_env, connect_exe_env
from airsim_env import AirSimGym_env
from .p_exp_replay import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
logger.info(f"---------- device  =  {device} ")

LOGG = True  # <------- change here

if LOGG:
    logger.add(
        f"{os.path.dirname(os.path.realpath(__file__))}/logs/log_{time.time()}.log"
    )


# LOAD_FROM = "./saved_weights/test_depth2/"  #'./saved_weights/ac_per_bd_5/'
# HIDDEN_GRU = 48
# LR_CRITIC = 3e-4
# LR_ACTOR = 3e-5
# BATCH_SIZE = 64  # 128 #32 # FROM REPLAY BUFFER
# BUFFER_SIZE = 50_000
# MIN_REPLAY_SIZE = 800
# MAX_STEPS_PER_EPISODE = 500
# LOGGING_INTERVAL = 50  # 150
# TAU = 5e-3  # SOFT NET UPDATE
# GAMMA = 0.99
# IMG_HW = [72, 128]
# ACTION_SIZE = 3
# CLIP_ACTION = [-2, 2]
# VELOCITY_SIZE = 3  # AS INPUT ADDITIONAL STATE
# RANDOM = 0.1
# TRAIN_RATE = 4
# SAVE_WEIGHTS = "./saved_weights/test_depth3/"
# LOGG_TB_DIR = "./logs/test_depth3/"


class TimeDistributed(nn.Module):
    def __init__(self, layer, time_steps, **kwargs):
        super(TimeDistributed, self).__init__()
        self.layers = nn.ModuleList([layer(**kwargs) for _ in range(time_steps)])

    def forward(self, x):
        # batch_size, time_steps, C, H, W = x.size()
        time_steps = x.size()[1]
        output = torch.tensor([]).to(device)
        for i in range(time_steps):
            output_t = self.layers[i](x[:, i, ...])
            if type(output_t) == tuple:
                output_t = output_t[0].unsqueeze(1)  # do not need last hidden state
            else:
                output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        return output


class Actor(nn.Module):
    def __init__(
        self,
        input_ch,
        img_hw,
        lr,
        seq_size,
        save_path,
        action_size,
        hidden_size,
        min_max_actions,
        velocity_size=None,
        name="",
    ):
        super(Actor, self).__init__()

        self.save_path = save_path
        self.name = name
        self.action_size = action_size
        self.min_max_actions = min_max_actions
        self.input_ch = input_ch
        self.img_hw = img_hw
        self.velocity_size = velocity_size
        self.seq_size = seq_size
        self.hidden_size = hidden_size

        self.gru_shaped = False

        self.get_shared_net()
        self.get_state_vel_process(vel=True)
        self.get_actor()

        self.init_weights()
        self.optimizer = None
        if lr:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def do_checkpoint(self, load=True, best=False, alias=""):
        signature = (
            "actor_best.pt" if best else "actor_last.pt"
        )  # save/load best or last
        signature = alias + signature
        if load:
            ckpt_info = torch.load(self.save_path + signature)
            logger.info(
                f"ckpt_info from path : {signature} \nloaded with keys = {ckpt_info.keys()}"
            )
            self.load_state_dict(ckpt_info["model_state_dict"], strict=False)
            if self.optimizer:
                self.optimizer.load_state_dict(ckpt_info["optimizer_state_dict"])
        else:  # save
            os.makedirs(self.save_path, exist_ok=True)
            ckpt_info = {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            torch.save(ckpt_info, self.save_path + signature)

    def load_checkpoint(self, path):
        ckpt_info = torch.load(path)
        logger.info(f"Actor ckpt_info loaded with keys = {ckpt_info.keys()}")
        self.load_state_dict(ckpt_info["model_state_dict"], strict=False)
        if self.optimizer:
            self.optimizer.load_state_dict(ckpt_info["optimizer_state_dict"])

    def forward(self, state, velocity=None):
        # logger.info(f'actor got {state.shape}. {velocity.shape}. {state[0].device}, {velocity.device}')
        out_shared = self.shared_net(state)
        out_shared = self.gru_block(out_shared)[0][
            :, -1, :
        ]  # get last timestamp output from GRU
        out_shared = self.process_state_block(out_shared)

        if self.velocity_size is not None:
            assert velocity is not None, "velocity must be passed"
            velocity_embed = self.process_vel(velocity)
            out_shared = out_shared + velocity_embed

        actions_pred = torch.tanh(self.actor(out_shared))
        # logger.info(f'actions_pred {actions_pred.shape}') # torch.Size([1, 3])
        # actions_pred = torch.clip(actions_pred, min= self.min_max_actions[0], max=self.min_max_actions[1])
        return actions_pred

    def init_weights(self):

        def _init_weights_uniform(block, value=None, gru=False, conv=False):
            if gru:  # no bias
                torch.nn.init.xavier_uniform_(block.weight_ih_l0)
                torch.nn.init.xavier_uniform_(block.weight_hh_l0)
            else:
                torch.nn.init.uniform_(block.weight.data, -value, value)
                if conv:
                    torch.nn.init.uniform_(block.bias.data, -value, value)

        logger.info("\nstart actor initialisation weights\n")
        for named_p in self.named_modules():
            if isinstance(named_p[1], nn.Conv2d):
                size_w = named_p[1].weight.data.size()
                value = 1.0 / np.sqrt(size_w[1] * size_w[2] * size_w[3])
                _init_weights_uniform(block=named_p[1], value=value, conv=True)

            elif isinstance(named_p[1], nn.Linear):
                size_w = named_p[1].weight.data.size()
                value = (
                    1.0 / np.sqrt(size_w[0]) if size_w[1] != self.action_size else 0.003
                )
                _init_weights_uniform(block=named_p[1], value=value)

            elif isinstance(named_p[1], nn.GRU):
                _init_weights_uniform(block=named_p[1], gru=True)

    def get_shared_net(self):
        self.shared_net = nn.Sequential(
            TimeDistributed(
                nn.Conv2d,
                time_steps=self.seq_size,
                in_channels=self.input_ch,
                out_channels=16,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.ELU(),
            TimeDistributed(
                nn.Conv2d,
                time_steps=self.seq_size,
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
            ),
            nn.ELU(),
            TimeDistributed(
                nn.Conv2d,
                time_steps=self.seq_size,
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
            ),
            nn.ELU(),
            TimeDistributed(nn.AvgPool2d, time_steps=self.seq_size, kernel_size=(2, 2)),
            TimeDistributed(
                nn.Conv2d,
                time_steps=self.seq_size,
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
            ),
            nn.ELU(),
            TimeDistributed(
                nn.Conv2d,
                time_steps=self.seq_size,
                in_channels=32,
                out_channels=32,
                kernel_size=(4, 4),
            ),
            nn.ELU(),
            TimeDistributed(nn.AvgPool2d, time_steps=self.seq_size, kernel_size=(2, 2)),
            TimeDistributed(
                nn.Conv2d,
                time_steps=self.seq_size,
                in_channels=32,
                out_channels=16,
                kernel_size=(3, 3),
            ),
            nn.ELU(),
            TimeDistributed(
                nn.Conv2d,
                time_steps=self.seq_size,
                in_channels=16,
                out_channels=8,
                kernel_size=(1, 1),
            ),
            nn.ELU(),
            TimeDistributed(nn.Flatten, time_steps=self.seq_size),
        )
        if not self.gru_shaped:
            self.gru_input_size = self.shared_net.to(device)(
                torch.rand(
                    2,
                    self.seq_size,
                    self.input_ch,
                    self.img_hw[0],
                    self.img_hw[1],
                    device=device,
                )
            ).size()[-1]

            logger.info(f"GRU input size defined as : {self.gru_input_size}")
        self.gru_block = nn.Sequential(
            nn.GRU(
                input_size=self.gru_input_size,
                hidden_size=self.hidden_size,
                bias=False,
                batch_first=True,
            )
        )

    def get_state_vel_process(self, vel=False):

        self.process_state_block = nn.Sequential(
            nn.LayerNorm(self.hidden_size), nn.Tanh()
        )
        if vel:
            assert self.velocity_size is not None, "velocity_size must be defined"
            self.process_vel = nn.Sequential(
                nn.Linear(self.velocity_size, self.hidden_size, bias=False),
                nn.LayerNorm(self.hidden_size),
                nn.Tanh(),
            )

    def get_actor(self):

        self.actor_backbone = nn.Sequential(
            nn.Linear(self.hidden_size, 32, bias=False),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Linear(32, 32, bias=False),
            nn.LayerNorm(32),
            nn.ELU(),
        )
        self.actor_head = nn.Linear(32, self.action_size, bias=False)

        self.actor = nn.Sequential(self.actor_backbone, self.actor_head)


class Critic(nn.Module):
    def __init__(
        self,
        input_ch,
        img_hw,
        lr,
        seq_size,
        save_path,
        hidden_size,
        action_size,
        velocity_size=None,
        name="",
    ):
        super(Critic, self).__init__()

        self.save_path = save_path
        self.action_size = action_size
        self.input_ch = input_ch
        self.img_hw = img_hw
        self.velocity_size = velocity_size
        self.seq_size = seq_size
        self.gru_shaped = False
        self.name = name
        self.hidden_size = hidden_size
        self.get_shared_net()
        self.get_state_vel_process(vel=True)

        self.get_critic()
        self.init_weights()
        self.optimizer = None
        if lr:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    @logger.catch()
    def forward(self, state, velocity=None, action=None):
        out_shared = self.shared_net(state)
        out_shared = self.gru_block(out_shared)[0][
            :, -1, :
        ]  # get last timestamp output from GRU
        out_shared = self.process_state_block(out_shared)

        if self.velocity_size is not None:
            assert velocity is not None, "velocity must be passed"
            velocity_embed = self.process_vel(velocity)
            out_shared = out_shared + velocity_embed

        critic_embed = self.critic_backbone(action)
        state_action = critic_embed + out_shared
        q_critic = self.critic_head(state_action)
        q_critic = self.critic_q(q_critic)

        return q_critic

    def get_shared_net(self):
        self.shared_net = nn.Sequential(
            TimeDistributed(
                nn.Conv2d,
                time_steps=self.seq_size,
                in_channels=self.input_ch,
                out_channels=16,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.ELU(),
            TimeDistributed(
                nn.Conv2d,
                time_steps=self.seq_size,
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
            ),
            nn.ELU(),
            TimeDistributed(
                nn.Conv2d,
                time_steps=self.seq_size,
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
            ),
            nn.ELU(),
            TimeDistributed(nn.AvgPool2d, time_steps=self.seq_size, kernel_size=(2, 2)),
            TimeDistributed(
                nn.Conv2d,
                time_steps=self.seq_size,
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
            ),
            nn.ELU(),
            TimeDistributed(
                nn.Conv2d,
                time_steps=self.seq_size,
                in_channels=32,
                out_channels=32,
                kernel_size=(4, 4),
            ),
            nn.ELU(),
            TimeDistributed(nn.AvgPool2d, time_steps=self.seq_size, kernel_size=(2, 2)),
            TimeDistributed(
                nn.Conv2d,
                time_steps=self.seq_size,
                in_channels=32,
                out_channels=16,
                kernel_size=(3, 3),
            ),
            nn.ELU(),
            TimeDistributed(
                nn.Conv2d,
                time_steps=self.seq_size,
                in_channels=16,
                out_channels=8,
                kernel_size=(1, 1),
            ),
            nn.ELU(),
            TimeDistributed(nn.Flatten, time_steps=self.seq_size),
        )
        if not self.gru_shaped:
            self.gru_input_size = self.shared_net.to(device)(
                torch.rand(
                    2,
                    self.seq_size,
                    self.input_ch,
                    self.img_hw[0],
                    self.img_hw[1],
                    device=device,
                )
            ).size()[-1]

            logger.info(f"GRU input size defined as : {self.gru_input_size}")
        self.gru_block = nn.Sequential(
            nn.GRU(
                input_size=self.gru_input_size,
                hidden_size=self.hidden_size,
                bias=False,
                batch_first=True,
            )
        )

    def get_state_vel_process(self, vel=False):

        self.process_state_block = nn.Sequential(
            nn.LayerNorm(self.hidden_size), nn.Tanh()
        )
        if vel:
            assert self.velocity_size is not None, "velocity_size must be defined"
            self.process_vel = nn.Sequential(
                nn.Linear(self.velocity_size, self.hidden_size, bias=False),
                nn.LayerNorm(self.hidden_size),
                nn.Tanh(),
            )

    def init_weights(self):

        def _init_weights_uniform(block, value=None, gru=False, conv=False):
            if gru:  # no bias
                torch.nn.init.xavier_uniform_(block.weight_ih_l0)
                torch.nn.init.xavier_uniform_(block.weight_hh_l0)
            else:
                torch.nn.init.uniform_(block.weight.data, -value, value)
                if conv:
                    torch.nn.init.uniform_(block.bias.data, -value, value)

        logger.info("\nstart critic initialisation weights\n")
        for named_p in self.named_modules():
            if isinstance(named_p[1], nn.Conv2d):
                size_w = named_p[1].weight.data.size()
                value = 1.0 / np.sqrt(size_w[1] * size_w[2] * size_w[3])
                _init_weights_uniform(block=named_p[1], value=value, conv=True)

            elif isinstance(named_p[1], nn.Linear):
                size_w = named_p[1].weight.data.size()
                value = 1.0 / np.sqrt(size_w[0]) if size_w[1] != 1 else 0.003
                _init_weights_uniform(block=named_p[1], value=value)

            elif isinstance(named_p[1], nn.GRU):
                _init_weights_uniform(block=named_p[1], gru=True)

    def do_checkpoint(self, load=True, best=False, alias=""):
        signature = (
            "critic_best.pt" if best else "critic_last.pt"
        )  # save/load best or last
        signature = alias + signature  # target or not
        weights_ = os.path.join(self.save_path + signature)
        if load:
            ckpt_info = torch.load(weights_)
            logger.info(
                f"ckpt_info from path : {signature} \nloaded with keys = {ckpt_info.keys()}"
            )
            self.load_state_dict(ckpt_info["model_state_dict"], strict=False)
            if self.optimizer:
                self.optimizer.load_state_dict(ckpt_info["optimizer_state_dict"])
        else:  # save
            os.makedirs(self.save_path, exist_ok=True)
            ckpt_info = {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            torch.save(ckpt_info, weights_)

    def load_checkpoint(self, path):
        ckpt_info = torch.load(path)
        logger.info(f"Critic ckpt_info loaded with keys = {ckpt_info.keys()}")
        self.load_state_dict(ckpt_info["model_state_dict"], strict=False)
        if self.optimizer:
            self.optimizer.load_state_dict(ckpt_info["optimizer_state_dict"])

    def get_critic(self):
        self.critic_backbone = nn.Sequential(
            nn.Linear(self.action_size, self.hidden_size, bias=False),
            nn.LayerNorm(self.hidden_size),
            nn.Tanh(),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(self.hidden_size, 32, bias=False),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Linear(32, 32, bias=False),
            nn.LayerNorm(32),
            nn.ELU(),
        )
        self.critic_q = nn.Linear(
            32, 1
        )  #  kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.07, theta=0.1, dt=1e-3, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return "OrnsteinUhlenbeckActionNoise(mu={}, sigma={})".format(
            self.mu, self.sigma
        )


class DDPG_Agent(nn.Module):
    def __init__(
        self,
        input_ch,
        img_hw,
        seq_size,
        action_size,
        min_max_actions,
        velocity_size,
        hidden_size,
        load_ckpt=None,
        save_path=None,
        gamma=None,
        lr_critic=None,
        tau=None,
        lr_actor=None,
        batch_size=None,
        memory_size=None,
    ):
        super(DDPG_Agent, self).__init__()

        self.save_path = save_path
        self.tau = tau
        self.gamma = gamma
        self.memory_size = memory_size
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.state_size = [input_ch, *img_hw]
        self.velocity_size = velocity_size
        self.action_size = action_size

        self.actor = Actor(
            input_ch,
            img_hw,
            lr_actor,
            seq_size,
            save_path,
            action_size,
            hidden_size,
            min_max_actions,
            velocity_size,
        )
        self.target_actor = Actor(
            input_ch,
            img_hw,
            lr_actor,
            seq_size,
            save_path,
            action_size,
            hidden_size,
            min_max_actions,
            velocity_size,
            name="target",
        )
        self.critic = Critic(
            input_ch,
            img_hw,
            lr_critic,
            seq_size,
            save_path,
            hidden_size,
            action_size,
            velocity_size,
        )
        self.target_critic = Critic(
            input_ch,
            img_hw,
            lr_critic,
            seq_size,
            save_path,
            hidden_size,
            action_size,
            velocity_size,
            name="critic",
        )

        if len(load_ckpt):
            self.actor.load_checkpoint(path=load_ckpt + "actor_last.pt")
            self.critic.load_checkpoint(path=load_ckpt + "critic_last.pt")
            logger.info(f"loaded chkpt from {load_ckpt}")

        self.update_network_parameters(tau=1, strict=False)
        self.noise = OUActionNoise(mu=np.zeros(action_size))
        if memory_size:
            self.replay_buffer_per = Memory(capacity=self.memory_size)

    def update_network_parameters(self, tau=None, strict=True):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = (
                tau * critic_state_dict[name].clone()
                + (1 - tau) * target_critic_dict[name].clone()
            )

        self.target_critic.load_state_dict(critic_state_dict, strict=strict)

        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_dict[name].clone()
            )

        self.target_actor.load_state_dict(actor_state_dict, strict=strict)

    def train(self):
        batch, idxs, _ = self.replay_buffer_per.sample(self.batch_size)

        images = torch.zeros(
            [self.batch_size] + [self.seq_size] + self.state_size, device=device
        )
        vels = torch.zeros([self.batch_size, self.velocity_size], device=device)
        actions = torch.zeros([self.batch_size, self.action_size], device=device)
        rewards = torch.zeros([self.batch_size, 1], device=device)
        next_images = torch.zeros(
            [self.batch_size] + [self.seq_size] + self.state_size, device=device
        )
        next_vels = torch.zeros([self.batch_size, self.velocity_size], device=device)
        dones = torch.zeros([self.batch_size, 1], device=device)

        for i, sample in enumerate(batch):
            # logger.info(f'\ntraining agent. sample type = {sample[1]}, {sample[2]}, {sample[4]}')
            images[i], vels[i] = transform_observation(sample[0])
            actions[i] = torch.as_tensor(sample[1], device=device, dtype=torch.float32)
            rewards[i] = torch.as_tensor(sample[2], device=device, dtype=torch.float32)
            next_images[i], next_vels[i] = transform_observation(sample[3])
            dones[i] = torch.as_tensor(sample[4], device=device, dtype=torch.float32)

        actions_clean = self.actor.forward(state=images, velocity=vels)
        target_actions = self.target_actor.forward(
            state=next_images, velocity=next_vels
        )
        target_next_q_s = self.target_critic.forward(
            state=next_images, velocity=next_vels, action=target_actions
        )
        targets = rewards + self.gamma * (1 - dones) * target_next_q_s
        targets = targets.view(self.batch_size, 1)

        self.actor.optimizer.zero_grad()
        # self.actor.train()
        actor_loss = -self.critic.forward(
            state=images, velocity=vels, action=actions_clean
        )
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        critic_value = self.critic.forward(
            state=images, velocity=vels, action=actions
        )  # pred
        td_error = torch.abs(critic_value - targets)  # error
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(targets, critic_value)  # loss
        critic_loss.backward()
        self.critic.optimizer.step()

        for i in range(self.batch_size):
            idx = idxs[i]
            self.replay_buffer_per.update(idx, td_error[i].detach().cpu().numpy()[0])
            # logger.info(f'td_error[{i}] in for_ loop in train = {td_error[i].detach().cpu().numpy()[0]}')

        self.update_network_parameters(strict=False)
        return critic_loss.detach().cpu().numpy(), actor_loss.detach().cpu().numpy()

    def append_replay_buffer_per(self, state, action, reward, new_state, done):
        state_tensor = transform_observation(state)
        new_state_tensor = transform_observation(new_state)

        action_tensor = torch.as_tensor(action, device=device, dtype=torch.float32)
        done_tensor = torch.as_tensor([done], device=device, dtype=torch.float32)
        reward_tensor = torch.as_tensor([reward], device=device, dtype=torch.float32)
        # logger.info(f'*-*, {state_tensor[0].shape}, {state_tensor[1].shape}')
        # logger.info(f'*-*, {action_tensor.shape}, done_tensor = {done_tensor} - {done_tensor.shape}, reward_tensor={reward_tensor},{reward_tensor.shape}')
        critic_value = self.critic.forward(
            state=state_tensor[0], velocity=state_tensor[1], action=action_tensor
        )

        target_actions = self.target_actor.forward(
            state=new_state_tensor[0], velocity=new_state_tensor[1]
        )
        target_next_q_s = self.target_critic.forward(
            state=new_state_tensor[0],
            velocity=new_state_tensor[1],
            action=target_actions,
        )
        # logger.info(f'**, {target_next_q_s.shape}, {done_tensor.shape}, {reward_tensor.shape}')

        targets = reward_tensor + self.gamma * (1 - done_tensor) * target_next_q_s
        targets = targets.view(1, 1)

        td = torch.abs(targets - critic_value)
        # logger.info(f'td in append memory = {td}')
        # logger.info(f'td[0] in append memory = {td[0].detach().cpu().numpy()[0]}')
        self.replay_buffer_per.add(
            td[0].detach().cpu().numpy(), (state, action, reward, new_state, done)
        )
        return td.detach().cpu().numpy()

    def take_action(self, observation):
        # self.actor.eval()
        actions = self.actor.forward(*observation)
        actions = actions + torch.tensor(self.noise(), dtype=torch.float32).to(device)
        actions = torch.clip(
            actions, self.actor.min_max_actions[0], self.actor.min_max_actions[1]
        )

        # self.actor.train()
        return actions.cpu().detach().numpy()


def transform_observation(observation):
    # observation = [history, vel]
    image = torch.as_tensor(observation[0], device=device, dtype=torch.float32).permute(
        0, 1, 4, 2, 3
    )
    # logger.info(f"image shape = {image.shape}")
    vel = torch.as_tensor(observation[1], device=device, dtype=torch.float32)
    # logger.info(f'vel shape = {vel.shape}')
    return [image, vel]


def training_ddpg_per(env: AirSimGym_env, cfg_agent):
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.shape[0]

    load_ckpt = cfg_agent.get("train", "load_from")

    save_path = cfg_agent.get("logg", "save_weights")
    logg_tb = cfg_agent.get("logg", "logg_tb_dir")
    logging_interval = cfg_agent.getint("logg", "logging_interval")

    buffer_size = cfg_agent.getint("train", "buffer_size")
    hidden_gru = cfg_agent.getint("train", "hidden_gru")
    gamma = cfg_agent.getfloat("train", "gamma")
    batch = cfg_agent.getint("train", "batch")
    gradient_steps = cfg_agent.getint("train", "gradient_steps")
    lr_actor = cfg_agent.getfloat("train", "lr_actor")
    lr_critic = cfg_agent.getfloat("train", "lr_critic")
    min_replay_size = cfg_agent.getint("train", "min_replay_size")
    max_steps_per_episode = cfg_agent.getint("train", "max_steps_per_episode")

    velocity_size = cfg_agent.getint("train", "velocity_size")
    tau = cfg_agent.getfloat("train", "tau")

    train_rate = cfg_agent.getint("train", "train_rate")
    seq_size = cfg_agent.getint("train", "seq_size")
    random_ = cfg_agent.getfloat("train", "random")
    clip_action = list(map(float, cfg_agent.get("train", "clip_action").split(",")))

    tb_summary = SummaryWriter(logg_tb)
    logg_hyperparams(tb_summary, cfg_agent)

    rddpg_agent = DDPG_Agent(
        save_path=save_path,
        tau=tau,
        input_ch=observation_shape[2],
        gamma=gamma,
        memory_size=buffer_size,
        batch_size=batch,
        img_hw=observation_shape[:2],
        seq_size=seq_size,
        action_size=action_shape,
        min_max_actions=clip_action,
        velocity_size=velocity_size,
        load_ckpt=load_ckpt,
        lr_critic=lr_critic,
        lr_actor=lr_actor,
        hidden_size=hidden_gru,
    ).to(device)

    global_step = 0
    logger.info(f"min_replay_size={min_replay_size}")

    while global_step < min_replay_size:
        observation, info = env.reset()
        done = False
        image, vel = observation
        history = np.stack(
            [image[np.newaxis, ...]] * seq_size, axis=1
        )  # Batch-sequence-h-w-channels
        vel = vel.reshape(1, -1)
        state = [history, vel]

        while not done and global_step < min_replay_size:
            try:
                global_step += 1
                state_tensor = transform_observation(state)
                random_rate = np.random.random() < random_
                if random_rate:
                    action = np.array(np.random.uniform(-1, 1, (1, action_shape)))
                else:
                    action = rddpg_agent.take_action(state_tensor)  # with noise

                observe, reward, terminated, truncated, info = env.step(
                    action[0]
                )  # step from noisy action / 1d list

                # append to buffer
                image_new, vel_new = observe  # numpy
                vel_new = vel_new.reshape(1, -1)
                history = np.append(
                    history[:, 1:], image_new[np.newaxis, np.newaxis, ...], axis=1
                )
                #  vel = vel.reshape(1, -1)
                new_state = [history, vel_new]
                done = terminated or truncated
                # logger.info(f'do append')
                rddpg_agent.append_replay_buffer_per(
                    state, action, reward, new_state, done
                )
                state = new_state
                if global_step % logging_interval == 0:
                    logger.info(f"COLLECTING REPLAY BUFFER at step = {global_step}")

            except ValueError as e:
                # if str(e) == 'cannot reshape array of size 1 into shape (0,0)':
                logger.info(
                    f"Recovering from AirSim error in replay buffer. i = {global_step}"
                )
                observation, info = env.reset()
                done = True
                image, vel = observation
                history = np.stack(
                    [image[np.newaxis, ...]] * seq_size, axis=1
                )  # Batch-sequence-h-w-channels
                vel = vel.reshape(1, -1)
                state = [history, vel]

    # main training:
    logger.info(f"\n****\nstart main training ...\n")
    global_step = 0
    actor_sum_loss_episodes, critic_sum_loss_episodes, sum_reward_episodes, tds_avg = (
        0,
        0,
        0,
        0,
    )

    for episode in itertools.count():
        (
            actor_sum_loss_per_ep,
            critic_sum_loss_per_ep,
            sum_reward_per_episode,
            tds_per_ep,
        ) = (0, 0, 0, 0)

        observation, _ = env.reset()
        image, vel = observation
        history = np.stack(
            [image[np.newaxis, ...]] * seq_size, axis=1
        )  # Batch-sequence-h-w-channels
        vel = vel.reshape(1, -1)
        state = [history, vel]
        done = False
        step_epoch = 0
        logger.info(f"episode={episode}")
        while not done and step_epoch < max_steps_per_episode:
            try:
                if global_step % train_rate == 0:
                    for grad_step in range(gradient_steps):
                        # do train (opt.step / backward) and update target networks
                        # env.client.simPause(True)
                        critic_loss, actor_loss = rddpg_agent.train()
                        actor_sum_loss_per_ep += actor_loss.item()
                        critic_sum_loss_per_ep += critic_loss.item()
                        if grad_step % 50 == 0:
                            logger.info(f"grad_step={grad_step}")
                        # env.client.simPause(False)
                step_epoch += 1
                global_step += 1

                # get action from state
                state_tensor = transform_observation(state)
                random_rate = np.random.random() < random_
                if random_rate:
                    action = np.array(np.random.uniform(-1, 1, (1, action_shape)))
                else:
                    action = rddpg_agent.take_action(state_tensor)  # with noise

                observe, reward, terminated, truncated, info = env.step(
                    action[0]
                )  # step from noisy action / 1d list

                tb_summary.add_scalar(
                    "reward_step_immediate", reward, global_step=global_step
                )
                sum_reward_per_episode += reward
                # append to buffer
                image_new, vel_new = observe  # numpy
                vel_new = vel_new.reshape(1, -1)
                history = np.append(
                    history[:, 1:], image_new[np.newaxis, np.newaxis, ...], axis=1
                )

                new_state = [history, vel_new]
                done = terminated or truncated
                td = rddpg_agent.append_replay_buffer_per(
                    state, action, reward, new_state, done
                )
                tds_per_ep += td
                state = new_state

            except ValueError as e:
                logger.info(str(e))
                # if str(e) == 'cannot reshape array of size 1 into shape (0,0)':
                logger.info(
                    f"Recovering from AirSim error in replay buffer. i = {global_step}"
                )
                observation, info = env.reset()
                done = True
                image, vel = observation
                history = np.stack(
                    [image[np.newaxis, ...]] * seq_size, axis=1
                )  # Batch-sequence-h-w-channels
                vel = vel.reshape(1, -1)
                state = [history, vel]

        # done
        sum_reward_episodes += sum_reward_per_episode
        tb_summary.add_scalar(
            "sum_reward_per_episode", sum_reward_per_episode, global_step=episode
        )

        actor_sum_loss_episodes += actor_sum_loss_per_ep
        critic_sum_loss_episodes += critic_sum_loss_per_ep

        tds_avg += tds_per_ep / step_epoch
        avg_per_episode = sum_reward_episodes / (episode + 1)
        actor_loss_per_episode = actor_sum_loss_episodes / (episode + 1)
        critic_loss_per_episode = critic_sum_loss_episodes / (episode + 1)

        # do checkpoint for last
        rddpg_agent.actor.do_checkpoint(load=False, best=False, alias="")
        rddpg_agent.target_actor.do_checkpoint(load=False, best=False, alias="target")
        rddpg_agent.critic.do_checkpoint(load=False, best=False, alias="")
        rddpg_agent.target_critic.do_checkpoint(load=False, best=False, alias="target")

        tb_summary.add_scalar(
            "avg_reward_current", avg_per_episode, global_step=episode
        )
        tb_summary.add_scalar(
            "avg_actor_loss_per_episode", actor_loss_per_episode, global_step=episode
        )
        tb_summary.add_scalar(
            "avg_critic_loss_per_episode", critic_loss_per_episode, global_step=episode
        )
        tb_summary.add_scalar(
            "actor_loss_per_episode", actor_sum_loss_per_ep, global_step=episode
        )
        tb_summary.add_scalar(
            "critic_loss_per_episode", critic_sum_loss_per_ep, global_step=episode
        )
        tb_summary.add_scalar(
            "tds_avg pper step ", tds_avg / (episode + 1), global_step=episode
        )


def train_DDPG_per(cfg_agent, exe_path, cfg_env_path, documents_path):
    env, env_process = connect_exe_env(
        cfg_agent,
        exe_path,
        cfg_env_path,
        documents_path,
    )

    res = training_ddpg_per(env, cfg_agent)
    if env_process is not None:
        close_env(env_process)
    return res


def infer_DDPG_per(cfg_agent, exe_path, cfg_env_path, documents_path):

    env, env_process = connect_exe_env(
        cfg_agent,
        exe_path,
        cfg_env_path,
        documents_path,
    )

    observation_shape = env.observation_space.shape
    action_shape = env.action_space.shape[0]

    load_ckpt = cfg_agent.get("train", "load_from")
    logg_tb = cfg_agent.get("infer", "logg_tb_dir")
    hidden_gru = cfg_agent.getint("train", "hidden_gru")
    velocity_size = cfg_agent.getint("train", "velocity_size")
    seq_size = cfg_agent.getint("train", "seq_size")
    clip_action = list(map(float, cfg_agent.get("train", "clip_action").split(",")))
    tb_summary = SummaryWriter(logg_tb)
    logg_hyperparams(tb_summary, cfg_agent)

    rddpg_agent = DDPG_Agent(
        input_ch=observation_shape[2],
        img_hw=observation_shape[:2],
        seq_size=seq_size,
        action_size=action_shape,
        min_max_actions=clip_action,
        velocity_size=velocity_size,
        load_ckpt=load_ckpt,
        hidden_size=hidden_gru,
    ).to(device)
    # rddpg_agent.noise = OUActionNoise(
    #     mu=np.zeros(rddpg_agent.action_size), sigma=0, theta=0, dt=0
    # )

    logger.info(f"\n****\nstart Inference DDPG ...\n")
    global_step = 0
    rddpg_agent.actor.eval()
    rddpg_agent.critic.eval()
    observation, info = env.reset()

    for episode in itertools.count():
        sum_reward_per_episode = 0
        observation, _ = env.reset()

        image, vel = observation
        history = np.stack(
            [image[np.newaxis, ...]] * seq_size, axis=1
        )  # Batch-sequence-h-w-channels
        vel = vel.reshape(1, -1)
        state = [history, vel]
        done = False
        step_epoch = 0
        while not done:
            step_epoch += 1
            global_step += 1
            # get action from state
            state_tensor = transform_observation(state)
            action = rddpg_agent.take_action(state_tensor)  # with noise
            observe, reward, terminated, truncated, _ = env.step(action[0])
            tb_summary.add_scalar("reward_step_immediate", reward, global_step=episode)
            sum_reward_per_episode += reward
            # append to buffer
            image_new, vel_new = observe  # numpy
            vel_new = vel_new.reshape(1, -1)
            history = np.append(
                history[:, 1:], image_new[np.newaxis, np.newaxis, ...], axis=1
            )
            new_state = [history, vel_new]
            done = terminated or truncated
            state = new_state

    if env_process is not None:
        close_env(env_process)


def chwck_noise():
    noise = OUActionNoise(mu=np.zeros(3))
    for _ in range(100):
        n = noise()
        print(n)


if __name__ == "__main__":
    # train_DDPG_per(logg_tb = LOGG_TB_DIR, save_path = SAVE_WEIGHTS, seq_size = 5, height_airsim_restart_positions = [-0.8339])
    chwck_noise()
