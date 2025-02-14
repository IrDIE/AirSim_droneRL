from torch import nn
import itertools
import random
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import warnings
import numpy as np
import os
import torch
from airsim_env import close_env, connect_exe_env
import time
from .p_exp_replay import *
from loguru import logger
from utils.utils import update_logg_reward, load_save_logg_reward
from utils.pytorch_wrappers import PytorchLazyFrames
from utils.utils import logg_hyperparams

warnings.filterwarnings("ignore")
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
logger.info(f"---------- device  =  {device} ")

class Double_Dueling_DQN_per(nn.Module):
    def __init__(self, env, device, gamma = None, save_path = None):
        super(Double_Dueling_DQN_per, self).__init__()
        self.device = device
        self.not_calculated_flatten = True
        self.outp_size = 1024
        self.action_shape = env.action_space.n
        self.convNet = self.get_conv_net(env)
        self.dueling_state = self.get_dueling_state()
        self.dueling_action = self.get_dueling_action()
        self.save_path = save_path
        self.gamma = gamma

    def forward(self, x):
        enconed = self.convNet(x)
        V = self.dueling_state(enconed)
        A = self.dueling_action(enconed)
        return V, A

    def action(self, states, epsilon, inference=False):
        states = torch.tensor(states, dtype=torch.float32, device=self.device).permute(
            0, 3, 2, 1
        )
        # logger.info(f'states shape tensors = {states.shape}')
        _, advantage = self.forward(states)  # in online net -> get action advantage
        max_indexs = torch.argmax(advantage, dim=1)
        actions = max_indexs.detach().tolist()
        if not inference:
            for i in range(len(actions)):
                if np.random.random() <= epsilon:
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
        # logger.info(f'env.observation_space.shape = {env.observation_space.shape}')
        self.in_channels = list([env.observation_space.shape[2]])
        self.convNet_ = nn.Sequential(
            nn.Conv2d(self.in_channels[0], 32, kernel_size=(5, 5), stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        if self.not_calculated_flatten:
            with torch.no_grad():
                sample = env.observation_space.sample()[None]

                torch_sample = torch.as_tensor(sample).permute(0, 3, 2, 1)
                logger.info(f"sample shape={torch_sample.shape}")
                self.flatten_size = self.convNet_(torch_sample.float()).shape[1]
                logger.info(f"self.flatten_size = {self.flatten_size}")
            self.not_calculated_flatten = False

        return nn.Sequential(
            self.convNet_,
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.outp_size),
            nn.ReLU(),
        )

    def compute_loss(self, target_net, trg, get_targets_q = False):

        # trg - state, action, reward, new_state, terminated or truncated
        states_ = [t[0] for t in trg]
        new_states_ = [t[3] for t in trg]

        actions = torch.as_tensor(
            np.asarray([t[1] for t in trg]), dtype=torch.int64, device=self.device
        ).unsqueeze(-1)

        rews = torch.as_tensor(
            np.asarray([t[2] for t in trg]), dtype=torch.float32, device=self.device
        ).unsqueeze(-1)
        dones = torch.as_tensor(
            np.asarray([t[4] for t in trg]),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)
        

        if isinstance(states_[0], PytorchLazyFrames):
            states = torch.as_tensor(
                np.stack([lazy_frames.get_frames() for lazy_frames in states_]),
                dtype=torch.float32,
                device=self.device,
            )
            states = states.permute(0, 3, 2, 1)
            new_states = torch.as_tensor(
                np.stack([lazy_frames.get_frames() for lazy_frames in new_states_]),
                dtype=torch.float32,
                device=self.device,
            )
            new_states = new_states.permute(0, 3, 2, 1)
        else:
            states = torch.as_tensor(states_, dtype=torch.float32, device=self.device)
            new_states = torch.as_tensor(
                np.asarray(new_states_), dtype=torch.float32, device=self.device
            )

        # for double:
        V_states, A_states = self.forward(states)

        V_new_states, A_new_states = target_net.forward(new_states)

        V_s_eval, A_s_eval = self.forward(new_states)
        q_pred = torch.add(
            V_states, (A_states - A_states.mean(dim=1, keepdim=True))
        )  # action_q_values
        q_next = torch.add(
            V_new_states, (A_new_states - A_new_states.mean(dim=1, keepdim=True))
        )
        q_eval = torch.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        best_q_index = q_eval.argmax(dim=1, keepdim=True)  # max_actions
        targets_selected_q_values = torch.gather(
            input=q_next, dim=1, index=best_q_index
        )  # = q_next[indices, max_actions]
        targets = rews + self.gamma * (1 - dones) * targets_selected_q_values
        # loss
        action_q_values = torch.gather(q_pred, dim=1, index=actions)
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)
        if get_targets_q: return loss, action_q_values, targets
        return loss

    def do_checkpoint(
        self, load=True, path_load=None, best=False, alias="", optimizer=None
    ):
        signature = "dqn_best.pt" if best else "dqn_last.pt"  # save/load best or last
        signature = alias + signature
        if load:
            weights_ = os.path.join(path_load, signature)
            ckpt_info = torch.load(weights_)

            logger.info(
                f"ckpt_info from path : {weights_} \nloaded with keys = {ckpt_info.keys()}"
            )
            self.load_state_dict(ckpt_info["model_state_dict"], strict=False)
            if optimizer: optimizer.load_state_dict(ckpt_info["optimizer_state_dict"])
        else:  # save
            assert optimizer is not None
            os.makedirs(self.save_path, exist_ok=True)
            ckpt_info = {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(ckpt_info, os.path.join(self.save_path , signature))


def append_replay_buffer_per(replay_buffer_per, online_net, target_net, t, gamma):

    # t - state, action, reward, terminated, truncated, new_state
    states_ = [t[0]]
    actions = torch.as_tensor(
        np.asarray([t[1]]), dtype=torch.int64, device=device
    ).unsqueeze(-1)
    rews = torch.as_tensor(
        np.asarray([t[2]]), dtype=torch.float32, device=device
    ).unsqueeze(-1)
    dones = torch.as_tensor(
        np.asarray([t[3] or t[4]]),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(-1)
    new_states_ = [t[5]]

    if isinstance(states_[0], PytorchLazyFrames):
        states = torch.as_tensor(
            np.stack([lazy_frames.get_frames() for lazy_frames in states_]),
            dtype=torch.float32,
            device=device,
        )
        states = states.permute(0, 3, 2, 1)
        new_states = torch.as_tensor(
            np.stack([lazy_frames.get_frames() for lazy_frames in new_states_]),
            dtype=torch.float32,
            device=device,
        )
        new_states = new_states.permute(0, 3, 2, 1)
    else:
        states = torch.as_tensor(states_, dtype=torch.float32, device=device)
        new_states = torch.as_tensor(
            np.asarray(new_states_), dtype=torch.float32, device=device
        )

    # for double:
    V_states, A_states = online_net.forward(states)
    V_new_states, A_new_states = target_net.forward(new_states)
    V_s_eval, A_s_eval = online_net.forward(new_states)
    q_pred = torch.add(
        V_states, (A_states - A_states.mean(dim=1, keepdim=True))
    )  # action_q_values
    q_next = torch.add(
        V_new_states, (A_new_states - A_new_states.mean(dim=1, keepdim=True))
    )
    q_eval = torch.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
    best_q_index = q_eval.argmax(dim=1, keepdim=True)  # max_actions
    targets_selected_q_values = torch.gather(
        input=q_next, dim=1, index=best_q_index
    )  # = q_next[indices, max_actions]
    targets = rews + gamma * (1 - dones) * targets_selected_q_values
    # loss
    action_q_values = torch.gather(q_pred, dim=1, index=actions)
    td = torch.abs(targets - action_q_values)[0].detach().cpu().numpy()[0]
    replay_buffer_per.add(td, (t[0], t[1], t[2], t[-1], t[3] or t[4]))
    # appended state, action, reward, new_state, terminated or truncated

    return


def training_dddqn_per(env, cfg_agent):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"---------- device  =  {device} ")
    load_ckpt = cfg_agent.get("train", "load_from")
    save_path = cfg_agent.get("logg", "save_weights")
    logg_tb = cfg_agent.get("logg", "logg_tb_dir")
    logging_interval = cfg_agent.getint("logg", "logging_interval")
    buffer_size = cfg_agent.getint("train", "buffer_size")
    gamma = cfg_agent.getfloat("train", "gamma")
    batch_size = cfg_agent.getint("train", "batch")
    lr = cfg_agent.getfloat("train", "lr")
    min_replay_size = cfg_agent.getint("train", "min_replay_size")
    max_steps_per_episode = cfg_agent.getint("train", "max_steps_per_episode")
    target_update_freq = cfg_agent.getint("train", "target_update_freq")
    epsilon_start = cfg_agent.getfloat("train", "epsilon_start")
    epsilon_end = cfg_agent.getfloat("train", "epsilon_end")
    epsilon_decay = cfg_agent.getint("train", "epsilon_decay")

    replay_buffer_per = Memory(capacity=buffer_size)

    # replay_buffer = deque(maxlen=BUFFER_SIZE)
    online_net = Double_Dueling_DQN_per(
        env=env, save_path=save_path,gamma=gamma, device=device, 
    ).to(device)
    optimizer = Adam(lr=lr, params=online_net.parameters())
    target_net = Double_Dueling_DQN_per(
        env=env, save_path=save_path, gamma=gamma,device=device, 
    ).to(device)
    if len(load_ckpt) > 0:
        online_net.do_checkpoint(load=True, optimizer=optimizer, path_load=load_ckpt)
    tb_summary = SummaryWriter(logg_tb)
    logg_hyperparams(tb_summary, cfg_agent)

    target_net.load_state_dict(online_net.state_dict())
    target_net.train()
    online_net.train()

    # init replay buffer before training
    global_step = 0
    while global_step < min_replay_size:
        states = env.reset()
        done = False

        while not done and global_step < min_replay_size:
            try:
                actions = [env.action_space.sample()]  # sample from env randomly
                new_states, rewards, terminateds, truncateds, infos = env.step(actions)
                done = terminateds or truncateds
                global_step += 1

                # append to buffer
                for state, action, reward, terminated, truncated, new_state in zip(
                    states, actions, rewards, terminateds, truncateds, new_states
                ):
                    transition = (
                        state,
                        action,
                        reward,
                        terminated,
                        truncated,
                        new_state,
                    )

                    append_replay_buffer_per(replay_buffer_per, online_net, target_net, transition, gamma)
                    # replay_buffer.append(transition)

                if global_step % logging_interval == 0:
                    logger.info(f"COLLECTING REPLAY BUFFER at step = {global_step}")

            except ValueError as e:
                if str(e) == "cannot reshape array of size 1 into shape (0,0)":
                    logger.info(
                        f"Recovering from AirSim error in replay buffer. global_step = {global_step}"
                    )
                    states = env.reset()
                    actions = [env.action_space.sample()]
                    new_states, rewards, terminateds, truncateds, infos = env.step(
                        actions
                    )

                # logger.info(f'replay {i}, terminateds={terminateds}, truncateds={truncateds}')

    # main training loop
    logger.info(f"\n***\nFinish collect buffer. Start main training....")
    global_step = 0
    online_net_sum_loss_episodes = 0
    sum_reward_episodes = 0


    for episode in itertools.count():
        (
            actor_sum_loss_per_ep,
            critic_sum_loss_per_ep,
            sum_reward_per_episode,
            tds_per_ep,
        ) = (0, 0, 0, 0)
        states = env.reset()
        done = False
        step_epoch = 0
        # select action
        while not done and step_epoch < max_steps_per_episode:

            epsilon = np.interp(
                episode,
                [0, epsilon_decay],
                [epsilon_start, epsilon_end],
            )
            # take action
            try:
                step_epoch += 1
                global_step += 1
                if isinstance(states[0], PytorchLazyFrames):
                    states_ = np.stack([lasy.get_frames() for lasy in states])
                    actions = online_net.action(states_, epsilon)
                else:
                    actions = online_net.action(
                        states, epsilon
                    )  # epsilon - random policy now inside .action
                new_states, rewards, terminateds, truncateds, infos = env.step(actions)
                done = terminateds or truncateds
                sum_reward_per_episode += rewards.item()
                tb_summary.add_scalar(
                    "reward_immediate", rewards.item(), global_step=global_step
                )

                for (
                    state,
                    action,
                    reward,
                    terminated,
                    truncated,
                    new_state,
                    info,
                ) in zip(
                    states, actions, rewards, terminateds, truncateds, new_states, infos
                ):
                    transition = (
                        state,
                        action,
                        reward,
                        terminated,
                        truncated,
                        new_state,
                    )
                    append_replay_buffer_per(replay_buffer_per, online_net, target_net, transition, gamma)
                    # replay_buffer.append(transition)

                states = new_states
                batch, idxs, _ = replay_buffer_per.sample(batch_size)
                # transition_sample = random.sample(replay_buffer, BATCH_SIZE)
                loss, action_q_values, targets = online_net.compute_loss(target_net, batch, get_targets_q=True)
                actor_sum_loss_per_ep += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                td_error = torch.abs(action_q_values - targets)
                for i in range(batch_size):
                    idx = idxs[i]
                    replay_buffer_per.update(idx, td_error[i].detach().cpu().numpy()[0])

            except ValueError as e:
                if str(e) == "cannot reshape array of size 1 into shape (0,0)":
                    logger.info(
                        f"Recovering from AirSim error in main training. step_epoch={step_epoch} ; global_step= {global_step}"
                    )
                    states = env.reset()
                    if isinstance(states[0], PytorchLazyFrames):
                        states_ = np.stack([lasy.get_frames() for lasy in states])
                        actions = online_net.action(states_, epsilon)
                    else:
                        actions = online_net.action(
                            states, epsilon
                        )  # epsilon - random policy now inside .action

                    new_states, rewards, terminateds, truncateds, infos = env.step(
                        actions
                    )

            if global_step % target_update_freq: 
                # TODO: should be at the times of backward() and .sample()
                target_net.load_state_dict(online_net.state_dict())

        sum_reward_episodes += sum_reward_per_episode
        tb_summary.add_scalar(
            "sum_reward_per_episode", sum_reward_per_episode, global_step=episode
        )
        online_net_sum_loss_episodes += actor_sum_loss_per_ep
        avg_per_episode = sum_reward_episodes / (episode + 1)
        actor_loss_per_episode = online_net_sum_loss_episodes / (episode + 1)
        tb_summary.add_scalar(
            "avg_reward_current", avg_per_episode, global_step=episode
        )
        tb_summary.add_scalar(
            "epsilon", epsilon, global_step=episode
        )
        tb_summary.add_scalar(
            "actor_loss_per_episode", actor_loss_per_episode, global_step=episode
        )
        online_net.do_checkpoint(load=False, optimizer=optimizer, best=False)


def infer_DDDQN_per(
    cfg_agent,
    exe_path,
    cfg_env_path,
    documents_path,
):
    env, env_process = connect_exe_env(
        cfg_agent,
        exe_path,
        cfg_env_path,
        documents_path,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"---------- device  =  {device} ")
    load_ckpt = cfg_agent.get("infer", "load_from")
    logg_tb = cfg_agent.get("infer", "logg_tb_dir")

    online_net = Double_Dueling_DQN_per(
        env=env,
        device=device,
    ).to(device)

    online_net.do_checkpoint(load=True, path_load=load_ckpt)
    tb_summary = SummaryWriter(logg_tb)
    logg_hyperparams(tb_summary, cfg_agent)
    online_net.eval()

    # =============================================
    global_step = 0
    sum_reward_episodes = 0
    states = env.reset()

    for episode in itertools.count():
        sum_reward_per_episode = 0
        done = False
        step_epoch = 0
        # select action
        while not done:
            epsilon = 0
            # take action
            try:
                step_epoch += 1
                global_step += 1
                if isinstance(states[0], PytorchLazyFrames):
                    states = np.stack([lasy.get_frames() for lasy in states])

                actions = online_net.action(states, epsilon)
                new_states, rewards, terminateds, truncateds, infos = env.step(actions)
                done = terminateds or truncateds
                sum_reward_per_episode += rewards.item()

                tb_summary.add_scalar(
                    "instant reward per step", rewards.item(), global_step=global_step
                )

                states = new_states

            except ValueError as e:
                if str(e) == "cannot reshape array of size 1 into shape (0,0)":
                    logger.info(
                        f"Recovering from AirSim error in main training. step_epoch={step_epoch} ; global_step= {global_step}"
                    )
                    states = env.reset()
                    if isinstance(states[0], PytorchLazyFrames):
                        states = np.stack([lasy.get_frames() for lasy in states])
                    actions = online_net.action(states, epsilon)
                    new_states, rewards, terminateds, truncateds, infos = env.step(
                        actions
                    )

        logger.info(f"Training logging ep {episode}......")
        sum_reward_episodes += sum_reward_per_episode
        tb_summary.add_scalar(
            "sum_reward_per_episode", sum_reward_per_episode, global_step=episode
        )
        avg_per_episode = sum_reward_episodes / (episode + 1)
        tb_summary.add_scalar(
            "avg_reward_current", avg_per_episode, global_step=episode
        )

    # =============================================

    close_env(env_process)


def train_DDDQN_per(
    cfg_agent,
    exe_path,
    cfg_env_path,
    documents_path,
):

    env, env_process = connect_exe_env(
        cfg_agent,
        exe_path,
        cfg_env_path,
        documents_path,
    )
    res = training_dddqn_per(env, cfg_agent)
    close_env(env_process)
    return res
