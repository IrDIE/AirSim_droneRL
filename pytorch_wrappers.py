from collections import deque
from loguru import logger
import gym
import numpy as np

from baselines_wrappers import VecEnvWrapper
from baselines_wrappers.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, ScaledFloatFrame, \
    ClipRewardEnv, WarpFrame
from baselines_wrappers.wrappers import TimeLimit


def make_atari_deepmind(env_id, max_episode_steps=None, scale_values=False, clip_rewards=True, render_mode="human"):
    env = gym.make(env_id, new_step_api =False , render_mode=render_mode) # apply_api_compatibility=False
    env = NoopResetEnv(env, noop_max=30)

    if 'NoFrameskip' in env.spec.id:
        env = MaxAndSkipEnv(env, skip=4)

    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    env = EpisodicLifeEnv(env)

    env = WarpFrame(env)

    if scale_values:
        env = ScaledFloatFrame(env)

    if clip_rewards:
        env = ClipRewardEnv(env)

    env = TransposeImageObs(env, axis_order=[2, 0, 1])  # Convert to torch order (C, H, W)

    return env



class TransposeImageObs(gym.ObservationWrapper):

    def __init__(self, env, axis_order):
        super().__init__(env)
        self.axis_order = axis_order
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [
                obs_shape[self.axis_order[0]],
                obs_shape[self.axis_order[1]],
                obs_shape[self.axis_order[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, obs):
        return obs.transpose(self.axis_order[0], self.axis_order[1], self.axis_order[2])


class BatchedPytorchFrameStack(VecEnvWrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        super().__init__(env)
        self.k = k
        self.batch_stacks = [deque([], maxlen=k) for _ in range(env.num_envs)]
        shp = env.observation_space.shape
        # logger.info(f'shp[1:] = {shp[1:]}')
        # logger.info(f'shp[0] = {shp[0]}')
        # logger.info(f'shape=((shp[0] * k,) + shp[1:]) = { (shp[0] * k,) + shp[1:] }')
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=((shp[0] * k,) + shp[1:]),
                                                dtype=env.observation_space.dtype)
        self.env = env

    def reset(self, **kwargs):
        obses = self.env.reset(**kwargs)
        for _ in range(self.k):
            for i, obs in enumerate(obses):
                self.batch_stacks[i].append(obs.copy())
        return self._get_ob()

    def step_wait(self):
        obses, reward, done, info = self.env.step_wait()
        for i, obs_frame in enumerate(obses):
            self.batch_stacks[i].append(obs_frame)

        ret_ob = self._get_ob()
        return ret_ob, reward, done, info

    def _get_ob(self):
        return [PytorchLazyFrames(list(batch_stack), axis=0) for batch_stack in self.batch_stacks]

    def _transform_batched_frame(self, frame):
        return [f for f in frame]

class PytorchLazyFrames(object):
    def __init__(self, frames, axis=0):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None
        self.axis = axis

    def __len__(self):
        return len(self.get_frames())

    def get_frames(self):
        """Get Numpy representation without dumping the frames."""
        return np.concatenate(self._frames, axis=self.axis)
