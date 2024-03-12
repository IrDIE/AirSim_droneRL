import numpy as np
import os
os.environ.setdefault('PATH', '')
import gymnasium as gym
import cv2
cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env,noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = env.unwrapped.noop_action
        #assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    # def _step(self, ac):
    #     return self.env.step(ac)


#
# class FireResetEnv(gym.Wrapper):
#     def __init__(self, env):
#         """Take action on reset for environments that are fixed until firing."""
#         gym.Wrapper.__init__(self, env)
#         assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
#         assert len(env.unwrapped.get_action_meanings()) >= 3
#
#     def reset(self, **kwargs):
#         self.env.reset(**kwargs)
#         obs, _, done, _ = self.env.step(1)
#         if done:
#             self.env.reset(**kwargs)
#         obs, _, done, _ = self.env.step(2)
#         if done:
#             self.env.reset(**kwargs)
#         return obs
#
#     def step(self, ac):
#         return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=2):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env) # gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        # self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype) #deque(maxlen=3)
        self._obs_buffer = np.zeros((1, *env.observation_space.shape),
                                    dtype=env.observation_space.dtype)  # deque(maxlen=3)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last 2 observations."""
        total_reward = 0.0
        terminated = None
        truncated = None
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action) #observation, reward, terminated, truncated, info
            # if i == self._skip - 2: self._obs_buffer[0] = obs
            # if i == self._skip - 1: self._obs_buffer[1] = obs
            if i == self._skip - 1: self._obs_buffer[0] = obs
            total_reward += reward
            if terminated or truncated:
                break
        # Note that the observation on the done=True frame
        # doesn't matter

        # self._obs_buffer[1].shape = (360, 640, 3)
        # logger.info(f'self._obs_buffer.shape = {self._obs_buffer.shape}')
        # max_frame = self._obs_buffer.max(axis=0) #  contains some temporal information
        max_frame = self._obs_buffer[0]
        return max_frame, total_reward, terminated, truncated, info



class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs