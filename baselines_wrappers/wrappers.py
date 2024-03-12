from loguru import logger
import gym

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, terminated, truncated, info = self.env.step(ac)
        self._elapsed_steps += 1
        # logger.info(f'self._elapsed_steps = {self._elapsed_steps}')
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
            info['TimeLimit.truncated'] = True
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)