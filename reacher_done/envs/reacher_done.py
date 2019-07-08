import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.mujoco.reacher import ReacherEnv
import numpy as np

class ReacherDoneEnv(ReacherEnv):
  metadata = {'render.modes': ['human']}

#   def __init__(self):
#     super().__init__()
  def step(self, action):
    vec = self.get_body_com("fingertip")-self.get_body_com("target")
    reward_dist = - np.linalg.norm(vec)
    reward_ctrl = - np.square(action).sum()
    reward = reward_dist + reward_ctrl
    self.do_simulation(action, self.frame_skip)
    ob = self._get_obs()
    done = reward_dist < 0.01
    return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
#   def reset(self):
#     super().reset()
#   def render(self, mode='human'):
#     ...
#   def close(self):
#     ...