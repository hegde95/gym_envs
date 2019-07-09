import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.mujoco.reacher import ReacherEnv
import numpy as np

class ReacherDoneEnv(ReacherEnv):
  metadata = {'render.modes': ['human']}

#   def __init__(self):
#     ...
  def step(self, action):
    self.do_simulation(action, self.frame_skip)
    vec = self.get_body_com("fingertip")-self.get_body_com("target")
    dist = np.linalg.norm(vec)
    reward_dist = - dist
    reward_ctrl = - 0.3 * np.square(action).sum()
    reward_time = -0.04
    done = dist < 0.04 # done if it's close enough
    done_reward = 2
    reward = reward_dist + reward_ctrl + reward_time + done*done_reward
    ob = self._get_obs()
    info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, dist=dist)
    return ob, reward, done, info
  
#   def reset(self):
    # super().reset()
#   def render(self, mode='human'):
#     ...
#   def close(self):
#     ...