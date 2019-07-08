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
    reward_ctrl = - np.square(action).sum()
    done = dist < 0.04 # done if it's close enough
    done_reward = 2
    reward = reward_dist + reward_ctrl + done*done_reward
    ob = self._get_obs()
    # print("Dist ", dist)
    if done:
        print("Done!")
    return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
  
#   def reset(self):
    # super().reset()
#   def render(self, mode='human'):
#     ...
#   def close(self):
#     ...