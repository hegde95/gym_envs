import gym
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
import os


class HalfCheetahEnv(HalfCheetahEnv):

  # defaulting to dense reward type, everything else is the same from parent env
  def __init__(self):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    xml_path = os.path.join(curr_dir, 'assets', 'half_cheetah_soft.xml')
    super(HalfCheetahEnv, self).__init__(xml_file=xml_path)

  # def step(self, action):
  #   action = np.clip(action, self.action_space.low, self.action_space.high)
  #   self._set_action(action)
  #   self.sim.step()
  #   self._step_callback()
  #   obs = self._get_obs()

  #   done = self._is_success(obs['achieved_goal'], self.goal)
  #   info = {
  #       # does not include done from TimeLimit (episode completion)
  #       'is_success': done,
  #       'dist': goal_distance(obs['achieved_goal'], self.goal)
  #   }

  #   reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

  #   # Time penalty to encourage faster reaching
  #   reward_time = -0.1  # TODO: Make tweakable hyperparam
  #   reward = reward + reward_time
  #   return obs, reward, done, info

  # def compute_reward(self, achieved_goal, goal, info):
  #   ...
  # def reset(self):
  #   ...
  # def render(self, mode='human'):
  #   ...
  # def close(self):
  #   ...
