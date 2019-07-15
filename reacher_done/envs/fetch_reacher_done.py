import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.robotics.fetch.reach import FetchReachEnv
from gym.envs.robotics.fetch_env import goal_distance
import numpy as np

class FetchReacherDoneEnv(FetchReachEnv):
  metadata = {'render.modes': ['human']}

  #   def __init__(self):
  #     ...

  def step(self, action):
    action = np.clip(action, self.action_space.low, self.action_space.high)
    self._set_action(action)
    self.sim.step()
    self._step_callback()
    obs = self._get_obs()

    done = self._is_success(obs['achieved_goal'], self.goal)
    info = {
        'is_success': done, # does not include done from TimeLimit (episode completion)
    }
    reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
    
    # Time penalty to encourage faster reaching
    reward_time = -0.1 # TODO: Make tweakable hyperparam
    reward = reward + reward_time
    return obs, reward, done, info

  def compute_reward(self, achieved_goal, goal, info):
    reward = 0
    completion_reward = 5 # TODO: Make tweakable hyperparam
    
    # Compute distance between goal and the achieved goal.
    d = goal_distance(achieved_goal, goal)
    reached_goal = d <= self.distance_threshold

    # add distance reward
    if self.reward_type == 'sparse':
        reward = -(not reached_goal).astype(np.float32)
    else: # dense distance reward
        reward = -d
    
    # add completion reward
    if reached_goal:
      reward += completion_reward

    return reward
  
  #   def reset(self):
      # super().reset()
  #   def render(self, mode='human'):
  #     ...
  #   def close(self):
  #     ...
