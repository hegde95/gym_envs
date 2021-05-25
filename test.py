import gym
import gym_envs
env = gym.make("half-cheetah-soft-v0")
env.reset()
for i in range(10000):
  action = env.action_space.sample() # no action
  env.step(action)
  env.render('human')