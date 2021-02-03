from gym.envs.registration import register

register(
    id='reacher-done-v0',
    entry_point='gym_envs.envs:ReacherDoneEnv',
    max_episode_steps=100,
)

register(
    id='fetch-reacher-done-v0',
    entry_point='gym_envs.envs:FetchReacherDoneEnv',
    max_episode_steps=100,
)

register(
    id='fetch-pnp-done-v0',
    entry_point='gym_envs.envs:FetchPnPDoneEnv',
    max_episode_steps=100,
)

register(
    id='half-cheetah-soft-v0',
    entry_point='gym_envs.envs:HalfCheetahEnv',
    max_episode_steps=100,
)
