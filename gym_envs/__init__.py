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
    entry_point='gym_envs.envs:HalfCheetahSoftEnv',
    max_episode_steps=100,
)

register(
    id='panda-reacher-v0',
    entry_point='gym_envs.envs:PandaReachEnv',
    max_episode_steps=1000,
)

register(
    id='panda-pnp-v0',
    entry_point='gym_envs.envs:PandaPickPlaceEnv',
    max_episode_steps=1000,
)

register(
    id='humanoid-soft-v0',
    entry_point='gym_envs.envs:HumanoidSoftEnv',
    max_episode_steps=1000,
)

register(
    id='laikago-v0',
    entry_point='gym_envs.envs:LaikagoEnv',
    max_episode_steps=1000,
)

register(
    id='laikago-v2',
    entry_point='gym_envs.envs:LaikagoEnv_v2',
    max_episode_steps=1000,
)