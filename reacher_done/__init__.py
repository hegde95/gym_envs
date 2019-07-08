from gym.envs.registration import register

register(
    id='reacher-done-v0',
    entry_point='reacher_done.envs:ReacherDoneEnv',
)
