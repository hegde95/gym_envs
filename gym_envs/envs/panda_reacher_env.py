import os
from gym import utils
from gym_envs.envs.panda_env import PandaEnv


# Ensure we get the path separator correct on windows
curr_dir = os.path.dirname(os.path.realpath(__file__))
xml_path = os.path.join(curr_dir, 'assets', 'panda_reacher.xml')

class PandaReachEnv(PandaEnv, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        # set joint values at startup
        initial_qpos = { # 'joint_name': float_value,
        'panda0_joint2': 0.282,
        'panda0_joint4': -0.934,
        }
        PandaEnv.__init__(self, 
            model_path=xml_path, 
            has_object=False, 
            block_gripper=True, 
            n_substeps=20,
            target_range=0.15, 
            target_offset=[0., 0., 0.],
            distance_threshold=0.05,
            initial_qpos=initial_qpos, 
            reward_type=reward_type)
        utils.EzPickle.__init__(self)
