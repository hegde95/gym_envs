import gym
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
import os


class HalfCheetahSoftEnv(HalfCheetahEnv):

    # HalfCheetah-v3 but with a different xml file
    def __init__(self):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(curr_dir, 'assets', 'half_cheetah_soft.xml')
        super(HalfCheetahSoftEnv, self).__init__(
            xml_file=xml_path, reset_noise_scale=0.)

    # gym uses Ezpickle for envs that uses C/C++ (like mujoco and atari envs)
    # These functions are for pickling and unpickling
    # To allow you to use deepcopy()
    def __getstate__(self):
        return {}  # no new variables to store from this derived class

    def __setstate__(self, d):
        out = type(self)()  # call constructor with no other arguments
        self.__dict__.update(out.__dict__)
