import gym
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
import os

# HalfCheetah-v3 but with a different xml file
class HalfCheetahSoftEnv(HalfCheetahEnv):
  _xml_path:str

  def __init__(self, xml_path = None):
    if xml_path is None:
      curr_dir = os.path.dirname(os.path.realpath(__file__))
      self._xml_path = os.path.join(curr_dir, 'assets', 'half_cheetah_soft.xml')
    else:
      self._xml_path = xml_path

    super(HalfCheetahSoftEnv, self).__init__(
      xml_file=self._xml_path, reset_noise_scale=0.)

  # gym uses Ezpickle for envs that uses C/C++ (like mujoco and atari envs)
  # These functions are for pickling and unpickling
  # To allow you to use deepcopy()
  def __getstate__(self):
    return {"_xml_path": self._xml_path}

  def __setstate__(self, d):
    # out = type(self)()  # call constructor with no other arguments
    out = type(self)(*d["_xml_path"])
    self.__dict__.update(out.__dict__)
