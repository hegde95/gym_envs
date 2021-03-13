import gym
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym.utils import EzPickle
import numpy as np
import os


# Humanoid with custom xml file
# TODO: See if DMControl's humanoid has useful functions?
# Qpos & Qvel are [torso, abdomen_y, right leg, left leg]
# torso qpos = x, y, z, quat (wxyz).
# torso qvel = v_x, v_y, v_z, omega_x, omega_y, omega_z
# right leg (qpos and qvel) = hip (x, z, y), knee, ankle (y, x)
# left leg  (qpos and qvel) = hip (x, z, y), knee, ankle (y, x)
class HumanoidSoftEnv(HumanoidEnv):
  _xml_path:str

  def __init__(self, xml_path = None):
    if xml_path is None:
      curr_dir = os.path.dirname(os.path.realpath(__file__))
      self._xml_path = os.path.join(curr_dir, 'assets', 'humanoidDM_soft.xml')
    else:
      self._xml_path = xml_path

    super(HumanoidSoftEnv, self).__init__(
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

  def step(self, action):
    # parent env doesn't clip actions!
    # action = np.clip(action, self.action_space.low, self.action_space.high)
    observation, reward, done, info = super(HumanoidSoftEnv, self).step(action=action)
    done = False # HACK: we don't use this param
    return observation, reward, done, info

  # TODO: these start and end incides are fixed. just store these in a dict?
  def get_sensor_value(self, sensor_name:str):
    sens_id = self.sim.model.sensor_name2id(sensor_name)
    sens_dims = self.sim.model.sensor_dim[sens_id]
    sens_start_idx = np.sum(self.sim.model.sensor_dim[0:sens_id]) - 1 # 0-indexed
    sens_end_idx = sens_start_idx + sens_dims
    return self.sim.data.sensordata[sens_start_idx:sens_end_idx]
