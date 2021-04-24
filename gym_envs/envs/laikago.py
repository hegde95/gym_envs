import numpy as np
from gym.utils import EzPickle
from gym.envs.mujoco.mujoco_env import MujocoEnv
import os

DEFAULT_CAMERA_CONFIG = {'distance': 3.0, 'trackbodyid': 1, 'elevation': 0}

class LaikagoEnv(MujocoEnv, EzPickle):
  def __init__(self,
               xml_path=None,
               frame_skip=5,
               forward_reward_weight=1.0,
               ctrl_cost_weight=0.1,
               reset_noise_scale=0.):
    EzPickle.__init__(**locals())
    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._reset_noise_scale = reset_noise_scale

    if xml_path is None:
      curr_dir = os.path.dirname(os.path.realpath(__file__))
      self._xml_path = os.path.join(curr_dir, 'assets', 'laikago', 'laikago.xml')
    else:
      self._xml_path = xml_path
    MujocoEnv.__init__(self, self._xml_path, frame_skip=frame_skip)
    self.init_qpos = self.sim.model.key_qpos.flatten()
    self.init_qvel = self.sim.model.key_qvel.flatten()


  def step(self, action):
    # x_position_before = self.sim.data.qpos[0]
    self.do_simulation(action, self.frame_skip)
    # x_position_after = self.sim.data.qpos[0]
    # x_velocity = ((x_position_after - x_position_before)
    # / self.dt)
    reward = 0
    observation = self._get_obs()
    done = False
    info = {
        # 'x_position': x_position_after,
        # 'x_velocity': x_velocity,
    }
    return observation, reward, done, info


  def _get_obs(self):
    position = self.sim.data.qpos.flat.copy()
    velocity = self.sim.data.qvel.flat.copy()
    observation = np.concatenate((position, velocity)).ravel()
    return observation


  def reset_model(self):
    noise_low = -self._reset_noise_scale
    noise_high = self._reset_noise_scale

    qpos = self.init_qpos + self.np_random.uniform(
      low=noise_low, high=noise_high, size=self.model.nq)
    qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
      self.model.nv)

    self.set_state(qpos, qvel)

    observation = self._get_obs()
    return observation


  def viewer_setup(self):
    for key, value in DEFAULT_CAMERA_CONFIG.items():
      if isinstance(value, np.ndarray):
        getattr(self.viewer.cam, key)[:] = value
      else:
        setattr(self.viewer.cam, key, value)
