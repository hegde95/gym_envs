import numpy as np
from gym.utils import EzPickle
from gym.envs.mujoco.mujoco_env import MujocoEnv
import os
from gym import error, spaces
import gym

DEFAULT_CAMERA_CONFIG = {'distance': 3.0, 'trackbodyid': 1, 'elevation': 0}
MOTOR_CONTROL_MODES = ["SYMMETRIC_TORQUE", "TORQUE", "PD", "SYMMETRIC_TORQUE_WH"]

class LaikagoEnv_v2(MujocoEnv, EzPickle):
  def __init__(self,
               xml_path=None,
               frame_skip=5,
               forward_reward_weight=1.0,
               ctrl_cost_weight=0.1,
               reset_noise_scale=0.,
               motor_control_mode = "SYMMETRIC_TORQUE"):
    EzPickle.__init__(**locals())
    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._reset_noise_scale = reset_noise_scale

    self._desired_height = 0.6


    assert motor_control_mode in MOTOR_CONTROL_MODES
    self._motor_control_mode = motor_control_mode


    if xml_path is None:
      curr_dir = os.path.dirname(os.path.realpath(__file__))
      self._xml_path = os.path.join(curr_dir, 'assets', 'laikago', 'laikago.xml')
    else:
      self._xml_path = xml_path
    MujocoEnv.__init__(self, self._xml_path, frame_skip=frame_skip)
    self.init_qpos = self.sim.model.key_qpos[0]
    self.init_qvel = self.sim.model.key_qvel[0]
    # self._change_observation_space()
    


      


  def step(self, action, output=True):
    # action = action["action"]
    torques = self.get_torque_action(action)

    self.do_simulation(torques, self.frame_skip)


    if output:
      # Remove for performance boost
      reward = self._get_reward()
      observation = self._get_obs()
      done = False
      info = {
      }
      return observation, reward, done, info
    else:
      return


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


  def get_torque_action(self, action):
    if self._motor_control_mode == "SYMMETRIC_TORQUE":
      assert(action.size == 6) # symmetric
      new_action = np.hstack([action, action])

    elif self._motor_control_mode == "TORQUE":
      new_action = action

    elif self._motor_control_mode == "PD":
      # TODO: Finish converting actions from PD to torque
      pass


    return new_action

  def _set_action_space(self):

    bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
    low, high = bounds.T
        
    if self._motor_control_mode == "TORQUE":
      # Stays same as this is default
      # self.action_space = gym.spaces.Dict({'action':spaces.Box(low=low, high=high, dtype=np.float32)})
      self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    elif self._motor_control_mode == "SYMMETRIC_TORQUE":
      # self.action_space = gym.spaces.Dict({'action':spaces.Box(low=low[:6], high=high[:6], dtype=np.float32)})
      self.action_space = spaces.Box(low=low[:6], high=high[:6], dtype=np.float32)

    elif self._motor_control_mode == "PD":
      # self.action_space = gym.spaces.Dict({'action':spaces.Box(low=low, high=high, dtype=np.float32)})      
      self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)     

    return self.action_space

  # def _set_observation_space(self, observation):
  #     self.observation_space = convert_observation_to_space(observation)
  #     return self.observation_space
  def _change_observation_space(self):
    self.observation_space = gym.spaces.Dict({'obs':self.observation_space})
    return self.observation_space

  def _get_reward(self):
    curr_pos = self.data.get_body_xpos("torso")
    curr_vel = self.data.get_body_xvelp("torso")
    
    curr_height = curr_pos[2]
    curr_z_vel = curr_vel[2]
    curr_lateral_vel = curr_vel[0:2]

    height_reward = -np.abs(curr_height - self._desired_height)
    z_velocity_reward = curr_z_vel
    lateral_vel_reward = -np.linalg.norm(curr_lateral_vel)

    total_reward = 0.1*height_reward #+ 0.01*curr_z_vel+ 0.001*curr_z_vel
    
    return total_reward