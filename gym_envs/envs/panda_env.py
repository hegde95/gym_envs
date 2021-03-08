import numpy as np
from gym.envs.robotics import rotations, robot_env, utils
import mujoco_py
import os


def goal_distance(goal_a, goal_b):
  assert goal_a.shape == goal_b.shape
  return np.linalg.norm(goal_a - goal_b, axis=-1)


class PandaEnv(robot_env.RobotEnv):
  """Superclass for all Panda environments.
  """

  def __init__(
    self, model_path:str, n_substeps:int, block_gripper:bool,
    has_object:bool, target_offset, target_range,
    distance_threshold, initial_qpos, reward_type,
  ):
    """Initializes a new Panda environment with the default gripper.
    Args:
      model_path (string): path to the environments XML file
      n_substeps (int): number of substeps the simulation runs on every call to step
      block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
      has_object (boolean): whether or not the environment has an object
      target_offset (float or array with 3 elements): offset of the target
      target_range (float): range of a uniform distribution for sampling a target
      distance_threshold (float): the threshold after which a goal is considered achieved
      initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
      reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
    """
    self.block_gripper = block_gripper
    self.has_object = has_object
    self.target_offset = target_offset
    self.target_range = target_range
    self.distance_threshold = distance_threshold
    self.reward_type = reward_type
    self.obj_range = 0 # no randomness in object position on reset

    if model_path is None:
      curr_dir = os.path.dirname(os.path.realpath(__file__))
      self._model_path = os.path.join(curr_dir, 'assets', 'panda_simple.xml')
    else:
      self._model_path = model_path

    super(PandaEnv, self).__init__(
      model_path=self._model_path,
      n_substeps=n_substeps,
      n_actions=8, # 7 dof arm + 1 dof for both gripper fingers
      initial_qpos=initial_qpos)

  # GoalEnv methods
  # ----------------------------

  def compute_reward(self, achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.
    d = goal_distance(achieved_goal, goal)
    if self.reward_type == 'sparse':
      return -(d > self.distance_threshold).astype(np.float32)
    else:
      return -d

  # RobotEnv methods
  # ----------------------------

  def _step_callback(self):
    if self.block_gripper:
      self.sim.data.set_joint_qpos('panda0_finger_joint1', 0.)
      self.sim.data.set_joint_qpos('panda0_finger_joint2', 0.)
      self.sim.forward()

  def _set_action(self, action):
    """ Converts relative action to absolute action needed in the mujoco model"""
    assert action.shape == self.action_space.shape
    qpos = self.sim.data.qpos
    arm_qpos = qpos[0:7]
    gripper_qpos = qpos[7:9]
    
    arm_action = arm_qpos + action[0:7]
    gripper_action = gripper_qpos + np.full(2, action[7])

    self.sim.data.ctrl[:] = np.hstack([arm_action, gripper_action])
    return

  def _get_obs(self):
    # positions
    grip_pos = self.sim.data.get_site_xpos('panda0_end_effector')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('panda0_end_effector') * dt
    if self.has_object:
      object_pos = self.sim.data.get_site_xpos('object0')
      # # rotations
      # object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
      # # velocities
      # object_velp = self.sim.data.get_site_xvelp('object0') * dt
      # object_velr = self.sim.data.get_site_xvelr('object0') * dt
      # # gripper state
      # object_rel_pos = object_pos - grip_pos
      # object_velp -= grip_velp
    else:
      object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
    # gripper_state = robot_qpos[-2:]
    # gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    if not self.has_object:
      achieved_goal = grip_pos.copy()
    else:
      achieved_goal = np.squeeze(object_pos.copy())
    obs = np.concatenate([self.sim.data.qpos, self.sim.data.qvel])

    return {
      'observation': obs.copy(),
      'achieved_goal': achieved_goal.copy(),
      'desired_goal': self.goal.copy(),
    }

  def get_ee_pos(self):
    return self.sim.data.get_site_xpos('panda0_end_effector')

  def _viewer_setup(self):
    body_id = self.sim.model.body_name2id('panda0_link7')
    lookat = self.sim.data.body_xpos[body_id]
    for idx, value in enumerate(lookat):
      self.viewer.cam.lookat[idx] = value
    self.viewer.cam.distance = 2.5
    self.viewer.cam.azimuth = 132.
    self.viewer.cam.elevation = -14.

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    # NOTE: each derived env should have a target0
    site_id = self.sim.model.site_name2id('target0')
    self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
    self.sim.forward()

  def _reset_sim(self):
    self.sim.set_state(self.initial_state)

    # Randomize start position of object.
    if self.has_object:
      object_xpos = self.initial_gripper_xpos[:2] # x,y pos only
      while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
          object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
      object_qpos = self.sim.data.get_joint_qpos('object0:joint') #TODO define object0
      assert object_qpos.shape == (7,) #xyz and quat?
      object_qpos[:2] = object_xpos
      self.sim.data.set_joint_qpos('object0:joint', object_qpos)

    self.sim.forward()
    return True

  # TODO: are these goal methods in pandaenv becuase of GoalEnv? Use carefully
  def _sample_goal(self):
    if self.has_object:
      goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
      goal += self.target_offset
      goal[2] = self.height_offset
      if self.target_in_the_air and self.np_random.uniform() < 0.5:
        goal[2] += self.np_random.uniform(0, 0.45)
    else:
      goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
    return goal.copy()

  def _is_success(self, achieved_goal, desired_goal):
    d = goal_distance(achieved_goal, desired_goal)
    return (d < self.distance_threshold).astype(np.float32)

  def _env_setup(self, initial_qpos):
    for name, value in initial_qpos.items():
      self.sim.data.set_joint_qpos(name, value)
    # utils.reset_mocap_welds(self.sim) # skip. no mocap welds for actuation
    self.sim.forward()

    # Extract information for sampling goals.
    self.initial_gripper_xpos = self.sim.data.get_site_xpos('panda0_end_effector').copy()
    if self.has_object:
      self.height_offset = self.sim.data.get_site_xpos('object0')[2]

  def render(self, mode='human', width=500, height=500):
    return super(PandaEnv, self).render(mode, width, height)

  # Methods for this class
  # ----------------------------

  def set_state(self, qpos, qvel):
    """ Set MjState given the position and velocity"""
    assert qpos.shape == (self.sim.model.nq,) and qvel.shape == (self.sim.model.nv,)
    old_state = self.sim.get_state()
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                      old_state.act, old_state.udd_state)
    self.sim.set_state(new_state)
    self.sim.forward()
