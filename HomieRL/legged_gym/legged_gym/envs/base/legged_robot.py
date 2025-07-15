# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import copy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.terrain_new import HumanoidTerrain
import threading
import time
import pytorch3d.transforms as pt3d
from rsl_rl.utils.global_switch import global_switch
from legged_gym.utils.common import quaternion_to_rpy

def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x.unsqueeze(1), pitch_y.unsqueeze(1), yaw_z.unsqueeze(1)


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs
        self.actor_history_length = self.cfg.env.num_actor_history
        self.critic_history_length = self.cfg.env.num_critic_history
        self.actor_proprioceptive_obs_length = self.num_one_step_obs * self.actor_history_length
        self.critic_proprioceptive_obs_length = self.num_one_step_privileged_obs * self.critic_history_length
        self.actor_use_height = True if self.num_obs > self.actor_proprioceptive_obs_length else False
        self.num_lower_dof = self.cfg.env.num_actions
        #TODO
        self.arm_one_step_obs = self.cfg.arm.arm_num_observations
        self.arm_actor_history_length = self.cfg.arm.arm_num_observation_history
        self.arm_actor_proprioceptive_obs_length = self.arm_one_step_obs * self.arm_actor_history_length
        self.num_actions_arm = self.cfg.arm.num_actions_arm
        self.num_plan_actions = self.cfg.arm.num_actions_arm_cd - self.cfg.arm.num_actions_arm

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True



    def plan(self, plan_actions):
        """ Plan actions based on the provided commands.
            This function is called by the algorithm to get actions for the robot.
        """

        self.commands[:,4] = plan_actions[...,3]# height
        self.rpy_commands[:,0:3] = plan_actions[...,0:3]  # roll, pitch, yaw
        self.plan_actions[:] = plan_actions

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        upper_actions  = actions[:,self.num_lower_dof:]
        actions = actions[:, :self.num_lower_dof]  # lower body actions
        clip_actions = self.cfg.normalization.clip_actions
        if (self.common_step_counter % self.cfg.domain_rand.upper_interval == 0):
            # (NOTE) implementation of upper-body curriculum
            self.random_upper_ratio = min(self.action_curriculum_ratio, 1.0)
            uu = torch.rand(self.num_envs, self.num_actions - self.num_lower_dof, device=self.device)
            self.random_upper_ratio = -1.0 / (20 * (1-self.random_upper_ratio*0.99))*torch.log(1 - uu + uu * np.exp(-20 * (1-self.random_upper_ratio*0.99)))
            self.random_joint_ratio = self.random_upper_ratio * torch.rand(self.num_envs, self.num_actions - self.num_lower_dof).to(self.device)
            rand_pos = torch.rand(self.num_envs, self.num_actions - self.num_lower_dof, device=self.device) - 0.5
            self.random_upper_actions = ((self.action_min[:, self.num_lower_dof:] * (rand_pos >= 0)) + (self.action_max[:, self.num_lower_dof:] * (rand_pos < 0) ))* self.random_joint_ratio
            self.delta_upper_actions = (self.random_upper_actions - self.current_upper_actions) / (self.cfg.domain_rand.upper_interval)
        self.current_upper_actions += self.delta_upper_actions
        actions = torch.cat((actions, self.current_upper_actions), dim=-1)
        # TODO
        if global_switch.switch_open:
            #训练上肢策略时使用输入上肢动作
            actions[:,self.num_lower_dof + 1:] = upper_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        #print(self.actions)
        self.origin_actions[:] = self.actions[:]
        self.delayed_actions = self.actions.clone().view(1, self.num_envs, self.num_actions).repeat(self.cfg.control.decimation, 1, 1)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
                
        # Randomize Joint Injections
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # upper-body with position control; lower-body with force control;
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.rew_buf_arm, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.arm_time_buf += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.torso_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.torso_index, 3:7], self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt
        
        self.feet_pos[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 3:7]
        self.feet_vel[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        #TODO 更新手部位姿
        self.left_hand_state = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.left_hand_index]  # shape: [num_envs, 13]
        self.right_hand_state = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.right_hand_index]

        #TODO 更新头部位姿
        self.head_state = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.head_index]  # shape: [num_envs, 13]
        # compute contact related quantities
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        self.first_contacts = (self.feet_air_time >= self.dt) * self.contact_filt
        self.feet_air_time += self.dt
        feet_height, feet_height_var = self._get_feet_heights()
        self.feet_max_height = torch.maximum(self.feet_max_height, feet_height)
        
        # compute joint power
        joint_power = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((self.joint_powers[:, 1:], joint_power), dim=1)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_plan_actions[:] = self.plan_actions[:]
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target [:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        
        # reset contact related quantities
        self.feet_air_time *= ~self.contact_filt
        self.feet_max_height *= ~self.contact_filt

        return env_ids, termination_privileged_obs

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 10., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.gravity_termination_buf = torch.any(torch.norm(self.projected_gravity[:, 0:2], dim=-1, keepdim=True) > 0.8, dim=1)
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.gravity_termination_buf

        self.reverse_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        if global_switch.switch_open:
            # NOTE If you suddenly resample the arm action, it will cause problems.
            #   The body still maintains the original posture,
            #   so make a judgment when you walk 0.6 of the process.
            time_exceed_half = (self.arm_time_buf / (self.T_trajs / self.dt)) > 0.6
            # 计算每个环境的arm动作时间是否超过总时间的0.6
            self.reverse_buf = self.reverse_buf & time_exceed_half
            # 只有当arm动作时间超过总时间的0.6时，才考虑反转条件
            self.reset_buf |= self.reverse_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        # update action curriculum for specific dofs
        if self.cfg.env.action_curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_action_curriculum(env_ids)
            
        self.refresh_actor_rigid_shape_props(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # resample commands
        self._resample_commands(env_ids)
        #resample arm commands 修改
        self._resample_arm_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.joint_powers[env_ids] = 0.
        self.random_upper_actions[env_ids] = 0. 
        self.current_upper_actions[env_ids] = 0.
        self.delta_upper_actions[env_ids] = 0.
        reset_roll, reset_pitch, reset_yaw = euler_from_quaternion(self.base_quat[env_ids])
        self.roll[env_ids] = reset_roll
        self.pitch[env_ids] = reset_pitch
        self.yaw[env_ids] = reset_yaw
        self.reset_buf[env_ids] = 1
        
         #reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (len(env_ids), self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            # self.extras["episode"]["height_curriculum_ratio"] = self.height_curriculum_ratio
        if self.cfg.env.action_curriculum:
            self.extras["episode"]["action_curriculum_ratio"] = self.action_curriculum_ratio
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):#TODO等划分讨论好了再写
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """

        reward_scales = global_switch.get_reward_scales()
        self.rew_buf[:] = 0.
        self.rew_buf_arm[:] = 0.
        # print("reward names:",self.reward_names)
        # print("reward scales:",reward_scales)
        # print("reward functions:",self.reward_functions)
        # input("press enter to continue")
        for i in range(len(reward_scales)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]

            if torch.isnan(rew).any():
                import ipdb; ipdb.set_trace()
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """
        imu_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.rigid_body_states[:, self.imu_index,10:13])
        imu_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.gravity_vec)
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,     #3  commands: vel_x, vel_y, ang_z
                                    self.commands[:, 4].unsqueeze(1),   #1  height
                                    imu_ang_vel  * self.obs_scales.ang_vel,     #3  detect vel_x, vel_y, ang_z
                                    imu_projected_gravity,      #3  rpy
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,    #27
                                    self.dof_vel * self.obs_scales.dof_vel,     #27
                                    self.actions[:, :12],       #12 lower body actions
                                    self.plan_actions# TODO: 上肢策略输出的rpy添加
                                    ),dim=-1)
        current_actor_obs = torch.clone(current_obs)

        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec[0:(10 + 2 * self.num_actions + self.num_lower_dof + self.num_plan_actions)]
        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_obs:self.actor_proprioceptive_obs_length], current_actor_obs[:, :self.num_one_step_obs]), dim=-1)#带history
        current_critic_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf[:, self.num_one_step_privileged_obs:self.critic_proprioceptive_obs_length], current_critic_obs), dim=-1)#带history

    def get_arm_observations(self):
        """ Computes arm observations
            Returns the arm observations, which are the last 12 actions
        """
        imu_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index, 3:7],
                                          self.rigid_body_states[:, self.imu_index, 10:13])
        imu_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index, 3:7], self.gravity_vec)
        obs_buf = torch.cat(((self.dof_pos[:, self.num_lower_dof + 1:self.num_dof]
                              - self.default_dof_pos[:,
                                self.num_lower_dof + 1:self.num_dof])
                             * self.obs_scales.dof_pos,
                             # self.dof_vel[:, self.num_actions_loco:self.num_actions_loco+self.num_actions_arm] * self.obs_scales.dof_vel,
                             self.actions[:, self.num_lower_dof + 1:self.num_dof], #upper_dof
                             self.plan_actions
                             ), dim=-1)
        obs_buf = torch.cat((obs_buf,
                             self.commands_arm_obs[:, : 3*6],
                             imu_projected_gravity
                             ), dim=-1)
        base_height_l = self.root_states[:, 2] - self.feet_pos[:, 0, 2]
        base_height_r = self.root_states[:, 2] - self.feet_pos[:, 1, 2]
        base_height = torch.max(base_height_l, base_height_r)
        critic_obs_buf = torch.cat((obs_buf, base_height.unsqueeze(-1)), dim=-1)
        self.arm_obs_history = torch.cat((self.arm_obs_history[:, self.arm_one_step_obs:self.arm_actor_proprioceptive_obs_length], obs_buf), dim=1)
        return {'obs': obs_buf, 'privileged_obs': critic_obs_buf, 'obs_history': self.arm_obs_history}


    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        imu_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.rigid_body_states[:, self.imu_index,10:13])
        imu_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.gravity_vec)
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.commands[:, 4].unsqueeze(1),
                                    imu_ang_vel  * self.obs_scales.ang_vel,
                                    imu_projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions[:, :12],
                                    self.plan_actions
                                    ),dim=-1)

        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(10 + 2 * self.num_actions + self.num_lower_dof + self.num_plan_actions)]
        current_critic_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        return torch.cat((self.privileged_obs_buf[:, self.num_one_step_privileged_obs:self.critic_proprioceptive_obs_length], current_critic_obs), dim=-1)[env_ids]
            
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def create_cameras(self):
        """ Creates camera for each robot
        """
        self.camera_params = gymapi.CameraProperties()
        self.camera_params.width = self.cfg.camera.width
        self.camera_params.height = self.cfg.camera.height
        self.camera_params.horizontal_fov = self.cfg.camera.horizontal_fov
        self.camera_params.enable_tensors = True
        self.cameras = []
        for env_handle in self.envs:
            camera_handle = self.gym.create_camera_sensor(env_handle, self.camera_params)
            torso_handle = self.gym.get_actor_rigid_body_handle(env_handle, 0, self.torso_index)
            camera_offset = gymapi.Vec3(self.cfg.camera.offset[0], self.cfg.camera.offset[1], self.cfg.camera.offset[2])
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(self.cfg.camera.angle_randomization * (2 * np.random.random() - 1) + self.cfg.camera.angle))
            self.gym.attach_camera_to_body(camera_handle, env_handle, torso_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
            self.cameras.append(camera_handle)
            
    def post_process_camera_tensor(self):
        """
        First, post process the raw image and then stack along the time axis
        """
        new_images = torch.stack(self.cam_tensors)
        new_images = torch.nan_to_num(new_images, neginf=0)
        new_images = torch.clamp(new_images, min=-self.cfg.camera.far, max=-self.cfg.camera.near)
        # new_images = new_images[:, 4:-4, :-2] # crop the image
        self.last_visual_obs_buf = torch.clone(self.visual_obs_buf)
        self.visual_obs_buf = new_images.view(self.num_envs, -1)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.actor_handles[env_id]
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)

            for i in range(len(rigid_shape_props)):
                if self.cfg.domain_rand.randomize_friction:
                    rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_restitution:
                    rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id==0:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass
                print(f"Mass of body {i}: {p.mass} (before randomization)")
            print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[self.torso_body_index].mass = self.default_rigid_body_mass[self.torso_body_index] + self.payload[env_id, 0]
            props[self.left_hand_index].mass = self.default_rigid_body_mass[self.left_hand_index] + self.hand_payload[env_id, 0]
            props[self.right_hand_index].mass = self.default_rigid_body_mass[self.right_hand_index] + self.hand_payload[env_id, 1]
            
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = self.default_com + gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])
        if self.cfg.domain_rand.randomize_body_displacement:
            props[self.torso_body_index].com = self.default_body_com + gymapi.Vec3(self.body_displacement[env_id, 0], self.body_displacement[env_id, 1], self.body_displacement[env_id, 2])

        
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        # 采样需要重采样轨迹的环境（如机械臂轨迹时间到达）arm_time_buf累加
        traj_ids = (self.arm_time_buf % (self.T_trajs / self.dt).long() == 0).nonzero(as_tuple=False).flatten()
        self._resample_arm_commands(traj_ids)

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
                
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()


    # TODO

    def _resample_arm_commands(self, env_ids):
        if len(env_ids) == 0 or not global_switch.switch_open:
            return

        # 采样左手位置
        self.commands_arm[env_ids, 0] = torch_rand_float(self.cfg.arm.arm_commands.l[0], self.cfg.arm.arm_commands.l[1],
                                                         (env_ids.shape[0], 1), device=self.device).squeeze()
        self.commands_arm[env_ids, 1] = torch_rand_float(self.cfg.arm.arm_commands.p[0], self.cfg.arm.arm_commands.p[1],
                                                         (env_ids.shape[0], 1), device=self.device).squeeze()
        self.commands_arm[env_ids, 2] = torch_rand_float(self.cfg.arm.arm_commands.y[0], self.cfg.arm.arm_commands.y[1],
                                                         (env_ids.shape[0], 1), device=self.device).squeeze()
        # 采样右手位置
        self.commands_arm[env_ids, 6] = torch_rand_float(self.cfg.arm.arm_commands.l[0], self.cfg.arm.arm_commands.l[1],
                                                         (env_ids.shape[0], 1), device=self.device).squeeze()
        self.commands_arm[env_ids, 7] = torch_rand_float(self.cfg.arm.arm_commands.p[0], self.cfg.arm.arm_commands.p[1],
                                                         (env_ids.shape[0], 1), device=self.device).squeeze()
        self.commands_arm[env_ids, 8] = torch_rand_float(self.cfg.arm.arm_commands.y[0], self.cfg.arm.arm_commands.y[1],
                                                         (env_ids.shape[0], 1), device=self.device).squeeze()
        # 采样头部位置
        self.commands_arm[env_ids, 12] = torch_rand_float(self.cfg.arm.head_commands.l[0],
                                                          self.cfg.arm.head_commands.l[1], (env_ids.shape[0], 1),
                                                          device=self.device).squeeze()
        self.commands_arm[env_ids, 13] = torch_rand_float(self.cfg.arm.head_commands.p[0],
                                                          self.cfg.arm.head_commands.p[1], (env_ids.shape[0], 1),
                                                          device=self.device).squeeze()
        self.commands_arm[env_ids, 14] = torch_rand_float(self.cfg.arm.head_commands.y[0],
                                                          self.cfg.arm.head_commands.y[1], (env_ids.shape[0], 1),
                                                          device=self.device).squeeze()

        # 写入观测
        self.commands_arm_obs[env_ids, 0:3] = self.commands_arm[env_ids, 0:3]
        self.commands_arm_obs[env_ids, 6:9] = self.commands_arm[env_ids, 6:9]
        self.commands_arm_obs[env_ids, 12:15] = self.commands_arm[env_ids, 12:15]

        # 左手姿态
        roll_left = torch_rand_float(self.cfg.arm.arm_commands.roll_ee[0], self.cfg.arm.arm_commands.roll_ee[1],
                                     (env_ids.shape[0], 1), device=self.device).squeeze()
        pitch_left = torch_rand_float(self.cfg.arm.arm_commands.pitch_ee[0], self.cfg.arm.arm_commands.pitch_ee[1],
                                      (env_ids.shape[0], 1), device=self.device).squeeze()
        yaw_left = torch_rand_float(self.cfg.arm.arm_commands.yaw_ee[0], self.cfg.arm.arm_commands.yaw_ee[1],
                                    (env_ids.shape[0], 1), device=self.device).squeeze()
        # 右手姿态
        roll_right = torch_rand_float(self.cfg.arm.arm_commands.roll_ee[0], self.cfg.arm.arm_commands.roll_ee[1],
                                      (env_ids.shape[0], 1), device=self.device).squeeze()
        pitch_right = torch_rand_float(self.cfg.arm.arm_commands.pitch_ee[0], self.cfg.arm.arm_commands.pitch_ee[1],
                                       (env_ids.shape[0], 1), device=self.device).squeeze()
        yaw_right = torch_rand_float(self.cfg.arm.arm_commands.yaw_ee[0], self.cfg.arm.arm_commands.yaw_ee[1],
                                     (env_ids.shape[0], 1), device=self.device).squeeze()
        # 头部姿态
        roll_head = torch_rand_float(self.cfg.arm.head_commands.roll_ee[0], self.cfg.arm.head_commands.roll_ee[1],
                                     (env_ids.shape[0], 1), device=self.device).squeeze()
        pitch_head = torch_rand_float(self.cfg.arm.head_commands.pitch_ee[0], self.cfg.arm.head_commands.pitch_ee[1],
                                      (env_ids.shape[0], 1), device=self.device).squeeze()
        yaw_head = torch_rand_float(self.cfg.arm.head_commands.yaw_ee[0], self.cfg.arm.head_commands.yaw_ee[1],
                                    (env_ids.shape[0], 1), device=self.device).squeeze()

        zero_vec = torch.zeros_like(roll_left)
        # 左手四元数
        q1_left = quat_from_euler_xyz(zero_vec, zero_vec, yaw_left)
        q2_left = quat_from_euler_xyz(zero_vec, pitch_left, zero_vec)
        q3_left = quat_from_euler_xyz(roll_left, zero_vec, zero_vec)
        quat_left = quat_mul(q1_left, quat_mul(q2_left, q3_left))
        # 右手四元数
        q1_right = quat_from_euler_xyz(zero_vec, zero_vec, yaw_right)
        q2_right = quat_from_euler_xyz(zero_vec, pitch_right, zero_vec)
        q3_right = quat_from_euler_xyz(roll_right, zero_vec, zero_vec)
        quat_right = quat_mul(q1_right, quat_mul(q2_right, q3_right))
        # 头部四元数
        q1_head = quat_from_euler_xyz(zero_vec, zero_vec, yaw_head)
        q2_head = quat_from_euler_xyz(zero_vec, pitch_head, zero_vec)
        q3_head = quat_from_euler_xyz(roll_head, zero_vec, zero_vec)
        quat_head = quat_mul(q1_head, quat_mul(q2_head, q3_head))

        # 写入四元数缓存（如有需要可分别存储）
        self.obj_quats_left[env_ids] = quat_left.reshape(-1, 4)
        self.obj_quats_right[env_ids] = quat_right.reshape(-1, 4)
        self.obj_quats_head[env_ids] = quat_head.reshape(-1, 4)

        # 写入观测
        rpy_left = self.quat_to_angle(self.obj_quats_left[env_ids]).to(self.device)
        rpy_right = self.quat_to_angle(self.obj_quats_right[env_ids]).to(self.device)
        rpy_head = self.quat_to_angle(self.obj_quats_head[env_ids]).to(self.device)
        self.target_abg[env_ids] = torch.cat((rpy_left, rpy_right, rpy_head), dim=-1)

        self.commands_arm_obs[env_ids, 3:6] = rpy_left
        self.commands_arm_obs[env_ids, 9:12] = rpy_right
        self.commands_arm_obs[env_ids, 15:18] = rpy_head
        self._resample_Traj_commands(env_ids)

    def _resample_Traj_commands(self, env_ids):
        time_range = (self.cfg.arm.arm_commands.T_traj[1] - self.cfg.arm.arm_commands.T_traj[0]) / self.dt
        time_interval = torch.from_numpy(np.random.choice(int(time_range + 1), len(env_ids))).to(self.device)

        self.T_trajs[env_ids] = torch.ones_like(self.T_trajs[env_ids]) * self.cfg.arm.arm_commands.T_traj[
            0] + time_interval * self.dt
        self.arm_time_buf[env_ids] = torch.zeros_like(self.arm_time_buf[env_ids])

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        set_x = torch.rand(len(env_ids), 1).to(self.device)
        is_height = set_x < 1/3
        is_vel = set_x > 1/2
        self.commands[env_ids, 0] = (torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device) * is_vel).squeeze(1) 
        self.commands[env_ids, 1] = (torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device) * is_vel).squeeze(1) 
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = (torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device) * is_vel).squeeze(1)
            self.commands[env_ids, 4] = (torch_rand_float(self.command_ranges["height"][0], self.command_ranges["height"][1], (len(env_ids), 1), device=self.device) * is_height).squeeze(1) + self.cfg.rewards.base_height_target # height
        else:
            self.commands[env_ids, 2] = (torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device) * is_vel).squeeze(1)
            self.commands[env_ids, 4] = (torch_rand_float(self.command_ranges["height"][0], self.command_ranges["height"][1], (len(env_ids), 1), device=self.device) * is_height).squeeze(1) + self.cfg.rewards.base_height_target # height
        
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        self.joint_pos_target = self.default_dof_pos + actions_scaled
        self.joint_pos_target_2 = self.default_dof_pos + actions_scaled * 0
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
            torques = torques + self.actuation_offset + self.joint_injection
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
            torques = torques + self.actuation_offset + self.joint_injection
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        elif control_type=="M":
            torques = self.p_gains * self.Kp_factors * (
                    self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
            
            torques = torques + self.actuation_offset + self.joint_injection
            torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
            # print("default dof pos:",self.default_dof_pos)
            return torch.cat((torques[..., :self.num_lower_dof], self.joint_pos_target[..., self.num_lower_dof:]), dim=-1)
            # return torch.cat((torques[..., :self.num_lower_dof], self.joint_pos_target_2[..., self.num_lower_dof:]),
            #                  dim=-1)
        
        else:
            raise NameError(f"Unknown controller type: {control_type}")

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
        if self.cfg.domain_rand.randomize_initial_joint_pos:
            init_dos_pos = self.default_dof_pos * torch_rand_float(self.cfg.domain_rand.initial_joint_pos_scale[0], self.cfg.domain_rand.initial_joint_pos_scale[1], (len(env_ids), self.num_dof), device=self.device)
            init_dos_pos += torch_rand_float(self.cfg.domain_rand.initial_joint_pos_offset[0], self.cfg.domain_rand.initial_joint_pos_offset[1], (len(env_ids), self.num_dof), device=self.device)
            self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch.ones((len(env_ids), self.num_dof), device=self.device)

        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            self.root_states[env_ids, 2:3] += torch_rand_float(0.0, 0.1, (len(env_ids), 1), device=self.device) # z position within 0.1m of the ground
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 75% of the maximum, increase the range of commands
        if (torch.mean(self.episode_sums["tracking_x_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_x_vel"]) and (torch.mean(self.episode_sums["tracking_y_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_y_vel"]):
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum)

        
    def update_action_curriculum(self, env_ids):
        """ Implements a curriculum of increasing action range

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if (torch.mean(self.episode_sums["tracking_x_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_x_vel"]):
            self.action_curriculum_ratio += 0.05
            self.action_curriculum_ratio = min(self.action_curriculum_ratio, 1.0)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(10 + 2*self.num_actions + self.num_lower_dof + self.num_plan_actions, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:4] = 0. # commands
        noise_vec[4:7] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[7:10] = noise_scales.gravity * noise_level
        noise_vec[10:(10 + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(10 + self.num_actions):(10 + 2 * self.num_actions)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(10 + 2 * self.num_actions):(10 + 2 * self.num_actions + self.num_lower_dof)] = 0. # previous actions
        noise_vec[(10 + 2 * self.num_actions + self.num_lower_dof):] = noise_scales.plan_actions * noise_level * self.obs_scales.plan_actions
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.origin_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_max_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.first_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # new add
        self.torso_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.torso_index, 3:7], self.gravity_vec)
        # end new add
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            print(f"Joint {self.gym.find_actor_dof_index(self.envs[0], self.actor_handles[0], name, gymapi.IndexDomain.DOMAIN_ACTOR)}: {name}")
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.action_max = (self.hard_dof_pos_limits[:, 1].unsqueeze(0) - self.default_dof_pos) / self.cfg.control.action_scale
        self.action_min = (self.hard_dof_pos_limits[:, 0].unsqueeze(0) - self.default_dof_pos) / self.cfg.control.action_scale
        self.action_curriculum_ratio = self.cfg.domain_rand.init_upper_ratio
        self.target_heights = torch.ones((self.num_envs), device=self.device) * self.cfg.rewards.base_height_target
        print(f"Action min: {self.action_min}")
        print(f"Action max: {self.action_max}")
        
        self.random_upper_actions = torch.zeros((self.num_envs, self.num_actions - self.num_lower_dof), device=self.device)
        self.current_upper_actions = torch.zeros((self.num_envs, self.num_actions - self.num_lower_dof), device=self.device)
        self.delta_upper_actions = torch.zeros((self.num_envs, 1), device=self.device)
        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_injection = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuation_offset = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
            self.hand_payload = torch_rand_float(self.cfg.domain_rand.hand_payload_mass_range[0], self.cfg.domain_rand.hand_payload_mass_range[1], (self.num_envs ,2), device=self.device)

        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
        if self.cfg.domain_rand.randomize_body_displacement:
            self.body_displacement = torch_rand_float(self.cfg.domain_rand.body_displacement_range[0], self.cfg.domain_rand.body_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        
        #joint powers
        self.joint_powers = torch.zeros(self.num_envs, 100, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        # arm commands
        self.commands_arm = torch.zeros(self.num_envs, self.cfg.arm.arm_num_commands, dtype=torch.float,
                                        device=self.device, requires_grad=False)  # lpy, rpy for transfer to camera base
        self.commands_arm_obs = torch.zeros(self.num_envs, self.cfg.arm.arm_num_commands, dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)  # lpy, rpy for transfer to camera base
        #TODO改维度
        self.obj_quats_left = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
        self.obj_quats_right = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
        self.obj_quats_head = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
        self.visual_rpy = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.target_abg = torch.zeros((self.num_envs, 9), device=self.device, dtype=torch.float) # target angles in the arm end effector frame

        self.T_trajs = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # command time
        self.arm_time_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.force_time_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        #TODO plan

        self.plan_actions = torch.zeros(self.num_envs, self.num_plan_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_plan_actions = torch.zeros(self.num_envs, self.num_plan_actions, dtype=torch.float,
                                             device=self.device, requires_grad=False)
        self.rpy_commands = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float) # rpy commands for the arm end effector
        #TODO reward buf
        self.rew_buf_arm = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                                      device=self.device, requires_grad=False)
        self.commands_arm_lpy_range = torch.tensor(
            [self.cfg.arm.arm_commands.l[1] - self.cfg.arm.arm_commands.l[0],
             self.cfg.arm.arm_commands.p[1] - self.cfg.arm.arm_commands.p[0],
             self.cfg.arm.arm_commands.y[1] - self.cfg.arm.arm_commands.y[0]],
            device=self.device,
            requires_grad=False,
        ).reshape(1, -1)
        self.commands_head_lpy_range = torch.tensor(
            [self.cfg.arm.head_commands.l[1] - self.cfg.arm.head_commands.l[0],
             self.cfg.arm.head_commands.p[1] - self.cfg.arm.head_commands.p[0],
             self.cfg.arm.head_commands.y[1] - self.cfg.arm.head_commands.y[0]],
            device=self.device,
            requires_grad=False,
        ).reshape(1, -1)
        self.commands_arm_rpy_range = torch.tensor(
            [self.cfg.arm.arm_commands.roll_ee[1] - self.cfg.arm.arm_commands.roll_ee[0],
             self.cfg.arm.arm_commands.pitch_ee[1] - self.cfg.arm.arm_commands.pitch_ee[0],
             self.cfg.arm.arm_commands.yaw_ee[1] - self.cfg.arm.arm_commands.yaw_ee[0]],
            device=self.device,
            requires_grad=False,
        ).reshape(1, -1)
        self.commands_head_rpy_range = torch.tensor(
            [self.cfg.arm.head_commands.roll_ee[1] - self.cfg.arm.head_commands.roll_ee[0],
             self.cfg.arm.head_commands.pitch_ee[1] - self.cfg.arm.head_commands.pitch_ee[0],
             self.cfg.arm.head_commands.yaw_ee[1] - self.cfg.arm.head_commands.yaw_ee[0]],
            device=self.device,
            requires_grad=False,
        ).reshape(1, -1)

        #TODO measure height
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.measured_heights = self._get_heights()
        else:
            self.measured_heights = 0

        # TODO arm_get_obs
        self.arm_obs_history = torch.zeros((self.num_envs, self.cfg.arm.arm_num_observation_history * self.cfg.arm.arm_num_observations), device=self.device, dtype=torch.float)
        #print(f"Arm obs history shape: {self.arm_obs_history.shape}")
    def _prepare_reward_function(self): #TODO 划分还要再讨论
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt

        for key in list(self.hybrid_reward_scales.keys()):
            self.hybrid_reward_scales[key] *= self.dt

        # update pretrained reward scales with hybrid reward scales
        for name, scale in self.reward_scales.items():
            if name not in self.hybrid_reward_scales:
                self.hybrid_reward_scales[name] = scale

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.hybrid_reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, '_reward_' + name))
            if name not in self.reward_scales:
                self.reward_scales[name] = 0.

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        # print(f"Reward scales: {self.reward_scales.keys()}")
        # print(f"Hybrid reward scales: {self.hybrid_reward_scales.keys()}")
        # input("Press Enter to continue...")
        global_switch.set_reward_scales(self.hybrid_reward_scales, self.reward_scales)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(self.body_names)
        self.num_dof = len(self.dof_names)
        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        left_foot_names = [s for s in self.body_names if self.cfg.asset.left_foot_name in s]
        right_foot_names = [s for s in self.body_names if self.cfg.asset.right_foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])
            
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.hand_payload = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
            self.hand_payload = torch_rand_float(self.cfg.domain_rand.hand_payload_mass_range[0], self.cfg.domain_rand.hand_payload_mass_range[1], (self.num_envs, 2), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
        if self.cfg.domain_rand.randomize_body_displacement:
            self.body_displacement = torch_rand_float(self.cfg.domain_rand.body_displacement_range[0], self.cfg.domain_rand.body_displacement_range[1], (self.num_envs, 3), device=self.device)

        print('body_names',self.body_names)
        self.torso_body_index = self.body_names.index("torso_link")
        self.head_index = self.body_names.index("head_link")
        # self.left_hand_index = self.body_names.index("left_hand_palm_link")
        # self.right_hand_index = self.body_names.index("right_hand_palm_link")
        self.left_hand_index = self.body_names.index("left_rubber_hand")
        self.right_hand_index = self.body_names.index("right_rubber_hand")
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            dof_props["driveMode"][12:].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"][12:] = [300., 200., 200., 200., 100.,  20.,  20.,  20., 200., 200., 200., 100.,  20.,  20.,  20.]
            dof_props["damping"][12:] = [5.0000, 4.0000, 4.0000, 4.0000, 1.0000, 0.5000, 0.5000,
                                            0.5000, 4.0000, 4.0000, 4.0000, 1.0000, 0.5000, 0.5000, 0.5000]
        
        
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            if i == 0:
                self.default_com = copy.deepcopy(body_props[0].com)
                self.default_body_com = copy.deepcopy(body_props[self.torso_body_index].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
            
        knee_names = self.cfg.asset.knee_names
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])
            
        self.left_foot_indices = torch.zeros(len(left_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_foot_names)):
            self.left_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_foot_names[i])
        
        self.right_foot_indices = torch.zeros(len(right_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_foot_names)):
            self.right_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_foot_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
      
        self.left_leg_joint_indices = torch.zeros(len(self.cfg.asset.left_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_leg_joints)):
            self.left_leg_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_leg_joints[i])
            
        self.right_leg_joint_indices = torch.zeros(len(self.cfg.asset.right_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_leg_joints)):
            self.right_leg_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_leg_joints[i])
            
        self.leg_joint_indices = torch.cat((self.left_leg_joint_indices, self.right_leg_joint_indices))
        
        self.left_hip_joint_indices = torch.zeros(len(self.cfg.asset.left_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_hip_joints)):
            self.left_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_joints[i])
            
        self.right_hip_joint_indices = torch.zeros(len(self.cfg.asset.right_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_hip_joints)):
            self.right_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_joints[i])
            
        self.hip_joint_indices = torch.cat((self.left_hip_joint_indices, self.right_hip_joint_indices))
        
        self.hip_pitch_joint_indices = torch.zeros(len(self.cfg.asset.hip_pitch_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.hip_pitch_joints)):
            self.hip_pitch_joint_indices[i] = self.dof_names.index(self.cfg.asset.hip_pitch_joints[i])
    
            
        self.ankle_joint_indices = torch.zeros(len(self.cfg.asset.ankle_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.ankle_joints)):
            self.ankle_joint_indices[i] = self.dof_names.index(self.cfg.asset.ankle_joints[i])
            
        self.knee_joint_indices = torch.zeros(len(self.cfg.asset.knee_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.knee_joints)):
            self.knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.knee_joints[i])
            
        self.upper_body_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.upper_body_link)
        self.imu_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.imu_link)
        # new add
        self.torso_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.terminate_after_contacts_on[0])
        print("torso index:",self.torso_index)
        # end new add

    def _get_env_origins(self):
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(
                torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.hybrid_reward_scales = class_to_dict(self.cfg.hybrid.reward_scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.upper_interval = np.ceil(self.cfg.domain_rand.upper_interval_s / self.dt)

    def _get_feet_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :3].clone()
        if self.cfg.terrain.mesh_type == 'plane':
            left_foot_height = torch.mean(left_foot_pos[:, :, 2], dim = -1, keepdim=True)
            left_foot_height_var = torch.var(left_foot_pos[:, :, 2], dim = -1, keepdim=True)
            right_foot_height = torch.mean(right_foot_pos[:, :, 2], dim = -1, keepdim=True)
            right_foot_height_var = torch.var(right_foot_pos[:, :, 2], dim = -1, keepdim=True)
            return torch.cat((left_foot_height, right_foot_height), dim=-1), torch.cat((left_foot_height_var, right_foot_height_var), dim=-1)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            left_points = left_foot_pos[env_ids].clone()
            right_points = right_foot_pos[env_ids].clone()
        else:
            left_points = left_foot_pos.clone()
            right_points = right_foot_pos.clone()

        left_points += self.terrain.cfg.border_size
        right_points += self.terrain.cfg.border_size
        left_points = (left_points/self.terrain.cfg.horizontal_scale).long()
        right_points = (right_points/self.terrain.cfg.horizontal_scale).long()
        left_px = left_points[:, :, 0].view(-1)
        right_px = right_points[:, :, 0].view(-1)
        left_py = left_points[:, :, 1].view(-1)
        right_py = right_points[:, :, 1].view(-1)
        left_px = torch.clip(left_px, 0, self.height_samples.shape[0]-2)
        right_px = torch.clip(right_px, 0, self.height_samples.shape[0]-2)
        left_py = torch.clip(left_py, 0, self.height_samples.shape[1]-2)
        right_py = torch.clip(right_py, 0, self.height_samples.shape[1]-2)

        left_heights1 = self.height_samples[left_px, left_py]
        left_heights2 = self.height_samples[left_px+1, left_py]
        left_heights3 = self.height_samples[left_px, left_py+1]
        left_heights = torch.min(left_heights1, left_heights2)
        left_heights = torch.min(left_heights, left_heights3)
        left_heights = left_heights.view(self.num_envs, -1) * self.cfg.terrain.vertical_scale
        left_foot_heights =  left_foot_pos[:, :, 2] - left_heights

        right_heights1 = self.height_samples[right_px, right_py]
        right_heights2 = self.height_samples[right_px+1, right_py]
        right_heights3 = self.height_samples[right_px, right_py+1]
        right_heights = torch.min(right_heights1, right_heights2)
        right_heights = torch.min(right_heights, right_heights3)
        right_heights = right_heights.view(self.num_envs, -1) * self.cfg.terrain.vertical_scale
        right_foot_heights =  right_foot_pos[:, :, 2] - right_heights

        feet_heights = torch.cat((torch.mean(left_foot_heights, dim=-1, keepdim=True), torch.mean(right_foot_heights, dim=-1, keepdim=True)), dim=-1)
        feet_heights_var = torch.cat((torch.var(left_foot_heights, dim=-1, keepdim=True), torch.var(right_foot_heights, dim=-1, keepdim=True)), dim=-1)

        return torch.clip(feet_heights, min=0.), feet_heights_var

    #TODO measure height 来自humanoid
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heightXBotL = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heightXBotL)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def get_lpy_in_base_coord(self, env_ids, effector = 'left_hand'):
        #input("get_lpy_in_base_coord")
        if effector == 'left_hand':
            effector_state = self.left_hand_state
        elif effector == 'right_hand':
            effector_state = self.right_hand_state
        elif effector == 'head':
            effector_state = self.head_state
        else:
            raise ValueError('effector must be left_hand, right_hand, or head')

        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])

        self.grasper_move = torch.tensor([0.1, 0, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
        self.grasper_move_in_world = quat_rotate(effector_state[env_ids, 3:7], self.grasper_move)
        self.grasper_in_world = effector_state[env_ids, :3] + self.grasper_move_in_world

        x = torch.cos(yaw) * (self.grasper_in_world[:, 0] - self.root_states[env_ids, 0]) \
            + torch.sin(yaw) * (self.grasper_in_world[:, 1] - self.root_states[env_ids, 1])
        y = -torch.sin(yaw) * (self.grasper_in_world[:, 0] - self.root_states[env_ids, 0]) \
            + torch.cos(yaw) * (self.grasper_in_world[:, 1] - self.root_states[env_ids, 1])
        z = torch.mean(self.grasper_in_world[:, 2].unsqueeze(1) - self.measured_heights, dim=1) - 0.64

        l = torch.sqrt(x**2 + y**2 + z**2)
        p = torch.atan2(z, torch.sqrt(x**2 + y**2))
        y_aw = torch.atan2(y, x)

        return torch.stack([l, p, y_aw], dim=-1)

    def get_alpha_beta_gamma_in_base_coord(self, env_ids, effector = 'left_hand'):
        # input("get_alpha_beta_gamma_in_base_coord")
        if effector == 'left_hand':
            effector_state = self.left_hand_state
        elif effector == 'right_hand':
            effector_state = self.right_hand_state
        elif effector == 'head':
            effector_state = self.head_state
        else:
            raise ValueError('effector must be left_hand, right_hand, or head')
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        base_quats = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        ee_in_base_quats = quat_mul(quat_conjugate(base_quats), effector_state[:, 3:7])
        abg = self.quat_to_angle(ee_in_base_quats)

        return abg

    def quat_to_angle(self, quat):
        quat = quat.to(self.device)
        y_vector = to_torch([0., 1., 0.], device=self.device).repeat((quat.shape[0], 1))
        z_vector = to_torch([0., 0., 1.], device=self.device).repeat((quat.shape[0], 1))
        x_vector = to_torch([1., 0., 0.], device=self.device).repeat((quat.shape[0], 1))
        roll_vec = quat_apply(quat, y_vector)  # [0,1,0]
        alpha = torch.atan2(roll_vec[:, 2], roll_vec[:, 1])  # alpha angle = arctan2(z, y)
        pitch_vec = quat_apply(quat, z_vector)  # [0,0,1]
        beta = torch.atan2(pitch_vec[:, 0], pitch_vec[:, 2])  # beta angle = arctan2(x, z)
        yaw_vec = quat_apply(quat, x_vector)  # [1,0,0]
        gamma = torch.atan2(yaw_vec[:, 1], yaw_vec[:, 0])  # gamma angle = arctan2(y, x)

        return torch.stack([alpha, beta, gamma], dim=-1)
        #------------ reward functions----------------
    def _reward_tracking_x_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :1] - self.base_lin_vel[:, :1]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_y_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, 1:2] - self.base_lin_vel[:, 1:2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2]) *  (self.commands[:, 4] >= 0.735)
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):      #todo 修改
        # Penalize non flat base orientation
        #print(self.projected_gravity.shape)
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    # new add
    def _reward_torso_orientation(self):    #todo 修改
        # Penalize non flat torso orientation
        #print(self.torso_projected_gravity)
        return torch.sum(torch.square(self.torso_projected_gravity[:, :2]), dim=1)
    # end new add
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        # print('last_action:',self.last_actions.shape)
        # print('action:',self.actions.shape)
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_tracking_base_height(self):
        base_height_l = self.root_states[:, 2] - self.feet_pos[:, 0, 2]
        base_height_r = self.root_states[:, 2] - self.feet_pos[:, 1, 2]
        base_height = torch.max(base_height_l, base_height_r)
        height_error = torch.abs(base_height - self.commands[:, 4] + self.cfg.asset.ankle_sole_distance)
        return torch.exp(-height_error * 4)
    
    def _reward_deviation_hip_joint(self):
        #print(self.hip_joint_indices)
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.hip_joint_indices], dim=-1) *  (self.commands[:, 4] >= 0.735)
    
    def _reward_deviation_ankle_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.ankle_joint_indices], dim=-1) *  (self.commands[:, 4] >= 0.735)
    
    def _reward_deviation_knee_joint(self):
        height_error = (self.root_states[:, 2] - self.commands[:, 4])
        knee_action_min = self.default_dof_pos[:, self.knee_joint_indices] + self.cfg.control.action_scale * self.action_min[:, self.knee_joint_indices]
        knee_action_max = self.default_dof_pos[:, self.knee_joint_indices] + self.cfg.control.action_scale * self.action_max[:, self.knee_joint_indices]
        joint_deviation = (self.dof_pos[:, self.knee_joint_indices] - knee_action_min) / (knee_action_max - knee_action_min) # always positive
        return torch.sum(torch.abs((joint_deviation-0.5) * height_error.unsqueeze(-1)), dim=-1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0])[:, :self.num_actions].clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1])[:, :self.num_actions].clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * self.first_contacts, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.1 # no reward for zero command
        return rew_airTime
    
    def _reward_feet_clearance(self):
        cur_feetvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        feetvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            feetvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_feetvel_translated[:, i, :])
        feet_height, feet_height_var = self._get_feet_heights()
        height_error = torch.square(feet_height - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        feet_leteral_vel = torch.sqrt(torch.sum(torch.square(feetvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * feet_leteral_vel, dim=1) * (self.commands[:, 4]>=0.71)
    
    def _reward_feet_distance_lateral(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        foot_leteral_dis = torch.abs(footpos_in_body_frame[:, 0, 1] - footpos_in_body_frame[:, 1, 1])
        #return torch.clamp(foot_leteral_dis - self.cfg.rewards.least_feet_distance_lateral, max=0) + torch.clamp(-foot_leteral_dis + self.cfg.rewards.most_feet_distance_lateral, max=0) * (self.commands[:, 4] >= 0.735)
        return torch.clamp(foot_leteral_dis - 0.2, max=0) + torch.clamp(
        -foot_leteral_dis + 0.2, max=0) * (self.commands[:, 4] >= 0.735)

    def _reward_knee_distance_lateral(self):
        cur_knee_pos_translated = self.rigid_body_states[:, self.knee_indices, :3].clone() - self.root_states[:, 0:3].unsqueeze(1)
        knee_pos_in_body_frame = torch.zeros(self.num_envs, len(self.knee_indices), 3, device=self.device)
        for i in range(len(self.knee_indices)):
            knee_pos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_knee_pos_translated[:, i, :])
        knee_lateral_dis = torch.abs(knee_pos_in_body_frame[:, 0, 1] - knee_pos_in_body_frame[:, 2, 1]) + torch.abs(knee_pos_in_body_frame[:, 1, 1] - knee_pos_in_body_frame[:, 3, 1])
        return torch.clamp(knee_lateral_dis - self.cfg.rewards.least_knee_distance_lateral * 2, max=0) + torch.clamp(-knee_lateral_dis + self.cfg.rewards.most_knee_distance_lateral * 2, max=0) * (self.commands[:, 4] >= 0.735)
    
    def _reward_feet_ground_parallel(self):
        feet_heights, feet_heights_var = self._get_feet_heights()
        continue_contact = (self.feet_air_time >= 3* self.dt) * self.contact_filt
        return torch.sum(feet_heights_var * continue_contact, dim=1)
    
    def _reward_feet_parallel(self):
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices[0:3], :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices[0:3], :3].clone()
        feet_distances = torch.norm(left_foot_pos - right_foot_pos, dim=2)
        feet_distances_var = torch.var(feet_distances, dim=1)
        return feet_distances_var * (self.commands[:, 4] >= 0.735)
    
    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
    
    def _reward_joint_power(self):
        #Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1) / torch.clip(torch.sum(torch.square(self.commands[:, 0:2]), dim=-1) + 0.2 * torch.square(self.commands[:, 2]), min=0.1)

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 3 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square((self.torques / self.p_gains.unsqueeze(0))[:, :self.num_lower_dof]), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[:, :self.num_lower_dof]), dim=1)
    
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit)[:, :self.num_lower_dof].clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit)[:, :self.num_lower_dof].clip(min=0.), dim=1)
    
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.5
        single_contact = torch.sum(1.*contacts, dim=1)==1
        rew_no_fly = 1.0 * single_contact
        rew_no_fly = torch.max(rew_no_fly, 1. * (torch.norm(self.commands[:, :3], dim=1) < 0.1)) # full reward for zero command
        return rew_no_fly
    
    def _reward_joint_tracking_error(self):
        return torch.sum(torch.square(self.joint_pos_target[:, :self.num_lower_dof] - self.dof_pos[:, :self.num_lower_dof]), dim=-1)
    
    def _reward_feet_slip(self): 
        # Penalize feet slipping
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        return torch.sum(torch.norm(self.feet_vel[:,:,:2], dim=2) * contact, dim=1)
    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_contact_momentum(self):
        # encourage soft contacts
        feet_contact_momentum_z = torch.clip(self.feet_vel[:, :, 2], max=0) * torch.clip(self.contact_forces[:, self.feet_indices, 2] - 50, min=0)
        return torch.sum(feet_contact_momentum_z, dim=1)
    
    def _reward_action_vanish(self):
        upper_error = torch.clip(self.origin_actions[:, :self.num_lower_dof] - self.action_max[:, :self.num_lower_dof], min=0)
        lower_error = torch.clip(self.action_min[:, :self.num_lower_dof] - self.origin_actions[:, :self.num_lower_dof], min=0)
        return torch.sum(upper_error + lower_error, dim=-1)
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        contacts = torch.sum(self.contact_forces[:, self.feet_indices, 2] < 0.1, dim=-1)
        error_sim = (contacts) * (self.commands[:, 4] >= 0.735)
        return error_sim * (torch.norm(self.commands[:, :3], dim=1) < 0.1)

    #TODO
    #------------ hybrid reward functions----------------
    def _reward_arm_control_limits(self):      #27
        out_of_limits = -(self.plan_actions[:, 0] - self.cfg.hybrid.rewards.limit_body_pitch[0]).clip(max=0.)  # lower limit
        out_of_limits += (self.plan_actions[:, 0] - self.cfg.hybrid.rewards.limit_body_pitch[1]).clip(min=0.)
        out_of_limits += -(self.plan_actions[:, 1] - self.cfg.hybrid.rewards.limit_body_roll[0]).clip(max=0.)  # lower limit
        out_of_limits += (self.plan_actions[:, 1] - self.cfg.hybrid.rewards.limit_body_roll[1]).clip(min=0.)
        out_of_limits += -(self.plan_actions[:, 1] - self.cfg.hybrid.rewards.limit_body_yaw[0]).clip(max=0.)  # lower limit
        out_of_limits += (self.plan_actions[:, 1] - self.cfg.hybrid.rewards.limit_body_yaw[1]).clip(min=0.)
        return out_of_limits

    def _reward_arm_control_smoothness(self): #26

        # Penalize changes in actions
        diff = torch.square(self.plan_actions - self.last_plan_actions)
        diff = diff * (self.last_plan_actions != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_arm_energy(self):       #23
        energy_sum = torch.sum(
            torch.square(self.torques[:, self.num_lower_dof + 1:]*self.dof_vel[:, self.num_lower_dof + 1:])
            , dim=1)
        return energy_sum

    def _reward_arm_dof_vel(self):      #24
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[..., self.num_lower_dof + 1:]), dim=1)

    def _reward_arm_dof_acc(self):      #25
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel)[..., self.num_lower_dof + 1:] / self.dt), dim=1)

    def _reward_arm_action_rate(self):  #30
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions)[..., self.num_lower_dof + 1:], dim=1)

    def _reward_arm_manip_commands_tracking_combine(self):  # 22
        lpy = self.get_lpy_in_base_coord(torch.arange(self.num_envs, device=self.device), effector = 'left_hand')
        lpy_error = torch.sum((torch.abs(lpy - self.commands_arm_obs[:, 0:3])) / self.commands_arm_lpy_range,dim=1)
        lpy2 = self.get_lpy_in_base_coord(torch.arange(self.num_envs, device=self.device), effector='right_hand')
        lpy_error += torch.sum((torch.abs(lpy2 - self.commands_arm_obs[:, 6:9])) / self.commands_arm_lpy_range,dim=1)
        lpy3 = self.get_lpy_in_base_coord(torch.arange(self.num_envs, device=self.device), effector='head')
        lpy_error += torch.sum((torch.abs(lpy3 - self.commands_arm_obs[:, 12:15])) / self.commands_arm_lpy_range,dim=1)

        rpy = self.get_alpha_beta_gamma_in_base_coord(torch.arange(self.num_envs, device=self.device), effector = 'left_hand')
        rpy_error = torch.sum((torch.abs(rpy - self.target_abg[:,0:3])) / self.commands_arm_rpy_range, dim=1)
        rpy2 = self.get_alpha_beta_gamma_in_base_coord(torch.arange(self.num_envs, device=self.device),effector='right_hand')
        rpy_error += torch.sum((torch.abs(rpy2 - self.target_abg[:,3:6])) / self.commands_arm_rpy_range, dim=1)
        rpy3 = self.get_alpha_beta_gamma_in_base_coord(torch.arange(self.num_envs, device=self.device),effector='head')
        rpy_error += torch.sum((torch.abs(rpy3 - self.target_abg[:,6:9])) / self.commands_arm_rpy_range, dim=1)

        return torch.exp(-(self.cfg.hybrid.rewards.manip_weight_lpy * lpy_error + self.cfg.hybrid.rewards.manip_weight_rpy * rpy_error))

    def _reward_vis_manip_commands_tracking_lpy(self):  #不用
        lpy = self.get_lpy_in_base_coord(torch.arange(self.num_envs, device=self.device))
        lpy_error = torch.sum((torch.abs(lpy - self.commands_arm_obs[:, 0:3])) / self.commands_arm_lpy_range, dim=1)
        return torch.exp(-lpy_error)

    def _reward_vis_manip_commands_tracking_rpy(self):  #不用
        rpy = self.get_alpha_beta_gamma_in_base_coord(torch.arange(self.num_envs, device=self.device))
        rpy_error = torch.sum((torch.abs(rpy - self.commands_arm_obs[:, 3:6])) / self.commands_arm_rpy_range, dim=1)
        return torch.exp(-rpy_error)

    def _reward_arm_action_smoothness_1(self):      #29
        # Penalize changes in actions
        diff = torch.square(self.joint_pos_target[:, self.num_lower_dof + 1:] - self.last_joint_pos_target[:, self.num_lower_dof + 1:])
        diff = diff * (self.last_actions[:, self.num_lower_dof + 1:] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_arm_action_smoothness_2(self):      #28
        # Penalize changes in actions
        diff = torch.square(self.joint_pos_target[:, self.num_lower_dof + 1:] - 2 * self.last_joint_pos_target[:, self.num_lower_dof + 1:] + self.last_last_joint_pos_target[:, self.num_lower_dof + 1:])
        diff = diff * (self.last_actions[:, self.num_lower_dof + 1:] != 0)  # ignore first step
        diff = diff * (self.last_last_actions[:, self.num_lower_dof + 1:] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

