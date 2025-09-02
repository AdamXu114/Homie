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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np


class G1RoughCfg(LeggedRobotCfg):
    # 修改
    class hybrid:
        num_actions = 18
        plan_vel = False
        use_vision = False

        # TODO
        class rewards:
            terminal_body_height = 0.75
            use_terminal_body_height = True
            use_terminal_roll = False
            use_terminal_pitch = True
            use_terminal_roll_pitch = False  # TODO
            terminal_body_roll = 0.10
            terminal_body_pitch = 0.2
            terminal_body_pitch_roll = 80. / 180. * np.pi
            headupdown_thres = 0.1
            manip_weight_lpy = 0.75
            manip_weight_rpy = 0.25
            limit_body_roll = [-0.04, 0.04]
            limit_body_pitch = [-0.04, 0.04]
            limit_body_yaw = [-0.04, 0.04]
            orientation_tracking_sigma = 1.0
            manip_tracking_sigma = 0.25
            height_command_limits = [0.65, 0.75]
            # TODO
        class reward_scales:
            # hip_joint_penality = -0.
            # arm_control_limits = -5.
            # arm_control_smoothness = -0.2
            arm_energy = -0.004
            # arm_dof_vel = -1e-2 * 10
            # arm_dof_acc = -2.5e-4 * 10
            # arm_action_rate = -0.01 * 10
            arm_action_smoothness_1 = -0.05 * 5
            # arm_action_smoothness_2 = -0.0


            arm_manip_commands_tracking_combine = 2.
            # vis_manip_commands_tracking_lpy = 0.
            # vis_manip_commands_tracking_rpy = 0.
            # orientation_tracking = -0.005
            # height_command_limits = -0.5
            # height_command_smoothness = -0.25
            # orientation_heuristic = -2.0
            # orientation_control = -10.
            # penalize_yaw = -0.5
            # penalize_pitch = -0.5

            # raibert_heuristic = -0.0



    # TODO
    class arm:
        num_actions_arm = 14  # upper_dof
        arm_num_privileged_obs = 53 + 1  # 暂定 + height
        arm_num_observation_history = 6
        arm_num_observations = 53  # upper_dof14，action18, upper_commands3*6, rpy3, ?height1
        arm_num_obs_history = arm_num_observations * arm_num_observation_history
        arm_num_commands = 3 * 6
        num_actions_arm_cd = 14 + 3 + 1  # upper_dof，rpy，height

        class arm_commands:
            #TODO Torch 使用class_to_dict报错
            angle75 = np.deg2rad(75)
            angle60 = np.deg2rad(60)
            # TODO 待修改
            # l = [0.40, 0.43]        #default 0.413
            # p = [0.38, 0.42]       #经度default0.40
            # y = [0.40, 0.44]             #维度default0.42
            # roll_ee = [-0.03, 0.03]
            # pitch_ee = [-0.03, 0.03]
            # yaw_ee = [-0.03, 0.03]
            l = [0.35, 0.48]        #default 0.413
            p = [0.32, 0.48]       #经度default0.40
            y = [0.35, 0.48]             #维度default0.42
            roll_ee = [-0.1, 0.1]
            pitch_ee = [-0.1, 0.1]
            yaw_ee = [-0.1, 0.1]
            T_traj = [2, 3.]
            T_force_range = [1, 4.]
            add_force_thres = 0.3

        class head_commands:
            angle75 = np.deg2rad(75)
            angle60 = np.deg2rad(60)
            # TODO 待修改
            l = [0.13, 0.17]         #default 0.15
            p = [0.57, 0.61]   #default 0.59
            y = [-0.03, 0.03]        #default 0
            roll_ee = [-0.03, 0.03]
            pitch_ee = [-0.03, 0.03]
            yaw_ee = [-0.03, 0.03]
            T_traj = [2, 3.]
            T_force_range = [1, 4.]
            add_force_thres = 0.3

        class obs_scales:
            l = 1.
            p = 1.
            y = 1.
            wx = 1.
            wy = 1.
            wz = 1.

    class init_state(LeggedRobotCfg.init_state):
        # pos = [0.0, 0.0, 0.85] # x,y,z [m]
        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #     'left_hip_yaw_joint' : 0. ,
        #    'left_hip_roll_joint' : 0,
        #    'left_hip_pitch_joint' : -0.1,
        #    'left_knee_joint' : 0.3,
        #    'left_ankle_pitch_joint' : -0.2,
        #    'left_ankle_roll_joint' : 0,
        #    'right_hip_yaw_joint' : 0.,
        #    'right_hip_roll_joint' : 0,
        #    'right_hip_pitch_joint' : -0.1,
        #    'right_knee_joint' : 0.3,
        #    'right_ankle_pitch_joint': -0.2,
        #    'right_ankle_roll_joint' : 0,
        #     "waist_yaw_joint":0.,
        #     "waist_roll_joint": 0.,
        #     "waist_pitch_joint": 0.,
        #     "left_shoulder_pitch_joint": 0.,
        #     "left_shoulder_roll_joint": 0.,
        #     "left_shoulder_yaw_joint": 0.,
        #     "left_elbow_joint": 0.,
        #     "left_wrist_roll_joint": 0.,
        #     "left_wrist_pitch_joint": 0.,
        #     "left_wrist_yaw_joint": 0.,
        #     "left_hand_index_0_joint": 0.,
        #     "left_hand_index_1_joint": 0.,
        #     "left_hand_middle_0_joint": 0.,
        #     "left_hand_middle_1_joint": 0.,
        #     "left_hand_thumb_0_joint": 0.,
        #     "left_hand_thumb_1_joint": 0.,
        #     "left_hand_thumb_2_joint": 0.,
        #     "right_shoulder_pitch_joint": 0.,
        #     "right_shoulder_roll_joint": -0.,#-0.3
        #     "right_shoulder_yaw_joint": 0.,
        #     "right_elbow_joint": 0.,#0.8
        #     "right_wrist_roll_joint": 0.,
        #     "right_wrist_pitch_joint": 0.,
        #     "right_wrist_yaw_joint": 0.,
        #     "right_hand_index_0_joint": 0.,
        #     "right_hand_index_1_joint": 0.,
        #     "right_hand_middle_0_joint": 0.,
        #     "right_hand_middle_1_joint": 0.,
        #     "right_hand_thumb_0_joint": 0.,
        #     "right_hand_thumb_1_joint": 0.,
        #     "right_hand_thumb_2_joint": 0.,
        #
        # }
        pos = [0.0, 0.0, 0.75]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_yaw_joint': 0.,
            'left_hip_roll_joint': 0,
            'left_hip_pitch_joint': -0.312,
            'left_knee_joint': 0.669,
            'left_ankle_pitch_joint': -0.363,
            'left_ankle_roll_joint': 0,
            'right_hip_yaw_joint': 0.,
            'right_hip_roll_joint': 0,
            'right_hip_pitch_joint': -0.312,
            'right_knee_joint': 0.669,
            'right_ankle_pitch_joint': -0.363,
            'right_ankle_roll_joint': 0,
            "waist_yaw_joint": 0.,
            # "waist_roll_joint": 0.,
            # "waist_pitch_joint": 0.,
            "left_shoulder_pitch_joint": 0.,
            "left_shoulder_roll_joint": 0.,
            "left_shoulder_yaw_joint": 0.,
            "left_elbow_joint": 0.,
            "left_wrist_roll_joint": 0.,
            "left_wrist_pitch_joint": 0.,
            "left_wrist_yaw_joint": 0.,

            # "left_hand_index_0_joint": 0.,
            # "left_hand_index_1_joint": 0.,
            # "left_hand_middle_0_joint": 0.,
            # "left_hand_middle_1_joint": 0.,
            # "left_hand_thumb_0_joint": 0.,
            # "left_hand_thumb_1_joint": 0.,
            # "left_hand_thumb_2_joint": 0.,
            "right_shoulder_pitch_joint": 0.,
            "right_shoulder_roll_joint": -0.,  # -0.3
            "right_shoulder_yaw_joint": 0.,
            "right_elbow_joint": 0.,  # 0.8
            "right_wrist_roll_joint": 0.,
            "right_wrist_pitch_joint": 0.,
            "right_wrist_yaw_joint": 0.,

            # "right_hand_index_0_joint": 0.,
            # "right_hand_index_1_joint": 0.,
            # "right_hand_middle_0_joint": 0.,
            # "right_hand_middle_1_joint": 0.,
            # "right_hand_thumb_0_joint": 0.,
            # "right_hand_thumb_1_joint": 0.,
            # "right_hand_thumb_2_joint": 0.,

        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'M'
        # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,

                     "waist": 300,
                     "shoulder": 200,
                     "wrist": 20,
                     "elbow": 100,
                     "hand": 10

                     }  # [N*m/rad]
        damping = {'hip_yaw': 2,
                   'hip_roll': 2,
                   'hip_pitch': 2,
                   'knee': 4,
                   'ankle': 2,
                   "waist": 5,
                   "shoulder": 4,
                   "wrist": 0.5,
                   "elbow": 1,
                   "hand": 2
                   }  # [N*m/rad]  # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False  # NOTE set True later
        max_curriculum = 1.4
        num_commands = 5  # lin_vel_x, lin_vel_y, ang_vel_yaw, heading, height, orientation
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        heading_to_ang_vel = False

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.8, 1.2]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-0.8, 0.8]  # min max [rad/s]
            heading = [-3.14, 3.14]
            height = [-0.5, 0.0]

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_exhand.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        left_foot_name = "left_foot"
        right_foot_name = "right_foot"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ['torso_link']
        curriculum_joints = []
        left_leg_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint',
                           'left_ankle_pitch_joint']
        right_leg_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint',
                            'right_ankle_pitch_joint']
        left_hip_joints = ['left_hip_roll_joint',
                           "left_hip_yaw_joint"]  # ['left_hip_roll_joint', "left_hip_pitch_joint", "left_hip_yaw_joint"]  change this!!!!
        right_hip_joints = ['right_hip_roll_joint',
                            "right_hip_yaw_joint"]  # ['right_hip_roll_joint', "right_hip_pitch_joint", "right_hip_yaw_joint"]  change this!!!!
        hip_pitch_joints = ['right_hip_pitch_joint', 'left_hip_pitch_joint']
        knee_joints = ['left_knee_joint', 'right_knee_joint']
        ankle_joints = ["left_ankle_roll_joint", "right_ankle_roll_joint"]
        upper_body_link = "torso_link"
        imu_link = "imu_in_pelvis"
        knee_names = ["left_knee_link", "left_hip_yaw_link", "right_knee_link", "right_hip_yaw_link"]
        self_collision = 1
        flip_visual_attachments = False
        ankle_sole_distance = 0.02

    class domain_rand(LeggedRobotCfg.domain_rand):
        use_random = True

        randomize_joint_injection = use_random
        joint_injection_range = [-0.05, 0.05]

        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]

        randomize_payload_mass = use_random
        payload_mass_range = [-5, 10]

        hand_payload_mass_range = [-0.1, 0.3]

        randomize_com_displacement = True  # False change this !!!
        com_displacement_range = [-0.03, 0.03]  # [-0.1, 0.1] change this !!!

        randomize_body_displacement = use_random
        body_displacement_range = [-0.1, 0.1]

        randomize_link_mass = use_random
        link_mass_range = [0.8, 1.2]

        randomize_friction = use_random
        friction_range = [0.1, 3.0]

        randomize_restitution = use_random
        restitution_range = [0.0, 1.0]

        randomize_kp = use_random
        kp_range = [0.9, 1.1]

        randomize_kd = use_random
        kd_range = [0.9, 1.1]

        randomize_initial_joint_pos = use_random
        initial_joint_pos_scale = [0.8, 1.2]
        initial_joint_pos_offset = [-0.1, 0.1]

        push_robots = use_random
        push_interval_s = 4
        upper_interval_s = 1
        max_push_vel_xy = 0.5

        init_upper_ratio = 0.
        delay = use_random

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            tracking_x_vel = 1.5
            tracking_y_vel = 1.
            tracking_ang_vel = 2.
            lin_vel_z = -0.5
            ang_vel_xy = -0.05  # -0.025 change this!!
            orientation = -1.5
            torso_orientation = -2.
            action_rate = -0.01
            tracking_base_height = 2.
            deviation_hip_joint = -0.75  # -0.2 change this!!!
            deviation_ankle_joint = -0.5
            deviation_knee_joint = -0.75
            dof_acc = -2.5e-7
            dof_pos_limits = -2.
            feet_air_time = 0.5
            # feet_clearance = -0.25
            feet_distance_lateral = 0.5
            # knee_distance_lateral = 1.0
            feet_ground_parallel = -2.0
            feet_parallel = -3.0
            smoothness = -0.05
            joint_power = -2e-5
            feet_stumble = -1.5
            torques = -2.5e-6
            dof_vel = -1e-4
            dof_vel_limits = -2e-3
            torque_limits = -0.1
            no_fly = 0.75
            joint_tracking_error = -0.1
            feet_slip = -0.25
            feet_contact_forces = -0.00025
            contact_momentum = 2.5e-4
            action_vanish = -1.0
            stand_still = -0.15

        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.975
        soft_dof_vel_limit = 0.80
        soft_torque_limit = 0.95
        base_height_target = 0.74
        max_contact_force = 400.
        least_feet_distance = 0.2
        least_feet_distance_lateral = 0.18  # 2
        most_feet_distance_lateral = 0.35
        most_knee_distance_lateral = 0.35
        least_knee_distance_lateral = 0.2
        clearance_height_target = 0.14

    class env(LeggedRobotCfg.rewards):
        num_envs = 4096
        num_actions = 12
        num_dofs = 27
        num_one_step_observations = 2 * num_dofs + 10 + num_actions + 4  # 54 + 10 + 12 = 22 + 54 = 76 + 4plan
        num_one_step_privileged_obs = num_one_step_observations + 3
        num_actor_history = 6
        num_critic_history = 1
        num_observations = num_actor_history * num_one_step_observations
        num_privileged_obs = num_critic_history * num_one_step_privileged_obs
        action_curriculum = True
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0., 1., 0., 0., 0.]

    class noise(LeggedRobotCfg.terrain):
        add_noise = True
        noise_level = 1.0

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.02
            dof_vel = 2.0
            lin_vel = 0.1
            ang_vel = 0.5
            gravity = 0.05
            height_measurement = 0.1
            plan_actions = 0.1


class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        use_flip = True
        entropy_coef = 0.01
        symmetry_scale = 1.0
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        # 修改
        arm_policy_class_name = 'ArmActorCritic'
        arm_algorithm_class_name = 'ArmPPO'
        save_interval = 100
        num_steps_per_env = 50
        max_iterations = 10000
        run_name = ''
        experiment_name = 'test'
        wandb_project = "MyHomie"
        logger = "wandb"
        # logger = "tensorboard"
        wandb_user = "xu147266" # enter your own wandb user name here

# load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkptex

