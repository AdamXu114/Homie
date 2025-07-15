import time

import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import G1RoughCfg
import torch
import yaml
from collections import deque
import onnx
import onnxruntime as ort

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    #cfg = G1RoughCfg()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_dofs = config["num_dofs"]
        num_actor_history = config["num_actor_history"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    hist_obs = deque(maxlen=num_actor_history)
    for _ in range(num_actor_history):
        hist_obs.append(np.zeros(num_obs, dtype=np.float32))  # ✅ 1D


    # Load robot model
    model = onnx.load(
        f"{LEGGED_GYM_ROOT_DIR}/logs/exported/policies/policy.onnx")  # policy_1_21dof_newnoraml.onnx")#policy_1_21dof_testnew.onnx")policy_lab_new_bizhang_earlysmall2
    # print(model)
    onnx.checker.check_model(model)
    policy = ort.InferenceSession(
        f"{LEGGED_GYM_ROOT_DIR}/logs/exported/policies/policy.onnx")
    m = mujoco.MjModel.from_xml_path(xml_path)

    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    # policy = torch.jit.load(policy_path)
    import mujoco_viewer

    viewer = mujoco_viewer.MujocoViewer(m, d)
    flag = False
    if True:
        # with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps,
                             np.zeros_like(kds), d.qvel[6:], kds)
            #tau = np.clip(tau, -200, 200)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.
                t0 = time.time()
                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)


                obs[:3] = cmd * cmd_scale
                obs[3] = 0.75   #height
                obs[4:7] = omega
                obs[7:10] = gravity_orientation
                obs[10: 10 + num_dofs] = qj
                obs[10 + num_dofs: 10 + 2 * num_dofs] = dqj
                obs[10 + 2 * num_dofs: 10 + 2 * num_dofs + num_actions] = action
                #obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
                if counter == 1:
                    for _ in range(num_actor_history):
                        hist_obs.append(obs)
                # print("omega:", omega)
                # print("obs:", obs)
                # print("关节位置:", d.qpos[7:])
                # print("目标位置:", target_dof_pos)
                # print("qvel:",d.qvel[6:])
                # print("重力方向:", gravity_orientation)
                # print("tau:", tau)
                # print("d.ctrl:", d.ctrl)
                # print("obs mean:", np.mean(obs))
                hist_obs.append(obs.copy())
                # 拼接为连续数组
                obs_tensor = np.concatenate(list(hist_obs)).astype(np.float32)[None,:]
                # obs_tensor = torch.from_numpy(obs_tensor).unsqueeze(0)
                # policy inference
                # action = policy(obs_tensor).detach().numpy().squeeze()
                action = policy.run(None, {'input': obs_tensor})[0][0]
                #action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                # print("action:",action)
                # transform action to target_dof_pos

                #TODO:change the action, add zero upper body action
                # target_dof_pos = action * action_scale + default_angles[:num_actions]
                target_dof_pos = np.concatenate([
                    action ,
                    np.zeros(num_dofs - num_actions)
                ])* action_scale + default_angles
                t1 = time.time()
                #print(t1-t0)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.

            # viewer.sync()
            viewer.render()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)