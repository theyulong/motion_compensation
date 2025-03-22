import numpy as np
from stable_baselines3 import DDPG

def get_obs():
    previous_simulink_state = np.array((-0.0023, -0.1002, 0.0724,
                                            0,0,0,
                                            0.9393, 1.1161, 0.9981, 1.0051,
                                            0,0,0,0,
                                            0,0,0,0),dtype=np.float32)

    current_simulink_state = np.array(( -0.0023, -0.1002, 0.0724,
                                            -0.0023, -0.1002, 0.0724,
                                            0.9393, 1.1161, 0.9981, 1.0051,
                                            0.9393, 1.1161, 0.9981, 1.0051,
                                            0,0,0,0),dtype=np.float32)  
    target_theta = current_simulink_state[0:3]
    actual_theta = current_simulink_state[3:6]
    previous_theta = previous_simulink_state[3:6]
    actual_length = current_simulink_state[10:14]
    pervious_action = previous_simulink_state[14:18]
    # observation: actual_length, previous_action
    # observation: target_theta, err_theta, previous_action
    # observation: sin(actual_theta), cos(actual_theta), sin(err_theta), cos(err_theta)
    err_theta = target_theta - actual_theta
    # observation = np.array([err_theta])
    observation = np.concatenate((err_theta, pervious_action))
    # print('[_get_obs]-pervious_action',pervious_action)

    return { "observation": observation,
            "desired_goal": target_theta,
            "achieved_goal": actual_theta}

def get_action():
    obs = get_obs()
    day = 1215
    num = 1
    model = DDPG.load(f'./simulation/{day}/model/cable_v1_DDPG_{num}', env=None)
    action, _state = model.predict(obs, deterministic=True)
    # 没有环境情况下可以调用model并输出动作
    print('[load_test]-action',action)
    return action