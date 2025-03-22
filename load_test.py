import numpy as np
from stable_baselines3 import DDPG

class LoadTest:
    def __init__(self):
        self.previous_simulink_state = np.array((-0.0023, -0.1002, 0.0724,
                                                0,0,0,
                                                0.9393, 1.1161, 0.9981, 1.0051,
                                                0,0,0,0,
                                                0,0,0,0),dtype=np.float32)
    
        self.current_simulink_state = np.array(( -0.0023, -0.1002, 0.0724,
                                                -0.0023, -0.1002, 0.0724,
                                                0.9393, 1.1161, 0.9981, 1.0051,
                                                0.9393, 1.1161, 0.9981, 1.0051,
                                                0,0,0,0),dtype=np.float32)   
    def _get_obs(self):
            self.target_theta = self.current_simulink_state[0:3]
            self.actual_theta = self.current_simulink_state[3:6]
            self.previous_theta = self.previous_simulink_state[3:6]
            self.actual_length = self.current_simulink_state[10:14]
            self.pervious_action = self.previous_simulink_state[14:18]
            # observation: actual_length, previous_action
            # observation: target_theta, err_theta, previous_action
            # observation: sin(actual_theta), cos(actual_theta), sin(err_theta), cos(err_theta)
            self.err_theta = self.target_theta - self.actual_theta
            # self.observation = np.array([self.err_theta])
            self.observation = np.concatenate((self.err_theta, self.pervious_action))
            # print('[_get_obs]-pervious_action',self.pervious_action)

            return { "observation": self.observation,
                    "desired_goal": self.target_theta,
                    "achieved_goal": self.actual_theta}

if __name__ == "__main__":
    LT = LoadTest()
    obs = LT._get_obs()
    day = 1215
    num = 1
    model = DDPG.load(f'./simulation/{day}/model/cable_v1_DDPG_{num}', env=None)
    action, _state = model.predict(obs, deterministic=True)
    # 没有环境情况下可以调用model并输出动作
    print('[load_test]-action',action)