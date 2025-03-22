import gym
from gym import spaces
from gym.utils import seeding
import matlab.engine
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import os
from typing import Any, Dict, Union
from panda_gym.utils import distance
 
class CableLengthHerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self):
        # obs and action
        # 14 Observations are dictionaries with the "theta", "pervious_theta", "target_length", "pervious_action"
        self.actionHigh = np.array([0.004, 0.004, 0.004, 0.004], dtype=np.float32)
        self.thetaHigh = np.array([0.5, 0.1, 0.3], dtype=np.float32)
        self.thetaLow = np.array([-0.4, -1.0, -0.5], dtype=np.float32)
        # observation: actual_length, previous_action
        # observation: actual_theta, err_theta, previous_action
        # observation: target_theta, actual_theta, err_theta
        # observation: err_theta, actual_theta, err_theta
        # observation: err_theta, previous_action
        self.observationHigh = np.array([ 2.0, 2.0, 2.0, 0.004, 0.004, 0.004, 0.004], dtype=np.float32)
        self.observationLow = np.array([ -2.0, -2.0, -2.0, -0.004, -0.004, -0.004, -0.004], dtype=np.float32)

        # observation: err_theta, previous_action, desired_goal: target_theta, achieved_goal: actual_theta, 
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=self.observationLow, high=self.observationHigh, shape=(7,), dtype=np.float32),
                "desired_goal": spaces.Box(low=self.thetaLow, high=self.thetaHigh, shape=(3,), dtype=np.float32),
                "achieved_goal": spaces.Box(low=self.thetaLow, high=self.thetaHigh, shape=(3,), dtype=np.float32),
            }
        )
        # 4 actions, corresponding to "length1", "length2", "length3", "length4"
        self.action_space = spaces.Box(
            low=-self.actionHigh, high=self.actionHigh, shape=(4,), dtype=np.float32
        )

        # simulation model param
        self.simulation_time = 2.7
        self.step_time = 0.015
        self.cycle_num = int(self.simulation_time / self.step_time) + 1

        self.pause_time_total = 0
        self.pause_time = 0.5

        self.pasue_flag = 0
        self.repeat_flag = 0
        self.first_flag = 1
        self.step_num = 0
        self.end_flag = 1
        self.reset_flag = 0

        # is_success threshold: distance_threshold and simulation_stop threshold: deltaTheta
        self.distance_threshold = np.array([0.005,0.015,0.04])
        self.deltaTheta = np.array([0.15,0.15,0.35])

        # save theta and action as picture
        self.day = 2101
        self.plt_theta_action = 1 # theta 和 action的txt文件
        self.plt_reward = 0 # reward的txt文件
        self.plt_area_reward = 0 # area_reward的txt文件
        self.figure_path_theta = f"D:/heyulong/OneDrive/code/motion_compensation/simulation/{self.day}/picture/theta"
        self.figure_path_action = f"D:/heyulong/OneDrive/code/motion_compensation/simulation/{self.day}/picture/action"
        self.txt_action_path = f"D:/heyulong/OneDrive/code/motion_compensation/simulation/{self.day}/txt/action"
        self.txt_theta_path = f"D:/heyulong/OneDrive/code/motion_compensation/simulation/{self.day}/txt/theta"
        self.txt_reward_path = f"D:/heyulong/OneDrive/code/motion_compensation/simulation/{self.day}/txt/reward"
        self.txt_all_reward_path = f"D:/heyulong/OneDrive/code/motion_compensation/simulation/{self.day}/txt/reward"
        self.target_theta_plt = np.zeros((3,self.cycle_num),dtype=float)
        self.actual_theta_plt = np.zeros((3,self.cycle_num),dtype=float)
        self.action_plt = np.zeros((4,self.cycle_num),dtype=float)

        # init simulation model
        self.eng = matlab.engine.start_matlab()
        self.eng.cd('E:/ANACONDA/envs/py37-vs/Lib/site-packages/gym/envs/classic_control/cable',nargout=0)
        self.model_name = 'computed_torque_rl_Theta_length'
        self.eng.load_system(self.model_name)
        self.eng.simulink_using_theta(nargout=0)
        self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        self.eng.set_param(self.model_name + '/pause_time', 'value', str(self.step_time), nargout=0)
        self.eng.set_param(self.model_name, 'StopTime', str(self.simulation_time), nargout=0)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
 
    def _get_obs(self):
        # 归一化，放缩至同一数量级
        target_theta = self.current_simulink_state[0:3]
        target_theta_max = np.array([0.3679,-0.0449,0.1303],dtype=np.float32)
        target_theta_min = np.array([-0.2340,-0.9024,-0.3378],dtype=np.float32)
        self.target_theta = 2*(target_theta-target_theta_min)/(target_theta_max-target_theta_min) - 1
        actual_theta = self.current_simulink_state[3:6]
        self.actual_theta = 2*(actual_theta-target_theta_min)/(target_theta_max-target_theta_min) - 1

        pervious_action = self.previous_simulink_state[14:18]
        self.pervious_action = pervious_action*250
        # observation: err_theta, previous_action
        err_theta = target_theta - actual_theta
        # success的阈值
        err_theta_max = np.array([0.01, 0.03, 0.15])
        err_theta_min = np.array([-0.01, -0.03, -0.15])
        self.err_theta = 2*(err_theta-err_theta_min)/(err_theta_max-err_theta_min) - 1
        self.observation = np.concatenate((self.err_theta, self.pervious_action))

        return { "observation": self.observation,
                 "desired_goal": self.target_theta,
                 "achieved_goal": self.actual_theta,}
 
    def _get_info(self):
        # no return
        # info：target_theta, actual_theta, err_theta
        self.target_theta = self.current_simulink_state[0:3]
        self.actual_theta = self.current_simulink_state[3:6]
        # print("[cable]-target_theta",self.target_theta)
        # print("[cable]-actual_theta",self.actual_theta)
        self.err_theta = self.target_theta - self.actual_theta
        return 

    def plot_theta(self):
        # plot result
        colorList = ('r',':r','--r')
        targetLabelList = ('thetaTarget1','thetaTarget2','thetaTarget3')
        actualLabelList = ('thetaActual1','thetaActual2','thetaActual3')
        fig,axes = plt.subplots(3,3,figsize=(9,6),dpi=75,facecolor="w",sharex=True,sharey=False)

        plt.subplot2grid((3,3),(0,0),colspan=2,rowspan=3)   # 总大小，起始点，列，行
        for i in range(3):
            plt.plot(self.target_theta_plt[i,:],'k',lw='1.5',label=targetLabelList[i])  
            plt.plot(self.actual_theta_plt[i,:],colorList[i],lw='1.5',label=actualLabelList[i])
        plt.xlim(0,181)
        plt.ylim(-1,0.4) 

        plt.legend(frameon=False)
        plt.xlabel('cycle(%)')
        plt.ylabel('theta(rad)')

        # ylimList = np.array([[0,5e-3],[-4e-2,0],[0,0.2]])
        err_theta = self.target_theta_plt-self.actual_theta_plt
        maxErr = np.zeros(3,dtype=float)
        minErr = np.zeros(3,dtype=float)
        for i in range(3):
            maxErr[i] = max(err_theta[i,:])
            minErr[i] = min(err_theta[i,:])
        for i in range(3):
            plt.subplot(3,3,3*(i+1))
            plt.plot(err_theta[i,:],colorList[i],lw='1.5',label='thetaErr')    
            plt.xlabel('cycle(%)')
            plt.ylabel('theta(rad)')  
            plt.legend(frameon=False)
            plt.ylim([minErr[i],maxErr[i]]) 
        plt.xlim(0,181)
        
        # 保存theta图片
        figure_name = "cable_{}".format(str(self.reset_flag))
        if not os.path.exists(self.figure_path_theta):
            os.makedirs(self.figure_path_theta) # 如果不存在目录figure_save_path，则创建
        plt.savefig(os.path.join(self.figure_path_theta, figure_name))#第一个是指存储路径，第二个是图片名字
        # plt.draw()
        plt.close('all')

        if self.plt_theta_action:
            # 保存theta-txt文件
            if not os.path.exists(self.txt_theta_path):
                os.makedirs(self.txt_theta_path)
            txt_name = f'theta_{str(self.reset_flag)}.txt'
            file_path = os.path.join(self.txt_theta_path, txt_name)
            # 数据一次性写入
            txt_data = np.vstack([np.arange(181),
                                self.target_theta_plt[0, :], self.target_theta_plt[1, :], self.target_theta_plt[2, :], 
                                self.actual_theta_plt[0, :], self.actual_theta_plt[1, :], self.actual_theta_plt[2, :] ]).T
            # 格式化数据并写入文件
            np.savetxt(file_path, txt_data, fmt='%d %.6f %.6f %.6f %.6f %.6f %.6f',
                    header='step target_theta_1 target_theta_2 target_theta_3 actual_theta_1 actual_theta_2 actual_theta_3',
                    comments='', delimiter=' ')

    def plot_action(self):
        # plot result
        colorList = ('-r',':r','-.r','--r')
        actionLabelList = ('action1','action2','action3','action4')
        fig,axes = plt.subplots(4,4,figsize=(9,6),dpi=75,facecolor="w",sharex=True,sharey=False)

        plt.subplot2grid((4,4),(0,0),colspan=3,rowspan=4)   # 总大小，起始点，列，行
        for i in range(4):
            plt.plot(self.action_plt[i,:],colorList[i],lw='1.5',label=actionLabelList[i])
        plt.xlim(0,181)
        plt.ylim([-self.actionHigh[0],self.actionHigh[0]]) 

        plt.legend(frameon=False)
        plt.xlabel('cycle(%)')
        plt.ylabel('action(m)')

        err_action = self.action_plt
        for i in range(4):
            plt.subplot(4,4,4*(i+1))
            plt.plot(err_action[i,:],colorList[i],lw='1.5',label='action')    
            plt.xlabel('cycle(%)')
            plt.ylabel('action(m)')  
            plt.legend(frameon=False)
            plt.ylim([-self.actionHigh[i],self.actionHigh[i]]) 
            # plt.ylim([-0.005,0.005]) 
        plt.xlim(0,181)

        # 保存action图片
        figure_name = "action_{}".format(str(self.reset_flag))
        if not os.path.exists(self.figure_path_action):
            os.makedirs(self.figure_path_action) # 如果不存在目录figure_save_path，则创建
        plt.savefig(os.path.join(self.figure_path_action, figure_name))#第一个是指存储路径，第二个是图片名字
        # plt.draw()
        plt.close('all')

        if self.plt_theta_action:
            # 保存action-txt文件
            if not os.path.exists(self.txt_action_path):
                os.makedirs(self.txt_action_path)
            txt_name = f'action_{str(self.reset_flag)}.txt'
            file_path = os.path.join(self.txt_action_path, txt_name)
            # 数据一次性写入
            txt_data = np.vstack([np.arange(181),
                                self.action_plt[0, :], self.action_plt[1, :], self.action_plt[2, :], self.action_plt[3, :],]).T
            # 格式化数据并写入文件
            np.savetxt(file_path, txt_data, fmt='%d %.6f %.6f %.6f %.6f',
                    header='step action_1 action_2 action_3 action_4',
                    comments='', delimiter=' ')
 
    def reset(self, seed=None, options=False):
        # We need the following line to seed self.np_random
        # super().reset()
        print("[cable]-reset()")
        
        self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        self.eng.set_param(self.model_name + '/pause_time', 'value', str(self.step_time), nargout=0)
        self.eng.set_param(self.model_name, 'StopTime', str(self.simulation_time), nargout=0)
        self.pause_time_total = self.step_time
        self.pasue_flag = 0
        self.repeat_flag = 0
        self.first_flag = 1
        self.step_num = 0
        
        # reset state[1x18]: target_theta, actual_theta, target_length, actual_length, action
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
        observation = self._get_obs()

        if self.reset_flag:
            self.plot_theta()
            self.plot_action()
        self.target_theta_plt = np.zeros((3,self.cycle_num),dtype=float)
        self.actual_theta_plt = np.zeros((3,self.cycle_num),dtype=float)
        self.action_plt = np.zeros((4,self.cycle_num),dtype=float)

        self.reset_flag = self.reset_flag + 1
        
        return observation
    
    def step(self, action):
        # k_out*math.e**(k_in*err_theta[:,2])导致奖励惩罚极大，使得模型训练不正确，输出action=nan，此处做筛选
        action[np.isnan(action)] = 0
        action = np.clip(action, -self.actionHigh, self.actionHigh)

        # print("[cable]-action:",action)
        # 【set new action】
        self.eng.set_param(self.model_name + '/action', 'value', str(action), nargout=0)
        # 【start simulation】
        if self.first_flag:
            # 第一次进入step，start开始，暂停2s
            self.eng.set_param(self.model_name , 'SimulationCommand', 'start', nargout=0)
            time.sleep(2)
            self.first_flag = 0 
        else:
            # 非第一次进入step，continue下一步，暂停0.5s
            self.eng.set_param(self.model_name , 'SimulationCommand', 'continue', nargout=0)
            time.sleep(self.pause_time)
        
        # 【确保定步长的执行】
        while True:
            self.repeat_flag = self.repeat_flag + 1
            self.pasue_flag = self.eng.eval('pause_flag')
            if int(self.pasue_flag):
                self.pasue_flag = 0
                self.repeat_flag = 0
                done = 0
                break
            if self.repeat_flag > 3:
                time.sleep(self.pause_time)
        # 【end simulation】
        # 【save state and picture】
        # simulink_state[1x18]: target_theta, actual_theta, target_length, actual_length, action
        self.simulation_state = np.array(self.eng.eval("state'")) # [1x18]
        self.next_simulink_state = self.simulation_state[:,-1]
        self.previous_simulink_state = self.current_simulink_state
        self.current_simulink_state = self.next_simulink_state
        self.target_theta_plt[:,self.step_num] = self.current_simulink_state[0:3].reshape((3,))
        self.actual_theta_plt[:,self.step_num] = self.current_simulink_state[3:6].reshape((3,))
        self.action_plt[:,self.step_num] = action.reshape((4,))
        # print('[step]-step_num',self.step_num)
        # print('[step]-target_theta_plt',self.target_theta_plt[:,self.step_num])
        # print('[step]-actual_theta_plt',self.actual_theta_plt[:,self.step_num])
        # print('[step]-action_plt',self.action_plt[:,self.step_num])
        
        # 【更新下一步pause_time】
        self.pause_time_total = self.pause_time_total + self.step_time
        self.eng.set_param(self.model_name + '/pause_time', 'value', str(self.pause_time_total), nargout=0)
        self.step_num = self.step_num + 1
        
        # 【更新obs】
        observation = self._get_obs()
        # print('[step]-observation',observation)
        info = self._get_info()
        
        # 【判断出界】超出边界直接done=1
        actual_theta = self.current_simulink_state[3:6]
        target_theta = self.current_simulink_state[0:3]
        
        stepThetaLow = target_theta - self.deltaTheta
        stepThetaHigh = target_theta + self.deltaTheta
        for i in range(3):
            area_flag = np.array([actual_theta[i] >= stepThetaHigh[i], actual_theta[i] <= stepThetaLow[i]])
            if area_flag.any():
                print(f"[cable]-setp[{self.step_num}]-actual_theta[{i}] over limt area:{actual_theta[i]}")
                print(f"[cable]-stepThetaLow-stepThetaHigh:{stepThetaLow[i]}-{stepThetaHigh[i]}")
                done = 1

        # 【判断小于阈值】更新info
        info = {"is_success": self.is_success(actual_theta, target_theta)}

        reward = self.compute_reward(actual_theta, target_theta, info)
 
        return observation, reward, done, info
    
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray):
        d_is = np.array([0,0,0],dtype=np.float64)
        is_flag = np.array([0,0,0],dtype=np.float64)
        for i in range(3):
            d_is[i] = abs(achieved_goal[i]-desired_goal[i])

        is_flag = np.array([d_is[0] <= self.distance_threshold[0],
                            d_is[1] <= self.distance_threshold[1],
                            d_is[2] <= self.distance_threshold[2]])
        # 全部都小于阈值视为成功，后期可动态变化阈值
        return np.array(is_flag.all())
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        # 【平滑奖励】
        delta_action = (self.previous_simulink_state[14:18]-self.current_simulink_state[14:18])/self.step_time
        action_reward = -3.5*math.sqrt(delta_action[0]**2+delta_action[1]**2+delta_action[2]**2+delta_action[3]**2)
        
        err_theta = abs(desired_goal-achieved_goal)

        previous_actual_theta = self.previous_simulink_state[3:6]
        previous_target_theta = self.previous_simulink_state[0:3]
        previous_err_theta = abs(previous_target_theta-previous_actual_theta)
        current_actual_theta = self.current_simulink_state[3:6]
        current_target_theta = self.current_simulink_state[0:3]
        current_err_theta = abs(current_target_theta-current_actual_theta) 
        delta_err_theta = abs(previous_err_theta - current_err_theta)/self.step_time

        arr_size = err_theta.shape
        # print('[reward]-arr_size,',arr_size)

        # np.array([0.15,0.15,0.35])
        stepThetaHigh0 = self.deltaTheta
        stepThetaHigh1 = np.array([0.05, 0.05, 0.1])
        stepThetaHigh2 = np.array([0.015, 0.03, 0.05])
        stepThetaHigh3 = np.array([0.005, 0.01, 0.02])

        stepThetaDHigh0 = np.array([0.27, 1, 2])
        stepThetaDHigh1 = np.array([0.05, 0.3, 1])
        stepThetaDHigh2 = np.array([0.02, 0.1, 0.35])
        stepThetaDHigh3 = np.array([0.01, 0.01, 0.04])

        area_reward = np.array([0,0,0,0], dtype=np.float32)
        area_d_reward = np.array([0,0,0,0], dtype=np.float32)
        err_theta_d_reward = 0
        k_out = 30
        k_biss = 0.5
        k_in = 40
        # HER使用
        if max(arr_size) > 4:
            # 【误差奖励（err_reward）】
            # err_theta_d_reward = -0.1 * (0.1*math.e**delta_err_theta[:,0] + 0.5*math.e**delta_err_theta[:,1] + 10*math.e**delta_err_theta[:,2])
            err_reward = -0.1 * (0.1*math.e**err_theta[:,0] + 0.5*math.e**err_theta[:,1] + k_out/(k_biss+np.exp(-k_in*delta_err_theta[2])))
            # 【误差范围奖励（err_theta）】
            for j in range(max(arr_size)):
                area_flag0 = np.array([err_theta[j,0] >= stepThetaHigh0[0],err_theta[j,1] >= stepThetaHigh0[1],err_theta[j,2] >= stepThetaHigh0[2]])
                area_flag1 = np.array([err_theta[j,0] <= stepThetaHigh1[0],err_theta[j,1] <= stepThetaHigh1[1],err_theta[j,2] <= stepThetaHigh1[2]])
                area_flag2 = np.array([err_theta[j,0] <= stepThetaHigh2[0],err_theta[j,1] <= stepThetaHigh2[1],err_theta[j,2] <= stepThetaHigh2[2]])
                area_flag3 = np.array([err_theta[j,0] <= stepThetaHigh3[0],err_theta[j,1] <= stepThetaHigh3[1],err_theta[j,2] <= stepThetaHigh3[2]])
                if area_flag0.any():
                    area_reward[0] = area_reward[0] - 2
                else:                                                   # < area_flag0
                    if area_flag1.all():                                # < area_flag1
                        area_reward[1] = area_reward[1] - 0.2 
                        if area_flag2.all():                            # < area_flag2
                            area_reward[2] = area_reward[2] + 0.1 
                            if area_flag3.all():                        # < area_flag3
                                area_reward[3] = area_reward[3] + 0.5 
                            else:                                       # < area_flag3
                                area_reward[3] = area_reward[3] - 0.1 
                        else:                                           # > area_flag2
                            area_reward[2] = area_reward[2] - 0.2
                    else:                                               # > area_flag1
                        area_reward[1] = area_reward[1] - 0.5
                if not os.path.exists(self.txt_reward_path):
                    os.makedirs(self.txt_reward_path)
                txt_name = f'area_rewaed_her_{str(self.reset_flag)}.txt'
                file_path = os.path.join(self.txt_reward_path, txt_name)
                # 数据一次性写入
                txt_data = []
                txt_data.append(f'{j} {area_reward[0]} {area_reward[1]} {area_reward[2]} {area_reward[3]}\n')
            if self.plt_area_reward:
                with open(file_path, 'w') as file:
                    file.writelines(txt_data)
                area_reward = 0.5*area_reward
                with open(file_path, 'a') as file:
                    file.write(f'{j+1} {area_reward[0]} {area_reward[1]} {area_reward[2]} {area_reward[3]}\n')

                # area_d_flag0 = np.array([delta_err_theta[j,0] >= stepThetaDHigh0[0],
                #                          delta_err_theta[j,1] >= stepThetaDHigh0[1],delta_err_theta[j,2] >= stepThetaDHigh0[2]])
                # area_d_flag1 = np.array([delta_err_theta[j,0] <= stepThetaDHigh1[0],
                #                          delta_err_theta[j,1] <= stepThetaDHigh1[1],delta_err_theta[j,2] <= stepThetaDHigh1[2]])
                # area_d_flag2 = np.array([delta_err_theta[j,0] <= stepThetaDHigh2[0],
                #                          delta_err_theta[j,1] <= stepThetaDHigh2[1],delta_err_theta[j,2] <= stepThetaDHigh2[2]])
                # area_d_flag3 = np.array([delta_err_theta[j,0] <= stepThetaDHigh3[0],
                #                          delta_err_theta[j,1] <= stepThetaDHigh3[1],delta_err_theta[j,2] <= stepThetaDHigh3[2]])
                # if area_d_flag0.any():
                #     area_d_reward[0] = area_d_reward[0]-10
                # if area_d_flag1.all():
                #     area_d_reward[1] = area_d_reward[1]-1 
                # if area_d_flag2.all():
                #     area_d_reward[2] = area_d_reward[2]+0.1  
                # if area_d_flag3.all():
                #     area_d_reward[3] = area_d_reward[3]+0.5      
        # 普通使用 
        else:
            # 【误差速度（err_theta_d_reward）和误差（err_reward）奖励】
            err_theta_d_reward = -0.1 * (0.1*math.e**delta_err_theta[0] + 0.5*math.e**delta_err_theta[1] + k_out/(k_biss+np.exp(-k_in*delta_err_theta[2])))
            err_reward = -0.1 * (0.1*math.e**err_theta[0] + 0.5*math.e**err_theta[1] + k_out/(k_biss+np.exp(-k_in*delta_err_theta[2])))
            # 【误差范围奖励（err_theta）】    
            area_flag0 = np.array([err_theta[0] >= stepThetaHigh0[0],err_theta[1] >= stepThetaHigh0[1],err_theta[2] >= stepThetaHigh0[2]])
            area_flag1 = np.array([err_theta[0] <= stepThetaHigh1[0],err_theta[1] <= stepThetaHigh1[1],err_theta[2] <= stepThetaHigh1[2]])
            area_flag2 = np.array([err_theta[0] <= stepThetaHigh2[0],err_theta[1] <= stepThetaHigh2[1],err_theta[2] <= stepThetaHigh2[2]])
            area_flag3 = np.array([err_theta[0] <= stepThetaHigh3[0],err_theta[1] <= stepThetaHigh3[1],err_theta[2] <= stepThetaHigh3[2]])
            # print('[reward]-area_flag',area_flag)
            if area_flag0.any():
                    area_reward[0] = - 2
            else:                                                   # < area_flag0
                if area_flag1.all():                                # < area_flag1
                    area_reward[1] = - 0.2 
                    if area_flag2.all():                            # < area_flag2
                        area_reward[2] = + 0.1 
                        if area_flag3.all():                        # < area_flag3
                            area_reward[3] = + 0.5 
                        else:                                       # < area_flag3
                            area_reward[3] = - 0.1 
                    else:                                           # > area_flag2
                        area_reward[2] = - 0.2
                else:                                               # > area_flag1
                    area_reward[1] = - 0.5 

            # 【误差速度范围奖励（delta_err_theta）】
            area_d_flag0 = np.array([delta_err_theta[0] >= stepThetaDHigh0[0],
                                     delta_err_theta[1] >= stepThetaDHigh0[1],delta_err_theta[2] >= stepThetaDHigh0[2]])
            area_d_flag1 = np.array([delta_err_theta[0] <= stepThetaDHigh1[0],
                                     delta_err_theta[1] <= stepThetaDHigh1[1],delta_err_theta[2] <= stepThetaDHigh1[2]])
            area_d_flag2 = np.array([delta_err_theta[0] <= stepThetaDHigh2[0],
                                     delta_err_theta[1] <= stepThetaDHigh2[1],delta_err_theta[2] <= stepThetaDHigh2[2]])
            area_d_flag3 = np.array([delta_err_theta[0] <= stepThetaDHigh3[0],
                                     delta_err_theta[1] <= stepThetaDHigh3[1],delta_err_theta[2] <= stepThetaDHigh3[2]])
            if area_d_flag0.any():
                    area_d_reward[0] = - 2
            else:                                                   # < area_flag0
                if area_d_flag1.all():                              # < area_flag1
                    area_d_reward[1] = - 0.2 
                    if area_d_flag2.all():                          # < area_flag2
                        area_d_reward[2] = + 0.1 
                        if area_d_flag3.all():                      # < area_flag3
                            area_d_reward[3] = + 0.5 
                        else:                                       # < area_flag3
                            area_d_reward[3] = - 0.1 
                    else:                                           # > area_flag2
                        area_d_reward[2] = - 0.2
                else:                                               # > area_flag1
                    area_d_reward[1] = - 0.5 
            if self.plt_area_reward:
                if not os.path.exists(self.txt_reward_path):
                    os.makedirs(self.txt_reward_path)
                txt_name = f'area_rewaed_none_{str(self.reset_flag)}.txt'
                file_path = os.path.join(self.txt_reward_path, txt_name)
                # 数据一次性写入
                txt_data = []
                txt_data.append(f'{self.step_num} {area_reward[0]} {area_reward[1]} {area_reward[2]} {area_reward[3]} '
                                f'{area_d_reward[0]} {area_d_reward[1]} {area_d_reward[2]} {area_d_reward[3]}\n')
                with open(file_path, 'a') as file:
                    file.writelines(txt_data)
            
        # 【仿真步数奖励（step_reward）】
        step_reward = 2 * self.step_num / self.cycle_num

        # 【成功阈值奖励】
        success_reward = 0
        if info['is_success']:
            success_reward = 1

        all_reward = action_reward + err_reward + step_reward + np.sum(area_reward) + np.sum(area_d_reward) + err_theta_d_reward + success_reward
        if self.plt_reward:
            if not os.path.exists(self.txt_all_reward_path):
                os.makedirs(self.txt_all_reward_path)
            txt_name = 'reward.txt'
            file_path = os.path.join(self.txt_all_reward_path, txt_name)
            # 数据一次性写入
            reward_data = []
            reward_data.append(f'{self.step_num + self.cycle_num*(self.reset_flag-1)} {all_reward}\n')
            with open(file_path, 'a') as file:
                file.writelines(reward_data)
        
        return all_reward

    def render(self, mode="human"):
        # 暂时不做，有时间可以做动态绘图
        # if self.viewer is None:
        #     from gym.envs.classic_control import rendering

        #     self.viewer = rendering.Viewer(500, 500)
        #     self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
        #     rod = rendering.make_capsule(1, 0.2)
        #     rod.set_color(0.8, 0.3, 0.3)
        #     self.pole_transform = rendering.Transform()
        #     rod.add_attr(self.pole_transform)
        #     self.viewer.add_geom(rod)
        #     axle = rendering.make_circle(0.05)
        #     axle.set_color(0, 0, 0)
        #     self.viewer.add_geom(axle)
        #     fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        #     self.img = rendering.Image(fname, 1.0, 1.0)
        #     self.imgtrans = rendering.Transform()
        #     self.img.add_attr(self.imgtrans)

        # self.viewer.add_onetime(self.img)
        # self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        # if self.last_u:
        #     self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        # return self.viewer.render(return_rgb_array=mode == "rgb_array")
        pass

    def close(self):
        # 关闭动态绘图
        if self.viewer:
            self.viewer.close()
            self.viewer = None

