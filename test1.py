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
        self.actionHigh = np.array([0.003, 0.003, 0.003, 0.003], dtype=np.float32)
        self.thetaHigh = np.array([0.5, 0.1, 0.3], dtype=np.float32)
        self.thetaLow = np.array([-0.4, -1.0, -0.5], dtype=np.float32)
        # observation: actual_length, previous_action
        # observation: actual_theta, err_theta, previous_action
        # observation: target_theta, actual_theta, err_theta
        self.observationHigh = np.array([ 2.0, 2.0, 2.0, 0.003, 0.003, 0.003, 0.003], dtype=np.float32)
        self.observationLow = np.array([ -2.0, -2.0, -2.0, -0.003, -0.003, -0.003, -0.003], dtype=np.float32)
        # self.observationSin = np.ones((12,), dtype=np.float32)
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

        # model
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

        self.distance_threshold = np.array([0.002,0.005,0.005])
        self.deltaTheta = np.array([0.15,0.15,0.35])

        self.figure_path = "D:/heyulong/OneDrive/code/motion_compensation/simulation/1216/picture"
        self.target_theta_plt = np.zeros((3,self.cycle_num),dtype=float)
        self.actual_theta_plt = np.zeros((3,self.cycle_num),dtype=float)

        # init model
        self.eng = matlab.engine.start_matlab()
        self.eng.cd('E:/ANACONDA/envs/py37/Lib/site-packages/gym/envs/classic_control/cable',nargout=0)
        self.model_name = 'computed_torque_rl_theta'
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
    
    # def normalize_state(self,)

    def plot_theta(self):
        # plot result
        colorList = ('r',':r','--r')
        targetLabelList = ('thetaTarget1','thetaTarget2','thetaTarget3')
        actualLabelList = ('thetaActual1','thetaActual2','thetaActual3')
        fig,axes = plt.subplots(3,3,figsize=(9,6),dpi=75,facecolor="w",sharex=True,sharey=False)
        # for i in range(3):
        #     print(colorList[i])

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

        figure_name = "cable_v0_{}".format(str(self.reset_flag))
        if not os.path.exists(self.figure_path):
            os.makedirs(self.figure_path) # 如果不存在目录figure_save_path，则创建
        plt.savefig(os.path.join(self.figure_path, figure_name))#第一个是指存储路径，第二个是图片名字

        # plt.draw()
        # plt.close()
 
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
        
        # reset
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

        self.plot_theta()
        self.target_theta_plt = np.zeros((3,self.cycle_num),dtype=float)
        self.actual_theta_plt = np.zeros((3,self.cycle_num),dtype=float)

        self.reset_flag = self.reset_flag + 1
        
        return observation
    
    def step(self, action):
        # action:[1x4],float
        action = np.clip(action, -self.actionHigh, self.actionHigh)
        # print("[cable]-action:",action)
        self.eng.set_param(self.model_name + '/action', 'value', str(action), nargout=0)
        if self.first_flag:
            # 第一次进入step，start开始，暂停2s
            self.eng.set_param(self.model_name , 'SimulationCommand', 'start', nargout=0)
            time.sleep(2)
            self.first_flag = 0 
        else:
            # 非第一次进入step，continue下一步，暂停0.5s
            self.eng.set_param(self.model_name , 'SimulationCommand', 'continue', nargout=0)
            time.sleep(self.pause_time)
        
        # test
        # if self.step_num % 30 == 0:
        #     print("[cable]-step_num:",self.step_num)

        # 确保定步长的执行
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
        
        # simulink_state[1x18]: target_theta, actual_theta, target_length, actual_length, action
        self.simulation_state = np.array(self.eng.eval("state'")) # [1x18]
        self.next_simulink_state = self.simulation_state[:,-1]
        self.previous_simulink_state = self.current_simulink_state
        self.current_simulink_state = self.next_simulink_state
        self.target_theta_plt[:,self.step_num] = self.current_simulink_state[0:3].reshape((3,))
        self.actual_theta_plt[:,self.step_num] = self.current_simulink_state[3:6].reshape((3,))
        
        # 更新下一步pause_time
        self.pause_time_total = self.pause_time_total + self.step_time
        self.eng.set_param(self.model_name + '/pause_time', 'value', str(self.pause_time_total), nargout=0)
        self.step_num = self.step_num + 1
        
        observation = self._get_obs()
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

        # 阈值奖励
        success_reward = 0
        if not info['is_success']:
            success_reward = -0.1

        reward = self.compute_reward(actual_theta, target_theta, info) + success_reward
 
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
        # 平滑奖励
        delta_action = self.previous_simulink_state[14:18]-self.current_simulink_state[14:18]
        action_reward = -math.sqrt(delta_action[0]**2+delta_action[1]**2+delta_action[2]**2+delta_action[3]**2)
        
        err_theta = abs(desired_goal-achieved_goal)
        previous_theta = self.previous_simulink_state[3:6]
        current_theta = self.current_simulink_state[3:6]
        err_theta_d = abs(previous_theta-current_theta)/self.step_time
        arr_size = err_theta.shape

        stepThetaLow = desired_goal - self.deltaTheta
        stepThetaHigh = desired_goal + self.deltaTheta
        stepThetaLow1 = desired_goal - 0.1
        stepThetaHigh1 = desired_goal + 0.1
        stepThetaLow2 = desired_goal - 0.05
        stepThetaHigh2 = desired_goal + 0.05
        stepThetaLow3 = desired_goal - 0.01
        stepThetaHigh3 = desired_goal + 0.01
        print('[reward]-stepThetaHigh3',stepThetaHigh3)

        area_reward = np.array([0,0,0,0], dtype=np.float32)
        # HER使用
        if max(arr_size) > 4:
            # 误差奖励
            d = 0.1 * (0.5*math.e**err_theta[:,0] + 1*math.e**err_theta[:,1] + 5*math.e**err_theta[:,2])
            err_reward = -d
            # 范围奖励
            for j in range(max(arr_size)):
                for i in range(3):
                    area_flag = np.array([achieved_goal[j,i] <= stepThetaLow[i], achieved_goal[j,i] >= stepThetaHigh[i]])
                    area_flag1 = np.array([achieved_goal[j,i] <= stepThetaHigh1[i], achieved_goal[j,i] >= stepThetaLow1[i]])
                    area_flag2 = np.array([achieved_goal[j,i] <= stepThetaHigh2[i], achieved_goal[j,i] >= stepThetaLow2[i]])
                    area_flag3 = np.array([achieved_goal[j,i] <= stepThetaHigh3[i], achieved_goal[j,i] >= stepThetaLow3[i]])
                    if area_flag.any():
                        area_reward[0] = -10
                    if area_flag1.all():
                        area_reward[1] = -1 
                    if area_flag2.all():
                        area_reward[2] = 0.1
                        if i != 2:
                            area_reward[2] = 0.02    
                    if area_flag3.all():
                        area_reward[3] = 0.5  
                        if i != 2:
                            area_reward[3] = 0.1  
        # 普通使用 
        else:
            # 误差奖励
            d = 0.1 * (0.5*math.e**err_theta[0] + 1*math.e**err_theta[1] + 2*math.e**err_theta[2])
            err_reward = -d
            # 范围奖励
            for i in range(3):      
                area_flag = np.array([achieved_goal[i] <= stepThetaLow[i], achieved_goal[i] >= stepThetaHigh[i]])
                area_flag1 = np.array([achieved_goal[i] <= stepThetaHigh1[i], achieved_goal[i] >= stepThetaLow1[i]])
                area_flag2 = np.array([achieved_goal[i] <= stepThetaHigh2[i], achieved_goal[i] >= stepThetaLow2[i]])
                area_flag3 = np.array([achieved_goal[i] <= stepThetaHigh2[i], achieved_goal[i] >= stepThetaLow2[i]])
                if area_flag.any():
                    area_reward[0] = -10
                if area_flag1.all():
                    area_reward[1] = -1 
                if area_flag2.all():
                    area_reward[2] = 0.1
                    if i != 2:
                        area_reward[2] = 0.02    
                if area_flag3.all():
                    area_reward[3] = 0.5  
                    if i != 2:
                        area_reward[3] = 0.1  

        
        step_reward = 2 * self.step_num / self.cycle_num
        return action_reward + err_reward + step_reward + np.sum(area_reward)
        # return step_reward + area_reward + err_reward

    def render(self, mode="human"):
        # 暂时不做，有时间可以做动态绘图
        pass
 
    def close(self):
        # 关闭动态绘图
        # if self.viewer:
        #     self.viewer.close()
        #     self.viewer = None
        pass
