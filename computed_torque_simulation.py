import matlab.engine
import time
import gym
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO

# test matlab engine
# 注意点1：python调用并运行Maltab的m文件，同时直接读取工作区中的参数数据
# 注意点2：Matlab通过代码控制Simulink启停，重置参数，并读取Simlink的仿真结果
mat_eng = matlab.engine.start_matlab()  # 启动Matlab引擎，命名为eng
env_name = 'computed_torque_rl'
mat_eng.load_system(env_name)
mat_eng.model_using(nargout=0)  # 通过eng运行写好的m文件 matlab_init.m”，nargout=0表示不返回输出。
                                # Note: 此m文件需要在ptyhon的启动界面下。
simulation_time = 2.7
step_time = 0.015
cycle_num = int(simulation_time / step_time) + 1
mat_eng.set_param(env_name + '/pause_time', 'value', str(step_time), nargout=0)
mat_eng.set_param(env_name , 'StopTime', str(simulation_time), nargout=0)
mat_eng.set_param(env_name , 'SimulationCommand', 'start', nargout=0)

target_theta = np.zeros((3,cycle_num),dtype=float)
actual_theta = np.zeros((3,cycle_num),dtype=float)

for i in range(cycle_num):
    print("num:",i,"\n")
    mat_eng.set_param(env_name + '/pause_time', 'value', str(step_time + i*step_time), nargout=0)
    mat_eng.set_param(env_name , 'SimulationCommand', 'continue', nargout=0)
    sig_actual_theta = np.array(mat_eng.eval("get(actual_q).Data'")).shape(-1)
    sig_actual_theta = sig_actual_theta[:,-1]
    actual_theta[:,i] = sig_actual_theta

for i in 3:
    plt.plot(actual_theta[i,:])  
plt.show()