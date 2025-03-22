# 导入matlab.engine，安装方法可以参考官方文档
# https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
import matlab.engine
import torch
import time
import argparse
import numpy as np
import os
# 我用的是pytorch,装一个tensorboardX用于可视化
from tensorboardX import SummaryWriter
# SAC算法在另外的文件里面，代码太长就不放了，这个可以直接github参考下star比较多的即可，都大差不差
# from algo.sac import SAC
# 随机数种子，这个不难搜
# from algo.utils import setup_seed
from stable_baselines3 import DDPG, TD3, SAC, HerReplayBuffer, PPO
from RL_brain import PPO
from gym.utils import seeding
def seed(seed=None):
        seed = seeding.np_random(seed)
        return [seed]

def main(args):
    seed(20)
    # define the directory for model saving
    model_name = 'simu_cable_length'
    logdir = ' ' #自己定义存储用的文件夹，包括存储模型、数据以及tensorboardX查看用的
    if args.tensorboard:
        writer = SummaryWriter(logdir)

    # define the agent
    # state_dim改成了2维的，与simulink保持一直。2维3维应该都可以
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.current_device())
    # device = torch.device('cpu')

    # ----------------------------------------- #
    # 参数设置
    # ----------------------------------------- #

    num_episodes = 100  # 总迭代次数
    gamma = 0.9  # 折扣因子
    actor_lr = 1e-3  # 策略网络的学习率
    critic_lr = 1e-2  # 价值网络的学习率
    n_hiddens = 16  # 隐含层神经元个数
    env_name = 'Pendulum-v0'  # 连续环境
    return_list = []  # 保存每个回合的return
    n_states = 13
    n_actions = 4
    action_bound = 0.01

    agent = PPO(n_states=n_states,  # 状态数3
            n_hiddens=n_hiddens,  # 隐含层数
            n_actions=n_actions,  # 动作数1
            actor_lr=actor_lr,  # 策略网络学习率
            critic_lr=critic_lr,  # 价值网络学习率
            lmbda = 0.95,  # 优势函数的缩放因子
            epochs = 10,  # 一组序列训练的轮次
            eps = 0.2,  # PPO中截断范围的参数
            gamma=gamma,  # 折扣因子
            device = device
            )

    # define the sample time and stop time
    sample_time = 0.015
    stop_time = 2.7
    step_max = int(stop_time / sample_time) + 1

    # define the training parameters
    batch_size = 256
    auto_entropy = True
    max_episodes = 600

    # define the environment
    eng = matlab.engine.start_matlab()
    # 这个pendulum_v3中的v3没啥意义，就是我在建simulink时保存的第三版。前两版有点小问题，但是木有删除。
    env_name = 'computed_torque_rl' 
    eng.load_system(env_name)

    # the training process
    num_training = 0
    for ep in range(max_episodes):
        t1 = time.time()
        # reset the environment
        # python和matlab-simulink的交互语句
        eng.set_param(env_name, 'StopTime', str(stop_time), nargout=0)  # 21 for 20 seconds 先设定总的仿真时间
        eng.set_param(env_name + '/pause_time', 'value', str(sample_time), nargout=0) # 设定初始的pause_time
        eng.set_param(env_name, 'SimulationCommand', 'start', nargout=0) # 开始跑仿真
        pause_time = 0
        # 这些数据是要存的，为了方便把每个step生成的（state,action, reward, next_state, done)放到replay buffer里面
        # 不同于openAI gym中的模型，simulink模型中控制量的反馈没那么快，0.06s时的控制信号action对应的是0.11s的reward和next_state
        # 所以我们需要把每个step的先存下来，等一个episode结束后统一push进buffer里面
        obs_list, action_list, reward_list, done_list = [], [], [], []
        clock_list = []

        # 构造数据集，保存每个回合的状态数据
        transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
        }

        for step in range(step_max):
            # obtain the status of env 先获取状态
            model_status = eng.get_param(env_name, 'SimulationStatus')
            # 这个if建议换成 while True哈，这样子就不用下面那个elif以防万一了。
            if model_status == 'paused':
                # obtain the latest observation
                # 这是和matlab工作区交互数据的过程
                clock = np.array(eng.eval('out.time.Data'))[-1]
                next_state = np.array(eng.eval('out.obs.Data'))[-1]
                reward = np.array(eng.eval('out.reward.Data'))[-1]
                # control_singal = np.array(eng.eval('out.control.Data'))[-1]
                action = agent.take_action(state)
                state = next_state

                act = action * action_bound
                act = np.clip(act, -action_bound, action_bound)

                clock_list.append(clock)

                obs_list.append(obs)
                action_list.append(action)
                reward_list.append(reward)
                done_list.append(0.0)

                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append()


                # 这个就是把pause_time每次都加上0.05s,实现最后的时序控制
                pause_time += sample_time
                # training process
                # 并没有等到simulink模型跑完，等跑完也是可以的，然后换成其他的判断。
                # 或者get_para='stopped’应该也行，我没试，感觉行。
                if (pause_time + 0.5) > stop_time:
                    done_list[-1] = 1.0
                    eng.set_param(env_name, 'SimulationCommand', 'stop', nargout=0)
                    # 这就是上面说的，开始一个个往buffer里面push
                    len_list = len(obs_list)
                    for i1 in range(len_list - 1):
                        obs = obs_list[i1]
                        action = action_list[i1]
                        reward = reward_list[i1 + 1]
                        next_obs = obs_list[i1 + 1]
                        done = done_list[i1 + 1]
                        agent.replay_buffer.push(obs, action, reward, next_obs, done)
                    buffer_length = len(agent.replay_buffer)
                    # 训练过程
                    if buffer_length > batch_size:
                        for _ in range(100):
                            value_loss, q_value_loss1, q_value_loss2, policy_loss = agent.update(batch_size,
                                                                                                reward_scale=0.1,
                                                                                                auto_entropy=True,
                                                                                                target_entropy=-1. * n_actions)
                            if args.tensorboard:
                                writer.add_scalar('Loss/V_loss', value_loss, global_step=num_training)
                                writer.add_scalar('Loss/Q1_loss', q_value_loss1, global_step=num_training)
                                writer.add_scalar('Loss/Q2_loss', q_value_loss2, global_step=num_training)
                                writer.add_scalar('Loss/pi_loss', policy_loss, global_step=num_training)
                                num_training += 1

                    ep_rd = np.sum(reward_list[1:])
                    print('\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'
                          .format(ep, max_episodes, ep_rd, time.time() - t1))

                    if args.tensorboard:
                        writer.add_scalar('Reward/train_rd', ep_rd, global_step=ep)

                    break
                # 这里接的是上面正常的一个step，此时应该还是paused，然后把控制量、pause_time写入，再continue就行了。
                eng.set_param(env_name + '/input', 'value', str(act), nargout=0)  # initial control signal
                eng.set_param(env_name + '/pause_time', 'value', str(pause_time), nargout=0)
                eng.set_param(env_name, 'SimulationCommand', 'continue', nargout=0)

            elif model_status == 'running':
                eng.set_param(env_name, 'SimulationCommand', 'continue', nargout=0)

        if ep % 100 == 0 and ep != 0 and args.save_model:
            model_path_pip_epoch =  ' '# 改一下路径哈
            if not os.path.exists(model_path_pip_epoch):
                os.makedirs(model_path_pip_epoch)
            agent.save_model(model_path_pip_epoch)
            print('=============The model is saved at epoch {}============='.format(ep))
    if args.save_model:
        model_path_final =''# 改一下路径哈
        agent.save_model(model_path_final)
        print('=============The final model is saved!==========')

    eng.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument('--save_model', default=True, action="store_true")
    parser.add_argument('--save_data', default=False, action="store_true")
    args = parser.parse_args()
    main(args)