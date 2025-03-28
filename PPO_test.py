import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from RL_brain import PPO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(torch.cuda.current_device())
# device = torch.device('cpu')

# ----------------------------------------- #
# 参数设置
# ----------------------------------------- #

num_episodes = 200  # 总迭代次数
gamma = 0.9  # 折扣因子
actor_lr = 1e-4  # 策略网络的学习率
critic_lr = 1e-3  # 价值网络的学习率
n_hiddens = 32  # 隐含层神经元个数
env_name = 'Pendulum-v0'  # 连续环境
return_list = []  # 保存每个回合的return

# ----------------------------------------- #
# 环境加载
# ----------------------------------------- #

env = gym.make(env_name)
n_states = env.observation_space.shape[0]  # 状态数 3
n_actions = env.action_space.shape[0]  # 动作数 1

# ----------------------------------------- #
# 模型构建
# ----------------------------------------- #

agent = PPO(n_states=n_states,  # 状态数3
            n_hiddens=n_hiddens,  # 隐含层数
            n_actions=n_actions,  # 动作数1
            actor_lr=actor_lr,  # 策略网络学习率
            critic_lr=critic_lr,  # 价值网络学习率
            lmbda = 0.95,  # 优势函数的缩放因子
            epochs = 20,  # 一组序列训练的轮次
            eps = 0.2,  # PPO中截断范围的参数
            gamma=gamma,  # 折扣因子
            device = device
            )

# ----------------------------------------- #
# 训练--回合更新 on_policy
# ----------------------------------------- #

for i in range(num_episodes): 
    # print(env.reset())
    state = env.reset()  # 环境重置
    done = False  # 任务完成的标记
    episode_return = 0  # 累计每回合的reward

    # 构造数据集，保存每个回合的状态数据
    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
    }

    while not done:
        action = agent.take_action(state)  # 动作选择
        next_state, reward, done, _  = env.step(action)  # 环境更新
        # 保存每个时刻的状态\动作\...
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        # 更新状态
        state = next_state
        # 累计回合奖励
        episode_return += reward

    # 保存每个回合的return
    return_list.append(episode_return)
    # 模型训练
    agent.update(transition_dict)

    # 打印回合信息
    print(f'iter:{i}, return:{np.mean(return_list[-10:])}')

# -------------------------------------- #
# 绘图
# -------------------------------------- #
for i_episode in range(10):     # cycle times
        obs = env.reset()
        for i in range(50):         # step
            action = agent.take_action(obs)
            obs, reward, done, info = env.step(action)
            env.render('human')
            print(done)
            if done:
                print('Done')
                obs = env.reset()

plt.plot(return_list)
plt.title('return')
plt.show()