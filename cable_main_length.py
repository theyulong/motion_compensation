import gym
from stable_baselines3 import DDPG, TD3, SAC, HerReplayBuffer, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import os

# NOTE
# 1、cmd:tensorboard --logdir ./panda_reach_v2_tensorboard/
#    web:http://localhost:6006/
# 2、policy_kwargs = dict(activation_fn=th.nn.ReLU,
#                  net_arch=dict(pi=[128, 32], vf=[256, 128, 64, 32]))
#    没有办法使用层字典为 DDPG 的策略和评论网络指定不同的架构，建议使用TD3，TD3也可以用HER
#    使用policy='MultiInputPolicy'，多种不同输入，是为了使用HER经验回放
# 3、多个输入转换为由net_arch网络处理的单个向量。
# 4、total_timesteps使用整数

class CableMain:
    def __init__(self, simulation_day, log_name):
        self.env = gym.make("Cable-v1") # panda_gym step渲染 render=True
        self.simulation_day = simulation_day
        self.log_dir = f'./simulation_length/{self.simulation_day}/log/'
        self.models_dir = f'./simulation_length/{self.simulation_day}/model'
        self.simulation_num = 100
        self.tb_log_name = log_name
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
    # 1、TD3
    def cable_TD3_mult(self):
        # policy_kwargs = dict(vf=[128, 64, 32], pi=[128, 64, 32])
        policy_kwargs = dict(net_arch=[256, 256, 128])
        model = TD3('MultiInputPolicy', 
                    self.env, buffer_size=int(1e6),tau=0.0015,learning_starts=181*10,
                    learning_rate=1e-5, batch_size=1024, gamma= 0.99, train_freq=(int(181/4),"step"),
                    # replay_buffer_class=HerReplayBuffer, 
                    policy_kwargs=policy_kwargs, #replay_buffer_kwargs=replay_buffer_kwargs,
                    verbose=1, tensorboard_log=self.log_dir)
        
        for i in range(self.simulation_num):
            model.learn(total_timesteps=181*10*(i+1), reset_num_timesteps=False, tb_log_name=self.tb_log_name)
            model.save(f"{self.models_dir}/cable_v0_TD3_{i+1}")
            print('simulation_num:',i)

    def cable_TD3_load(self, day, num):
        model = TD3.load(f'./simulation_length/{day}/model/cable_v0_TD3_{num}', env=self.env)
        for i_episode in range(3):     # cycle times
            obs = self.env.reset()
            print(f"[load]-i_episode:{i_episode}")
            for i_step in range(181):         # step
                # if i_step%30 == 0:
                print(f"[load]-i_step:{i_step}")
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)

    def cable_TD3_again(self, day, num):
        model = TD3.load('./simulation_length/{day}/model/cable_v0_TD3_{num}', env=self.env)
        model.set_env(self.env, force_reset=True)
        # 回调函数，定期保存模型
        checkpoint_callback = CheckpointCallback(save_freq=100, save_path="./simulation/1213/log/", name_prefix="rl_model")
        model.learn(total_timesteps=5000, log_interval=10, callback=checkpoint_callback)

    # 2、DDPG
    def cable_DDPG_mult(self):
        # policy_kwargs = dict(net_arch=[256, 256])
        # 12 200 16
        policy_kwargs = dict(net_arch=[256, 256])
        # replay_buffer_kwargs=dict(
        # n_sampled_goal=5,
        # goal_selection_strategy="future",
        # # IMPORTANT: because the env is not wrapped with a TimeLimit wrapper
        # # we have to manually specify the max number of steps per episode
        # max_episode_length=100,
        # online_sampling=True,
        # )
        n_actions = self.env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
        model = DDPG('MultiInputPolicy', 
                    self.env, buffer_size=int(1e6),action_noise=action_noise,
                    learning_rate=1e-5, batch_size=256, gamma= 0.99, train_freq=(3,"episode"),
                    replay_buffer_class=HerReplayBuffer, #replay_buffer_kwargs=replay_buffer_kwargs,
                    policy_kwargs=policy_kwargs, 
                    verbose=1, tensorboard_log=self.log_dir)
    
        for i in range(self.simulation_num):
            model.learn(total_timesteps=181*10*(i+1), reset_num_timesteps=False, tb_log_name=self.tb_log_name)
            model.save(f"{self.models_dir}/cable_v1_DDPG_{i+1}")
            print('simulation_num:', i)

    def cable_DDPG_load(self,day,num):
        model = DDPG.load(f'./simulation_length/{day}/model/cable_v1_DDPG_{num}', env=self.env)
        for i_episode in range(3):     # cycle times
            obs = self.env.reset()
            print(f"[load]-i_episode:{i_episode}")
            for i_step in range(181):         # step
                if i_step%30 == 0:
                    print(f"[load]-i_step:{i_step}")
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                # if done:
                #     obs = env.reset()
    
    # 3、SAC
    def cable_SAC_mult(self):
        policy_kwargs = dict(net_arch=[256, 256])
        model = SAC('MultiInputPolicy', 
                    self.env, buffer_size=int(1e6),tau=0.001,learning_starts=181*10,
                    learning_rate=1e-5, batch_size=181*5, gamma= 0.99, train_freq=(int(181/5),"step"), 
                    replay_buffer_class=HerReplayBuffer, 
                    policy_kwargs=policy_kwargs, 
                    verbose=1, tensorboard_log=self.log_dir)
        
        for i in range(self.simulation_num):
            model.learn(total_timesteps=181*10*(i+1), reset_num_timesteps=False, tb_log_name=self.tb_log_name)
            model.save(f"{self.models_dir}/cable_v0_SAC_{i+1}")
            print('simulation_num:', i+1)

    # 4、PPO
    def cable_PPO_mult(self):
        policy_kwargs = dict(net_arch=[256, 256])
        model = PPO('MultiInputPolicy', 
                    self.env, buffer_size=int(1e6),learning_rate=1e-5,n_steps=181*5,
                    batch_size=512, gamma= 0.99,gae_lambda=0.98,ent_coef=0.01,  
                    policy_kwargs=policy_kwargs, 
                    verbose=1, tensorboard_log=self.log_dir)
        
        for i in range(self.simulation_num):
            model.learn(total_timesteps=181*10*(i+1), reset_num_timesteps=False, tb_log_name=self.tb_log_name)
            model.save(f"{self.models_dir}/cable_v0_SAC_{i+1}")
            print('simulation_num:', i+1)

    # 5、choose
    def swtich_mode(self, mode, day=1212, num=1):
        # DDPG_train、
        cableMain = CableMain(self.simulation_day, self.tb_log_name)
        if mode == 'DDPG_train':
            # train with DDPG and HER
            cableMain.cable_DDPG_mult()
        elif mode == 'DDPG_load':
            # load with DDPG and HER
            cableMain.cable_DDPG_load(day,num)
        elif mode == 'SAC_train':
            cableMain.cable_SAC_mult()
        elif mode == 'TD3_train':
            cableMain.cable_TD3_mult()
        elif mode == 'TD3_train_again':
            cableMain.cable_TD3_again()
        elif mode == 'TD3_load':
            cableMain.cable_TD3_load(day,num)


if __name__ == "__main__":
    # tensorboard --logdir ./simulation_length/1220/log
    # 每次训练需要更新时间：[cable_main.py]-simulation_day and [cableLengthHer.py]-self.day
    cableMain = CableMain(simulation_day='2103',log_name='TD3')
    # cableMain.swtich_mode('DDPG_train')
    cableMain.swtich_mode('TD3_train')
    # cableMain.swtich_mode('SAC_train')
    # cableMain.swtich_mode(mode='TD3_load',day=1228,num=5)
