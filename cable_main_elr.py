from ElegantRL.agents import AgentPPO, AgentDDPG, AgentTD3, AgentSAC
from ElegantRL.train.config import get_gym_env_args, Config
from ElegantRL.train.run import *
import numpy as np
import gym
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
        self.simulation_day = simulation_day
        self.log_dir = f'./simulation/{self.simulation_day}/log/'
        self.models_dir = f'./simulation/{self.simulation_day}/model'
        self.simulation_num = 100
        self.tb_log_name = log_name
        # 1、get env info
        get_gym_env_args(gym.make("Cable-v1"), if_print=True)
    
    def train_PPO(self):
        # 2、init agent and env
        self.env_func = gym.make
        self.env_args = {
            "env_num": 1,
            "env_name": "Cable-v1",
            "max_step": 1600,
            "state_dim": 10,
            "action_dim": 4,
            "if_discrete": False,
            "target_return": 300,
            "id": "Cable-v1",
        }
        self.args = Config(AgentPPO, env_class=self.env_func, env_args=self.env_args)

        # 3、Specify hyper-parameters
        self.args.max_step = 181*200
        self.args.target_step = self.args.max_step
        self.args.gamma = 0.99
        self.srgs.reward_scale = 2**-1
        self.args.eval_times = 2 ** 4
        self.args.gpu_id = 0
    
    def train(self, agent):
        # 4、train
        if agent == 'PPO':
            self.train_PPO()
            train_agent(self.args)
            
   

if __name__ == "__main__":
    # tensorboard --logdir ./panda_reach_v2_tensorboard/
    cableMain = CableMain(simulation_day='1220',log_name='TD3')
    cableMain.train('TD3_train')

