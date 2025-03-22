import gym
import panda_gym
import time
from stable_baselines3 import DDPG, TD3, SAC, HerReplayBuffer, PPO

env = gym.make("PandaReach-v2", render=True) # panda_gym step渲染 render=True
log_dir = './panda_reach_v2_tensorboard/'
# cmd:tensorboard --logdir ./panda_reach_v2_tensorboard/
# web:http://localhost:6006/

MODE = ''
# MODE = 'without'
# MODE = 'with'
MODE = 'load'
# MODE = 'multiplewith'

if MODE == 'without':
    # model = SAC('MlpPolicy', env, verbose=1,tensorboard_log=log_dir)
    model = PPO('MultiInputPolicy', env, verbose=1,tensorboard_log=log_dir)
    # DDPG
    # model = DDPG(policy="MultiInputPolicy", env=env, buffer_size=100000, replay_buffer_class=HerReplayBuffer, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=20000)
    model.save("./daran0/without/ppo_daran_reach_v0_{}".format(str(1)))

elif MODE == 'with':
    model = DDPG(policy="MultiInputPolicy", 
                env=env,buffer_size=1000000,
                learning_rate=2e-4,batch_size=256,
                replay_buffer_class=HerReplayBuffer,
                verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=5000)
    model.save("./ddpg_panda_reach_v2_{}".format(str('1')))

elif MODE == 'load':
    model = DDPG.load('./ddpg_panda_reach_v2_{}'.format(str('1')), env=env)
    for i_episode in range(10):     # cycle times
        obs = env.reset()
        for i in range(50):         # step
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render('human')
            print(done)
            if done:
                print('Done')
                obs = env.reset()
           
elif MODE == 'multiplewith':
    models_dir = 'daran0/with/ddpg_daran_reach_v0_'
    model = DDPG('MultiInputPolicy', 
                 env, buffer_size=1000000,learning_rate=1e-4, batch_size=256,replay_buffer_class=HerReplayBuffer, verbose=1, tensorboard_log=log_dir)
    TIMESTEPS = 4000
    for i in range(10):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DDPG")
        model.save(f"{models_dir}/{TIMESTEPS *(i+1)}")

# env = gym.make("CartPole-v1") # gym中无render=True，渲染由env.render()每次step提供

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10_000)

# obs = env.reset()
# for i in range(2):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     # VecEnv resets automatically
#     # if done:
#     #   obs = env.reset()
#     time.sleep(1)

# env.close()