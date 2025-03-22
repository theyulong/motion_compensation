import gym
from stable_baselines3 import PPO, DDPG, HerReplayBuffer

# Parallel environments
# env = gym.make("Pendulum-v0")

# model = DDPG("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=5000)
# model.save("Pendulum")

# del model # remove to demonstrate saving and loading

# model = DDPG.load("Pendulum")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()


env = gym.make("Cable-v1") # panda_gym step渲染 render=True
policy_kwargs = dict(net_arch=[256, 256])
model = DDPG('MultiInputPolicy', 
                env, buffer_size=1e5,
                learning_rate=1e-3, batch_size=256, gamma= 0.995, train_freq=(6,"step"),
                replay_buffer_class=HerReplayBuffer, 
                policy_kwargs=policy_kwargs, 
                verbose=1,)


model.learn(total_timesteps=181*5, reset_num_timesteps=False, tb_log_name="DDPG")
model.save(f"cable_v1_DDPG_")

