# 0103-TD3 with HER 
# edit: 
# 1、k_out*math.e**(k_in*err_theta[:,2])导致奖励惩罚极大，使得模型训练不正确，输出action=nan，此处做筛选
    action[np.isnan(action)] = 0
# 2、Sigmoid 函数（如 Logistic 函数或 tanh 函数）有平缓的增长区间，并且输出被限制在一个有限的范围内。
#    这种类型的函数在x的小变化下可以敏感地变化，而其值总是被限制在[0,1][0,1] 或 [−1,1][−1,1] 的范围内。
    k_out = 30
    k_biss = 0.5
    k_in = 40
    k_out/np.exp(k_in*delta_err_theta[2]) -> k_out/(k_biss+np.exp(-k_in*delta_err_theta[2]))
# 3、添加HER的累加
# 4、action震荡，添加同级数惩罚
    delta_action = (self.previous_simulink_state[14:18]-self.current_simulink_state[14:18])/self.step_time
    action_reward = -3.5*math.sqrt(delta_action[0]**2+delta_action[1]**2+delta_action[2]**2+delta_action[3]**2)
    return action_reward + err_reward + step_reward + np.sum(area_reward) + np.sum(area_d_reward) + err_theta_d_reward

# 5、调整learning_starts=181*10，net_arch=[256, 256, 128]
policy_kwargs = dict(net_arch=[256, 256, 128])
model = TD3('MultiInputPolicy', 
            self.env, buffer_size=int(1e6),tau=0.0015,learning_starts=181*10,
            learning_rate=1e-5, batch_size=1024, gamma= 0.99, train_freq=(int(181/4),"step"),
            # replay_buffer_class=HerReplayBuffer, 
            policy_kwargs=policy_kwargs, #replay_buffer_kwargs=replay_buffer_kwargs,
            verbose=1, tensorboard_log=self.log_dir)

# 6、归一化
# 1取消全部统一到[-1,1]
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

# 改为：
target_theta = self.current_simulink_state[0:3]
target_theta_max = np.array([0.3679,-0.0449,0.1303],dtype=np.float32)
target_theta_min = np.array([-0.2340,-0.9024,-0.3378],dtype=np.float32)
self.target_theta = (target_theta-target_theta_min)/(target_theta_max-target_theta_min)
actual_theta = self.current_simulink_state[3:6]
self.actual_theta = (actual_theta-target_theta_min)/(target_theta_max-target_theta_min)
pervious_action = self.previous_simulink_state[14:18]
self.pervious_action = pervious_action*100

err_theta = self.target_theta - self.actual_theta
err_theta_max = np.array([0.01, 0.03, 0.15])
err_theta_min = np.array([-0.01, -0.03, -0.15])
self.err_theta = err_theta*(err_theta_max-err_theta_min) + err_theta_min
self.observation = np.concatenate((self.err_theta, self.pervious_action))

# 7、取消success_reward 加入comupte_reward
    all_reward = action_reward + err_reward + step_reward + np.sum(area_reward) + np.sum(area_d_reward) + err_theta_d_reward

# 8、area_reward更新
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
# 改为：
if area_flag0.any():
    area_reward[0] = area_reward[0] - 4
else:                                                   # < area_flag0
    if area_flag1.all():                                # < area_flag1
        area_reward[1] = area_reward[1] - 1 
        if area_flag2.all():                            # < area_flag2
            area_reward[2] = area_reward[2] + 1 
            if area_flag3.all():                        # < area_flag3
                area_reward[3] = area_reward[3] + 2 
            else:                                       # < area_flag3
                area_reward[3] = area_reward[3] - 0.5 
        else:                                           # > area_flag2
            area_reward[2] = area_reward[2] - 1
    else:                                               # > area_flag1
        area_reward[1] = area_reward[1] - 2

