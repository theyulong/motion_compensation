% 2024-12-31
% 画reward
% clc;clear;close all;

ep_rew_mean = readtable("simulation\1228\data\1228_TD3_0_ep_rew_mean.csv");
ep_rew_mean.WallTime = datetime(ep_rew_mean.WallTime, 'ConvertFrom', 'posixtime');
success_rate = readtable("simulation\1228\data\1228_TD3_0_success_rate.csv");
success_rate.WallTime = datetime(success_rate.WallTime, 'ConvertFrom', 'posixtime');
[maxValue, maxIndex] = max(ep_rew_mean.Value);
[minValue, minIndex] = min(ep_rew_mean.Value);
figure;
indexDelta = 20;
hold on;box on;legend('box','off');% grid minor;
xlim([0 ep_rew_mean.Step(maxIndex+indexDelta)]);ylim([minValue 0]);
plot(ep_rew_mean.Step(1:maxIndex+indexDelta),ep_rew_mean.Value(1:maxIndex+indexDelta),'k',LineWidth=1.5);
xlabel('step');ylabel('reward');
legend('reward','Location','southeast');
