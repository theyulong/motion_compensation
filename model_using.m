% python using matlab for init
clc;clear;close all;

disp(['matlab init!']);
% 加载变量
load("q1.mat");
load("q2.mat");
load("q3.mat");
load("T.mat");

kp = 50700;
kd = 2*sqrt(kp);
n = 100;