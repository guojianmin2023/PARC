%% config  优化主程序

clc;
clear;
load('traindata.mat');
load('testdata.mat');
iter_max = 400; %优化次数
repeat_num =4; % ensemble average size
% 1~5: eig_rho, W_in_a, a, reg, d. 超参数范围
lb = [0 0 0 10^-10 0];
ub = [3 3 1 10^-2  1];
options = optimoptions('surrogateopt','MaxFunctionEvaluations',iter_max,'PlotFcn','surrogateoptplot');
filename = ['opt_attractor_2_' datestr(now,30) '_' num2str(randi(999)) '.mat']; %保存当前程序的所有数据，文件名随机生成
min_rmse = @(x) (func_train_repeat_attractor_with_lable(x,repeat_num,traindata,testdata));
%% main (don't need to change this part)

tic
[opt_result,opt_fval,opt_exitflag,opt_output,opt_trials] = surrogateopt(min_rmse,lb,ub,options);%Matlab内置的优化函数
toc
save(filename)
if ~ispc
    exit;
end