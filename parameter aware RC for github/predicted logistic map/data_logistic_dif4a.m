
% 生成训练数据traindata（对应四个分岔参数的数据，2*8000）
% 随机给系统赋初值，每个分岔参数a得到后2000步的logistic数据
% 每个分岔参数的变量时间序列下一行为该分岔参数值（作为参数引导的通道输入数据）
clc;
clear;
totle_time=3000;  % 总的迭代时间
de_time=1000;     % 去掉1000步暂态
time=totle_time-de_time; % 每个分岔参数下，最终保留的时间步长,用于训练RC
a1=3.3;   % 周期2
a2=3.5;   % 周期4
a3=3.6;   % 混沌
a4=3.8;   % 混沌
D_in=2;   % 输入数据变量个数，logistic map 本身的变量x和对应的分岔参数a
u=zeros(D_in,4*time);     % 为训练集数据预留存储空间
% u1,u2,u3,u4是以上分岔参数对应的后2000步时间序列，每个矩阵为1*2000
%% 分别为每个分岔参数的系统状态变量时间序列预留存储空间
u1=zeros(1,time); % 第一个参数a1对应的时间序列存储空间 
u2=zeros(1,time);
u3=zeros(1,time);
u4=zeros(1,time);

% 计算不同分岔参数对应的 Logistic map 的后一段时间序列
data_logistic1=fun_logistic(totle_time,de_time,a1);
data_logistic2=fun_logistic(totle_time,de_time,a2);
data_logistic3=fun_logistic(totle_time,de_time,a3);
data_logistic4=fun_logistic(totle_time,de_time,a4);

u1=data_logistic1;  % u1保存第一个分岔参数a1对应的后2000步数据
u2=data_logistic2;  
u3=data_logistic3; 
u4=data_logistic4; 
% 将不同分岔参数对应的数据串起来，形成训练数据u（第一行为不同参数下的x，
% 第二行为对应的分岔参数），用于训练RC
u(1,1:time)=u1;         
u(1,time+1:2*time)=u2; 
u(1,2*time+1:3*time)=u3;
u(1,3*time+1:end)=u4; 
% 训练数据矩阵第二行为第一行状态向量对应的分岔参数
u(2,1:time)=a1;             
u(2,time+1:2*time)=a2; 
u(2,2*time+1:3*time)=a3;
u(2,3*time+1:end)=a4; 
traindata=u;
% 保存训练数据为traindata.mat，以便训练 RC 时调用
save('traindata.mat','traindata'); 
% 如果要使用该文件的数据，需要先使用load函数加载该文件的数据，方法如下
% load('traindata.mat');

%% 生成 logistic map system 时间序列程序（函数程序）
% 输入参数：totle_time（总时长）,de_time（去掉的暂态时长）,a（分岔参数）
function data_log=fun_logistic(totle_time,de_time,a)
    x=zeros(1,totle_time); 
    rng(1);            % 随机数种子 
    x(1)=rand(1,1);    % x的初值为（0,1）之间的随机数
     for t=1:totle_time-1
        x(t+1)=a*x(t)*(1-x(t)); 
     end
    data_log=x(de_time+1:end);  % 保留去掉前de_time暂态时间后的数据
end
