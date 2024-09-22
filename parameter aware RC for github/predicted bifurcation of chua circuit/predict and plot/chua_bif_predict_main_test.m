% 优化后结果   0.0231    0.9720    0.5316    0.0003    0.7477 
% rng=130 , 误差为 0.2182 

clc;clear;
load('traindata.mat');
dt=0.001;      % 解ode方程时的时间步长
mod_num=20;    % 数据采样间隔 
drive_num=50;  % 驱动数据时间步长
load opt_attractor_2_20240716T172246_814.mat 
load min_rng_set.mat

%%  贝叶斯优化结果
result = getfield ( opt_trials,'Fval');
 param= getfield ( opt_trials,'X');
 [sort_result,result_num]=sort(result);
 sort_param=param(result_num,:);
 opt_result=sort_param(1,:);
 sort_rng=min_rng_set(result_num);
 opt_rng=sort_rng(1);  
  rng(opt_rng);


%%  超参数值调用贝叶斯优化后的结果

eig_rho =opt_result(1);   % RC隐藏层连接矩阵的谱半径
W_in_a = opt_result(2);   % W_in的范围
a = opt_result(3);
reg = opt_result(4);      % Wout计算中的正则化系数
density =opt_result(5);   % RC隐藏层连接矩阵密度
hyperpara_set=[eig_rho,W_in_a,a,reg,density];   % 超参数集合
rng_num=opt_rng;       % 随机数种子
resSize =500;          % size of the reservoir   
initLen = 100;         % 训练RC时，需要去掉的数据长度
trainLen=length(traindata(1,:))-1;    % 训练RC时，需要的输入层数据长度
inSize = 4;        % RC输入的数据维数
outSize = 3;       % RC输出的数据维数（标签不输出）
nonliner_num=2;    % 将RC拟合成线性和非线性（平方项）结合
%%
  indata=traindata;  % 训练数据矩阵大小4*14000
  X = zeros(nonliner_num*resSize+1,trainLen);  % 将机器拟合成F(x+x^2+1)的非线性函数
  Yt = indata(1:outSize,2:trainLen+1);      % RC 预测数据对应的目标系统数据
%%
Win = (2.0*rand(resSize,inSize)-1.0)*W_in_a;
WW = zeros(resSize,resSize);     % RC隐藏层连接矩阵W未作处理前的矩阵
% 生成对称稀疏矩阵WW
for i=1:resSize
    for j=i:resSize
            if (rand()<density)
             WW(i,j)=(2.0*rand()-1.0);
             WW(j,i)=WW(i,j);
            end
    end
end
rhoW = eigs(WW,1);          % 求WW的最大特征值
W = WW .* (eig_rho /rhoW);  % 对RC隐藏层连接矩阵进行缩放得到最终的连接矩阵
x=2*rand(resSize,1)-1;      % 给RC隐藏层状态随机取值
for t = 1:trainLen
    u = indata(:,t);        % 把时间步长为t的输入数据赋给u
    x = (1-a)*x + a*tanh( Win*u + W*x );  % 更新RC隐藏层状态向量
    X(:,t) = [1;x;x.^2;];   % 经过RC输出函数作用后的节点状态向量
end
% 去掉initLen步长的 RC 预测数据和预测数据对应的目标系统真实数据
    X(:,1:initLen)=[];  % initLen = 100;
   Yt(:,1:initLen)=[];

% 随机打乱对应变量的迭代数据先后顺序，有助于RC提高泛化能力，防止时间依赖性
rank=randperm( size(X,2) );   % 生成一个包含这些列索引的随机排列
X=X(:, rank);          % 使用随机排列的索引对矩阵 X 的列进行重新排列。
Yt=Yt(:, rank); 
X_T = X';
% 用随机打乱后的数据训练Wout
Wout = Yt*X_T / (X*X_T + reg*eye(nonliner_num*resSize+1));
%% predict chua circuit bifurcation
bif_chua_pre5=[];  % 预留预测分岔图数据矩阵
% 共2列，第一列为分岔参数，第二列存放分岔参数对应的最大值点
r=15.1:0.001:15.7; % 分岔参数范围
n_r=length(r);     % 分岔参数取值个数
testLen=12000;
y = Wout*[1;x;x.^2;];  % RC预测的数据
u(1:3,1)=y;            % 将RC的输出时间序列返回输入层时间序列通道
for k=1:n_r
    u(4,1)=r(k);       % 在控制通道加对应的分岔参数值
    Y= zeros(outSize,testLen);    
    % 模拟测试阶段
    for t = 1:100      % 先让reservoir运行 100 个时间步长（暂态）
        x= (1-a)*x + a*tanh( Win*u + W*x );
        y = Wout*[1;x;x.^2;];      
        Y(:,t) = y;         % 将输出结果保存到 Y 矩阵
        u(1:3,1) = y;
        u(4,1)=r(k);
    end
    for t = 1:testLen        % 预测testLen时长的时间序列
        x= (1-a)*x + a*tanh( Win*u + W*x );
        y = Wout*[1;x;x.^2;];  
        Y(:,t) = y;          
        u(1:3,1) = y;
        u(4,1)=r(k);
    end
    data_chua=Y(1,end-8000:end);      
    %% 找第一个变量x的局部最大值
    for t=3:length(data_chua(1,:))  % 从第一个变量x的第三个点开始找局部最大值
        if data_chua(1,t-1)>data_chua(1,t-2)&&data_chua(1,t-1)>data_chua(1,t)
            bif_chua_pre5=[bif_chua_pre5;r(k),data_chua(1,t-1)];
        end
    end
end
save('bif_chua_pre5.mat','bif_chua_pre5');
load('bif_chua_processed_data.mat');           % 目标系统实际分岔图数据
r_values=[15.15, 15.323, 15.496, 15.67 ];      % 训练集分岔值采样点
figure(1)   % 画预测分岔图
plot(bif_chua_pre5(:,1),bif_chua_pre5(:,2),'k.','markersize',0.5);
hold on;
% 绘制4条垂直线并标注r值
for i=1:length(r_values)
    xline(r_values(i), '--b', 'LineWidth', 2); % 使用蓝色虚线，线宽为2
    % 添加 r 值的标注
    text(r_values(i),0.812, ['r = ' num2str(r_values(i))], 'VerticalAlignment',...
        'bottom', 'HorizontalAlignment', 'right', 'FontSize', 12, 'Color', 'red');
end
title(' Predicted Chua circuit bifurcation','Fontsize',20,'Color','b');
% 在图的左上角添加标注 (a)
text(0.01, 0.95, '(a)', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');
% axis tight;
xlabel('r','FontName','Times New Roman','FontSize',20);
ylim([0.81 , 1.02]);
ylabel('x','FontName','Times New Roman','FontSize',20);

figure(2) % 画实际分岔图
plot(bif_chua_processed_data(:,1),bif_chua_processed_data(:,2),'k.','markersize',0.5);
hold on;
% 绘制采样点竖线并标注
for i=1:length(r_values)
    xline(r_values(i), '--b', 'LineWidth', 2); % 使用蓝色虚线，线宽为2
    % 添加 r 值的标注
    text(r_values(i), 0.812, ['r = ' num2str(r_values(i))], 'VerticalAlignment',...
        'bottom', 'HorizontalAlignment', 'right', 'FontSize', 12, 'Color', 'red');
end
title(' True Chua circuit bifurcation','Fontsize',20,'Color','b');
% 在图的左上角添加标注 (b)
text(0.01, 0.95, '(b)', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');
% axis tight;
xlabel('r','FontName','Times New Roman','FontSize',20);
ylim([0.81 , 1.02]);
ylabel('x','FontName','Times New Roman','FontSize',20);





