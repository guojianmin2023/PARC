% 训练RC
function [rmse,rmse_dynamic,Y]=func_train_attractor_with_lable(hyperpara_set,traindata,rng_num,testdata)

%%
rng(rng_num);   % 随机数种子
drive_num=50; 
param_num=4;    % 分叉参数个数
numCopies =1;   % 数据需要复制的次数,复制次数为1时，数据不变
testdata = repmat(testdata, numCopies, 1);  % testdata仍然时原来产生的测试数据

%%
eig_rho =hyperpara_set(1);   % RC隐藏层连接矩阵的谱半径
W_in_a = hyperpara_set(2);   % W_in的范围
a = hyperpara_set(3);
reg = hyperpara_set(4);      % Wout计算中的正则化系数
density =hyperpara_set(5);   % RC隐藏层连接矩阵的密度
resSize =500;                % size of the reservoir  
initLen = 100;
trainLen=length(traindata(1,:))-1; % 训练数据长度
testLen = 2000;     % 预测时间步长
inSize = 4;         % RC输入的数据维数
outSize = 3;        % RC输出的数据维数（标签不输出）
nonliner_num=2;     % 将RC拟合成线性和非线性（平方项）结合
%%
  indata=traindata;  % 4*14000
  X = zeros(nonliner_num*resSize+1,trainLen);  % 将机器拟合成x+x^2+1的非线性函数
  Yt = indata(1:outSize,2:trainLen+1);
%%
Win = (2.0*rand(resSize,inSize)-1.0)*W_in_a;
WW = zeros(resSize,resSize);  % RC隐藏层连接矩阵W未作处理前的矩阵
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
x = zeros(resSize,1);       % RC隐藏层神经元状态向量存储空间
for t = 1:trainLen
    u = indata(:,t);        % 把训练集时间步长为t的变量数据赋给u
    x = (1-a)*x + a*tanh( Win*u + W*x );  % 更新RC隐藏层神经元状态
    X(:,t) = [1;x;x.^2];               % 经过RC输出函数作用后的节点状态向量
end
% 去掉initLen步长的 RC 预测数据和预测数据对应的目标系统真实数据
    X(:,1:initLen)=[];  % initLen = 100;
   Yt(:,1:initLen)=[];

% 随机打乱对应变量的迭代数据先后顺序，有助于RC提高泛化能力，防止时间依赖性
rank=randperm( size(X,2) );   % 生成一个包含这些列索引的随机排列
X=X(:, rank);                 % 使用随机排列的索引对矩阵 X 的列进行重新排列。
Yt=Yt(:, rank); 
X_T = X';
% 用随机打乱后的数据训练Wout
Wout = Yt*X_T / (X*X_T + reg*eye(nonliner_num*resSize+1));

%%  测试
Y=zeros(4,param_num*testLen);     % 用于存放所有的预测数据
rmse_dynamic=0;  

for j=0:param_num-1             % 生成每个参数的预测数据
    Y1= zeros(outSize,testLen); % 存放每个参数对应的tesLen长度的预测数据
    x1=2*rand(resSize,1)-1;     % 预测每个参数的数据时，都要重新初始化RC隐藏层状态
    Y1(1:3,1)=rand(3,1);
    Y1(4,1)=testdata(4,2000*j+1); % 将对应分叉参数标签赋给Y1时间序列的下一行
    u=testdata(:,2000*j+1);       % 把测试集每个参数的第一列给u
   for t = 1:testLen-1
        x1 = (1-a)*x1 + a*tanh( Win*u + W*x1 ); % 初始时刻的u为真实数据的第一列
        y = Wout*[1;x1;x1.^2;];                 % RC产生的数据
        if t<=drive_num  % 当启动数据在drive_num步长以内，则用测试数据的每一步驱动下一步
            u=testdata(:,2000*j+t+1);   
        else
            u = y;        % drive_num步长以后，RC闭环产生数据
        end
        Y1(1:3,t+1) = y;  % 系统产生的数据收集起来，作为系统输入数据的下一步
        Y1(4,t+1)=testdata(4,2000*j+t+1);
        u(4,1)=testdata(4,2000*j+t+1);  % 每个吸引子数据最后一行为标签行，不变
   end        
 Y(:,2000*j+1:2000*j+2000)=Y1;
% 下面是计算预测误差
pre_num=1550;  % 计算第drive_num时长开始加pre_num时长之间的平均误差
rmse_dynamic=rmse_dynamic+mean(abs(Y1(3,drive_num:drive_num+pre_num)-...
             testdata(3,2000*j+drive_num:2000*j+drive_num+pre_num)))+...
            mean(abs(Y1(2,drive_num:drive_num+pre_num)-...
            testdata(2,2000*j+drive_num:2000*j+drive_num+pre_num)))+...
            mean(abs(Y1(1,drive_num:drive_num+pre_num)-...
            testdata(1,2000*j+drive_num:2000*j+drive_num+pre_num)));
end
rmse=rmse_dynamic;
if isnan(rmse) || rmse>10
    rmse=10;
end
