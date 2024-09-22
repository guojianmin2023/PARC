% correct program to predict logistic map bifurcation
clc;clear;
load('traindata.mat');
eig_rho =0.05;  % RC 隐藏层连接矩阵 W 的谱半径
W_in_a = 1;     % 输入层和隐藏层之间的连接矩阵W_in的范围
a = 1;          % 泄露率
reg = 1e-5;     % Wout计算中的正则化系数
density =0.2;   % RC 隐藏层连接矩阵的密度
% hyperpara_set=[eig_rho,W_in_a,a,reg,density];
rng(1);
resSize =100;   % size of the reservoir
initLen = 100;
trainLen=length(traindata(1,:))-1;       % 训练数据长度
inSize = 2;         % RC 输入层数据维数
outSize = 1;        % RC 输出层数据维数（参数通道数据不输出）
nonliner_num=2;     % 将 RC 拟合成线性和非线性（平方项）的集合
%%
  indata=traindata;  % 2*8000
  X = zeros(nonliner_num*resSize+1,trainLen);  % 将机器拟合成F(x+x^2+1)的非线性函数
  Yt = indata(1:outSize,2:trainLen+1);         % RC 预测数据对应的目标系统数据
%%
Win = (2.0*rand(resSize,inSize)-1.0)*W_in_a;
WW = zeros(resSize,resSize);  % 设置 RC 隐藏层连接矩阵W未作处理前的存储空间
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
W = WW .* (eig_rho /rhoW);  % 对 RC 连接矩阵进行缩放得到最终的连接矩阵
%% 训练 Wout
x=2*rand(resSize,1)-1;    % 随机初始化 RC 隐藏层神经元状态
for t = 1:trainLen
    u = indata(:,t);      % 用每一列训练数据更新RC神经元状态并利用RC预测下一步数据
    x = (1-a)*x + a*tanh( Win*u + W*x );  % 更新 RC 节点的状态
    X(:,t) = [1;x;x.^2;];   % 经过RC隐藏层神经元输出函数作用后的节点状态
    %收集每个时间步长下的隐藏层神经元状态
end
% 去掉initLen步长的 RC 预测数据和预测数据对应的目标系统真实数据
    X(:,1:initLen)=[];  % initLen = 100;
   Yt(:,1:initLen)=[];
% 随机打乱对应变量的迭代数据先后顺序，有助于提高RC预测的泛化能力，防止RC对时间的依赖性
rank=randperm( size(X,2) );   % 生成一个包含这些列索引的随机排列
X=X(:, rank);                 % 使用随机排列的索引对矩阵 X 的列进行重新排列
Yt=Yt(:, rank); 
X_T = X';
% 用随机打乱后的数据训练Wout
Wout = Yt*X_T / (X*X_T + reg*eye(nonliner_num*resSize+1));
%% predict logistic map bifurcation
predicted_logistic_bif=[];  
% 设置预测的分岔图数据矩阵为空矩阵，便于矩阵大小随实际预测数据改变
r=3.2:0.001:4;         % 分岔图对应分岔参数范围及取值间隔
n_r=length(r);         % 分岔参数r的取值个数
testLen=5000;          % 预测数据长度
y = Wout*[1;x;x.^2;];  % 训练RC时，最后一列训练集数据输入RC后，RC的输出数据
for k=1:n_r    % 生成每个分岔参数的分岔图数据
    u(1,1)=y;  % 让RC生成的数据作为下一步的输入
    u(2,1)=r(k);  % 给RC输入时间序列加上对应的分岔参数
    Y= zeros(outSize,testLen);  % Y矩阵中只存放预测的变量时间序列
    %% 模拟测试阶段（先让RC工作一段时间）
    for t = 1:100  % 先 RC 运行 100 个时间步长(暂态)
        x= (1-a).*x + a.*tanh( Win*u + W*x );
        y = Wout*[1;x;x.^2;];  % RC产生的数据
        Y(:,t) = y;            % 将输出结果保存到 Y矩阵
        u(1,1) = y;
        u(2,1)=r(k);   % 因为在预测第k个分岔参数的分岔图数据，所以参数通道值为该分岔参数
    end
    %% 预测
     % 开始预测的第一个数据的输入时间序列为上一步RC的输出
    for t = 1:testLen          
        x= (1-a).*x + a.*tanh( Win*u + W*x );  
        y = Wout*[1;x;x.^2;];  % 预测数据
        Y(:,t) = y;            % 将预测数据保存到Y矩阵对应列
        u(1,1) = y;
        u(2,1)=r(k);
    end
    predicted_logistic_bif(k,:)=Y(:,end-500+1:end);  
    % 保留每个分岔参数的最后500个点画分岔图
end
save('predicted_logistic_bif.mat','predicted_logistic_bif');
plot(r,predicted_logistic_bif,'k.','markersize',0.5);
hold on;
% 绘制4条垂直虚线，每条虚线代表分岔参数采样点位置
a_values=[3.3, 3.5, 3.6, 3.8];  % 每条竖线对应的分岔参数值
posi=[0.02, 0.05, 0.02, 0.05];  % 每个采样点的位置
% 绘制多条竖线并标注
for i=1:length(a_values)
    xline(a_values(i), '--b', 'LineWidth', 2); % 使用蓝色虚线，线宽为2
    % 添加 x 值的标注
    text(a_values(i), posi(i), ['a = ' num2str(a_values(i))], 'VerticalAlignment',...
        'bottom', 'HorizontalAlignment', 'right', 'FontSize', 12, 'Color', 'red');
end
title('Logistic map bifurcation predict(da=0.001)','Fontsize',19, 'FontWeight', 'bold','Color','b');
% axis tight;
xlabel('a','FontName','Times New Roman','FontSize',24, 'FontWeight', 'bold');
ylabel('x','FontName','Times New Roman','FontSize',24, 'FontWeight', 'bold');
xlim([3.2  4]);
% 在图的左上角添加 (a)
text(3.21, 0.95, '(a)', 'FontSize', 20, 'FontWeight', 'bold', 'Color', ...
      'black', 'FontName', 'Times New Roman');



