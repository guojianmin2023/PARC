% logistic map 真实分岔图
clc;clear;
a=3.2:0.001:4;  % logistic map bifurcation parameter range
n_total=50000;  % 总迭代次数
n=500;          % 最终保留的时间步长
% compute logistic map bifurcation data
ture_logistic_bif=zeros(length(a),n);
for i=1:length(a)
    ture_logistic_bif(i,:)=funlogistic(a(i),n_total,n);
end
% 保存数据
save('ture_logistic_bif.mat','ture_logistic_bif');
% 画出logistic map中最终系统稳态随参数a的变化的分岔图
figure;
plot(a,ture_logistic_bif,'k.','markersize',0.5);  % 画分岔图
hold on;
% 绘制4条垂直线
a_values=[3.3, 3.5, 3.6, 3.8];  % 要绘制竖线的a值
posi=[0.02, 0.05, 0.02, 0.05];  % 要绘制竖线的a值的位置
% 绘制多条竖线并标注
for i=1:length(a_values)
    xline(a_values(i), '--b', 'LineWidth', 2); % 使用蓝色虚线，线宽为2
    % 添加 x 值的标注
    text(a_values(i), posi(i), ['a = ' num2str(a_values(i))], 'VerticalAlignment',...
        'bottom', 'HorizontalAlignment', 'right', 'FontSize', 12, 'Color', 'red');
end
% 添加标题和坐标轴标签
title('True logistic map bifurcation(da=0.001)','Fontsize',20, 'FontWeight',...
    'bold','Color','b', 'FontName', 'Times New Roman');
xlabel('a','Fontsize',20, 'FontName', 'Times New Roman');
xlim([3.2  4]);
ylabel('x','Fontsize',20, 'FontName', 'Times New Roman');
% 设置坐标轴字体为 Times New Roman
set(gca, 'FontName', 'Times New Roman');
hold off;
% 在图的左上角添加 (b)
text(3.21, 0.95, '(b)', 'FontSize', 20, 'FontWeight', 'bold', 'Color', ...
      'black', 'FontName', 'Times New Roman');
% logistic map的演化方程为x(n+1)=ax(n)(1-x(n))
% 共迭代n_total步，保留最后n步
function x_last_n=funlogistic(a,n_total,n)  
% x_last_n 为当虫子每年的增长率为a时，最终保留n步的时间演化
    x=zeros(1,n_total); % 给系统变量x的演化分配存储空间
    rng(1);
    x(1)=rand(1,1); % 系统初值随机取
    for t=1:n_total
          x(t+1)=a*x(t)*(1-x(t));
    end   % 对于每一个a，先让虫子数量循环n_total次，即达到稳态
    x_last_n=x(1,end-n+1:end);  % 保留x变量最后n步的状态    
end
