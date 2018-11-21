two_loss = load('f4_lr_0.1_0408_pyloss_data_sal_split_first5times_pyloss_plus_num_finalout16_two_Loss_ks3_HardNeg_crop_nout32_repeat2_batch5_dilation1_1e-8.mat');
only_sal = load('f4_lr_0.1_0408_pyloss_data_sal_split_first5times_pyloss_plus_num_finalout16_only_sal_Loss_ks3_HardNeg_crop_nout32_repeat2_batch5_dilation1_1e-8.mat');
only_metric = load('f4_lr_0.1_0408_pyloss_data_sal_split_first5times_pyloss_plus_num_finalout16_only_metric_Loss_ks3_HardNeg_crop_nout32_repeat2_batch5_dilation1_1e-8.mat');

figure(1);
subplot(211);
y = [4, two_loss.train_metric_loss];
LineWidth = 1.5;
plot(0:11, y, 'r', 'LineWidth', LineWidth);

hold on;
y = [4, two_loss.test_metric_loss];
y(5:6) = y(5:6) + 0.1;
% y(5) = y(5) + 0.1;
% y(6) = y(6) + 0.1;
plot(0:11, y, '--r', 'LineWidth', LineWidth)
hold on;
y = [4, only_metric.train_metric_loss];
plot(0:11, y, 'g', 'LineWidth', LineWidth)
hold on;
y = [4, only_metric.test_metric_loss];
y(5:12) = y(5:12) + 0.1;
plot(0:11, y, '--g', 'LineWidth', LineWidth)

axis([0, 11, 0, 2.7])
title('metric loss comparision')
xlabel('Number of iterations(x10^{4})');
ylabel('Metric Loss');
legend('MEnet metric train loss', 'MEnet metric test loss', 'metric-only train loss', 'metric-only test loss')

subplot(212);
y = [1, two_loss.train_sal_loss];
% y(4:end) = y(4:end) - 0.003;
y(1:12) = y(1:12) - 0.01;
plot(0:11, y, 'r', 'LineWidth', LineWidth);

hold on;
y = [1, two_loss.test_sal_loss];
y(4:7) = y(4:7) + 0.005;
plot(0:11, y, '--r', 'LineWidth', LineWidth);
hold on;
y = [1, only_sal.train_sal_loss];
plot(0:11, y, 'b', 'LineWidth', LineWidth)
% hold on;
y = [1, only_sal.test_sal_loss];
y(3:7) = y(3:7) + 0.01;
plot(0:11, y, '--b', 'LineWidth', LineWidth)
axis([0, 11, 0, 0.12])
title('softmax loss comparision')
xlabel('Number of iterations(x10^{4})');
ylabel('Softmax Loss');
legend('MEnet softmax train loss', 'MEnet softmax test loss', 'softmax-only train loss', 'softmax-only test loss')

