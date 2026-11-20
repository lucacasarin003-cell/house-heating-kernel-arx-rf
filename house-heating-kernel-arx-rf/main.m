
clear all
clc
close all

%% 1. Data loading
load ("house.mat");
Ts = 1;
data = iddata(y, u, Ts);
N = size(y, 1);

%% 2. Data plotting
figure;
subplot(2,1,1);
plot(y);
title('Output Data y^N (House Temperature)');
ylabel('Temperature');
xlabel('Time');

subplot(2,1,2);
plot(u);
title('Input Data u^N (Heating System)');
ylabel('Input');
xlabel('Time');

%% 3. Data detrending
% Compute the means of the data
mu = getTrend(data, 0);

% Perform data detrending
data_d = detrend(data, mu);

%% 4. Model structure design

orders_arxReg = [50 50 1]; 

% M1
opt_M1 = arxRegulOptions('RegulKernel', 'DI');
[Lambda1, R1, llh1, df1] = arxRegul(data_d, orders_arxReg, opt_M1);

opt1 = arxOptions;
opt1.Regularization.Lambda = Lambda1;
opt1.Regularization.R = R1; 
M1 = arx(data_d, orders_arxReg, opt1);

% M2
opt_M2 = arxRegulOptions('RegulKernel', 'TC');
[Lambda2, R2, llh2, df2] = arxRegul(data_d, orders_arxReg, opt_M2);

opt2 = arxOptions;
opt2.Regularization.Lambda = Lambda2;
opt2.Regularization.R = R2; 
M2 = arx(data_d, orders_arxReg, opt2);

% M3
opt_M3 = arxRegulOptions('RegulKernel', 'SS');
[Lambda3, R3, llh3, df3] = arxRegul(data_d, orders_arxReg, opt_M3);

opt3 = arxOptions;
opt3.Regularization.Lambda = Lambda3;
opt3.Regularization.R = R3; 
M3 = arx(data_d, orders_arxReg, opt3);

% M4
mrand = 5;
[Lambda4, R4, llh4, df4] = arxRegulRF2(data_d, orders_arxReg, mrand);

opt4 = arxOptions;
opt4.Regularization.Lambda = Lambda4;
opt4.Regularization.R = R4; 
M4 = arx(data_d, orders_arxReg, opt4);

%% 5. Plot the impulse response

% Extract the coefficients of A(z) and B(z) for all models
a1 = M1.a;  b1 = M1.b;
a2 = M2.a;  b2 = M2.b;
a3 = M3.a;  b3 = M3.b;
a4 = M4.a;  b4 = M4.b;

% Create lag (time) vectors
lags_a = 0:(length(a1)-1);
lags_b = 0:(length(b1)-1);

figure('Name', 'Impulse Responses of A(z) and B(z)', 'NumberTitle', 'off');
sgtitle('Impulse Responses of A(z) and B(z)');

% Subplot 1: M1 (DI Kernel)
subplot(2, 2, 1);
plot(lags_a, a1, 'r-', 'LineWidth', 1.5); hold on;
plot(lags_b, b1, 'b--', 'LineWidth', 1.5); hold off;
title('M1 (DI Kernel)');
xlabel('Lag (k)'); ylabel('Amplitude');
legend('A(z)', 'B(z)', 'Location', 'best');
grid on;

% Subplot 2: M2 (TC Kernel)
subplot(2, 2, 2);
plot(lags_a, a2, 'r-', 'LineWidth', 1.5); hold on;
plot(lags_b, b2, 'b--', 'LineWidth', 1.5); hold off;
title('M2 (TC Kernel)');
xlabel('Lag (k)'); ylabel('Amplitude');
legend('A(z)', 'B(z)', 'Location', 'best');
grid on;

% Subplot 3: M3 (SS Kernel)
subplot(2, 2, 3);
plot(lags_a, a3, 'r-', 'LineWidth', 1.5); hold on;
plot(lags_b, b3, 'b--', 'LineWidth', 1.5); hold off;
title('M3 (SS Kernel)');
xlabel('Lag (k)'); ylabel('Amplitude');
legend('A(z)', 'B(z)', 'Location', 'best');
grid on;

% Subplot 4: M4
subplot(2, 2, 4);
plot(lags_a, a4, 'r-', 'LineWidth', 1.5); hold on;
plot(lags_b, b4, 'b--', 'LineWidth', 1.5); hold off;
title('M4 (SS-RF Kernel)');
xlabel('Lag (k)'); ylabel('Amplitude');
legend('A(z)', 'B(z)', 'Location', 'best');
grid on;

%% 6. Marginal likelihood evaluation

llh_values = [llh1, llh2, llh3, llh4];
best_llh = max(llh_values);

%% 7. Dividing the dataset

training_data = data_d(1:floor(N/2));
evaluation_data = data_d(floor(N/2)+1:end);

%% 8. Reconstructing the models

M1_te = arx(training_data, orders_arxReg, opt1);
M2_te = arx(training_data, orders_arxReg, opt2);
M3_te = arx(training_data, orders_arxReg, opt3);
M4_te = arx(training_data, orders_arxReg, opt4);

opt = compareOptions('InitialCondition','z');


for h = 1:4
   
    figure("Name", ['Cross-Validation - h = ' num2str(h)], 'NumberTitle', 'off');
    compare(evaluation_data, M1_te, M2_te, M3_te, M4_te, h, opt);

end

%% 9. Residual analysis

figure("Name","Residual Analysis - M1: DI", 'NumberTitle', 'off');
resid(M1, data_d, 'corr');
figure("Name","Residual Analysis - M2: TC", 'NumberTitle', 'off');
resid(M2, data_d, 'corr');
figure("Name","Residual Analysis - M3: SS", 'NumberTitle', 'off');
resid(M3, data_d, 'corr');
figure("Name","Residual Analysis - M4: Random SS", 'NumberTitle', 'off');
resid(M4, data_d, 'corr');

%% 10. SURE criterium

sure1 = surek(M1, df1);
sure2 = surek(M2, df2);
sure3 = surek(M3, df3);
sure4 = surek(M4, df4);

%% 11. Computation time analysis

fprintf('Running computational time analysis (20 iterations)...\n');

% Pre-allocate arrays to store the execution times for 20 iterations
iterations = 20;
time_M3 = zeros(iterations, 1);
time_M4 = zeros(iterations, 1);

for i = 1:iterations
    
    % --- Timing for Model 3 (SS Kernel) ---
    tic;
    [~, ~, ~, ~] = arxRegul(evaluation_data, orders_arxReg, opt_M3);
   
    %[~, ~] = arxRegul(evaluation_data, orders_arxReg, opt_M3);

    time_M3(i) = toc;
    
    % --- Timing for Model 4 (SS Kernel with Random Features) ---
    tic;
    [~, ~, ~, ~] = arxRegulRF2(evaluation_data, orders_arxReg, mrand);

    %[~, ~] = arxRegulRF2(evaluation_data, orders_arxReg, mrand);

    time_M4(i) = toc;
    
end

% Compute the mean of the execution times
mean_time_M3 = mean(time_M3);
mean_time_M4 = mean(time_M4);

% Compute the standard deviation of the execution times
std_time_M3 = std(time_M3);
std_time_M4 = std(time_M4);

% Display the results
fprintf('\n--- Execution Time Results (over %d iterations) ---\n', iterations);
fprintf('Model 3 (SS Kernel):\n');
fprintf('  Mean Time: %f seconds\n', mean_time_M3);
fprintf('  Std Dev:   %f seconds\n\n', std_time_M3);

fprintf('Model 4 (SS Kernel with Random Features):\n');
fprintf('  Mean Time: %f seconds\n', mean_time_M4);
fprintf('  Std Dev:   %f seconds\n', std_time_M4);

