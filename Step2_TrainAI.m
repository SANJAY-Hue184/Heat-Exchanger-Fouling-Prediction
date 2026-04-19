% =========================================================================
% STEP 2: TRAIN THE PHYSICS-INFORMED NEURAL NETWORK
% =========================================================================
% WHAT THIS SCRIPT DOES:
%   - Loads the 8000 scenarios from Step 1
%   - Normalizes all data (critical for neural network accuracy)
%   - Builds a 3-layer neural network [64 → 32 → 16 → 1]
%   - Trains it using Levenberg-Marquardt optimization
%   - Reports R² and RMSE accuracy on the held-out test set
%   - Saves the trained model to TrainedAI.mat
%
% RUN TIME: ~30-60 seconds
% =========================================================================

load('ProjectData.mat');   % Load everything from Step 1

fprintf('=========================================\n');
fprintf('  STEP 2: Training Physics-Informed AI\n');
fprintf('=========================================\n');
fprintf('  Input features: 9 (6 raw + 3 physics)\n');
fprintf('  Network layers: [64, 32, 16]\n');
fprintf('  Training data : %d scenarios\n', N);
fprintf('  Please wait...\n\n');

% -----------------------------------------------------------------
% NORMALIZATION — WHY THIS IS MANDATORY
%
% Problem without normalization:
%   T_bulk  ranges  200-380  → span ~180
%   q_flux  ranges  10000-40000 → span ~30000
%
% The gradient during training is dominated by q_flux (100× bigger).
% The network effectively ignores T_bulk because its numbers barely
% move the loss function. Result: terrible accuracy.
%
% Z-score normalization: x_norm = (x - mean) / std
% After normalization: every feature has mean=0 and std=1.
% Every feature gets equal initial attention from the network.
%
% RULE: always normalize y (target) too — not just X.
% RULE: save mu_X and sig_X — you need them to normalize new data later.
% -----------------------------------------------------------------
[X_norm, mu_X, sig_X] = zscore(X_phys);    % Normalize inputs (9 features)
[y_norm, mu_y, sig_y] = zscore(y);          % Normalize target (hours)

% -----------------------------------------------------------------
% BUILD THE NEURAL NETWORK
%
% Architecture: 9 inputs → 64 neurons → 32 neurons → 16 neurons → 1 output
%
% Layer 1 (64 neurons): learns broad patterns
%   e.g., "high Ts AND low Re together = likely fast fouling"
%
% Layer 2 (32 neurons): refines patterns from layer 1
%   e.g., "this combination of layer 1 patterns = 3000-hour cleaning time"
%
% Layer 3 (16 neurons): final refinement before output
%
% Output (1 neuron): predicted t_clean in normalized hours
%
% trainlm = Levenberg-Marquardt optimizer
%   Fastest convergence for medium-sized networks like this one
%   Automatically balances gradient descent and Newton's method
%
% max_fail = 15: early stopping
%   If validation error hasn't improved in 15 epochs, stop training
%   This prevents overfitting (memorizing training data)
% -----------------------------------------------------------------
net = fitnet([64, 32, 16]);
net.trainFcn = 'trainlm';

net.trainParam.showWindow  = false;   % No pop-up window
net.trainParam.epochs      = 1000;    % Max training iterations
net.trainParam.goal        = 1e-7;    % Stop if MSE reaches this
net.trainParam.max_fail    = 15;      % Early stopping patience
net.trainParam.min_grad    = 1e-8;    % Stop if gradient is flat

% L2 regularization (prevents overfitting by penalizing large weights)
% Value 0.01 = mild regularization, good for this problem size
net.performParam.regularization = 0.01;

% Data split: 70% training, 15% validation (for early stopping), 15% testing
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;

% -----------------------------------------------------------------
% TRAIN THE NETWORK
% Input must be transposed: fitnet expects [features × samples]
% Our X_norm is [samples × features], so we transpose with '
% -----------------------------------------------------------------
[net, tr] = train(net, X_norm', y_norm');

% -----------------------------------------------------------------
% EVALUATE ON TEST SET
%
% tr.testInd contains the indices of the 15% held-out test samples.
% These were NEVER seen by the network during training.
% This gives an honest estimate of real-world accuracy.
%
% Denormalization: prediction = (norm_prediction * sig_y) + mu_y
% This converts normalized output back to real hours.
% -----------------------------------------------------------------
test_idx = tr.testInd;
y_actual = y(test_idx);
y_pred   = (net(X_norm(test_idx,:)'))' * sig_y + mu_y;

% R² (coefficient of determination): 1.0 = perfect, 0.0 = useless
% How much of the variance in t_clean does the model explain?
R2 = 1 - sum((y_actual - y_pred).^2) / ...
         sum((y_actual - mean(y_actual)).^2);

% RMSE (root mean square error): average prediction error in HOURS
% Lower is better. For this problem, < 400 hours is good, < 200 is excellent
RMSE = sqrt(mean((y_pred - y_actual).^2));

% Mean Absolute Percentage Error (MAPE) — intuitive percentage accuracy
MAPE = mean(abs(y_pred - y_actual) ./ y_actual) * 100;

fprintf('Training Results (on 15%% held-out test set):\n');
fprintf('  R²   = %.4f  (target: > 0.95)\n', R2);
fprintf('  RMSE = %.0f hours  (target: < 300 hours)\n', RMSE);
fprintf('  MAPE = %.1f%%  (target: < 10%%)\n', MAPE);

if R2 > 0.95
    fprintf('\n  RESULT: EXCELLENT accuracy achieved!\n');
elseif R2 > 0.90
    fprintf('\n  RESULT: Good accuracy.\n');
else
    fprintf('\n  RESULT: Moderate accuracy. Consider running Step 1 again.\n');
end

% Save everything including the trained network
save('TrainedAI.mat');
fprintf('\nStep 2 COMPLETE. Trained AI saved to TrainedAI.mat\n');
fprintf('=========================================\n\n');