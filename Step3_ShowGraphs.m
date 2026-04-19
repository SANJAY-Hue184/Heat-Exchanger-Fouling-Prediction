% =========================================================================
% STEP 3: GENERATE ALL 5 PUBLICATION-QUALITY GRAPHS
% =========================================================================
% WHAT THIS SCRIPT DOES:
%   Graph 1 — Physics Threshold Map (signature Ebert-Panchal plot)
%   Graph 2 — Prediction Accuracy Parity Plot
%   Graph 3 — Feature Importance (proves physics features are useful)
%   Graph 4 — Dataset Distribution (t_clean histogram)
%   Graph 5 — Fouling Rate Map (EP_rate across all scenarios)
%
% RUN TIME: ~1-2 minutes (Random Forest for Graph 3 takes most time)
% =========================================================================

load('TrainedAI.mat');   % Loads all variables from Steps 1 and 2

fprintf('=========================================\n');
fprintf('  STEP 3: Generating Graphs\n');
fprintf('=========================================\n');

% Shared color palette (clean, professional look)
C_RED    = [0.82 0.12 0.10];
C_GREEN  = [0.08 0.52 0.28];
C_BLUE   = [0.12 0.38 0.72];
C_ORANGE = [0.88 0.50 0.08];
C_GRAY   = [0.55 0.55 0.55];
C_PURPLE = [0.45 0.18 0.65];

% =================================================================
% GRAPH 1: PHYSICS THRESHOLD MAP (Ts vs Re, colored by t_clean)
%
% WHAT IT SHOWS:
%   - X-axis: Reynolds number — measures flow turbulence
%   - Y-axis: Wall surface temperature Ts (°C) — drives fouling
%   - Color: how fast the exchanger fouls (red=fast, blue=safe)
%
% PHYSICAL MEANING:
%   - Top-left (high Ts, low Re) = DANGER ZONE
%     Hot wall + slow flow = asphaltenes cook and stick fast
%   - Bottom-right (low Ts, high Re) = SAFE ZONE
%     Cool wall + fast turbulent flow = no significant fouling
%
% THIS IS THE MOST IMPORTANT PHYSICS FIGURE.
% Any ExxonMobil/Shell/IOCL engineer immediately recognizes this
% plot — it proves you understand the Ebert-Panchal theory.
% =================================================================
fprintf('  Plotting Graph 1: Physics Threshold Map...\n');

figure('Name','Graph 1 — Physics Threshold Map', ...
       'Position',[50 50 900 580], 'Color','w');

% Classify scenarios by fouling speed
fast_mask = y < 2500;
med_mask  = y >= 2500 & y < 6000;
safe_mask = y >= 6000;

hold on; box on; grid on; grid minor;

% Plot safe scenarios first (behind), danger last (on top/visible)
scatter(Re_feat(safe_mask), Ts_feat(safe_mask)-273.15, 7, ...
        C_BLUE, 'filled', 'MarkerFaceAlpha', 0.4, ...
        'DisplayName', sprintf('Safe (t_{clean}>6000h, n=%d)', sum(safe_mask)));

scatter(Re_feat(med_mask),  Ts_feat(med_mask)-273.15,  7, ...
        C_ORANGE, 'filled', 'MarkerFaceAlpha', 0.5, ...
        'DisplayName', sprintf('Moderate (2500-6000h, n=%d)', sum(med_mask)));

scatter(Re_feat(fast_mask), Ts_feat(fast_mask)-273.15, 9, ...
        C_RED, 'filled', 'MarkerFaceAlpha', 0.6, ...
        'DisplayName', sprintf('Danger (t_{clean}<2500h, n=%d)', sum(fast_mask)));

% Draw the Ebert-Panchal threshold boundary
% At threshold: deposition = suppression → dRf/dt = 0
% Solving for Ts: Ts = Ea / (R * ln(alpha*Re^(-0.33) / (gamma*tau_mean)))
Re_line   = linspace(4000, 90000, 500);
tau_mean  = mean(tau_feat);
arg_inner = alpha .* Re_line.^(-0.33) ./ (gamma * tau_mean);
valid     = arg_inner > 1;   % log only defined where argument > 1
Ts_thresh = zeros(size(Re_line));
Ts_thresh(valid) = Ea ./ (R_gas .* log(arg_inner(valid))) - 273.15; % back to °C
in_range  = Ts_thresh > 150 & Ts_thresh < 450 & valid;
plot(Re_line(in_range), Ts_thresh(in_range), 'k-', 'LineWidth', 3.0, ...
     'DisplayName', 'Ebert-Panchal threshold');

% Zone annotation boxes
text(72000, 240, {'NO FOULING ZONE'; '← high Re, low T_s'}, ...
     'FontSize', 10.5, 'Color', C_BLUE, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'center', ...
     'BackgroundColor', [0.93 0.96 1.0], 'EdgeColor', C_BLUE);

text(12000, 390, {'HEAVY FOULING ZONE'; 'low Re, high T_s →'}, ...
     'FontSize', 10.5, 'Color', C_RED, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'center', ...
     'BackgroundColor', [1.0 0.93 0.92], 'EdgeColor', C_RED);

text(45000, 302, '← Threshold boundary', ...
     'FontSize', 9, 'Color', [0.2 0.2 0.2], 'FontAngle', 'italic');

xlabel('Reynolds Number  Re  [–]', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Wall Surface Temperature  T_s  [°C]', 'FontSize', 12, 'FontWeight', 'bold');
title({'Graph 1 — Ebert-Panchal Threshold Map'; ...
       'Operating space colored by fouling severity — boundary separates safe from danger zone'}, ...
      'FontSize', 13);
legend('Location', 'northeast', 'FontSize', 10);
set(gca, 'FontSize', 11);

% =================================================================
% GRAPH 2: PREDICTION ACCURACY (Parity Plot)
%
% WHAT IT SHOWS:
%   - X-axis: Actual t_clean (hours) — ground truth from simulation
%   - Y-axis: AI predicted t_clean — what the neural network says
%   - Black diagonal: perfect prediction line (y = x)
%   - Dashed lines: ±500 hour error band
%
% HOW TO READ IT:
%   - Points near the diagonal = accurate predictions
%   - Points scattered widely = poor accuracy
%   - Tight cluster around diagonal + high R² = excellent model
%   - The denser the scatter around the line, the better
% =================================================================
fprintf('  Plotting Graph 2: Prediction Accuracy...\n');

figure('Name','Graph 2 — Prediction Accuracy', ...
       'Position',[100 50 820 620], 'Color','w');

hold on; box on; grid on;

% Color points by prediction error magnitude
err_mag  = abs(y_pred - y_actual);
err_norm = err_mag / max(err_mag);  % 0=small error (blue), 1=large error (red)
clr_map  = [err_norm, 1-err_norm, 0.3*ones(size(err_norm))];  % RGB

scatter(y_actual, y_pred, 18, clr_map, 'filled', 'MarkerFaceAlpha', 0.55);

% Perfect prediction line
plot([0 8760], [0 8760], 'k-', 'LineWidth', 2.5);

% Error band lines (±500 hours)
plot([0 8760], [500 9260],  'k:', 'LineWidth', 1.2);
plot([0 8760], [-500 8260], 'k:', 'LineWidth', 1.2);
text(200, 9000, '±500 hr band', 'FontSize', 9, 'Color', [0.4 0.4 0.4]);

% Accuracy statistics text box
bias = mean(y_pred - y_actual);
stats_str = {sprintf('R²  =  %.4f', R2); ...
             sprintf('RMSE = %.0f hrs', RMSE); ...
             sprintf('MAPE = %.1f%%', MAPE); ...
             sprintf('Bias = %+.0f hrs', bias); ...
             sprintf('N test = %d', length(y_actual))};
annotation('textbox', [0.15 0.63 0.22 0.24], ...
           'String', stats_str, 'FontSize', 11, ...
           'BackgroundColor', [0.96 1.0 0.96], ...
           'EdgeColor', C_GREEN, 'LineWidth', 1.5, ...
           'FitBoxToText', 'on');

% Color bar explanation
colormap(gca, [linspace(0,1,64)', linspace(1,0,64)', 0.3*ones(64,1)]);
cb = colorbar;
cb.Label.String = 'Prediction Error (Blue=Small, Red=Large)';
cb.Label.FontSize = 10;
caxis([0 1]);

xlabel('Actual Hours to Cleaning  (t_{clean})', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('AI Predicted Hours to Cleaning', 'FontSize', 12, 'FontWeight', 'bold');
title({sprintf('Graph 2 — Prediction Accuracy  |  R² = %.4f  |  RMSE = %.0f hrs', R2, RMSE); ...
       'Points near the diagonal = accurate predictions | Color = error magnitude'}, ...
      'FontSize', 13);
xlim([0 9200]); ylim([0 9200]);
set(gca, 'FontSize', 11);

% =================================================================
% GRAPH 3: FEATURE IMPORTANCE
%
% WHAT IT SHOWS:
%   - Ranks all 9 input features by how much they help prediction
%   - Uses a Random Forest (100 trees) for reliable importance scores
%   - Physics features (green bars) should rank HIGHEST
%
% WHY THIS MATTERS FOR YOUR PROJECT:
%   This is your "proof" that the physics features are genuinely
%   informative — not just extra noise. When a reviewer or interviewer
%   asks "why do physics features help?", show them this graph.
%   If Ts_feat and tau_feat rank in top 3, your approach is validated.
%
% NOTE: Takes ~1 minute to train the Random Forest
% =================================================================
fprintf('  Plotting Graph 3: Feature Importance (training Random Forest)...\n');

mdl_rf = fitrensemble(X_norm, y_norm, 'Method', 'Bag', ...
                      'NumLearningCycles', 100, ...
                      'Learners', templateTree('MinLeafSize', 5));
imp = oobPermutedPredictorImportance(mdl_rf);

feat_names = {'T_{bulk}', 'u_{flow}', 'q_{flux}', ...
              '\rho_{oil}', '\mu_{oil}', 'D_{tube}', ...
              'Re (Physics)', 'T_s (Physics)', '\tau_w (Physics)'};

[imp_sorted, sort_idx] = sort(imp, 'descend');
names_sorted = feat_names(sort_idx);

% Color: green for physics features (indices 7,8,9), gray for raw inputs
bar_colors = zeros(9, 3);
for k = 1:9
    if sort_idx(k) >= 7    % Index 7,8,9 = physics features
        bar_colors(k,:) = C_GREEN;   % Green = physics
    else
        bar_colors(k,:) = C_GRAY;    % Gray = raw sensor
    end
end

figure('Name','Graph 3 — Feature Importance', ...
       'Position',[150 50 900 520], 'Color','w');

b = bar(imp_sorted, 0.72, 'FaceColor', 'flat');
b.CData = bar_colors;
hold on; box on; grid on; grid minor;

% Value labels on top of each bar
for k = 1:9
    text(k, imp_sorted(k)+0.002, sprintf('%.3f', imp_sorted(k)), ...
         'HorizontalAlignment', 'center', 'FontSize', 9.5, 'FontWeight', 'bold', ...
         'Color', bar_colors(k,:) * 0.7);
end

% Rank labels (1st, 2nd etc.)
rank_labels = {'1st','2nd','3rd','4th','5th','6th','7th','8th','9th'};
for k = 1:9
    text(k, -0.005, rank_labels{k}, 'HorizontalAlignment', 'center', ...
         'FontSize', 8.5, 'Color', [0.4 0.4 0.4]);
end

set(gca, 'XTick', 1:9, 'XTickLabel', names_sorted, 'XTickLabelRotation', 20);
ylabel('Permuted Predictor Importance Score', 'FontSize', 11, 'FontWeight', 'bold');
title({'Graph 3 — Feature Importance  (Random Forest, 100 trees)'; ...
       'Green = physics-derived features | Gray = raw sensor inputs'; ...
       'Physics features ranking high VALIDATES the physics-informed approach'}, ...
      'FontSize', 13);

% Legend
patch(NaN, NaN, C_GREEN, 'DisplayName', 'Physics feature (Re, T_s, \tau_w)');
patch(NaN, NaN, C_GRAY,  'DisplayName', 'Raw sensor input');
legend('Location', 'northeast', 'FontSize', 11);
set(gca, 'FontSize', 11); 
ymax = max(imp_sorted) * 1.18;
if ymax <= 0
    ymax = 1;
end
ylim([0, ymax]);

% Count how many top-3 are physics features
physics_in_top3 = sum(sort_idx(1:3) >= 7);
if physics_in_top3 >= 2
    fprintf('    Physics features in top 3: %d/3 — EXCELLENT validation!\n', physics_in_top3);
else
    fprintf('    Physics features in top 3: %d/3\n', physics_in_top3);
end

% =================================================================
% GRAPH 4: DATASET DISTRIBUTION (t_clean Histogram)
%
% WHAT IT SHOWS:
%   - How many scenarios fall into each cleaning time bucket
%   - Red zone (<2000h) = high-risk, frequent cleaning needed
%   - Orange zone (2000-5000h) = moderate fouling
%   - Blue zone (>5000h) = safe, long time between cleanings
%
% WHAT A GOOD DISTRIBUTION LOOKS LIKE:
%   A spread across all zones with a peak near 8760 (never fouled).
%   This proves the Latin Hypercube Sampling covered the full range
%   of operating conditions and the dataset is balanced.
% =================================================================
fprintf('  Plotting Graph 4: Dataset Distribution...\n');

figure('Name','Graph 4 — Dataset Distribution', ...
       'Position',[200 50 840 520], 'Color','w');

hold on; box on; grid on;
bins  = 0:300:8760;
[cnts, edgs] = histcounts(y, bins);
ctrs = (edgs(1:end-1) + edgs(2:end)) / 2;

for k = 1:length(cnts)
    if ctrs(k) < 2000
        fc = C_RED;
    elseif ctrs(k) < 5000
        fc = C_ORANGE;
    else
        fc = C_BLUE;
    end
    bar(ctrs(k), cnts(k), 295, 'FaceColor', fc, 'FaceAlpha', 0.80, 'EdgeColor', 'white');
end

xline(mean(y),   'k-',  'LineWidth', 2.2);
xline(median(y), 'k--', 'LineWidth', 1.6);
text(mean(y)+100,   max(cnts)*0.94, sprintf('Mean = %.0f hrs',   mean(y)),   'FontSize', 10, 'FontWeight', 'bold');
text(median(y)+100, max(cnts)*0.84, sprintf('Median = %.0f hrs', median(y)), 'FontSize', 10);

text(1000, max(cnts)*0.72, sprintf('High-risk\n%d scenarios', sum(y<2000)), ...
     'FontSize', 9.5, 'Color', C_RED, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(3500, max(cnts)*0.72, sprintf('Moderate\n%d scenarios', sum(y>=2000 & y<5000)), ...
     'FontSize', 9.5, 'Color', C_ORANGE, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(7000, max(cnts)*0.72, sprintf('Safe\n%d scenarios', sum(y>=5000)), ...
     'FontSize', 9.5, 'Color', C_BLUE, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

xlabel('Hours Until Cleaning Required  (t_{clean})', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Number of Scenarios  (out of 8000)', 'FontSize', 12, 'FontWeight', 'bold');
title({'Graph 4 — Distribution of Cleaning Times Across 8000 Scenarios'; ...
       'Red = frequent cleaning needed | Blue = safe long operation'}, ...
      'FontSize', 13);
set(gca, 'FontSize', 11);

% =================================================================
% GRAPH 5: FOULING RATE MAP (EP_rate across operating conditions)
%
% WHAT IT SHOWS:
%   - A 2D heatmap showing the Ebert-Panchal fouling rate
%   - X-axis: wall temperature Ts (the dominant driver)
%   - Y-axis: flow velocity u (the suppression driver)
%   - Color: net fouling rate (yellow=fast, blue=zero/threshold)
%
% PHYSICAL STORY:
%   Top-right corner: high temp AND high flow — the two forces fight
%   Bottom-right:     high temp + slow flow = WORST fouling
%   Upper-left:       low temp + fast flow = BEST conditions
%   White/light area: below fouling threshold (no fouling at all)
% =================================================================
fprintf('  Plotting Graph 5: Fouling Rate Map...\n');

figure('Name','Graph 5 — Fouling Rate Map', ...
       'Position',[250 50 860 560], 'Color','w');

% Create a grid of Ts and u values to evaluate EP_rate
Ts_grid_C = linspace(200, 420, 120);   % Wall temperature [°C]
u_grid    = linspace(0.3, 3.2,  120);   % Flow velocity [m/s]

[Ts_G, u_G] = meshgrid(Ts_grid_C, u_grid);
Ts_G_K = Ts_G + 273.15;

% Use representative values for the other parameters
rho_rep = mean(rho_oil);
mu_rep  = mean(mu_oil);
D_rep   = mean(D_tube);

Re_G    = rho_rep .* u_G .* D_rep ./ mu_rep;
f_G     = 0.316 .* Re_G.^(-0.25);
tau_G   = (f_G/2) .* rho_rep .* u_G.^2;
EP_G    = max(0, alpha .* Re_G.^(-0.33) .* exp(-Ea./(R_gas.*Ts_G_K)) - gamma.*tau_G);

% Cap the colormap for visual clarity (very high rates look same as moderately high)
EP_plot = min(EP_G, prctile(EP_G(:), 95));

imagesc(Ts_grid_C, u_grid, EP_plot);
axis xy; hold on; box on;
colormap(flipud(hot));
cb2 = colorbar;
cb2.Label.String = 'Net Fouling Rate  dR_f/dt  [m²K/W per hour]';
cb2.Label.FontSize = 10;

% Overlay the threshold contour (where EP_rate = 0)
contour(Ts_grid_C, u_grid, EP_G, [1e-10 1e-10], 'w-', 'LineWidth', 2.5);
text(240, 2.8, 'Fouling threshold', 'Color', 'white', 'FontSize', 10, ...
     'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.4]);
text(240, 2.55, '(below = safe)', 'Color', 'white', 'FontSize', 9.5, ...
     'BackgroundColor', [0 0 0 0.3]);

% Arrow annotations
annotation('textarrow', [0.82 0.72], [0.82 0.65], 'String', 'Danger zone', ...
           'FontSize', 10, 'Color', 'white', 'FontWeight', 'bold');

xlabel('Wall Surface Temperature  T_s  [°C]', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Flow Velocity  u  [m/s]',              'FontSize', 12, 'FontWeight', 'bold');
title({'Graph 5 — Ebert-Panchal Fouling Rate Map  (EP Rate Heatmap)'; ...
       'Yellow/hot = fast fouling | Dark blue = below fouling threshold (safe)'; ...
       sprintf('(Fixed: \\rho=%.0f kg/m³, \\mu=%.4f Pa·s, D=%.3f m)', rho_rep, mu_rep, D_rep)}, ...
      'FontSize', 13);
set(gca, 'FontSize', 11);

fprintf('\nStep 3 COMPLETE. All 5 graphs displayed.\n');
fprintf('  → Press Ctrl+S on each graph window to save as image\n');
fprintf('  → Or type: saveas(figure(1),''Graph1_ThresholdMap.png'')\n');
fprintf('=========================================\n\n');