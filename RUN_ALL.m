% =========================================================================
% STEP 5: MASTER RUNNER — RUNS THE ENTIRE PROJECT IN ONE CLICK
% =========================================================================
%
%  ██████╗ ██╗   ██╗███╗   ██╗    ████████╗██╗  ██╗██╗███████╗
%  ██╔══██╗██║   ██║████╗  ██║    ╚══██╔══╝██║  ██║██║██╔════╝
%  ██████╔╝██║   ██║██╔██╗ ██║       ██║   ███████║██║███████╗
%  ██╔══██╗██║   ██║██║╚██╗██║       ██║   ██╔══██║██║╚════██║
%  ██║  ██║╚██████╔╝██║ ╚████║       ██║   ██║  ██║██║███████║
%  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝       ╚═╝   ╚═╝  ╚═╝╚═╝╚══════╝
%
%  PHYSICS-INFORMED MACHINE LEARNING FOR HEAT EXCHANGER FOULING
%  IIT Guwahati — B.Tech Chemical Engineering Research Project
%
% =========================================================================

clc; clear; close all;
rng(2024);   % Fixed seed — every run gives identical results
fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════╗\n');
fprintf('║   PHYSICS-INFORMED ML — HEAT EXCHANGER FOULING            ║\n');
fprintf('║   Full Project Run — All 4 Steps                          ║\n');
fprintf('║   IIT Guwahati | B.Tech Chemical Engineering              ║\n');
fprintf('╚═══════════════════════════════════════════════════════════╝\n\n');

total_start = tic;

% =========================================================================
%  CHECK: All required files exist in the current folder
% =========================================================================
required_files = {'Step1_GenerateData.m','Step2_TrainAI.m', ...
                  'Step3_ShowGraphs.m','Step4_LiveDemo.m'};
all_found = true;
for k = 1:length(required_files)
    if ~exist(required_files{k}, 'file')
        fprintf('  ERROR: Cannot find %s in current folder!\n', required_files{k});
        fprintf('  Make sure all 5 .m files are in the same folder.\n');
        all_found = false;
    end
end
if ~all_found
    error('Missing files. Please check that all Step*.m files are in the same folder.');
end
fprintf('  All required files found. Starting...\n\n');

% =========================================================================
%  STEP 1: GENERATE DATA
% =========================================================================
fprintf('╔═══════════════╗\n');
fprintf('║    STEP 1     ║  Generate 8000 training scenarios\n');
fprintf('╚═══════════════╝\n');
t1 = tic;
Step1_GenerateData;
fprintf('  [Step 1 completed in %.1f seconds]\n\n', toc(t1));

% =========================================================================
%  STEP 2: TRAIN THE NEURAL NETWORK
% =========================================================================
fprintf('╔═══════════════╗\n');
fprintf('║    STEP 2     ║  Train the Physics-Informed Neural Network\n');
fprintf('╚═══════════════╝\n');
t2 = tic;
Step2_TrainAI;
fprintf('  [Step 2 completed in %.1f seconds]\n\n', toc(t2));

% =========================================================================
%  STEP 3: GENERATE ALL 5 GRAPHS
% =========================================================================
fprintf('╔═══════════════╗\n');
fprintf('║    STEP 3     ║  Generate 5 publication-quality graphs\n');
fprintf('╚═══════════════╝\n');
t3 = tic;
Step3_ShowGraphs;
fprintf('  [Step 3 completed in %.1f seconds]\n\n', toc(t3));

% =========================================================================
%  STEP 4: LIVE DEMO WITH 3 DIFFERENT SCENARIOS
% =========================================================================
fprintf('╔═══════════════╗\n');
fprintf('║    STEP 4     ║  Live Demo Predictions\n');
fprintf('╚═══════════════╝\n');

load('TrainedAI.mat');

% Shared constants for prediction
Cp_oil_demo = 2000;
k_oil_demo  = 0.13;

scenarios = [
    370, 0.6, 38000, 890, 0.004, 0.022;   % Scenario A: FAST fouling
    340, 1.0, 30000, 820, 0.002, 0.025;   % Scenario B: moderate
    220, 2.8, 12000, 800, 0.001, 0.045;   % Scenario C: NEVER fouls
];

scen_names = {'A — High-risk   (hot wall, slow flow)', ...
              'B — Moderate    (typical refinery)', ...
              'C — Safe        (cool wall, fast flow)'};

fprintf('\n  Live predictions for 3 test scenarios:\n');
fprintf('  %-40s  %s\n', 'Scenario', 'Predicted t_clean');
fprintf('  %s\n', repmat('-',1,60));

for s = 1:3
    pred_s = run_single_prediction(scenarios(s,:), net, mu_X, sig_X, mu_y, sig_y, Cp_oil_demo, k_oil_demo);
    fprintf('  %-40s  %.0f hours  (%.1f months)\n', scen_names{s}, pred_s, pred_s/720);
end
fprintf('\n');

% =========================================================================
%  FINAL SUMMARY
% =========================================================================
total_time = toc(total_start);
fprintf('╔═══════════════════════════════════════════════════════════╗\n');
fprintf('║              PROJECT COMPLETE — FINAL SUMMARY             ║\n');
fprintf('╚═══════════════════════════════════════════════════════════╝\n\n');
fprintf('TOTAL RUN TIME: %.1f minutes\n\n', total_time/60);
fprintf('╔═══════════════════════════════════════════════════════════╗\n');
fprintf('║  ALL DONE! Press Ctrl+S on each graph window to save.     ║\n');
fprintf('╚═══════════════════════════════════════════════════════════╝\n\n');

% =========================================================================
% LOCAL FUNCTIONS (Must be at the absolute bottom of the script)
% =========================================================================
function result = run_single_prediction(inputs, net, mu_X, sig_X, mu_y, sig_y, Cp_oil, k_oil)                                       
    T=inputs(1); u=inputs(2); q=inputs(3); rho=inputs(4); mu=inputs(5); D=inputs(6);
    Re  = (rho*u*D)/mu;
    Pr  = (mu*Cp_oil)/k_oil;
    Nu  = 0.023*Re^0.8*Pr^0.33;
    h   = Nu*k_oil/D;
    Ts  = (T+273.15) + q/h;
    f   = 0.316*Re^(-0.25);
    tau = (f/2)*rho*u^2;
    
    row  = [T, u, q, rho, mu, D, Re, Ts, tau];
    norm_row = (row - mu_X) ./ sig_X;
    pred_norm = net(norm_row');
    result = max(0, min(8760, pred_norm*sig_y + mu_y));
end