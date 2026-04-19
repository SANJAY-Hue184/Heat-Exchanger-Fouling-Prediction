% =========================================================================
% STEP 1: GENERATE DATA & CALCULATE PHYSICS FEATURES
% =========================================================================
% WHAT THIS SCRIPT DOES:
%   - Creates 8000 synthetic refinery heat exchanger scenarios
%   - Calculates all physics features (Re, Ts, tau_w etc.)
%   - Computes how many hours each exchanger takes to foul
%   - Saves everything to ProjectData.mat
%
% RUN TIME: ~3-5 minutes
% =========================================================================

 rng(2024);
fprintf('=========================================\n');
fprintf('  STEP 1: Generating Data\n');
fprintf('=========================================\n');

% -----------------------------------------------------------------
% PHYSICAL CONSTANTS (Ebert-Panchal crude oil fouling model)
% Published ExxonMobil values for refinery preheat trains
% -----------------------------------------------------------------
alpha   = 0.02778;    % Deposition constant     [m^2*K/J]
Ea      = 48000;    % Activation energy        [J/mol]
R_gas   = 8.314;    % Universal gas constant   [J/mol/K]
gamma   = 4.17e-9;  % Suppression constant     [m^2*K/J]
Cp_oil  = 2000;     % Heat capacity crude oil  [J/kg/K]
k_oil   = 0.13;     % Thermal conductivity     [W/m/K]
Rf_clean = 0.0002;  % TEMA cleaning threshold  [m^2*K/W]

% -----------------------------------------------------------------
% GENERATE 8000 SCENARIOS USING LATIN HYPERCUBE SAMPLING
% LHS gives better coverage than random — like a well-designed
% experiment covering all operating conditions evenly
% -----------------------------------------------------------------
N   = 8000;
lhs = lhsdesign(N, 6);

% Map 0-to-1 LHS values to real refinery operating ranges
T_bulk  = lhs(:,1) * 180 + 200;    % Bulk temperature   [°C]  200-380
u_flow  = lhs(:,2) * 2.5 + 0.5;    % Flow velocity      [m/s] 0.5-3.0
q_flux  = lhs(:,3) * 3e4  + 1e4;   % Heat flux      [W/m²] 10k-40k
rho_oil = lhs(:,4) * 120  + 780;   % Oil density      [kg/m³] 780-900
mu_oil  = lhs(:,5) * 4e-3 + 1e-3;  % Oil viscosity    [Pa.s] 1e-3 to 5e-3
D_tube  = lhs(:,6) * 0.03 + 0.02;  % Tube diameter      [m]   0.02-0.05

% -----------------------------------------------------------------
% CALCULATE ALL PHYSICS FEATURES (vectorized — no loop needed)
%
% WHY WE CALCULATE THESE:
% These are the "translator" features. They convert raw sensor
% readings into physically meaningful quantities that directly
% govern the fouling process. This is what makes Model B smarter
% than a black-box — it gives the neural network pre-digested
% chemistry knowledge, not just raw numbers.
% -----------------------------------------------------------------

% FORMULA 1: Reynolds Number
% Re = rho * u * D / mu
% Tells us: is the flow smooth (laminar) or chaotic (turbulent)?
% High Re = turbulent = flow scrubs the wall = LESS fouling
Re_feat = (rho_oil .* u_flow .* D_tube) ./ mu_oil;

% FORMULA 2: Prandtl Number
% Pr = (mu * Cp) / k
% Ratio of momentum to thermal diffusivity for crude oil (~15-30)
Pr_feat = (mu_oil .* Cp_oil) ./ k_oil;

% FORMULA 3: Nusselt Number (Dittus-Boelter correlation)
% Nu = 0.023 * Re^0.8 * Pr^0.33
% How much better does MOVING fluid transfer heat vs still fluid?
Nu_feat = 0.023 .* Re_feat.^0.8 .* Pr_feat.^0.33;

% FORMULA 4: Heat Transfer Coefficient
% h = Nu * k / D   [W/m²K]
% Rate of heat transfer from pipe wall into the crude oil
h_feat = Nu_feat .* k_oil ./ D_tube;

% FORMULA 5: Wall Surface Temperature *** MOST CRITICAL ***
% Ts = T_bulk + 273.15 + q_flux/h
% The pipe wall is ALWAYS hotter than bulk fluid (heat flows wall→fluid)
% +273.15 converts °C to Kelvin — MANDATORY for Arrhenius equation
% Small rise in Ts → HUGE rise in fouling (exponential relationship)
Ts_feat = (T_bulk + 273.15) + q_flux ./ h_feat;   % [Kelvin]

% FORMULA 6: Blasius Friction Factor
% f = 0.316 * Re^(-0.25)
% Friction in turbulent pipe flow (Blasius, valid 4000 < Re < 100000)
f_feat = 0.316 .* Re_feat.^(-0.25);

% FORMULA 7: Wall Shear Stress
% tau_w = (f/2) * rho * u²   [Pa]
% The "scrubbing force" of the flow on the pipe wall
% High tau_w = strong scrubbing = deposits get removed = less fouling
tau_feat = (f_feat/2) .* rho_oil .* u_flow.^2;

% FORMULA 8: Ebert-Panchal Fouling Rate
% dRf/dt = alpha*Re^(-0.33)*exp(-Ea/R*Ts) - gamma*tau_w
%
% DEPOSITION TERM: alpha*Re^(-0.33)*exp(-Ea/R*Ts)
%   Chemical reaction at the hot wall (Arrhenius equation)
%   Hot wall + moderate flow = asphaltenes cook and stick
%
% SUPPRESSION TERM: gamma*tau_w
%   Fast flow physically rips deposits off the wall
%
% NET RATE = Deposition minus Suppression
% max(0,...) = fouling can't go negative (deposits don't vanish)
EP_feat = max(0, alpha .* Re_feat.^(-0.33) .* ...
          exp(-Ea ./ (R_gas .* Ts_feat)) - gamma .* tau_feat);

% -----------------------------------------------------------------
% CALCULATE CLEANING TIME (TARGET LABEL)
%
% If fouling rate is constant (steady operating conditions):
%   Rf(t) = EP_rate * t
%   Cleaning happens when Rf = Rf_clean
%   Therefore: t_clean = Rf_clean / EP_rate
%
% This is analytically exact for constant operating conditions.
% Much faster than running ODE45 8000 times.
% -----------------------------------------------------------------
t_clean = zeros(N, 1);
for i = 1:N
    if EP_feat(i) > 0
        % Time = threshold / rate (simple linear equation)
        t_clean(i) = min(8760, Rf_clean / EP_feat(i));
    else
        % EP_rate = 0 means operating below fouling threshold
        % Suppression wins → no fouling → runs full year safely
        t_clean(i) = 8760;
    end
end

% -----------------------------------------------------------------
% BUILD FEATURE MATRIX FOR MODEL B (9 features)
%
% 6 RAW SENSOR INPUTS:
%   T_bulk, u_flow, q_flux, rho_oil, mu_oil, D_tube
%
% 3 PHYSICS-DERIVED FEATURES:
%   Re_feat  — Reynolds number (flow regime)
%   Ts_feat  — Wall temperature in Kelvin (Arrhenius driver)
%   tau_feat — Wall shear stress (suppression driver)
%
% NOTE: We intentionally exclude EP_feat as an input.
% If we include it, the model just reads the answer directly —
% that would be "cheating" and would not generalize to new crudes
% where EP_feat from training crude parameters would be wrong.
% -----------------------------------------------------------------
X_phys = [T_bulk, u_flow, q_flux, rho_oil, mu_oil, D_tube, ...
          Re_feat, Ts_feat, tau_feat];

% Add 3% Gaussian noise to simulate real sensor measurement error
% Real refinery sensors drift, vibrate, and get contaminated.
% Training with noise makes the model robust to imperfect readings.
y = t_clean .* (1 + 0.03 * randn(N, 1));
y = max(100, min(8760, y));   % Clamp: minimum 100 hrs, maximum 1 year

% -----------------------------------------------------------------
% PRINT SUMMARY AND SAVE
% -----------------------------------------------------------------
fprintf('\nDataset Summary:\n');
fprintf('  Scenarios generated  : %d\n', N);
fprintf('  Features per scenario: %d (6 raw + 3 physics)\n', size(X_phys,2));
fprintf('  Min t_clean          : %.0f hours\n', min(y));
fprintf('  Max t_clean          : %.0f hours\n', max(y));
fprintf('  Mean t_clean         : %.0f hours\n', mean(y));
fprintf('  Never fouled (8760h) : %d scenarios (%.0f%%)\n', ...
        sum(y >= 8700), sum(y >= 8700)/N*100);

% Save ALL variables to ProjectData.mat
% The next scripts (Step2, Step3, Step4) will load this file
save('ProjectData.mat');
fprintf('\nStep 1 COMPLETE. Saved to ProjectData.mat\n');
fprintf('=========================================\n\n');