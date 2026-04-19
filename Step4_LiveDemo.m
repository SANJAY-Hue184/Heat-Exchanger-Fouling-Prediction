
% =========================================================================
% STEP 4: LIVE PREDICTION DEMO
% =========================================================================
% WHAT THIS SCRIPT DOES:
%   - You enter real refinery operating conditions
%   - The AI predicts how many hours before cleaning is needed
%   - Shows the physics calculation step-by-step
%   - Tells you whether it is in the safe or danger zone
%
% HOW TO USE:
%   1. Change the numbers in the "YOUR INPUT" section below
%   2. Press F5 to run
%   3. Read the prediction in the command window
%
% TRY THESE SCENARIOS TO SEE THE PHYSICS WORKING:
%   Scenario A (should foul FAST):  T=370, u=0.6, q=38000, rho=890, mu=0.004, D=0.022
%   Scenario B (should NEVER foul): T=220, u=2.8, q=12000, rho=800, mu=0.001, D=0.045
%   Scenario C (moderate):          T=340, u=1.5, q=25000, rho=850, mu=0.002, D=0.030
% =========================================================================

load('TrainedAI.mat');   % Load trained AI and normalization parameters

fprintf('=========================================\n');
fprintf('  STEP 4: Live Prediction Demo\n');
fprintf('=========================================\n');

% =================================================================
% >> YOUR INPUT — CHANGE THESE NUMBERS TO TEST NEW CONDITIONS <<
% =================================================================
In_T_bulk = 370;       % Bulk fluid temperature [°C]    (range: 200-380)
In_u_flow = 0.6;       % Flow velocity         [m/s]   (range: 0.5-3.0)
In_q_flux = 38000;     % Heat flux             [W/m²]  (range: 10000-40000)
In_rho    = 800;       % Oil density           [kg/m³] (range: 780-900)
In_mu     = 0.004;     % Oil viscosity         [Pa.s]  (range: 0.001-0.005)
In_D_tube = 0.022;     % Tube inner diameter   [m]     (range: 0.02-0.05)
% =================================================================

fprintf('\nYour Input Conditions:\n');
fprintf('  Bulk temperature : %.0f °C\n',  In_T_bulk);
fprintf('  Flow velocity    : %.2f m/s\n', In_u_flow);
fprintf('  Heat flux        : %.0f W/m²\n',In_q_flux);
fprintf('  Oil density      : %.0f kg/m³\n',In_rho);
fprintf('  Oil viscosity    : %.4f Pa.s\n', In_mu);
fprintf('  Tube diameter    : %.3f m\n',    In_D_tube);

% -----------------------------------------------------------------
% CALCULATE PHYSICS FEATURES FOR THE INPUT
% (same formulas as Step 1 — must be identical)
% -----------------------------------------------------------------
fprintf('\nPhysics Calculation:\n');

% Reynolds Number
In_Re = (In_rho * In_u_flow * In_D_tube) / In_mu;
fprintf('  Reynolds number Re  = %.0f', In_Re);
if In_Re > 10000
    fprintf('  ← fully turbulent (good for suppression)\n');
elseif In_Re > 4000
    fprintf('  ← transitional flow\n');
else
    fprintf('  ← laminar flow (WARNING: high fouling risk)\n');
end

% Prandtl Number (using same constants as training)
In_Pr = (In_mu * Cp_oil) / k_oil;
fprintf('  Prandtl number Pr   = %.1f\n', In_Pr);

% Nusselt Number (Dittus-Boelter)
In_Nu = 0.023 * In_Re^0.8 * In_Pr^0.33;
fprintf('  Nusselt number Nu   = %.0f\n', In_Nu);

% Heat Transfer Coefficient
In_h = In_Nu * k_oil / In_D_tube;
fprintf('  Heat transfer coeff = %.0f W/m²K\n', In_h);

% Wall Surface Temperature (MOST CRITICAL)
In_Ts = (In_T_bulk + 273.15) + In_q_flux / In_h;
fprintf('  Wall temperature Ts = %.1f K  (= %.1f °C)\n', In_Ts, In_Ts-273.15);
fprintf('  Wall excess above bulk = +%.1f °C\n', In_q_flux/In_h);

% Blasius Friction Factor
In_f = 0.316 * In_Re^(-0.25);

% Wall Shear Stress
In_tau = (In_f/2) * In_rho * In_u_flow^2;
fprintf('  Wall shear stress   = %.2f Pa\n', In_tau);

% Ebert-Panchal Fouling Rate (for context, NOT fed to model)
EP_check = alpha * In_Re^(-0.33) * exp(-Ea/(R_gas*In_Ts)) - gamma*In_tau;
fprintf('  EP net fouling rate = %.2e m²K/W per hour\n', EP_check);
if EP_check <= 0
    fprintf('  STATUS: BELOW FOULING THRESHOLD → No fouling expected!\n');
else
    fprintf('  STATUS: Above fouling threshold → Fouling will occur\n');
end

% -----------------------------------------------------------------
% BUILD INPUT ROW AND NORMALIZE
% Must use EXACTLY the same feature order as training (9 features)
% Must normalize with training set statistics (mu_X, sig_X)
% -----------------------------------------------------------------
My_Input_Row  = [In_T_bulk, In_u_flow, In_q_flux, In_rho, In_mu, In_D_tube, ...
                 In_Re, In_Ts, In_tau];
My_Input_Norm = (My_Input_Row - mu_X) ./ sig_X;

% -----------------------------------------------------------------
% GET PREDICTION FROM NEURAL NETWORK
% Network expects column vector input → transpose with '
% Output is normalized → denormalize with sig_y and mu_y
% Clamp result between 0 and 8760 hours
% -----------------------------------------------------------------
Pred_norm    = net(My_Input_Norm');
Final_Answer = max(0, min(8760, Pred_norm * sig_y + mu_y));

% -----------------------------------------------------------------
% DISPLAY RESULTS WITH INTERPRETATION
% -----------------------------------------------------------------
fprintf('\n=========================================\n');
fprintf('  AI PREDICTION RESULT\n');
fprintf('=========================================\n');
fprintf('  >> Predicted cleaning time: %.0f HOURS\n\n', Final_Answer);

% Convert to days and months for intuition
fprintf('  = %.0f days\n', Final_Answer/24);
fprintf('  = %.1f months\n', Final_Answer/720);
fprintf('  = %.2f years\n\n', Final_Answer/8760);

% Risk classification and recommendation
if Final_Answer < 1000
    fprintf('  RISK LEVEL: HIGH\n');
    fprintf('  The exchanger fouls very fast under these conditions.\n');
    fprintf('  RECOMMENDATION: Increase flow velocity (u_flow) or reduce\n');
    fprintf('  heat flux to lower the wall temperature Ts.\n');
elseif Final_Answer < 3000
    fprintf('  RISK LEVEL: MODERATE\n');
    fprintf('  Schedule maintenance at approximately %.0f hours.\n', Final_Answer*0.9);
    fprintf('  RECOMMENDATION: Monitor closely. Consider slightly higher flow.\n');
elseif Final_Answer < 7000
    fprintf('  RISK LEVEL: LOW\n');
    fprintf('  Good operating conditions. Schedule cleaning around %.0f hours.\n', Final_Answer*0.95);
elseif Final_Answer >= 7000
    fprintf('  RISK LEVEL: VERY LOW\n');
    fprintf('  Excellent conditions. Exchanger can operate for near-full year.\n');
    fprintf('  Standard annual maintenance schedule is appropriate.\n');
end

% Show which zone it falls in on the threshold map
fprintf('\n  Wall temperature: %.1f°C\n', In_Ts-273.15);
fprintf('  Reynolds number:  %.0f\n', In_Re);
if EP_check > 0
    t_physics = Rf_clean / EP_check;
    fprintf('\n  Physics model direct estimate: %.0f hours\n', min(8760, t_physics));
    fprintf('  AI model prediction          : %.0f hours\n', Final_Answer);
    fprintf('  Difference                   : %.0f hours (AI adds robustness from data)\n', ...
            abs(Final_Answer - min(8760, t_physics)));
else
    fprintf('\n  Physics: below fouling threshold → t_clean = 8760 hours\n');
    fprintf('  AI prediction: %.0f hours\n', Final_Answer);
end

fprintf('\n=========================================\n');
fprintf('TIP: Change the input numbers at the top of\n');
fprintf('this file and press F5 again to test a new scenario.\n\n');

fprintf('Try Scenario A (fast fouling):  T=370, u=0.6, q=38000\n');
fprintf('Try Scenario B (never fouls):   T=220, u=2.8, q=12000\n');
fprintf('Try Scenario C (moderate):      T=340, u=1.5, q=25000\n');
fprintf('=========================================\n');