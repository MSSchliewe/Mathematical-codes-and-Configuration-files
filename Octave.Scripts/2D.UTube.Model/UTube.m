% UTube.m - Estimator based on quadratic objective (RMSE + moments)

clearvars; close all; clc;

%% ---------------- Globals and initialization ----------------
global solver_call_count solver_total_time solver_last_print solver_fail_count global_start_time
global objective_eval_count objective_last_print verbose
solver_call_count = 0;
solver_total_time = 0;
solver_last_print = tic;
solver_fail_count = 0;
global_start_time = tic;
objective_eval_count = 0;
objective_last_print = tic;
verbose = true;

%% ---------------- Configuration (edit as needed) ----------------
% Files
synth_csv = 'quadratic_signal.csv';   % synthetic signal (generator output)
sim_csv   = 'ResultsSheet2.csv';      % SPH simulation data (target in optimize)
debug_csv = 'debug_calibration.csv';  % optional debug CSV from generator

% Mode: 'validation' (use synthetic as data) or 'optimize' (use ResultSheet as data)
estimation_mode = 'optimize';  % 'validation' or 'optimize'

% Option: detrend SPH target for optimize (useful when synth is baseline 0)
use_detrend_target_for_optimize = true;

% physical parameters (should match generator)
rho = 1000;              % kg/m^3
D = 0.006;               % m
L_c = 0.47;              % m characteristic length used to convert per-length -> 3D

% drag exponent
use_alpha = 1.0;         % B * |v|^alpha * v

% numerical
dt_substeps = 400;
dt_substeps_coarse = 80;
max_abs_state = 1e6;
ds_coarse = 4;           % downsampling for coarse scan (overridden in validation)

% B search
B_min = 1e-12; B_max = 1e6;
B_safe_max = 1e6;
n_Bscan = 24;

% objective weights
w_z = 1.0; w_2 = 0.5; w_3 = 1.0;

%% ---------------- Defensive initializations (avoid undefined) ----------
% plotting / intermediate variables
baseline_shift_y0 = NaN;
y_final = [];
y_final_proc = [];
y_final_plot = [];
y_final_plot_proc = [];
z_synth_plot = [];
t_synth_shifted = [];
z_raw_orig = [];
z_raw = [];
z_proc = [];
m2_data = NaN; m3_data = NaN;
env = [];

%% ---------------- Load synthetic CSV (always) ----------------
if ~exist(synth_csv,'file')
  error('Synthetic file %s not found.', synth_csv);
end
Ms = importdata(synth_csv);
if isstruct(Ms) && isfield(Ms,'data'), Ms = Ms.data; end
t_synth = Ms(:,1); z_synth = Ms(:,2);

% set defaults for synth plot/time
z_synth_plot = z_synth;
t_synth_shifted = t_synth;

%% ---------------- Load target data depending on mode ----------------
if strcmpi(estimation_mode,'optimize')
  if ~exist(sim_csv,'file'), error('Result file %s not found (optimize mode).', sim_csv); end
  Mr = importdata(sim_csv);
  if isstruct(Mr) && isfield(Mr,'data'), Mr = Mr.data; end
  t = Mr(:,1);
  if size(Mr,2) < 2, error('Target file %s does not contain a second column with z.', sim_csv); end
  z_raw_orig = Mr(:,2);
else
  % validation: use synthetic as SPH target
  t = t_synth;
  z_raw_orig = z_synth;
end

% basic checks
dt = median(diff(t));
N = numel(t);
if dt <= 0, error('Invalid time vector'); end
fprintf('Using data: N=%d dt=%.12g  (mode=%s)\n', N, dt, estimation_mode);

%% ---------------- Ensure z_raw variables exist (defensive) ----------------
if isempty(z_raw_orig)
  error('z_raw_orig is empty after loading target data. Check input files.');
end
% default z_raw is original; may be replaced by detrended version below
z_raw = z_raw_orig;

%% ---------------- Optionally detrend SPH target for optimize (baseline 0) ----
if strcmpi(estimation_mode,'optimize') && exist('use_detrend_target_for_optimize','var') && use_detrend_target_for_optimize
  z_raw = detrend(z_raw_orig, 'linear');   % baseline ~0 for optimization and plotting
  fprintf('Optimize mode: using detrended SPH target (baseline ~0) for optimization and plots\n');
else
  z_raw = z_raw_orig;
end

%% ---------------- Preprocessing (compute z_proc, moments) ----------------
% baseline_mode kept simple; z already set to used-for-optimization version
z_proc = detrend(z_raw, 'linear');   % processed signal used for metrics
vz_proc = gradient(z_proc, dt);
m2_data = mean(vz_proc.^2);
m3_data = mean(abs(vz_proc).^3);
env = abs(hilbert(z_proc));
fprintf('Data moments: m2_data=%.12g  m3_data=%.12g\n', m2_data, m3_data);

% initial conditions for solver (based on the signal used for optimization)
y0_guess = z_raw(1);
if N>=2, v0_guess = (z_raw(2) - z_raw(1)) / dt; else v0_guess = 0; end

%% ---------------- Estimate dominant frequency (validation override) ----------------
if strcmpi(estimation_mode,'validation')
  omega_n = sqrt(2.0 * 9.81 / L_c);
  fdom = omega_n / (2*pi);
  period_est = 2*pi / omega_n;
  fprintf('Validation mode: forcing omega from generator: fdom=%.6g Hz  omega=%.6g rad/s\n', fdom, omega_n);
else
  if exist('hann','file')~=2
    hann_win = @(Nw) (0.5 - 0.5*cos(2*pi*(0:Nw-1)'/(Nw-1)));
    win = hann_win(N);
  else
    win = hann(N);
  end
  Nfft = 2^nextpow2(max(1024, 2*N));
  Y = fft(z_proc .* win, Nfft);
  P = abs(Y(1:floor(Nfft/2)+1)).^2;
  f = (0:floor(Nfft/2))' / (Nfft*dt);
  [~, imax] = max(P(2:end)); imax = imax + 1;
  fdom = f(imax);
  omega_n = 2*pi*fdom;
  period_est = 2*pi / omega_n;
  fprintf('Estimated omega: fdom=%.6g Hz  omega=%.6g rad/s\n', fdom, omega_n);
end
omega2 = omega_n^2;

%% ---------------- Prepare coarse time/signal vectors ----------------
if strcmpi(estimation_mode,'validation')
  ds_coarse = 1;
  fprintf('Validation mode: disabling coarse downsampling (ds_coarse=1)\n');
end
t_coarse = t(1:ds_coarse:end);
z_coarse = z_proc(1:ds_coarse:end);

%% ---------------- Coarse B-scan (RMSE temporal) ----------------
if m3_data <= 0
  B_guess_from_data = 1.0;
else
  B_guess_from_data = (m2_data) / max(eps, m3_data);
end
B_guess_from_data = min(max(B_guess_from_data, B_min), B_safe_max);
Bs = logspace(log10(max(B_min, B_guess_from_data*1e-2)), log10(max(B_min, min(B_safe_max, B_guess_from_data*1e2))), n_Bscan);

fprintf('Running coarse B-scan (RMSE)...\n');
ok = false(size(Bs)); Ez_vals = nan(size(Bs));
for i=1:numel(Bs)
  Bi = Bs(i);
  try
    ytmp = call_solver(Bi, 0, omega2, y0_guess, v0_guess, t_coarse, use_alpha, rho, D, dt_substeps_coarse, max_abs_state);
    ytmp_proc = detrend(ytmp - mean(ytmp), 'linear');
    Ez_vals(i) = sqrt(mean((ytmp_proc - z_coarse).^2));
    ok(i) = true;
  catch
    ok(i) = false;
    Ez_vals(i) = Inf;
    if verbose, fprintf('[scan] B=%.3g simulation failed\n', Bi); end
  end
end
valid_idx = find(ok);
if isempty(valid_idx)
  B_coarse_best = B_guess_from_data;
else
  [~, im] = min(Ez_vals(valid_idx));
  B_coarse_best = Bs(valid_idx(im));
end
fprintf('Coarse best B (RMSE) = %.12g\n', B_coarse_best);

%% ---------------- Select B depending on mode ----------------
if strcmpi(estimation_mode,'validation')
  B_opt = B_coarse_best;
  fprintf('Validation mode: using B_coarse_best = %.12g as B_opt\n', B_opt);
else
  fprintf('Optimize mode: minimizing quadratic objective J(B)\n');
  dt_coarse = median(diff(t_coarse));
  dt_sub_coarse = max(1, round(dt_substeps_coarse / ds_coarse));
  Jfun = @(logB) local_J_quad_logB(logB, t_coarse, z_coarse, dt_coarse, dt_sub_coarse, ...
                                   m2_data, m3_data, w_z, w_2, w_3, B_min, B_safe_max, ...
                                   omega2, y0_guess, v0_guess, use_alpha, rho, D, max_abs_state);
  x0 = log(max(B_min, min(B_safe_max, B_coarse_best)));
  opts = optimset('Display','iter','TolX',1e-8,'TolFun',1e-8,'MaxFunEvals',2000);
  try
    xopt = fminsearch(@(x) Jfun(x), x0, opts);
    if isempty(xopt) || any(~isfinite(xopt))
      warning('fminsearch returned invalid xopt; using B_coarse_best.');
      B_quad_opt = B_coarse_best;
    else
      B_quad_opt = exp(xopt(1));
    end
  catch ME
    warning('fminsearch failed: %s. Using B_coarse_best.', ME.message);
    B_quad_opt = B_coarse_best;
  end
  B_opt = B_quad_opt;
  fprintf('Optimize mode: selected B_opt = %.12g (quadratic objective)\n', B_opt);
end

%% ---------------- Final forward with B_opt (high resolution) ----------------
try
  y_final = call_solver(B_opt, 0, omega2, y0_guess, v0_guess, t, use_alpha, rho, D, dt_substeps, max_abs_state);
  y_final_proc = detrend(y_final - mean(y_final), 'linear');
  rmse_final = sqrt(mean((y_final_proc - z_proc).^2));
  vz_final = gradient(y_final_proc, dt);
  m2_final = mean(vz_final.^2); m3_final = mean(abs(vz_final).^3);
catch ME
  warning('Final simulation failed: %s', ME.message);
  y_final = nan(size(t)); y_final_proc = nan(size(t));
  rmse_final = NaN; m2_final = NaN; m3_final = NaN;
end

%% ---------------- Ensure y_final_plot exists (defensive) ----------------
if ~exist('y_final_plot','var') || isempty(y_final_plot)
  if exist('y_final_proc','var') && ~isempty(y_final_proc)
    % default plotting convention: if optimize used detrend, we will adjust below
    y_final_plot = y_final_proc;
    y_final_plot_proc = detrend(y_final_plot - mean(y_final_plot), 'linear');
  elseif exist('y_final','var') && ~isempty(y_final)
    y_final_plot = y_final;
    y_final_plot_proc = detrend(y_final_plot - mean(y_final_plot), 'linear');
  else
    y_final_plot = NaN(size(t));
    y_final_plot_proc = NaN(size(t));
  end
end

%% ---------------- Prepare synth and model traces for plotting (robust) ----------
% Two plotting conventions:
% - optimize + use_detrend_target_for_optimize: baseline 0 for all plotted traces
% - otherwise: align synth and model to original SPH baseline (z_raw_orig(1))

if strcmpi(estimation_mode,'optimize') && exist('use_detrend_target_for_optimize','var') && use_detrend_target_for_optimize
  % ensure synth baseline 0 (align first sample to 0)
  z_synth_plot = z_synth - z_synth(1);
  % ensure model plot baseline 0 (align first sample to 0)
  if exist('y_final_plot','var') && ~isempty(y_final_plot) && numel(y_final_plot)==numel(t)
    y_final_plot = y_final_plot - y_final_plot(1);
    y_final_plot_proc = detrend(y_final_plot - mean(y_final_plot), 'linear');
  end
  baseline_shift_y0 = 0;
else
  % align to original SPH baseline for plotting
  base_sim_initial = z_raw_orig(1);
  baseline_shift_y0 = base_sim_initial - z_synth(1);
  z_synth_shifted = z_synth + baseline_shift_y0;
  % optional lag correction (safe: only if signals have same sampling)
  try
    [c2, lags2] = xcorr(z_raw - mean(z_raw), z_synth - mean(z_synth));
    [~, imax2] = max(abs(c2)); lag_synth = lags2(imax2);
  catch
    lag_synth = 0;
  end
  dt_local = median(diff(t));
  if lag_synth ~= 0
    time_shift = lag_synth * dt_local;
    t_synth_shifted = t_synth - time_shift;
  else
    t_synth_shifted = t_synth;
  end
  if ~isequal(numel(t_synth_shifted), numel(t)) || any(abs(t_synth_shifted - t) > 1e-12)
    z_synth_plot = interp1(t_synth_shifted, z_synth_shifted, t, 'linear', 'extrap');
  else
    z_synth_plot = z_synth_shifted;
  end
  if exist('y_final_plot','var') && ~isempty(y_final_plot)
    y_final_plot = y_final_plot + base_sim_initial;
    y_final_plot_proc = detrend(y_final_plot - mean(y_final_plot), 'linear');
  end
end

%% ---------------- Linear equivalent for reporting (reference only) ----------------
if isfinite(m2_final) && m2_final>0 && isfinite(m3_final)
  mu_equiv = (rho * D^2 / 32.0) * B_opt * (m3_final / max(eps, m2_final));
  nu_equiv = mu_equiv / rho;
else
  mu_equiv = NaN; nu_equiv = NaN;
end
if isfinite(mu_equiv) && L_c>0
  mu_equiv_per_length = mu_equiv / L_c;
  nu_equiv_per_length = mu_equiv_per_length / rho;
else
  mu_equiv_per_length = NaN;
  nu_equiv_per_length = NaN;
end

note_2D = 'Results from 2D simulation are per unit depth; convert to 3D using L_c = V/A.';
fprintf('\nFinal diagnostics:\n');
fprintf('estimation_mode = %s\n', estimation_mode);
fprintf('B_opt=%.12g  B_coarse_best=%.12g\n', B_opt, B_coarse_best);
fprintf('mu_equiv=%.12g  nu_equiv=%.12g\n', mu_equiv, nu_equiv);
fprintf('Uc_data=%.6g  Uc_model=%.6g  RMSE_final=%.6g\n', sqrt(max(0,m2_data)), sqrt(max(0,m2_final)), rmse_final);
fprintf('m2_data=%.6g  m2_model=%.6g  m3_data=%.6g  m3_model=%.6g\n', m2_data, m2_final, m3_data, m3_final);
fprintf('[Summary] solver calls=%d  total_solver_time=%.3fs\n', solver_call_count, solver_total_time);
fprintf('L_c = %.6g m\n', L_c);
fprintf('mu_equiv_per_length = %.12g\n', mu_equiv_per_length);
if ~isnan(baseline_shift_y0)
  fprintf('baseline_shift_y0 (plot convention) = %.12g\n', baseline_shift_y0);
else
  fprintf('baseline_shift_y0 (plot convention) = NaN\n');
end
fprintf('Note: %s\n', note_2D);

%% ---------------- Export and plots ----------------
out_dir = '.';
fid = fopen(fullfile(out_dir,'UTube_timeseries.csv'),'w');
fprintf(fid,'t,z_raw_used_for_opt,z_proc,y_final_plot\n');
for i=1:numel(t)
  v4 = NaN;
  if exist('y_final_plot','var') && numel(y_final_plot)==numel(t)
    v4 = y_final_plot(i);
  elseif exist('y_final','var') && numel(y_final)==numel(t)
    v4 = y_final(i);
  end
  fprintf(fid,'%.12g,%.12g,%.12g,%.12g\n', t(i), z_raw(i), z_proc(i), v4);
end
fclose(fid);

fid = fopen(fullfile(out_dir,'UTube_summary.txt'),'w');
fprintf(fid,'UTube summary\n=============\n\n');
fprintf(fid,'estimation_mode = %s\n', estimation_mode);
fprintf(fid,'B_opt = %.12g\n', B_opt);
fprintf(fid,'B_coarse_best = %.12g\n', B_coarse_best);
fprintf(fid,'mu_equiv = %.12g\n', mu_equiv);
fprintf(fid,'nu_equiv = %.12g\n', nu_equiv);
fprintf(fid,'Uc_data = %.12g  Uc_model = %.12g\n', sqrt(max(0,m2_data)), sqrt(max(0,m2_final)));
fprintf(fid,'RMSE_final = %.12g\n', rmse_final);
fprintf(fid,'m2_data = %.12g  m2_model = %.12g\n', m2_data, m2_final);
fprintf(fid,'m3_data = %.12g  m3_model = %.12g\n', m3_data, m3_final);
if ~isnan(baseline_shift_y0)
  fprintf(fid,'baseline_shift_y0 = %.12g\n', baseline_shift_y0);
else
  fprintf(fid,'baseline_shift_y0 = NaN (plot convention)\n');
end
fclose(fid);

% ---------------- Time series (mean-zero only) ----------------
% Substituir o bloco de plot atual por este. Garante variáveis definidas e salva z_synth_plot/y_final_plot.

% Defensive checks
if ~exist('t','var'), error('time vector t not found before plotting.'); end
if ~exist('z_synth','var'), error('z_synth not found before plotting.'); end
if ~exist('z_raw','var') && exist('z_raw_orig','var'), z_raw = z_raw_orig; end
if ~exist('z_raw','var'), error('z_raw not found before plotting.'); end

% Ensure column vectors
t = t(:);
z_synth = z_synth(:);
z_raw = z_raw(:);

% Build mean-zero traces (convenção escolhida)
sph_plot = z_raw - mean(z_raw);                % SPH used-for-opt (mean ~0)
synth_plot = z_synth - mean(z_synth);          % synth mean-zero
if exist('y_final_proc','var') && numel(y_final_proc)==numel(t)
  model_plot = y_final_proc(:) - mean(y_final_proc(:));
elseif exist('y_final','var') && numel(y_final)==numel(t)
  model_plot = y_final(:) - mean(y_final(:));
else
  model_plot = NaN(size(t));
end

% Safe diagnostics
sph_start = NaN; synth_start = NaN; model_start = NaN;
if ~isempty(sph_plot), sph_start = sph_plot(1); end
if ~isempty(synth_plot), synth_start = synth_plot(1); end
if ~isempty(model_plot), model_start = model_plot(1); end
fprintf('Mean-zero conv: SPH start=%.12g  synth start=%.12g  model start=%.12g\n', sph_start, synth_start, model_start);
fprintf('Mean-zero means: SPH mean=%.12g  synth mean=%.12g  model mean=%.12g\n', ...
        mean(sph_plot(~isnan(sph_plot))), mean(synth_plot(~isnan(synth_plot))), mean(model_plot(~isnan(model_plot))));

% Plot (single, definitive)
figure('Name','Time series comparison (mean-zero)','NumberTitle','off'); clf;
plot(t, sph_plot, '-k', 'LineWidth', 1.0); hold on;
plot(t, model_plot, '-r', 'LineWidth', 1.0);
hold off;
legend('SPH (mean 0)', 'Calibrated model (mean 0)', 'Location', 'northeast');
xlabel('t (s)'); ylabel('y (m)');
title('Time series comparison (mean-zero convention)');
grid on;

% Persist final plotting variables to avoid accidental overwrites later
z_synth_plot = synth_plot;
y_final_plot = model_plot;

% Sanity prints (immediately before plotting)
synth_start = synth_plot(1);
model_start = NaN;
sph_start = NaN;
if ~isempty(sph_plot), sph_start = sph_plot(1); end
if ~isempty(model_plot), model_start = model_plot(1); end
fprintf('Sanity: SPH start=%.12g  synth_plot start=%.12g  model_plot start=%.12g\n', sph_start, synth_start, model_start);
fprintf('Sanity means: SPH mean=%.12g  synth mean=%.12g  model mean=%.12g\n', mean(sph_plot(~isnan(sph_plot))), mean(synth_plot(~isnan(synth_plot))), mean(model_plot(~isnan(model_plot))));

% --- Figure 2: envelope (instantaneous amplitude) ---
figure('Name','Envelope comparison','NumberTitle','off'); clf;
if exist('y_final_plot','var') && ~isempty(y_final_plot)
  env_model = abs(hilbert(detrend(y_final_plot,'linear')));
else
  env_model = abs(hilbert(detrend(y_final,'linear')));
end
plot(t, env, '--k', t, env_model, '-r');
legend('Env SPH (used for opt/plot)', 'Env Model', 'Location', 'northeast');
xlabel('t (s)'); ylabel('A (m)'); title('Envelope (instantaneous amplitude)'); grid on;

% --- Figure 3: spectral comparison (normalized FFT) ---
figure('Name','Spectral comparison','NumberTitle','off'); clf;
Nfft2 = 2^nextpow2(max(1024, 2*N));
Fd = fft(detrend(z_proc), Nfft2);
if exist('y_final_plot_proc','var') && ~isempty(y_final_plot_proc)
  Fm = fft(detrend(y_final_plot_proc), Nfft2);
else
  Fm = fft(detrend(y_final_proc), Nfft2);
end
fvec = (0:Nfft2/2)'/(Nfft2*dt);
plot(fvec, abs(Fd(1:Nfft2/2+1))/max(abs(Fd(1:Nfft2/2+1))), '--k'); hold on;
plot(fvec, abs(Fm(1:Nfft2/2+1))/max(abs(Fm(1:Nfft2/2+1))), '-r'); hold off;
xlim([0, 5*(1/period_est)]);
xlabel('Freq (Hz)'); ylabel('Normalized FFT magnitude'); legend('FFT SPH (used for opt/plot)', 'FFT Model', 'Location', 'northeast');
title('Spectral comparison'); grid on;

fprintf('UTube finished\n');

%% ---------------- Local objective helper (portable) ----------------
function J = local_J_quad_logB(logB, t_coarse, z_coarse, dt_coarse, dt_sub_coarse, ...
                               m2_data, m3_data, w_z, w_2, w_3, B_min_local, B_safe_max, ...
                               omega2_local, y0_local, v0_local, use_alpha_local, rho_local, D_local, max_abs_state_local)
  persistent eval_count
  if isempty(eval_count), eval_count = 0; end
  eval_count = eval_count + 1;

  Btry = exp(logB);
  if Btry < B_min_local || Btry > B_safe_max
    J = 1e6; return;
  end
  try
    ysim = rk4_quad_only(Btry, 0, omega2_local, y0_local, v0_local, t_coarse, use_alpha_local, rho_local, D_local, dt_sub_coarse, max_abs_state_local);
  catch
    J = 1e5; return;
  end
  ysim_proc = detrend(ysim - mean(ysim), 'linear');
  Ez = sqrt(mean((ysim_proc - z_coarse).^2));
  vz = gradient(ysim_proc, dt_coarse);
  m2m = mean(vz.^2); m3m = mean(abs(vz).^3);
  Em2 = abs(m2m - m2_data) / max(eps, m2_data);
  Em3 = abs(m3m - m3_data) / max(eps, m3_data);
  J = w_z * Ez + w_2 * Em2 + w_3 * Em3;

  if mod(eval_count,10)==0
    fprintf('[J] eval=%d  B=%.6g  Ez=%.6g  Em2=%.6g  Em3=%.6g  J=%.6g\n', eval_count, Btry, Ez, Em2, Em3, J);
  end
end

%% ---------------- Local solver wrapper ----------------
function y = call_solver(B_local, mu_local, omega2_local, y0_local, v0_local, tvec_local, alpha_local, rho_local, D_local, dt_substeps_local, max_abs_state_local)
  global solver_call_count solver_total_time solver_last_print global_start_time solver_fail_count verbose
  tstart = tic;
  y = rk4_quad_only(B_local, mu_local, omega2_local, y0_local, v0_local, tvec_local, alpha_local, rho_local, D_local, dt_substeps_local, max_abs_state_local);
  elapsed = toc(tstart);
  solver_call_count = solver_call_count + 1;
  solver_total_time = solver_total_time + elapsed;
  if verbose && (toc(solver_last_print)>5 || mod(solver_call_count,50)==0)
    fprintf('[Solver] calls=%d  last=%.3fs  total=%.3fs\n', solver_call_count, elapsed, solver_total_time);
    solver_last_print = tic;
  end
end

%% ---------------- RK4 solver (local) ----------------
function y = rk4_quad_only(B_local, mu_local, omega2_local, y0_local, v0_local, tvec_local, alpha_local, rho_local, D_local, dt_substeps_local, max_abs_state_local)
  global solver_fail_count
  Nloc = numel(tvec_local);
  y = zeros(Nloc,1); v = zeros(Nloc,1);
  y(1) = y0_local; v(1) = v0_local;
  bailout_thresh = min(max_abs_state_local, 1e6);
  for ii = 2:Nloc
    dtk = (tvec_local(ii)-tvec_local(ii-1)) / dt_substeps_local;
    s = [y(ii-1); v(ii-1)];
    for ss = 1:dt_substeps_local
      vcur = s(2);
      k1 = [ vcur;
             -omega2_local*s(1) - B_local * (abs(vcur)^alpha_local) * vcur ];
      s2 = s + 0.5*dtk*k1;
      v2 = s2(2);
      k2 = [ v2;
             -omega2_local*s2(1) - B_local * (abs(v2)^alpha_local) * v2 ];
      k3 = k2;
      s4 = s + dtk*k3;
      v4 = s4(2);
      k4 = [ v4;
             -omega2_local*s4(1) - B_local * (abs(v4)^alpha_local) * v4 ];
      s = s + (dtk/6)*(k1 + 2*k2 + 2*k3 + k4);
      if ~isfinite(s(1)) || ~isfinite(s(2)) || abs(s(1))>max_abs_state_local || abs(s(2))>max_abs_state_local
        solver_fail_count = solver_fail_count + 1;
        error('RK4 blowup at substep');
      end
      if abs(s(1))>bailout_thresh || abs(s(2))>bailout_thresh
        solver_fail_count = solver_fail_count + 1;
        error('RK4 early bailout: state exceeded threshold');
      end
    end
    y(ii) = s(1); v(ii) = s(2);
  end
end
