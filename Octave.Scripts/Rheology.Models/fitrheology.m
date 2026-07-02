% fitrheology.m
% Input: vector gamma_dot (s^-1) and tau (Pa)
% Output: fitted parameters, SSE, RMSE and comparative plot

clear; close all; clc;

% === USER CONFIGURATION ===
do_bootstrap = false; % true to enable bootstrap (may take time)
nboot = 200;          % number of bootstrap resamples if enabled
m_grid_points = 40;   % grid points in logspace for m
rel_sse_threshold = 0.05; % minimum relative SSE reduction to consider m "identifiable"

% Raw data
data = [
    0.155829283457453   21.9334954385007;
    0.165195272868596   22.4800823595738;
    0.190035160350059   21.689461135092;
    0.344658821879563   23.2474411580064;
    0.378396259109734   23.8267677911355;
    0.639858158757321   25.3107007459033;
    0.639858158757321   29.0774854816382;
    1.27407966603975    27.9921300034974;
    1.5357192972781 31.5168127671062;
    2.65820200520645    30.2725906372506
   ];
  gamma = data(:,1);
  tau = data(:,2);

valid = isfinite(gamma) & isfinite(tau) & (gamma>0);
gamma = gamma(valid); tau = tau(valid);
ng = numel(gamma);

% === MODELS =====================
% Bingham (non-regularized): tau = tau_y + eta_p * gamma
bing = @(p,g) p(1) + p(2) .* g; % p = [tau_y, eta_p]

% Herschel-Bulkley (non-regularized): tau = tau_y + K * gamma^n
hb = @(p,g) p(1) + p(2) .* (g .^ p(3)); % p = [tau_y, K, n]

% Papanastasiou regularized Herschel-Bulkley:
% tau = tau_y * (1 - exp(-m * gamma)) + K * gamma^n
papa = @(p,g) p(1) .* (1 - exp(-p(4) .* g)) + p(2) .* (g .^ p(3));
% p = [tau_y, K, n, m] with constraints tau_y>=0, K>0, n>0, m>0

% SSE helpers
sse = @(model_p, model_fun) sum((tau - model_fun(model_p, gamma)).^2);

% === INITIAL GUESSES and base optimization ===
% Bingham initial
tau_y0 = max(min(tau) - 0.01*abs(min(tau)), 0);
eta_p0 = max((max(tau)-min(tau)) / (max(gamma)-min(gamma) + eps), eps);
p0_bing = [tau_y0, eta_p0];

% HB initial
n0 = 0.6;
K0 = max(median((tau - tau_y0) ./ (gamma.^n0)), eps);
p0_hb = [tau_y0, K0, n0];

% Papanastasiou initial: use HB guesses + m0
m0 = 10; % reasonable guess; subsequent sweep will evaluate
p0_papa = [tau_y0, K0, n0, m0];

opts = optimset('TolX',1e-8,'TolFun',1e-8,'MaxIter',1e4,'MaxFunEvals',1e4,'Display','off');

% reparameterizations for strictly positive parameters via logs
% optimize q = [tau_y, log(K), log(n), log(m)] for Papanastasiou
obj_hb_log = @(q) sse([q(1), exp(q(2)), exp(q(3))], hb);
q0_hb = [p0_hb(1), log(max(p0_hb(2),eps)), log(max(p0_hb(3),1e-2))];
qopt_h = fminsearch(obj_hb_log, q0_hb, opts);
ph_opt = [qopt_h(1), exp(qopt_h(2)), exp(qopt_h(3))];
SSE_h = sse(ph_opt, hb);

obj_bing_log = @(q) sse([q(1), exp(q(2))], bing);
q0_bing = [p0_bing(1), log(max(p0_bing(2),eps))];
qopt_b = fminsearch(obj_bing_log, q0_bing, opts);
pb_opt = [qopt_b(1), exp(qopt_b(2))];
SSE_b = sse(pb_opt, bing);

% === 1) GRID SWEEP IN m (conditioned Papanastasiou) ===
% choose interval for m: from m_min to m_max in logspace
m_min = 1e-3; m_max = 1e4;
m_grid = logspace(log10(m_min), log10(m_max), m_grid_points);
SSE_grid = zeros(size(m_grid));
p_grid = zeros(numel(m_grid),4); % store conditional parameters

for i = 1:numel(m_grid)
  m_fix = m_grid(i);
  % optimize only tau_y, K, n with m fixed (reparam: tau_y free, logK, logn)
  obj_cond = @(q) sse([q(1), exp(q(2)), exp(q(3)), m_fix], papa);
  q0 = [p0_papa(1), log(max(p0_papa(2),eps)), log(max(p0_papa(3),1e-2))];
  qopt = fminsearch(obj_cond, q0, opts);
  p_cond = [qopt(1), exp(qopt(2)), exp(qopt(3)), m_fix];
  p_grid(i,:) = p_cond;
  SSE_grid(i) = sse(p_cond, papa);
end

% locate best m in the grid
[SSmin_grid, idx_min] = min(SSE_grid);
m_best_grid = m_grid(idx_min);
p_best_grid = p_grid(idx_min,:);

% === 2) FREE OPTIMIZATION (all parameters) using grid guess ===
% reparameterize everything: q = [tau_y, log(K), log(n), log(m)]
q0_all = [p_best_grid(1), log(max(p_best_grid(2),eps)), log(max(p_best_grid(3),1e-2)), log(max(p_best_grid(4),1e-6))];
obj_all_log = @(q) sse([q(1), exp(q(2)), exp(q(3)), exp(q(4))], papa);
qopt_all = fminsearch(obj_all_log, q0_all, opts);
popt_all = [qopt_all(1), exp(qopt_all(2)), exp(qopt_all(3)), exp(qopt_all(4))];
SSE_papa_full = sse(popt_all, papa);
RMSE_papa = sqrt(SSE_papa_full/ng);

% === compare models (SSE) ===
RMSE_b = sqrt(SSE_b/ng); RMSE_h = sqrt(SSE_h/ng);
fprintf('\nBingham model: tau_y=%.6g Pa, eta_p=%.6g, SSE=%.6g, RMSE=%.6g\n', pb_opt(1), pb_opt(2), SSE_b, RMSE_b);
fprintf('HB model: tau_y=%.6g Pa, K=%.6g, n=%.6g, SSE=%.6g, RMSE=%.6g\n', ph_opt(1), ph_opt(2), ph_opt(3), SSE_h, RMSE_h);
fprintf('Papanastasiou-HB (optimized): tau_y=%.6g Pa, K=%.6g, n=%.6g, m=%.6g, SSE=%.6g, RMSE=%.6g\n', popt_all(1), popt_all(2), popt_all(3), popt_all(4), SSE_papa_full, RMSE_papa);

% === ASSESSMENT OF IDENTIFIABILITY OF m (simple criterion) ===
% compare SSE of best papa (SSE_papa_full) with SSE of HB (SSE_h)
% and also with SSE of papa with very large m (approaches Bingham/HB ideal)
% estimator: relative reduction compared to best model without regularizer
rel_red_vs_hb = (SSE_h - SSE_papa_full) / (SSE_h + eps);
rel_red_vs_bh = (min(SSE_b,SSE_h) - SSE_papa_full) / (min(SSE_b,SSE_h) + eps);

% check width of the valley in SSE_grid to assess curvature
% normalize SSE_grid and measure ratio between SSE at top and SSE minimum
SSEnorm = SSE_grid / (min(SSE_grid)+eps);
% find width where SSE <= (1 + tol) * min
tol = 0.10; % 10% SSE increase from minimum
idx_band = find(SSE_grid <= (1+tol)*SSmin_grid);
band_width = m_grid(max(idx_band)) / m_grid(min(idx_band));

fprintf('\nRelative SSE reduction by including Papanastasiou vs HB: %.3g\n', rel_red_vs_hb);
fprintf('Relative SSE reduction vs best among Bingham/HB: %.3g\n', rel_red_vs_bh);
fprintf('SSE(m) profile: best m_grid = %.3g, band (<=+%.0f%%) width = %.3g times\n', m_best_grid, tol*100, band_width);

% simple decision: if relative reduction vs HB > rel_sse_threshold and band not too wide,
% consider m identifiable
m_estimavel = (rel_red_vs_hb > rel_sse_threshold) && (band_width < 1e2);

if m_estimavel
  fprintf('\nParameter m is considered identifiable from the data.\n');
  fprintf('Estimated m: %.6g (SSE min on grid: %.6g; SSE optimized: %.6g)\n', popt_all(4), SSmin_grid, SSE_papa_full);
else
  fprintf('\nParameter m is NOT clearly identifiable: insufficient SSE gain or very flat profile.\n');
  fprintf('m (best on grid) = %.6g; band width signals low sensitivity.\n', m_best_grid);
end

% ------------------ Additional rule: prefer HB if m not identifiable ------------------
% (should be placed immediately after computing m_estimavel)

% if m is not identifiable, do not consider Papanastasiou as candidate, unless
% the gain is substantial and forcing a test is desired (here prefer HB)
if ~m_estimavel
  fprintf('m is NOT identifiable from the data -> prefer Herschel-Bulkley over Papanastasiou\n');
  consider_papanastasiou = false;
else
  fprintf('m appears identifiable -> Papanastasiou remains a candidate\n');
  consider_papanastasiou = true;
end
% ---------------------------------------------------------------------------------------
% === Diagnostic PLOTS ===
% 1) fit curves with models
gplot = logspace(log10(min(gamma)/1.5), log10(max(gamma)*1.5), 300);
figure('position',[100 100 900 500]);
loglog(gamma, tau, 'ko', 'markerfacecolor','k'); hold on;
loglog(gplot, bing(pb_opt,gplot), '-b', 'linewidth',1.6);
loglog(gplot, hb(ph_opt,gplot), '-g', 'linewidth',1.6);
loglog(gplot, papa(popt_all,gplot), '-r', 'linewidth',1.8);
legend('data','Bingham','HB','Papanastasiou-HB','location','northwest');
xlabel('Shear rate \gammȧ (s^{-1})'); ylabel('Shear stress \tau (Pa)');
title('Fits: Bingham (blue), HB (green), Papanastasiou-HB (red)');
grid on;

% 2) SSE profile vs m (grid)
figure('position',[100 650 700 350]);
loglog(m_grid, SSE_grid, 'ko-','markerfacecolor','k'); hold on;
plot(popt_all(4), SSE_papa_full, 'pr','markersize',10,'markerfacecolor','r');
xlabel('m (Papanastasiou)'); ylabel('Conditional SSE');
title('Conditioned SSE(m) profile (logspace sweep)');
grid on;

% 3) residuals of best papa
figure('position',[820 650 700 300]);
semilogx(gamma, tau - papa(popt_all,gamma), 'or'); hold on;
xl = xlim; plot(xl, [0 0], '--k'); xlim(xl);
xlabel('\gammȧ (s^{-1})'); ylabel('residual (Pa)'); title('Papanastasiou-HB Residuals');

% === BOOTSTRAP (optional) for CI of m ===
if do_bootstrap
  fprintf('\nBootstrap started (nboot = %d) — this may take time...\n', nboot);
  m_boot = nan(nboot,1);
  parfor b = 1:nboot
    idx = randi(ng, ng, 1);
    gb = gamma(idx); tb = tau(idx);
    % fit conditioned on grid for initial guess
    % reuse optimization routine but with bootstrap data
    SSE_grid_b = zeros(size(m_grid));
    p_grid_b = zeros(numel(m_grid),4);
    for i = 1:numel(m_grid)
      m_fix = m_grid(i);
      obj_cond_b = @(q) sum((tb - ([q(1), exp(q(2)), exp(q(3)), m_fix].*(1) - 0*0)).^2);
      % simplification: use same conditional obj with gb,tb (implement fully as needed)
      % For brevity, here a full bootstrap implementation is not provided
    end
    % For simplicity in this sketch, mark NA; implement full bootstrap as required
    m_boot(b) = NaN;
  end
  % compute quantiles and report (implement)
  fprintf('Bootstrap not fully implemented in this sketch. Enable and adjust per dataset size.\n');
end
% ------------------ Automated model selection (append) ------------------
% Physical filters
tau_max_meas = max(tau);
fator_limite_tau_y = 10;   % reject tau_y > fator_limite_tau_y * tau_max_meas

is_bing_phys = true;
is_hb_phys   = true;
is_papa_phys = true;

% Bingham: pb_opt = [tau_y, eta_p]
tau_y_b = pb_opt(1); eta_p_b = pb_opt(2);
if tau_y_b < 0 || eta_p_b <= 0 || tau_y_b > fator_limite_tau_y * tau_max_meas
  is_bing_phys = false;
end

% Herschel-Bulkley: ph_opt = [tau_y, K, n]
tau_y_h = ph_opt(1); K_h = ph_opt(2); n_h = ph_opt(3);
% require n positive and within a reasonable bound (avoid huge exponents)
if tau_y_h < 0 || K_h <= 0 || n_h <= 0 || n_h > 10 || tau_y_h > fator_limite_tau_y * tau_max_meas
  is_hb_phys = false;
end

% Papanastasiou: popt_all = [tau_y, K, n, m]
tau_y_p = popt_all(1); K_p = popt_all(2); n_p = popt_all(3); m_p = popt_all(4);
if tau_y_p < 0 || K_p <= 0 || n_p <= 0 || n_p > 10 || m_p <= 0 || tau_y_p > fator_limite_tau_y * tau_max_meas
  is_papa_phys = false;
end

% Collect admissible models and their RMSE
models = {};
rmses  = [];
params = {};

if is_bing_phys
  models{end+1} = 'Bingham';
  rmses(end+1) = RMSE_b;
  params{end+1} = struct('tau_y',tau_y_b,'eta_p',eta_p_b);
else
  fprintf('Bingham rejected by physical filter: tau_y=%.6g, eta_p=%.6g\n', tau_y_b, eta_p_b);
end

if is_hb_phys
  models{end+1} = 'Herschel-Bulkley';
  rmses(end+1) = RMSE_h;
  params{end+1} = struct('tau_y',tau_y_h,'K',K_h,'n',n_h);
else
  fprintf('Herschel-Bulkley rejected by physical filter: tau_y=%.6g, K=%.6g, n=%.6g\n', tau_y_h, K_h, n_h);
end

if is_papa_phys
  models{end+1} = 'Papanastasiou-HB';
  rmses(end+1) = RMSE_papa;
  params{end+1} = struct('tau_y',tau_y_p,'K',K_p,'n',n_p,'m',m_p);
else
  fprintf('Papanastasiou-HB rejected by physical filter: tau_y=%.6g, K=%.6g, n=%.6g, m=%.6g\n', tau_y_p, K_p, n_p, m_p);
end

% Decision among admissible models
if isempty(models)
  warning('No model passed the physical filters. Check parameters and validity criteria.');
  best_model = '';
  best_params = struct();
else
  [min_rmse, idx_best] = min(rmses);
  best_model = models{idx_best};
  best_params = params{idx_best};
  fprintf('\nSelected model: %s (RMSE = %.6g)\n', best_model, min_rmse);
  disp('Selected model parameters:');
  disp(best_params);
end

% Optionally save selection
results.selected.model = best_model;
results.selected.params = best_params;
results.selected.RMSE = min_rmse;
% ---------------- end automated model selection -----------------------

% save results
results.bingham.params = struct('tau_y',pb_opt(1),'eta_p',pb_opt(2));
results.bingham.SSE = SSE_b; results.bingham.RMSE = RMSE_b;
results.hb.params = struct('tau_y',ph_opt(1),'K',ph_opt(2),'n',ph_opt(3));
results.hb.SSE = SSE_h; results.hb.RMSE = RMSE_h;
results.papanastasiou.params = struct('tau_y',popt_all(1),'K',popt_all(2),'n',popt_all(3),'m',popt_all(4));
results.papanastasiou.SSE = SSE_papa_full; results.papanastasiou.RMSE = RMSE_papa;
results.SSE_grid = SSE_grid; results.m_grid = m_grid;
save('fit_results_papa.mat','results');

fprintf('\nResults saved in fit_results_papa.mat\n\n');

