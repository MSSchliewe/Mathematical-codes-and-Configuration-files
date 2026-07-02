% stat_m.m
%
% Distribution analysis and conditional outlier detection
%
% Output: results.report contains parameters and analytical statistics of the bestModel

%% --- USER INPUT: paste your data and label here ---
raw_data = [ ...

2.54;

2.54;

2.54;

2.57;

2.57;

2.57;

2.57;

2.57;

2.57;

2.587;

2.61;

2.61;

2.61;

2.62;

2.62;

2.62;

2.62;

2.62;

2.62;

2.62;

2.62;

2.635;

2.635;

2.63575;

2.6375;

2.639;

2.64;

2.64;

2.6415;

2.6415;

2.6425;

2.644;

2.645;

2.65;

2.65;

2.65;

2.65;

2.65;

2.65;

2.65;

2.65;

2.65;

2.65;

2.65;

2.666;

2.684;

2.684;

2.684;

2.684;

2.684;

2.684;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;
2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.7;

2.704;

2.704;

2.704;

2.704;

2.704;

2.708;

2.744

];

data_label = 'Density of solids - g/cm3';

%% Preparation and basic statistics
x = raw_data(:);

x = x(~isnan(x));
if isempty(x), error('raw_data is empty.'); end

N = numel(x);

if N < 3, error('Sample too small (N < 3).'); end

mean_x = mean(x);

median_x = median(x);

sd_x = std(x,0);
cv = sd_x / mean_x;

mad_x = mad(x,1);
min_x = min(x);
max_x = max(x);

fprintf('\nAnalyzing: %s (N=%d) \n', data_label, N);
fprintf('sample mean = %.6g, median = %.6g, sd = %.6g, CV = %.3g\n', mean_x, median_x, sd_x, cv);

%% --- Fit candidate models (moments / MLE) ---
% Normal
p_norm.mu = mean_x;
p_norm.sigma = sd_x;
ll_norm = sum(log(normpdf(x, p_norm.mu, p_norm.sigma) + eps));
AIC_norm = 2*2 - 2*ll_norm;

% LogNormal (MLE on log scale) if all positive
has_logn = all(x > 0);
if has_logn
  y = log(x);
  p_logn.mu = mean(y);
  p_logn.sigma = std(y,0);
  ll_logn = sum(log(lognpdf(x, p_logn.mu, p_logn.sigma) + eps));
  AIC_logn = 2*2 - 2*ll_logn;
else
  AIC_logn = Inf;
end

% Exponential (scale = mean)
has_exp = all(x >= 0);
if has_exp
  p_exp.scale = mean(x); % scale parameter (mean)
  ll_exp = sum(log(exppdf(x, p_exp.scale) + eps));
  AIC_exp = 2*1 - 2*ll_exp;
else
  AIC_exp = Inf;
end

% Triangular (mode from histogram) - optional
a_tri = min_x; b_tri = max_x;
[counts_hist, centers_hist] = hist(x, max(3, ceil(sqrt(N))));
[~, idxm] = max(counts_hist);
m_tri = centers_hist(idxm);
if exist('tri_pdf', 'file') == 2
  pdf_tri_vals = tri_pdf(x, a_tri, m_tri, b_tri);
  ll_tri = sum(log(pdf_tri_vals + eps));
  AIC_tri = 2*3 - 2*ll_tri;
else
  AIC_tri = Inf;
end

% Beta if in (0,1)
has_beta = all(x > 0) && all(x < 1);
if has_beta
  ab = betafit(x);
  p_beta.a = ab(1); p_beta.b = ab(2);
  ll_beta = sum(log(betapdf(x, p_beta.a, p_beta.b) + eps));
  AIC_beta = 2*2 - 2*ll_beta;
else
  AIC_beta = Inf;
end

% assemble model list and choose by AIC
models = {}; aic_vals = [];
models{end+1} = 'Normal'; aic_vals(end+1) = AIC_norm;
if has_logn; models{end+1} = 'LogNormal'; aic_vals(end+1) = AIC_logn; end
if has_exp; models{end+1} = 'Exponential'; aic_vals(end+1) = AIC_exp; end
if has_beta; models{end+1} = 'Beta'; aic_vals(end+1) = AIC_beta; end
if exist('tri_pdf', 'file') == 2; models{end+1} = 'Triangular'; aic_vals(end+1) = AIC_tri; end

[~, idxBest] = min(aic_vals);
bestModel = models{idxBest};
fprintf('Selected best model by AIC: %s\n', bestModel);

%% --- GOF (KS bootstrap) - optional if ks_boot available ---
if exist('ks_boot', 'file') == 2
  Bboot = 500;
  gof_results = struct();
  % Normal
  est_norm_handle = @(d) struct('type', 'Normal','mu',mean(d),'sigma', std(d,0));
  cdf_norm_handle = @(z,p) normcdf(z, p.mu, p.sigma);
  gof_results.Normal.pval = ks_boot(x, cdf_norm_handle, est_norm_handle, Bboot);
  % LogNormal
  if has_logn
    est_logn_handle = @(d) struct('type', 'LogNormal','mu',mean(log(d)),'sigma', std(log(d),0));
    cdf_logn_handle = @(z,p) logncdf(z, p.mu, p.sigma);
    gof_results.LogNormal.pval = ks_boot(x, cdf_logn_handle, est_logn_handle, Bboot);
  end
  if has_exp
    est_exp_handle = @(d) struct('type', 'Exponential', 'mu',mean(d));
    cdf_exp_handle = @(z,p) expcdf(z, p.mu);
    gof_results.Exponential.pval = ks_boot(x, cdf_exp_handle, est_exp_handle, Bboot);
  end
  if has_beta
    est_beta_handle = @(d) struct('type', 'Beta','a',betafit(d)(1),'b',betafit(d)(2));
    cdf_beta_handle = @(z,p) betacdf(z, p.a, p.b);
    gof_results.Beta.pval = ks_boot(x, cdf_beta_handle, est_beta_handle, Bboot);
  end
  if exist('cdf_tri_vector', 'file') == 2
    p_tri = struct('type','Triangular','a',a_tri,'m',m_tri,'b',b_tri);
    est_tri_wrapper = @(d) p_tri;
    cdf_tri_handle = @(z,p) cdf_tri_vector(z,p);
    gof_results.Triangular.pval = ks_boot(x, cdf_tri_handle, est_tri_wrapper, Bboot);
  end
  disp('GOF p-values (KS bootstrap):'); disp(gof_results);
else
  warning('ks_boot not found: skipping KS bootstrap GOF.');
end

%% --- Build unambiguous report for the bestModel ---
report = struct();
report.model = bestModel;
report.sample_mean = mean_x;
report.sample_sd = sd_x;
report.sample_cv = cv;

% compute analytic parameters and percentiles for the chosen model only
switch bestModel

case 'Normal'
  report.param_name = 'Normal (mean, sd)';
  report.mu = p_norm.mu;
  report.sigma = p_norm.sigma;
  report.mean_analytic = p_norm.mu;
  report.sd_analytic = p_norm.sigma;
  report.P5 = norminv(0.05, p_norm.mu, p_norm.sigma);
  report.P50 = p_norm.mu;
  report.P95 = norminv(0.95, p_norm.mu, p_norm.sigma);

case 'LogNormal'
  report.param_name = 'LogNormal (mu_ln, sigma_ln)';
  report.mu_ln = p_logn.mu;
  report.sigma_ln = p_logn.sigma;
  report.mean_analytic = exp(p_logn.mu + 0.5 * p_logn.sigma^2);
  report.sd_analytic = sqrt((exp(p_logn.sigma^2)-1)*exp(2*p_logn.mu + p_logn.sigma^2));
  report.P5 = exp(p_logn.mu + norminv(0.05)*p_logn.sigma);
  report.P50 = exp(p_logn.mu);
  report.P95 = exp(p_logn.mu + norminv(0.95)*p_logn.sigma);

case 'Exponential'
  report.param_name = 'Exponential (scale)';
  report.scale = p_exp.scale;
  report.mean_analytic = p_exp.scale;
  report.sd_analytic = p_exp.scale;
  report.P5 = - p_exp.scale * log(1-0.05);
  report.P50 = - p_exp.scale * log(0.5);
  report.P95 = - p_exp.scale * log(1-0.95);

case 'Beta'
  report.param_name = 'Beta (a, b)';
  report.a = p_beta.a;
  report.b = p_beta.b;
  report.mean_analytic = p_beta.a / (p_beta.a + p_beta.b);
  report.sd_analytic = sqrt((p_beta.a*p_beta.b)/((p_beta.a+p_beta.b)^2*(p_beta.a+p_beta.b+1)));
  report.P5 = betainv(0.05, p_beta.a, p_beta.b);
  report.P50 = betainv(0.5, p_beta.a, p_beta.b);
  report.P95 = betainv(0.95, p_beta.a, p_beta.b);

case 'Triangular'
  report.param_name = 'Triangular (a, m, b)';
  report.a = a_tri; report.m = m_tri; report.b = b_tri;
  report.mean_analytic = (a_tri + m_tri + b_tri)/3;
  report.sd_analytic = NaN;
  report.P5 = []; report.P50 = []; report.P95 = [];

otherwise
  error('Unknown bestModel: %s', bestModel);
end
results = struct();
results.report = report;

%% --- Parametric bootstrap only for the bestModel (when applicable)
if strcmp(bestModel, 'LogNormal')
  Bparam = 2000;
  rng('default');
  mu_boot = zeros(Bparam, 1); sigma_boot = zeros(Bparam, 1);

  for b = 1:Bparam
    sim = lognrnd(p_logn.mu, p_logn.sigma, N, 1);
    mu_boot(b) = mean(log(sim));
    sigma_boot(b) = std(log(sim),0);
  end
  results.report.mu_ln_CI = prctile(mu_boot, [2.5 97.5]);
  results.report.sigma_ln_CI = prctile(sigma_boot, [2.5 97.5]);
  fprintf('Parametric bootstrap (LogNormal) mu_ln CI: [%.6g, %.6g], sigma_ln CI: [%.6g, %.6g]\n', ...
    results.report.mu_ln_CI(1), results.report.mu_ln_CI(2), results.report.sigma_ln_CI(1), results.report.sigma_ln_CI(2));

elseif strcmp(bestModel, 'Normal')
  % optional parametric bootstrap for Normal
  Bparam = 1000;
  rng('default');
  mu_b = zeros(Bparam, 1); sd_b = zeros(Bparam,1);
  for b = 1:Bparam
    sim = normrnd(p_norm.mu, p_norm.sigma, N, 1);
    mu_b(b) = mean(sim); sd_b(b)= std(sim,0);
  end
  results.report.mu_CI = prctile(mu_b, [2.5 97.5]);
  results.report.sigma_CI = prctile(sd_b, [2.5 97.5]);
  fprintf('Parametric bootstrap (Normal) mean CI: [%.6g, %.6g], sd CI: [%.6g, %.6g]\n', ...
    results.report.mu_CI(1), results.report.mu_CI(2), results.report.sigma_CI(1), results.report.sigma_CI(2));

elseif strcmp(bestModel, 'Exponential')
  % optional bootstrap for exponential scale
  Bparam = 1000;
  rng('default');
  scale_b = zeros(Bparam, 1);
  for b = 1:Bparam
    sim = exprnd(p_exp.scale, N, 1);
    scale_b(b) = mean(sim);
  end
  results.report.scale_CI = prctile(scale_b, [2.5 97.5]);
  fprintf('Parametric bootstrap (Exponential) scale CI: [%.6g, %.6g]\n', results.report.scale_CI(1), results.report.scale_CI(2));
end

%% --- Final printout (only fields of the bestModel) ---
fprintf(' \n --- Final report for bestModel: %s --- \n', bestModel);
switch bestModel

case 'Normal'
  fprintf('Model: Normal\n');
  fprintf('mu = %.6g, sigma = %.6g\n', results.report.mu, results.report.sigma);
  fprintf('analytic mean = %.6g, analytic sd = %.6g\n', results.report.mean_analytic, results.report.sd_analytic);
  fprintf('P5, P50, P95 = %.6g, %.6g, %.6g\n', results.report.P5, results.report.P50, results.report.P95);

case 'LogNormal'
  fprintf('Model: LogNormal\n');
  fprintf('mu_ln = %.6g, sigma_ln = %.6g\n', results.report.mu_ln, results.report.sigma_ln);
  fprintf('analytic mean = %.6g, analytic sd = %.6g\n', results.report.mean_analytic, results.report.sd_analytic);
  fprintf('P5, P50, P95 = %.6g, %.6g, %.6g\n', results.report.P5, results.report.P50, results.report.P95);

case 'Exponential'
  fprintf('Model: Exponential\n');
  fprintf('scale = %.6g\n', results.report.scale);
  fprintf('analytic mean = %.6g, analytic sd = %.6g\n', results.report.mean_analytic, results.report.sd_analytic);
  fprintf('P5, P50, P95 = %.6g, %.6g, %.6g\n', results.report.P5, results.report.P50, results.report.P95);

case 'Beta'
  fprintf('Model: Beta\n');
  fprintf('a = %.6g, b = %.6g\n', results.report.a, results.report.b);
  fprintf('analytic mean = %.6g, analytic sd = %.6g\n', results.report.mean_analytic, results.report.sd_analytic);
  fprintf('P5, P50, P95 = %.6g, %.6g, %.6g\n', results.report.P5, results.report.P50, results.report.P95);

case 'Triangular'
  fprintf('Model: Triangular\n');
  fprintf('a = %.6g, m = %.6g, b = %.6g\n', results.report.a, results.report.m, results.report.b);
  fprintf('analytic mean = %.6g\n', results.report.mean_analytic);

otherwise
  fprintf('No final print implemented for model: %s\n', bestModel);
end

%% --- Conditional outlier detection (applied to bestModel) ---
function out_idx = detect_outliers_by_model_local(xvec, model_name, mad_thresh)
if nargin < 3; mad_thresh = 3.5; end
pos = find(~isnan(xvec));
x_pos = xvec(pos);

switch model_name
case 'LogNormal'
  y = log(x_pos);

case 'Beta'
  x_pos = min(max(x_pos, eps), 1-eps);
  y = log(x_pos ./ (1 - x_pos));

case 'Exponential'
  y = log(x_pos);

otherwise
  y = x_pos;
end

Q1 = prctile(y,25); Q3 = prctile(y,75); IQRval = Q3 - Q1;
iqr_low = Q1 - 1.5 * IQRval; iqr_high = Q3 + 1.5 * IQRval;
idx_iqr = find(y < iqr_low | y > iqr_high);

med_y = median(y); mad_y = mad(y,1);
if mad_y == 0
  idx_mad = [];
else
  robust_z = abs((y - med_y) / (1.4826 * mad_y));
  idx_mad = find(robust_z > mad_thresh);
end
idx_union = unique([idx_iqr; idx_mad]);

out_idx = pos(idx_union);
end

outliers_conditional = detect_outliers_by_model_local(x, bestModel, 3.5);
fprintf('\nOutlier detection (conditional on %s): \n', bestModel);
if isempty(outliers_conditional)
  fprintf(' (none) \n');
else
  for i = 1:numel(outliers_conditional)
    fprintf(' index %d, value = %.6g\n', outliers_conditional(i), x(outliers_conditional(i)));
  end
end

%% --- Plot: linear histogram + fitted pdf + bootstrap envelope (95% CI) ---
% We assume the following variables exist in the workspace:
% x, bestModel, p_norm, p_logn, p_exp, p_beta, a_tri, m_tri, b_tri

Bplot = 500;
xx = linspace(min(x), max(x), 400);

% --- histogram / KDE
figure('Name', ['Histogram + fitted pdf - ' data_label], 'NumberTitle', 'off');
if exist('histcounts', 'file') == 2
  [counts_plot, edges_plot] = histcounts(x, max(3, ceil(sqrt(N))));
  centers_plot = (edges_plot(1:end-1) + edges_plot(2:end))/2;
  binw_plot = edges_plot(2) - edges_plot(1);
else
  [counts_plot, centers_plot] = hist(x, max(3, ceil(sqrt(N))));
  binw_plot = centers_plot(2) - centers_plot(1);
end
density_plot = counts_plot / (N * binw_plot);
bar(centers_plot, density_plot, 1, 'FaceColor', [0.95 0.95 1], 'EdgeColor', 'k');
hold on;

% KDE (if available)
if exist('ksdensity', 'file') == 2 || exist('ksdensity', 'builtin') == 5
  try
    [f_kde, xi_kde] = ksdensity(x, 'Support', [min(x) max(x)]);
    plot(xi_kde, f_kde, 'Color', [0.2 0.2 0.8], 'LineWidth', 1.2);
  catch
    % ignore KDE errors
  end
end

% --- fitted pdf according to bestModel
pdf_hat = zeros(size(xx));
simfun = []; param1 = []; param2 = [];
switch bestModel
case 'Normal'
  pdf_hat = normpdf(xx, p_norm.mu, p_norm.sigma);
  simfun = @(mu, sig) normrnd(mu, sig, N, 1);
  param1 = p_norm.mu; param2 = p_norm.sigma;
case 'LogNormal'
  pdf_hat = lognpdf(xx, p_logn.mu, p_logn.sigma);
  simfun = @(mu, sig) lognrnd(mu, sig, N, 1);
  param1 = p_logn.mu; param2 = p_logn.sigma;
case 'Exponential'
  pdf_hat = exppdf(xx, p_exp.scale);
  simfun = @(mu,~) exprnd(mu, N, 1);
  param1 = p_exp.scale; param2 = NaN;
case 'Beta'
  pdf_hat = betapdf(xx, p_beta.a, p_beta.b);
  simfun = @(a,b) betarnd(a,b, N, 1);
  param1 = p_beta.a; param2 = p_beta.b;
case 'Triangular'
  if exist('tri_pdf', 'file') == 2
    pdf_hat = tri_pdf(xx, a_tri, m_tri, b_tri);
  else
    pdf_hat = zeros(size(xx));
  end
  simfun = []; param1 = []; param2 = [];
otherwise
  pdf_hat = zeros(size(xx));
end

% plot fitted pdf
if any(pdf_hat > 0)
  plot(xx, pdf_hat, 'r--', 'LineWidth', 1.6);
end

% --- parametric bootstrap envelope (95% CI) for the fitted pdf ---
pdf_boot = [];
if ~isempty(simfun)
  pdf_boot = zeros(Bplot, numel(xx));
  for b = 1:Bplot
    if strcmp(bestModel, 'LogNormal') || strcmp(bestModel, 'Normal')
      sim = simfun(param1, param2);
    elseif strcmp(bestModel, 'Exponential')
      sim = simfun(param1, []);
    elseif strcmp(bestModel, 'Beta')
      sim = simfun(param1, param2);
    else
      sim = [];
    end
    if ~isempty(sim)
      switch bestModel
      case 'LogNormal'
        mu_b = mean(log(sim)); sigma_b = std(log(sim),0);
        pdf_boot(b,:) = lognpdf(xx, mu_b, sigma_b);
      case 'Normal'
        mu_b = mean(sim); sigma_b = std(sim,0);
        pdf_boot(b,:) = normpdf(xx, mu_b, sigma_b);
      case 'Exponential'
        mu_b = mean(sim);
        pdf_boot(b,:) = exppdf(xx, mu_b);
      case 'Beta'
        ab_b = betafit(sim);
        pdf_boot(b,:) = betapdf(xx, ab_b(1), ab_b(2));
      end
    end
  end
  if ~isempty(pdf_boot)
    pdf_lo = prctile(pdf_boot, 2.5);
    pdf_hi = prctile(pdf_boot, 97.5);
    plot(xx, pdf_lo, 'r:', 'LineWidth', 1);
    plot(xx, pdf_hi, 'r:', 'LineWidth', 1);
    % shaded envelope (optional)
    try
      xenv = [xx, fliplr(xx)];
      yenv = [pdf_lo, fliplr(pdf_hi)];
      h = fill(xenv, yenv, [1 0.8 0.8], 'EdgeColor', 'none');
      set(h, 'FaceAlpha', 0.25);
    catch
      % ignore fill errors
    end
  end
end

% finalize plot
xlabel(data_label); ylabel('Density');
title(sprintf('%s - Histogram + fitted pdf (%s)', data_label, bestModel));
legend_entries = {'Histogram'};
if exist('f_kde', 'var'); legend_entries{end+1} = 'KDE'; end
if any(pdf_hat>0); legend_entries{end+1} = 'Fitted pdf'; end
if ~isempty(pdf_boot); legend_entries{end+1} = '95% parametric envelope'; end
legend(legend_entries, 'Location', 'northeast');
grid on; hold off;

%% --- Plot: fitted CDF + bootstrap envelope (95% CI) ---
figure('Name', ['Fitted CDF + envelope - ' data_label], 'NumberTitle', 'off');
xx_cdf = linspace(min(x), max(x), 400);
cdf_hat = zeros(size(xx_cdf));
cdf_boot = [];

switch bestModel
case 'Normal'
  cdf_hat = normcdf(xx_cdf, p_norm.mu, p_norm.sigma);
case 'LogNormal'
  cdf_hat = logncdf(xx_cdf, p_logn.mu, p_logn.sigma);
case 'Exponential'
  cdf_hat = expcdf(xx_cdf, p_exp.scale);
case 'Beta'
  cdf_hat = betacdf(xx_cdf, p_beta.a, p_beta.b);
case 'Triangular'
  if exist('cdf_tri_vector', 'file') == 2
    cdf_hat = cdf_tri_vector(xx_cdf, struct('a', a_tri, 'm', m_tri, 'b', b_tri));
  else
    cdf_hat = zeros(size(xx_cdf));
  end
otherwise
  cdf_hat = zeros(size(xx_cdf));
end

% empirical CDF for comparison
[f_emp, x_emp] = ecdf(x);

% plot empirical CDF and fitted CDF
plot(x_emp, f_emp, 'k.', 'MarkerSize', 8); hold on;
plot(xx_cdf, cdf_hat, 'r-', 'LineWidth', 1.6);

% bootstrap envelope for CDF (parametric) if simfun available
if ~isempty(simfun)
  cdf_boot = zeros(Bplot, numel(xx_cdf));
  for b = 1:Bplot
    if strcmp(bestModel, 'LogNormal') || strcmp(bestModel, 'Normal')
      sim = simfun(param1, param2);
    elseif strcmp(bestModel, 'Exponential')
      sim = simfun(param1, []);
    elseif strcmp(bestModel, 'Beta')
      sim = simfun(param1, param2);
    else
      sim = [];
    end
    if ~isempty(sim)
      switch bestModel
      case 'LogNormal'
        cdf_boot(b,:) = logncdf(xx_cdf, mean(log(sim)), std(log(sim),0));
      case 'Normal'
        cdf_boot(b,:) = normcdf(xx_cdf, mean(sim), std(sim,0));
      case 'Exponential'
        cdf_boot(b,:) = expcdf(xx_cdf, mean(sim));
      case 'Beta'
        ab_b = betafit(sim);
        cdf_boot(b,:) = betacdf(xx_cdf, ab_b(1), ab_b(2));
      end
    end
  end
  if ~isempty(cdf_boot)
    cdf_lo = prctile(cdf_boot, 2.5);
    cdf_hi = prctile(cdf_boot, 97.5);
    plot(xx_cdf, cdf_lo, 'r:', 'LineWidth', 1);
    plot(xx_cdf, cdf_hi, 'r:', 'LineWidth', 1);
    % shaded envelope
    try
      xenv = [xx_cdf, fliplr(xx_cdf)];
      yenv = [cdf_lo, fliplr(cdf_hi)];
      h2 = fill(xenv, yenv, [1 0.9 0.9], 'EdgeColor', 'none');
      set(h2, 'FaceAlpha', 0.25);
    catch
      % ignore fill errors
    end
  end
end

xlabel(data_label); ylabel('Cumulative probability');
title(sprintf('%s - Empirical CDF and fitted CDF (%s)', data_label, bestModel));
legend('Empirical CDF', 'Fitted CDF', '95% parametric envelope', 'Location', 'southeast');

grid on; hold off;

%% --- Save final results in "results" (already contains results.report) ---
results.outliers_conditional = outliers_conditional;
fprintf('\nResults saved in variable "results". Use results.report for bestModel parameters.\n');

% End of script

