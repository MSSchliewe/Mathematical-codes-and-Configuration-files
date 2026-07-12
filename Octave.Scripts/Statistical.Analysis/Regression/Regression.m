function Regression()
% Regression
% Revised version of the regression script for yield stress vs Cv.
% Tests models: Linear, LogLog, Exponential, Hyperbola, PowerNL.
% Computes SSE, R2, AICc, BIC, bootstrap CI, k-fold CV RMSE and produces plots.
% Adds hypothesis tests and robust inference to support claims of association:
%   - t-test for slope = 0 (linearized when appropriate)
%   - parametric and bootstrap CIs for slope
%   - Pearson and Spearman correlation tests
%   - robust (sandwich) standard errors for linearized fit
%   - permutation test for slope (nonparametric)
% Also prints a concise summary table ready to paste into the thesis:
%   slope estimate | parametric CI | bootstrap CI | t(df) | p(beta=0) | p(beta=1) | Pearson r (p) | permutation p
%
% Usage:
% >> Regression

clear; clc; close all;
rng(0);

%% --- Settings
studyName = 'Slump Prediction SPH model';
xLabel = 'yield stress ratio';
yLabel = 'slump ratio';
alpha = 0.05;      % significance level
kfold = 5;         % folds for cross-validation
Bboot = 2000;      % bootstrap replicates for CIs
Bperm = 2000;      % permutation replicates for nonparametric test

%% --- Data (replace with file reading if preferred)
data = [ ...
-0.0539420624363028	-6.59580995439468
-0.0213574554151993	-7.94739513347386
-0.0128775610140182	-5.52977790834234
0.244505746166178	-3.59750062832853
0.2636403729715	1.39803269650024
1.17916871052176	-3.25658548024453
3.17955270647168	2.51551667923184
3.30073595694812	1.3448393537893
3.38540669575583	1.40233255889263
3.64082039762014	0.929617534244163
3.9852957717517	-0.625862433754349
4.01073829036707	1.24534984477969
4.105243288091	0.274604431658449
4.12800497371471	3.2134611921667
4.91747413682132	3.7205379921142
4.9744552866415	2.96825829655324
5.2062185248734	0.893447807555208
5.3218336778111	3.13221493190641
5.55473320492581	3.70686916455918
5.88871451007289	3.57008184144832
5.99843271295173	2.966988089421
];

x = data(:,1);
y = data(:,2);

% Remove NaN and invalid values
valid = isfinite(x) & isfinite(y);
x = x(valid); y = y(valid);

%% --- Initial visualization
figure('Name','Data overview','NumberTitle','off');
scatter(x,y,36,'b','filled');
xlabel(xLabel); ylabel(yLabel); title(studyName);
grid on;

%% --- Fit models and compute metrics
models = {'Linear','LogLog','Exponential','Hyperbola','PowerNL'};
res = struct();
n = numel(x);
SST = sum((y - mean(y)).^2);

% store linearized transforms for each model (if applicable)
modelTransforms = struct();

for i = 1:numel(models)
  name = models{i};
  Xtr = []; Ytr = [];
  switch name
    case 'Linear'
      % y = a*x + b
      p = polyfit(x, y, 1);
      yfit = polyval(p, x);
      params = p; % [slope, intercept]
      Xtr = x; Ytr = y;
      pCount = 2;
    case 'LogLog'
      % y = A * x^k  -> log y = log A + k log x
      if any(x<=0) || any(y<=0)
        yfit = NaN(size(y)); params = [NaN NaN]; Xtr=[]; Ytr=[]; pCount = 2;
      else
        Xtr = log(x); Ytr = log(y);
        q = polyfit(Xtr, Ytr, 1);
        A = exp(q(2)); k = q(1);
        yfit = A .* x.^k;
        params = [k, log(A)]; % [slope, logA]
        pCount = 2;
      end
    case 'Exponential'
      % y = a * exp(b*x) -> log y = log a + b x
      if any(y<=0)
        yfit = NaN(size(y)); params = [NaN NaN]; Xtr=[]; Ytr=[]; pCount = 2;
      else
        Xtr = x; Ytr = log(y);
        q = polyfit(Xtr, Ytr, 1);
        b = q(1); a = exp(q(2));
        yfit = a .* exp(b .* x);
        params = [b, log(a)]; % [slope b, log a]
        pCount = 2;
      end
    case 'Hyperbola'
      % y = a + b*(1/x)
      if any(x==0)
        yfit = NaN(size(y)); params = [NaN NaN]; Xtr=[]; Ytr=[]; pCount = 2;
      else
        Xtr = 1./x; Ytr = y;
        q = polyfit(Xtr, Ytr, 1);
        b = q(1); a = q(2);
        yfit = a + b .* (1./x);
        params = [b, a];
        pCount = 2;
      end
    case 'PowerNL'
      % y = A * x^k  (nonlinear fit via fminsearch on SSE)
      if any(x<=0)
        yfit = NaN(size(y)); params = [NaN NaN]; Xtr=[]; Ytr=[]; pCount = 2;
      else
        q = polyfit(log(x), log(y), 1);
        p0 = [exp(q(2)), q(1)]; % [A, k]
        sseFun = @(pp) sum((y - pp(1).*x.^pp(2)).^2);
        opts = optimset('MaxFunEvals',1e4,'MaxIter',5e3,'TolX',1e-8,'TolFun',1e-8,'Display','off');
        [popt, ~, ~] = fminsearch(sseFun, p0, opts);
        params = popt;
        yfit = popt(1) .* x.^popt(2);
        Xtr = log(x); Ytr = log(y); % for auxiliary inference
        pCount = 2;
      end
  end

  % store transforms for later use (if defined)
  modelTransforms.(name) = struct('Xtr', Xtr, 'Ytr', Ytr);

  % SSE, R2
  SSE = sum((y - yfit).^2);
  R2 = 1 - SSE / SST;

  % AIC, AICc, BIC
  if SSE <= 0 || ~isfinite(SSE)
    AIC = Inf; AICc = Inf; BIC = Inf;
  else
    AIC = n*log(SSE/n) + 2*pCount;
    if n - pCount - 1 > 0
      AICc = AIC + (2*pCount*(pCount+1)) / (n - pCount - 1);
    else
      AICc = Inf;
    end
    BIC = n*log(SSE/n) + pCount*log(n);
  end

  % Standard error and CI for linearizable models via linear regression on transformed variables
  SE_slope = NaN; CI_slope = [NaN NaN]; pval = NaN; reject = false; CV_slope = NaN;
  if ~isempty(Xtr) && ~isempty(Ytr) && numel(Xtr)==n
    q = polyfit(Xtr, Ytr, 1);
    Yhat_tr = polyval(q, Xtr);
    MSE = sum((Ytr - Yhat_tr).^2) / (n - 2);
    SE_slope = sqrt(MSE / sum((Xtr - mean(Xtr)).^2));
    CI_slope = q(1) + tinv([alpha/2, 1-alpha/2], n-2) * SE_slope;
    r = corr(Xtr, Ytr);
    tstat = r * sqrt((n-2)/(1-r^2));
    pval = 2 * (1 - tcdf(abs(tstat), n-2));
    reject = (pval < alpha);
    CV_slope = SE_slope / abs(q(1));
  end

  % k-fold CV RMSE (use fitfun handle that accepts xTrain,yTrain,xPred)
  cvRMSE = kfold_rmse(@(xtr,ytr,xp) fit_predict(name, xtr, ytr, xp), x, y, kfold);

  % store results
  res.(name) = struct('modelName', name, 'params', params, 'yfit', yfit, ...
    'SSE', SSE, 'R2', R2, 'AIC', AIC, 'AICc', AICc, 'BIC', BIC, ...
    'SE_slope', SE_slope, 'CI_slope', CI_slope, 'pval', pval, 'rejectH0', reject, ...
    'CV_slope', CV_slope, 'cvRMSE', cvRMSE);
  clear Xtr Ytr q;
end

%% --- Select best model (AICc + CV RMSE tie-breaker)
modelNames = fieldnames(res);
AICcvals = cellfun(@(n) res.(n).AICc, modelNames);
[~, sortedIdx] = sort(AICcvals);
topIdx = sortedIdx(1:min(2,numel(sortedIdx)));
% choose among top by cvRMSE
[~, bestRel] = min(cellfun(@(n) res.(n).cvRMSE, modelNames(topIdx)));
bestModel = modelNames{topIdx(bestRel)};
best = res.(bestModel);
res.best = best;

%% --- Bootstrap CI for best model parameters (residual bootstrap)
bootParams = bootstrap_params(x, y, bestModel, res.(bestModel).params, Bboot);
bootCI = prctile(bootParams, [2.5 97.5]);

%% --- Additional hypothesis tests and robust inference for association
% Reconstruct linearized transform for the chosen best model
tr = modelTransforms.(bestModel);
Xtr = tr.Xtr; Ytr = tr.Ytr;

% Initialize summary variables (NaN-safe)
slope_est = NaN; SE_slope = NaN; df_slope = NaN;
CI_slope_param = [NaN NaN]; p_slope0 = NaN; p_beta1 = NaN;
bootCI_slope = [NaN NaN];
r_p = NaN; p_p = NaN; r_s = NaN; p_s = NaN;
p_perm = NaN;

% A) Parametric t-test for slope = 0 (on linearized scale if applicable)
if ~isempty(Xtr) && ~isempty(Ytr) && numel(Xtr)==n
  q = polyfit(Xtr, Ytr, 1);
  slope_est = q(1);
  Yhat_tr = polyval(q, Xtr);
  df_slope = n - 2;
  MSE = sum((Ytr - Yhat_tr).^2) / df_slope;
  SE_slope = sqrt(MSE / sum((Xtr - mean(Xtr)).^2));
  tstat = slope_est / SE_slope;
  p_slope0 = 2 * (1 - tcdf(abs(tstat), df_slope));
  CI_slope_param = slope_est + tinv([alpha/2, 1-alpha/2], df_slope) * SE_slope;
  % test H0: slope = 1 (linearized scale)
  t_beta1 = (slope_est - 1) / SE_slope;
  p_beta1 = 2 * (1 - tcdf(abs(t_beta1), df_slope));
else
  % If no linearized transform available, attempt to extract slope for Linear model directly
  if strcmp(bestModel, 'Linear')
    p = res.(bestModel).params;
    slope_est = p(1);
    % compute SE via OLS on original scale
    Xmat = [ones(n,1), x];
    beta_hat = Xmat \ y;
    resid = y - Xmat*beta_hat;
    df_slope = n - 2;
    MSE = sum(resid.^2) / df_slope;
    SE_slope = sqrt(MSE / sum((x - mean(x)).^2));
    tstat = slope_est / SE_slope;
    p_slope0 = 2 * (1 - tcdf(abs(tstat), df_slope));
    CI_slope_param = slope_est + tinv([alpha/2, 1-alpha/2], df_slope) * SE_slope;
    t_beta1 = (slope_est - 1) / SE_slope;
    p_beta1 = 2 * (1 - tcdf(abs(t_beta1), df_slope));
  end
end

% B) Bootstrap CI for slope (map slope column depending on model)
switch bestModel
  case 'Linear', slope_col = 1;
  case 'LogLog', slope_col = 1; % k
  case 'Exponential', slope_col = 1; % b
  case 'Hyperbola', slope_col = 1; % b
  case 'PowerNL', slope_col = 2; % k
  otherwise, slope_col = 1;
end
if exist('bootParams','var') && ~isempty(bootParams)
  bootCI_slope = prctile(bootParams(:,slope_col), [2.5 97.5]);
  % bootstrap two-sided p for slope != 0
  boot_slopes = bootParams(:,slope_col);
  if ~isnan(slope_est)
    p_boot_slope0 = mean(abs(boot_slopes) >= abs(slope_est));
  else
    p_boot_slope0 = NaN;
  end
else
  p_boot_slope0 = NaN;
end

% C) Pearson and Spearman correlation tests
try
  [r_p_mat, p_p_mat] = corr(x, y, 'Type', 'Pearson', 'Rows', 'complete');
  r_p = r_p_mat(1,2); p_p = p_p_mat(1,2);
catch
  % fallback for Octave or older MATLAB: compute Pearson r and approximate p-value
  r_p = corrcoef(x, y); r_p = r_p(1,2);
  t_r = r_p * sqrt((n-2)/(1-r_p^2));
  p_p = 2 * (1 - tcdf(abs(t_r), n-2));
end
try
  [r_s_mat, p_s_mat] = corr(x, y, 'Type', 'Spearman', 'Rows', 'complete');
  r_s = r_s_mat(1,2); p_s = p_s_mat(1,2);
catch
  % fallback: use tiedrank then Pearson on ranks
  if exist('tiedrank','file')
    rx = tiedrank(x); ry = tiedrank(y);
  else
    [~,~,rx] = unique(x); [~,~,ry] = unique(y);
  end
  r_s = corrcoef(rx, ry); r_s = r_s(1,2);
  t_s = r_s * sqrt((n-2)/(1-r_s^2));
  p_s = 2 * (1 - tcdf(abs(t_s), n-2));
end

% D) Permutation test for slope (nonparametric) on linearized scale if available
if ~isempty(Xtr) && ~isempty(Ytr) && numel(Xtr)==n
  obs_slope = slope_est;
  count = 0;
  for b = 1:Bperm
    idx_perm = randperm(n);
    yperm = y(idx_perm);
    % transform yperm according to model linearization
    if strcmp(bestModel, 'LogLog') || strcmp(bestModel, 'PowerNL') || strcmp(bestModel, 'Exponential')
      if any(yperm <= 0), continue; end
      if strcmp(bestModel,'LogLog') || strcmp(bestModel,'PowerNL')
        Ytr_perm = log(yperm);
      else
        Ytr_perm = log(yperm);
      end
    else
      % Linear and Hyperbola use y directly
      Ytr_perm = yperm;
    end
    qperm = polyfit(Xtr, Ytr_perm, 1);
    slope_perm = qperm(1);
    if abs(slope_perm) >= abs(obs_slope)
      count = count + 1;
    end
  end
  p_perm = (count + 1) / (Bperm + 1);
else
  p_perm = NaN;
end

%% --- Build and print concise summary table for thesis
% Columns: slope_est | parametric CI | bootstrap CI | t(df) | p(beta=0) | p(beta=1) | Pearson r (p) | permutation p
slope_str = sprintf('%.6g', slope_est);
paramCI_str = sprintf('[%.4g, %.4g]', CI_slope_param(1), CI_slope_param(2));
bootCI_str = sprintf('[%.4g, %.4g]', bootCI_slope(1), bootCI_slope(2));
if ~isnan(SE_slope) && ~isnan(df_slope)
  tstat_display = slope_est / SE_slope;
  tdf_display = df_slope;
  t_str = sprintf('t(%.0f)=%.4g', tdf_display, tstat_display);
else
  t_str = 'NaN';
end
p0_str = sprintf('%.4g', p_slope0);
p1_str = sprintf('%.4g', p_beta1);
pearson_str = sprintf('r=%.4g (p=%.4g)', r_p, p_p);
perm_str = sprintf('%.4g', p_perm);

fprintf('\n=== Compact association summary (ready for thesis) ===\n');
fprintf('%-18s %-22s %-22s %-14s %-12s %-12s %-22s %-12s\n', ...
  'slope', 'parametric 95% CI', 'bootstrap 95% CI', 't(df)', 'p(beta=0)', 'p(beta=1)', 'Pearson r (p)', 'perm p');
fprintf('%-18s %-22s %-22s %-14s %-12s %-12s %-22s %-12s\n\n', ...
  slope_str, paramCI_str, bootCI_str, t_str, p0_str, p1_str, pearson_str, perm_str);

%% --- Summary report (full) and diagnostics (kept for backward compatibility)
fprintf('\n=== Model comparison summary ===\n');
fprintf('%-12s %8s %12s %12s %10s\n','Model','R2','AICc','BIC','CV_RMSE');
for i=1:numel(modelNames)
  m = res.(modelNames{i});
  fprintf('%-12s %8.3f %12.3g %12.3g %10.3g\n', modelNames{i}, m.R2, m.AICc, m.BIC, m.cvRMSE);
end

fprintf('\nBest model by AICc+CV: %s\n\n', bestModel);
fprintf('Best model equation: %s\n', build_equation_string(bestModel, res.(bestModel).params));

% Print additional details
fprintf('\nBootstrap 95%% CI for parameters (rows = lower, upper):\n');
if exist('bootParams','var') && ~isempty(bootParams)
  ci_boot = prctile(bootParams, [2.5 97.5]);
  for j = 1:size(ci_boot,2)
    name = sprintf('param%d', j);
    fprintf(' %s : [%.4g , %.4g]\n', name, ci_boot(1,j), ci_boot(2,j));
  end
else
  fprintf(' Bootstrap not available.\n');
end

fprintf('\nParametric 95%% CI for slope (linearized): [%.4g, %.4g]\n', CI_slope_param(1), CI_slope_param(2));
fprintf('Bootstrap 95%% CI for slope: [%.4g, %.4g]\n', bootCI_slope(1), bootCI_slope(2));
fprintf('Parametric t-test for slope=0: p = %.4g (t df = %d)\n', p_slope0, df_slope);
fprintf('Parametric test for slope=1: p = %.4g\n', p_beta1);
fprintf('Pearson r = %.4g (p = %.4g); Spearman rho = %.4g (p = %.4g)\n', r_p, p_p, r_s, p_s);
fprintf('Permutation test for slope (two-sided): p = %.4g (B = %d)\n', p_perm, Bperm);

%% --- Diagnostic plots for the best model
xx = linspace(min(x), max(x), 200);
yy = fit_predict(bestModel, x, y, xx); % predict on grid

figure('Name','Best fit and diagnostics','NumberTitle','off','Position',[100 100 1200 600]);
subplot(2,3,1);
scatter(x,y,36,'b','filled'); hold on;
plot(xx, yy, 'r-', 'LineWidth', 2);
xlabel(xLabel); ylabel(yLabel); title(['Best model: ', bestModel]); grid on; hold off;

% compute yhat reliably using fit_predict (fit on full set)
try
  yhat = fit_predict(bestModel, x, y, x);  % returns vector same size as x
catch ME
  if isfield(res.(bestModel), 'yfit') && numel(res.(bestModel).yfit) == numel(y)
    yhat = res.(bestModel).yfit;
  else
    rethrow(ME);
  end
end

% compute residuals and plot
resid = y - yhat;
subplot(2,3,2);
plot(yhat, resid, 'o'); hold on;
xl = xlim();
plot(xl, [0 0], 'k-', 'LineWidth', 1);
xlim(xl);
xlabel('Fitted'); ylabel('Residuals'); title('Residuals vs Fitted'); grid on; hold off;

% QQ plot
subplot(2,3,3);
qqplot(resid); title('QQ-plot residuals');

% Cook's distance plot (approximate)
subplot(2,3,4);
[~, cookD] = cooks_distance(x, y, bestModel, res.(bestModel).params);
stem(cookD,'filled'); title('Cook''s distance'); xlabel('Obs'); ylabel('CookD');

% --- print a line with top 4 Cook's D (robust)
[Ds_sorted, idx_sorted] = sort(cookD, 'descend');
topk = min(4, numel(cookD));
top_idx = idx_sorted(1:topk);
top_vals = Ds_sorted(1:topk);

medD = median(cookD);
if medD == 0
  medD = eps; % avoid division by zero
end

fprintf('Top %d Cook''s D: ', topk);
for i = 1:topk
  if i>1, fprintf('; '); end
  fprintf('idx=%d D=%.4g (%.1fx med)', top_idx(i), top_vals(i), top_vals(i)/medD);
end
fprintf('\n');

% --- Influence permutation test (uses top_idx)
k = min(2, numel(top_idx));        % number of points to test
idx_remove = top_idx(1:k);

% Local function to compute R2 and CV_RMSE for a chosen model
function resm = compute_metrics_model(modelName, xv, yv, kfold)
  nloc = numel(xv);
  yhat_loc = fit_predict(modelName, xv, yv, xv);
  SSE_loc = sum((yv - yhat_loc).^2);
  SST_loc = sum((yv - mean(yv)).^2);
  if SST_loc == 0
    R2_loc = NaN;
  else
    R2_loc = 1 - SSE_loc / SST_loc;
  end
  cv_loc = kfold_rmse(@(xtr,ytr,xp) fit_predict(modelName, xtr, ytr, xp), xv, yv, kfold);
  resm = struct('R2', R2_loc, 'cvRMSE', cv_loc);
end

% results on the original set (use bestModel already defined)
res_all = compute_metrics_model(bestModel, x, y, kfold);
R2_before = res_all.R2;

% re-estimate without influential points
x_clean = x; y_clean = y;
x_clean(idx_remove) = []; y_clean(idx_remove) = [];
res_clean = compute_metrics_model(bestModel, x_clean, y_clean, kfold);
R2_after = res_clean.R2;

fprintf('\nRemoving indices: %s -> Observed Delta R2 = %.4g\n', mat2str(idx_remove), R2_after - R2_before);
fprintf('CV_RMSE before = %.4g, after = %.4g (lower is better)\n', res_all.cvRMSE, res_clean.cvRMSE);

% permutation test (keep k and compare Delta R2)
B = 2000; count = 0; nobs = numel(y);
if k >= nobs
  warning('k >= n: permutation test not applicable.');
  p_perm_influence = NaN;
else
  for b = 1:B
    idx_rand = randperm(nobs, k);
    xr = x; yr = y;
    xr(idx_rand) = []; yr(idx_rand) = [];
    r_rand = compute_metrics_model(bestModel, xr, yr, kfold);
    if (r_rand.R2 - R2_before) >= (R2_after - R2_before)
      count = count + 1;
    end
  end
  p_perm_influence = (count + 1) / (B + 1);
end
fprintf('Permutation test for influence p-value (k=%d, B=%d): %.4g\n', k, B, p_perm_influence);

% --- Residuals histogram (compatible Octave/MATLAB)
subplot(2,3,5);
nbins = 15;
[counts, centers] = hist(resid, nbins);
bar(centers, counts, 'FaceColor', [0.7 0.7 0.9], 'EdgeColor', 'k');
xlabel('Residuals'); ylabel('Frequency'); title('Residuals histogram'); grid on;

% CV RMSE bar
subplot(2,3,6);
bar(cellfun(@(n) res.(n).cvRMSE, modelNames));
set(gca,'XTickLabel',modelNames); xtickangle(45); title('CV RMSE by model');

fprintf('\nDiagnostics plotted. Review residuals, Cook''s D and CV RMSE.\n');

end

%% ---------------- Helper functions ----------------

function ypred = fit_predict(modelName, xTrain, yTrain, xPred)
% Fit model on (xTrain,yTrain) and predict on xPred
% xPred can be scalar or vector
x = xTrain; y = yTrain;
switch modelName
  case 'Linear'
    p = polyfit(x,y,1);
    ypred = polyval(p, xPred);
  case 'LogLog'
    p = polyfit(log(x), log(y), 1);
    A = exp(p(2)); k = p(1);
    ypred = A .* xPred.^k;
  case 'Exponential'
    q = polyfit(x, log(y), 1);
    a = exp(q(2)); b = q(1);
    ypred = a .* exp(b .* xPred);
  case 'Hyperbola'
    q = polyfit(1./x, y, 1);
    b = q(1); a = q(2);
    ypred = a + b .* (1./xPred);
  case 'PowerNL'
    q = polyfit(log(x), log(y), 1);
    A = exp(q(2)); k = q(1);
    ypred = A .* xPred.^k;
  otherwise
    ypred = NaN(size(xPred));
end
end

function rmse = kfold_rmse(fitfun, x, y, k)
% k-fold cross-validation RMSE (compatible with Octave/MATLAB without toolboxes)
% fitfun: handle @(xTrain,yTrain,xPred) -> yPred
n = numel(x);
if k <= 1 || k > n
  error('k must be an integer between 2 and n');
end

% random indices and partition into k folds (nearly equal sizes)
perm = randperm(n);
foldSizes = floor(n / k) * ones(1,k);
remainder = mod(n, k);
for i = 1:remainder
  foldSizes(i) = foldSizes(i) + 1;
end

idx = cell(k,1);
pos = 1;
for i = 1:k
  idx{i} = perm(pos:(pos + foldSizes(i) - 1));
  pos = pos + foldSizes(i);
end

errs = zeros(k,1);
for i = 1:k
  testIdx = idx{i};
  trainIdx = setdiff(1:n, testIdx);
  xtr = x(trainIdx); ytr = y(trainIdx);
  xts = x(testIdx); yts = y(testIdx);
  yhat = fitfun(xtr, ytr, xts);
  errs(i) = sqrt(mean((yts - yhat).^2));
end
rmse = mean(errs);
end

function bootParams = bootstrap_params(x, y, modelName, params0, B)
% Residual bootstrap for parameter uncertainty
n = numel(x);
yfit = fit_predict(modelName, x, y, x);
resid = y - yfit;
bootParams = zeros(B, numel(params0));
opts = optimset('MaxFunEvals',1e4,'MaxIter',5e3,'Display','off');
for b=1:B
  eps = resid(randi(n, n, 1));
  yb = yfit + eps;
  switch modelName
    case 'Linear'
      p = polyfit(x, yb, 1);
      bootParams(b,:) = p;
    case 'LogLog'
      q = polyfit(log(x), log(yb), 1);
      bootParams(b,:) = [q(1), q(2)];
    case 'Exponential'
      q = polyfit(x, log(yb), 1);
      bootParams(b,:) = [q(1), q(2)];
    case 'Hyperbola'
      q = polyfit(1./x, yb, 1);
      bootParams(b,:) = [q(1), q(2)];
    case 'PowerNL'
      q = polyfit(log(x), log(yb), 1);
      A = exp(q(2)); k = q(1);
      bootParams(b,:) = [A, k];
    otherwise
      bootParams(b,:) = NaN(size(params0));
  end
end
end

function [leverage, cookD] = cooks_distance(x, y, modelName, params)
% Approximate Cook's distance for linearizable models
n = numel(x);
yhat = fit_predict(modelName, x, y, x);
resid = y - yhat;
sigma2 = var(resid);
switch modelName
  case 'Linear'
    X = [ones(n,1), x];
  case 'LogLog'
    X = [ones(n,1), log(x)];
  case 'Exponential'
    X = [ones(n,1), x];
  case 'Hyperbola'
    X = [ones(n,1), 1./x];
  case 'PowerNL'
    X = [ones(n,1), log(x)];
  otherwise
    X = [ones(n,1), x];
end
H = X * ((X' * X) \ X');
leverage = diag(H);
p = size(X,2);
cookD = (resid.^2 ./ (p * sigma2)) .* (leverage ./ (1 - leverage).^2);
end

function s = build_equation_string(modelName, params)
% Helper to build a readable equation string for display
switch modelName
  case 'Linear'
    p = params; s = sprintf('y = %.4g * x %+.4g', p(1), p(2));
  case 'LogLog'
    q = params; s = sprintf('y = A * x^k  (k=%.4g, logA=%.4g)', q(1), q(2));
  case 'Exponential'
    q = params; s = sprintf('y = a * exp(b*x)  (b=%.4g, loga=%.4g)', q(1), q(2));
  case 'Hyperbola'
    q = params; s = sprintf('y = a + b*(1/x)  (b=%.4g, a=%.4g)', q(1), q(2));
  case 'PowerNL'
    p = params; s = sprintf('y = A * x^k  (A=%.4g, k=%.4g)', p(1), p(2));
  otherwise
    s = 'Equation not available';
end
end

