function Regression()
% Regression
% Revised version of the regression script for yield stress vs Cv.
% Tests models: Linear, LogLog, Exponential, Hyperbola, PowerNL.
% Computes SSE, R2, AICc, BIC, bootstrap CI, k-fold CV RMSE and produces plots.
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

%% --- Data (replace with file reading if preferred)
data = [ ...
-0.0539420624363028 -6.59580995439468
-0.0213574554151993 -7.94739513347386
-0.0128775610140182 -5.52977790834234
0.244505746166178   -3.59750062832853
0.2636403729715 1.39803269650024
1.17916871052176    -3.25658548024453
3.17955270647168    2.51551667923184
3.30073595694812    1.3448393537893
3.38540669575583    1.40233255889263
3.64082039762014    0.929617534244163
3.9852957717517 -0.625862433754349
4.01073829036707    1.24534984477969
4.105243288091  0.274604431658449
4.12800497371471    3.2134611921667
4.91747413682132    3.7205379921142
4.9744552866415 2.96825829655324
5.2062185248734 0.893447807555208
5.3218336778111 3.13221493190641
5.55473320492581    3.70686916455918
5.88871451007289    3.57008184144832
5.99843271295173    2.966988089421
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

for i = 1:numel(models)
  name = models{i};
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
        [popt, ~, exitflag] = fminsearch(sseFun, p0, opts);
        params = popt;
        yfit = popt(1) .* x.^popt(2);
        Xtr = log(x); Ytr = log(y); % for auxiliary inference
        pCount = 2;
      end
  end

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
  if exist('Xtr','var') && exist('Ytr','var') && numel(Xtr)==n
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

%% --- Summary report (equation and explicit CIs)
fprintf('\n=== Model comparison summary ===\n');
fprintf('%-12s %8s %12s %12s %10s\n','Model','R2','AICc','BIC','CV_RMSE');
for i=1:numel(modelNames)
  m = res.(modelNames{i});
  fprintf('%-12s %8.3f %12.3g %12.3g %10.3g\n', modelNames{i}, m.R2, m.AICc, m.BIC, m.cvRMSE);
end

fprintf('\nBest model by AICc+CV: %s\n\n', bestModel);

% --- Build readable equation for the best model
switch bestModel
  case 'Linear'
    p = res.(bestModel).params; % [slope, intercept]
    eqStr = sprintf('log(tau_y) = %.4g * Cv %+.4g', p(1), p(2));
    slopeName = 'slope';
    slopeVal = p(1);
    % bootstrap params: rows correspond to [slope, intercept]
    bootParamNames = {'slope','intercept'};
  case 'LogLog'
    q = res.(bestModel).params; % [k, logA]
    k = q(1); A = exp(q(2));
    eqStr = sprintf('log(tau_y) = log(A) + k * log(Cv)   (A=%.4g, k=%.4g)', A, k);
    slopeName = 'k (log-log slope)';
    slopeVal = k;
    bootParamNames = {'k','logA'};
  case 'Exponential'
    q = res.(bestModel).params; % [b, log(a)]
    b = q(1); a = exp(q(2));
    eqStr = sprintf('log(tau_y) = log(a) + b * Cv   (a=%.4g, b=%.4g)', a, b);
    slopeName = 'b (exp slope)';
    slopeVal = b;
    bootParamNames = {'b','loga'};
  case 'Hyperbola'
    q = res.(bestModel).params; % [b, a] where y = a + b*(1/x)
    b = q(1); a = q(2);
    eqStr = sprintf('log(tau_y) = %.4g + %.4g * (1/Cv)', a, b);
    slopeName = 'b (1/Cv coeff)';
    slopeVal = b;
    bootParamNames = {'b','a'};
  case 'PowerNL'
    p = res.(bestModel).params; % [A, k]
    A = p(1); k = p(2);
    eqStr = sprintf('log(tau_y) ~ log(A) + k * log(Cv)   (A=%.4g, k=%.4g)', A, k);
    slopeName = 'k (power exponent)';
    slopeVal = k;
    bootParamNames = {'A','k'};
  otherwise
    eqStr = 'Equation not available';
    slopeName = 'slope';
    slopeVal = NaN;
    bootParamNames = {};
end

fprintf('Best model equation: %s\n', eqStr);

% --- Confidence intervals via bootstrap (already computed in bootParams)
% bootParams: B x nParams
if exist('bootParams','var') && ~isempty(bootParams)
  ci_boot = prctile(bootParams, [2.5 97.5]);
  % print CI for each parameter (column by column)
  fprintf('\nBootstrap 95%% CI for parameters (rows = lower, upper):\n');
  for j = 1:size(ci_boot,2)
    name = 'param';
    if j <= numel(bootParamNames), name = bootParamNames{j}; end
    fprintf(' %s : [%.4g , %.4g]\n', name, ci_boot(1,j), ci_boot(2,j));
  end
  % If a parameter is identified as "slope", print it explicitly
  switch bestModel
    case {'Linear','LogLog','Exponential','Hyperbola'}
      slope_col = 1;
    case 'PowerNL'
      slope_col = 2;
    otherwise
      slope_col = 1;
  end
  if slope_col <= size(ci_boot,2)
    fprintf('\nCI 95%% for %s: [%.4g , %.4g]\n', slopeName, ci_boot(1,slope_col), ci_boot(2,slope_col));
  end
else
  fprintf('\nBootstrap parameters not available for CI.\n');
end

% --- Also print linearized CI if available (res.best.CI_slope)
if isfield(best,'CI_slope') && ~all(isnan(best.CI_slope))
  fprintf('CI slope (linearized): [%.4g, %.4g]\n', best.CI_slope(1), best.CI_slope(2));
end

fprintf('p-value (linearized test): %.4g, reject H0: %d\n', best.pval, best.rejectH0);
fprintf('\n');

%% --- Diagnostic plots for the best model
xx = linspace(min(x), max(x), 200);
yy = fit_predict(bestModel, x, y, xx); % predict on grid

figure('Name','Best fit and diagnostics','NumberTitle','off','Position',[100 100 1200 600]);
subplot(2,3,1);
scatter(x,y,36,'b','filled'); hold on;
plot(xx, yy, 'r-', 'LineWidth', 2);
xlabel(xLabel); ylabel(yLabel); title(['Best model: ', bestModel]); grid on; hold off;

% --- Residuals vs fitted (robust)
% ensure res and bestModel exist
if ~exist('res','var') || ~isstruct(res)
  error('Struct ''res'' not found. Run the fitting step before diagnostics.');
end
if ~exist('bestModel','var') || isempty(bestModel) || ~isfield(res, bestModel)
  % try to recover best choice in res.best or by AICc
  if isfield(res,'best')
    bestModel = res.best.modelName;
  else
    % fallback: choose model with smallest AICc
    names = fieldnames(res);
    aicc_vals = cellfun(@(n) res.(n).AICc, names);
    [~, idxmin] = min(aicc_vals);
    bestModel = names{idxmin};
  end
end

% compute yhat reliably using fit_predict (fit on full set)
try
  yhat = fit_predict(bestModel, x, y, x);  % returns vector same size as x
catch ME
  % fallback: try to use res.(bestModel).yfit if available
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
% draw horizontal line y = 0 (compatible with Octave/MATLAB)
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
% --- Integrated permutation test (uses fit_predict and kfold_rmse)
% top_idx must exist (from Top 4 block)
k = min(2, numel(top_idx));        % number of points to test (adjust if you want 1,2 or 3)
idx_remove = top_idx(1:k);

% Local function to compute R2 and CV_RMSE for a chosen model
function resm = compute_metrics_model(modelName, xv, yv, kfold)
  nloc = numel(xv);
  % prediction on the set (yfit)
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
B = 2000; count = 0; n = numel(y);
if k >= n
  warning('k >= n: permutation test not applicable.');
  p_perm = NaN;
else
  for b = 1:B
    idx_rand = randperm(n, k);
    xr = x; yr = y;
    xr(idx_rand) = []; yr(idx_rand) = [];
    r_rand = compute_metrics_model(bestModel, xr, yr, kfold);
    if (r_rand.R2 - R2_before) >= (R2_after - R2_before)
      count = count + 1;
    end
  end
  p_perm = (count + 1) / (B + 1);
end
fprintf('Permutation test p-value (k=%d, B=%d): %.4g\n', k, B, p_perm);
% --- Residuals histogram (compatible Octave/MATLAB)
subplot(2,3,5);
nbins = 15;
% use hist for compatibility
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

