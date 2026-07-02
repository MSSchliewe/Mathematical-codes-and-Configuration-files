function Rheo_correlation(fname)
% Rheo_correlation - compute correlations and regressions from rheology CSV
% Usage:
%   Rheo_correlation()            % uses default 'HB_grossos.csv'
%   Rheo_correlation('file.csv')  % specify input CSV filename
%
% The input CSV must contain columns (case-sensitive after whitespace->underscore):
% id, protocol, Cv, Ff, gamma_min, gamma_max, geo_gamma, tau_y, n, K
% Optional column: notes
%
% Outputs (CSV files with the same prefix as input):
%  *_pearson_corr.csv
%  *_spearman_corr.csv
%  *_corr_CI_low.csv
%  *_corr_CI_high.csv
%  *_regression_logTy.csv
%  *_bivar_logK_n.csv
%  *_enriched.csv

if nargin < 1
  fname = 'HB_grossos.csv';
end
rng('default');

%% 0) Read CSV
fid = fopen(fname, 'r');
if fid < 0, error('Could not open %s', fname); end
hdrline = fgetl(fid);
if ~ischar(hdrline), fclose(fid); error('Empty file'); end
hdr = strtrim(strsplit(hdrline, ','));
ncol = numel(hdr);
colIndex = containers.Map();
for i = 1:ncol
  key = regexprep(strtrim(hdr{i}), '\s+', '_');
  colIndex(key) = i;
end

required = {'id','protocol','Cv','Ff','gamma_min','gamma_max','geo_gamma','tau_y','n','K'};
for k = 1:numel(required)
  if ~isKey(colIndex, required{k})
    fclose(fid);
    error('Missing required column: %s', required{k});
  end
end

dataLines = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
fclose(fid);
dataLines = dataLines{1};
Nlines = numel(dataLines);

cols = cell(1, ncol);
for c = 1:ncol, cols{c} = cell(Nlines,1); end
for r = 1:Nlines
  parts = strsplit(dataLines{r}, ',');
  if numel(parts) < ncol, parts(end+1:ncol) = {''}; end
  for c = 1:ncol
    cols{c}{r} = strtrim(parts{c});
  end
end

getCol = @(name) cols{colIndex(name)};
id_col = getCol('id'); protocol_col = getCol('protocol');
Cv_col = getCol('Cv'); Ff_col = getCol('Ff');
gamma_min_col = getCol('gamma_min'); gamma_max_col = getCol('gamma_max');
geo_gamma_col = getCol('geo_gamma'); tau_y_col = getCol('tau_y');
n_col = getCol('n'); K_col = getCol('K');
if isKey(colIndex,'notes'), notes_col = getCol('notes'); else notes_col = repmat({''}, Nlines,1); end

% helper: convert text column to numeric (accepts comma or dot decimal)
toNum = @(cellv) cellfun(@(s) str2double(strrep(s, ',', '.')), cellv);
Cv = toNum(Cv_col); Ff = toNum(Ff_col);
gamma_min = toNum(gamma_min_col); gamma_max = toNum(gamma_max_col);
geo_gamma = toNum(geo_gamma_col); tau_y = toNum(tau_y_col);
n = toNum(n_col); K = toNum(K_col);

% filter rows with critical missing values
valid_mask = ~isnan(tau_y) & ~isnan(K) & ~isnan(n) & ~isnan(Cv) & ~isnan(Ff);
removed = sum(~valid_mask);
if removed > 0, fprintf('Warning: %d rows removed due to critical NAs\n', removed); end

id = id_col(valid_mask); protocol = protocol_col(valid_mask);
Cv = Cv(valid_mask); Ff = Ff(valid_mask);
gamma_min = gamma_min(valid_mask); gamma_max = gamma_max(valid_mask);
geo_gamma = geo_gamma(valid_mask); tau_y = tau_y(valid_mask);
n = n(valid_mask); K = K(valid_mask); notes = notes_col(valid_mask);

N = numel(tau_y);
if N == 0, error('No valid rows after filtering'); end
% if percentages were provided as 0-100, convert to fraction
if max(Cv) > 1, Cv = Cv/100; end
if max(Ff) > 1, Ff = Ff/100; end

%% 1) Transformations
logTy = log(tau_y); logK = log(K);
gamma0 = geo_gamma; gamma0(isnan(gamma0) | gamma0<=0) = 1;
V0 = K .* (gamma0 .^ n); logV0 = log(V0);
% sensitivity-like factors (useful for interpretation)
E_K = (K .* (gamma0.^n)) ./ (tau_y + K .* (gamma0.^n));
E_n = (K .* (gamma0.^n) .* log(gamma0)) ./ (tau_y + K .* (gamma0.^n));

%% 2) Variable matrix
vars = [logTy(:), logK(:), n(:), Cv(:), Ff(:), logV0(:)];
varNames = {'logTy','logK','n','Cv','Ff','logV0'};
valid_rows = all(~isnan(vars), 2);
vars_clean = vars(valid_rows, :);
Nc = size(vars_clean,1);

%% 3) Pearson correlation matrix
Rpear = corrcoef(vars_clean);
csvwrite([erase(fname,'.csv') '_pearson_corr.csv'], Rpear);

%% 4) Rank-based Spearman (column-wise ranks)
U = zeros(size(vars_clean));
for j = 1:size(vars_clean,2)
  col = vars_clean(:,j);
  if exist('tiedrank','file')
    U(:,j) = tiedrank(col);
  else
    [~,~,ranks] = unique(col);
    U(:,j) = ranks;
  end
end
Rspear = corrcoef(U);
csvwrite([erase(fname,'.csv') '_spearman_corr.csv'], Rspear);

%% 5) Bootstrap CI for Pearson correlations
B = 2000;
corr_boot = zeros(B, size(vars_clean,2), size(vars_clean,2));
for b = 1:B
  idx = randi(Nc, Nc, 1);
  samp = vars_clean(idx, :);
  corr_boot(b,:,:) = corrcoef(samp);
end
ci_low = zeros(size(vars_clean,2)); ci_high = zeros(size(vars_clean,2));
for i = 1:size(vars_clean,2)
  for j = 1:size(vars_clean,2)
    rvec = squeeze(corr_boot(:,i,j));
    ci = prctile(rvec, [2.5 97.5]);
    ci_low(i,j) = ci(1); ci_high(i,j) = ci(2);
  end
end
csvwrite([erase(fname,'.csv') '_corr_CI_low.csv'], ci_low);
csvwrite([erase(fname,'.csv') '_corr_CI_high.csv'], ci_high);

%% 6) Partial correlation: logK vs n controlling for Cv and Ff
X = [ones(N,1), Cv, Ff];
b1 = X \ logK(:); res_logK = logK(:) - X * b1;
b2 = X \ n(:); res_n = n(:) - X * b2;
r_partial_mat = corrcoef(res_logK, res_n);
partial_corr_value = r_partial_mat(1,2);

%% 7) Regression logTy ~ Cv + Ff + logV0 (robustfit fallback to OLS)
if exist('robustfit','file')
  [b_rob, stats_rob] = robustfit([Cv, Ff, logV0], logTy);
else
  Xreg = [ones(N,1), Cv, Ff, logV0];
  b_rob = Xreg \ logTy;
  stats_rob = [];
end
% bootstrap coefficients for CI
boot_b = zeros(B, numel(b_rob));
for b = 1:B
  idx = randi(N, N, 1);
  if exist('robustfit','file')
    bb = robustfit([Cv(idx), Ff(idx), logV0(idx)], logTy(idx));
  else
    Xb = [ones(N,1), Cv(idx), Ff(idx), logV0(idx)];
    bb = Xb \ logTy(idx);
  end
  boot_b(b,:) = bb';
end
ci_b_low = prctile(boot_b, 2.5); ci_b_high = prctile(boot_b, 97.5);

%% 8) Bivariate summary for logK and n
mu_bivar = [mean(logK), mean(n)];
Sigma_bivar = cov([logK(:), n(:)]);
boot_mu = zeros(B,2);
for b = 1:B
  idx = randi(N, N, 1);
  s = [logK(idx), n(idx)];
  boot_mu(b,:) = mean(s);
end
ci_mu_logK = prctile(boot_mu(:,1), [2.5 97.5]);
ci_mu_n = prctile(boot_mu(:,2), [2.5 97.5]);

%% 9) Save outputs
csvwrite([erase(fname,'.csv') '_pearson_corr.csv'], Rpear);
csvwrite([erase(fname,'.csv') '_spearman_corr.csv'], Rspear);
csvwrite([erase(fname,'.csv') '_corr_CI_low.csv'], ci_low);
csvwrite([erase(fname,'.csv') '_corr_CI_high.csv'], ci_high);

fid = fopen([erase(fname,'.csv') '_regression_logTy.csv'],'w');
fprintf(fid, 'term,beta,CI_low,CI_high\n');
terms = {'intercept','Cv','Ff','logV0'};
for k = 1:numel(terms)
  fprintf(fid, '%s,%.6g,%.6g,%.6g\n', terms{k}, b_rob(k), ci_b_low(k), ci_b_high(k));
end
fclose(fid);

fid = fopen([erase(fname,'.csv') '_bivar_logK_n.csv'],'w');
fprintf(fid, 'mu_logK,mu_n,var_logK,var_n,cov_logK_n\n');
fprintf(fid, '%.6g,%.6g,%.6g,%.6g,%.6g\n', mu_bivar(1), mu_bivar(2), Sigma_bivar(1,1), Sigma_bivar(2,2), Sigma_bivar(1,2));
fclose(fid);

fid = fopen([erase(fname,'.csv') '_enriched.csv'],'w');
fprintf(fid, 'id,protocol,Cv,Ff,gamma_min,gamma_max,geo_gamma,tau_y,n,K,logTy,logK,logV0,E_K,E_n,notes\n');
for i = 1:N
  fprintf(fid, '%s,%s,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%s\n', ...
    id{i}, protocol{i}, Cv(i), Ff(i), gamma_min(i), gamma_max(i), geo_gamma(i), tau_y(i), n(i), K(i), ...
    logTy(i), logK(i), logV0(i), E_K(i), E_n(i), notes{i});
end
fclose(fid);

fprintf('Rheo_correlation completed for %s. N=%d. Files saved with prefix %s_\n', fname, N, erase(fname,'.csv'));
end

