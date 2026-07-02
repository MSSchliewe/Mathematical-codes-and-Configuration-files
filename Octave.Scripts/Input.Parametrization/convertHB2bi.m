function convertHB2bi(zbottom, g, start_time)
% convertHB2bi.m - Octave/MATLAB compatible
% The script fits a bi-viscosity model to a Herschel-Bulkley rheology
% while keeping gtrans_ref fixed. Optimization is performed only on
% eta_uny, eta_yd and alpha. (Original: convertHB2bi.m)
%
% Note: This file is a direct translation of the original Portuguese
% comments and messages into English. The numerical logic is unchanged.

clear; close all; clc;

%% ------------------ PARAMETERS (edit here) ------------------------------
ID    = 'Data112';
rho   = 1983.98;         % kg/m^3
tau_y = 415.238;         % Pa
K     = 38.1866;         % Pa.s^n
n     = 0.682221;        % dimensionless

gmin = 0.05;             % s^-1 (minimum shear rate)
gmax = 31.45;            % s^-1 (maximum shear rate)
Npts = 50;               % points in the grid (starting points)

% continuous mesh for detection (if you already have gtrans_ref, ignore this step)
Ndense_full = 5000;

% Optimization parameters (now gtrans is fixed)
alpha_init = 8;          % initial guess for alpha
penalty_strength = 1e6;  % strong penalty to force coincidence at gtrans_ref
opts_fmin.MaxIter = 2000;
opts_fmin.MaxFunEvals = 5000;

% If you already have gtrans_ref calculated externally, set it here:
% gtrans_ref = <continuous_value>;
% If you want to recalculate via global power-law, leave gtrans_ref = [] and the script will compute it.
gtrans_ref = [];         % if empty, the script computes gtrans_ref automatically (continuous method)

% Output files
out_csv = sprintf('convertHB2bi_%s_summary_fixedgtrans.csv', char(ID));
out_curve = sprintf('curve_%s_fixedgtrans.csv', char(ID));
out_png = sprintf('convertHB2bi_%s_plot_fixedgtrans.png', char(ID));

epsv = 1e-14;
% -------------------------------------------------------------------------

%% ------------------ Grid and HB -----------------------------------------
if ~(gmin > 0 && gmax > gmin)
  error('Check gmin and gmax: 0 < gmin < gmax');
end
gamma = logspace(log10(gmin), log10(gmax), Npts)';   % column (initial points)
ggeom = exp(mean(log(gamma)));

eta_HB = (tau_y ./ gamma) + K .* (gamma.^(n-1));
eta_HB = max(eta_HB, epsv);

fprintf('Grid: %d log-spaced points between %.3e and %.3e s^-1\n', Npts, gmin, gmax);

%% ------------------ If needed: detect gtrans_ref (continuous method) -
if isempty(gtrans_ref)
  % global power-law fit (log-log)
  x_all = log10(gamma);
  y_all = log10(eta_HB);
  p_all = polyfit(x_all, y_all, 1);
  b_all = p_all(1); A_all = p_all(2); a_all = 10^(A_all);
  % continuous mesh and S(g)
  gamma_dense = linspace(gmin, gmax, Ndense_full)';
  S_dense = abs(a_all * b_all .* (gamma_dense.^(b_all - 1)) + epsv);
  Smin = min(S_dense); Smax = max(S_dense);
  S_target = sqrt(Smin * Smax);
  Sfun = @(g) abs(a_all * b_all .* (g.^(b_all - 1)) + epsv);
  obj_target = @(g) abs(Sfun(g) - S_target);
  opts_bnd = optimset('TolX',1e-12,'TolFun',1e-12,'Display','off');
  gtrans_ref = fminbnd(obj_target, gmin, gmax, opts_bnd);
  fprintf('Detected continuous gtrans_ref = %.12e s^-1 (automatic detection)\n', gtrans_ref);
else
  fprintf('Using provided gtrans_ref = %.12e s^-1\n', gtrans_ref);
end

% evaluate eta_HB at gtrans_ref analytically
eta_HB_at_gtrans = (tau_y / gtrans_ref) + K * (gtrans_ref^(n-1));

%% ------------------ Bi-viscosity fit with fixed gtrans -----------------
% initial plateau values at extremes
eta_gmin = (tau_y / gmin) + K * (gmin^(n-1));
eta_gmax = (tau_y / gmax) + K * (gmax^(n-1));

% parameter vector to optimize: [log10(eta_uny), log10(eta_yd), alpha]
p0_fix = [ log10(eta_gmin + epsv), log10(eta_gmax + epsv), alpha_init ];

% bi-viscosity model with fixed gtrans
bi_model_fix = @(plog, gg) ...
    ( 10.^(plog(1)) + (10.^(plog(2)) - 10.^(plog(1))) .* ...
      (0.5*(1 + tanh(plog(3)*(log10(gg+epsv) - log10(gtrans_ref+epsv)))) ) );

% weights: more weight near gtrans_ref
wvec = ones(size(gamma));
sigma_log = 0.6;
wvec = wvec .* (1 + 4*exp(-((log10(gamma)-log10(gtrans_ref)).^2)/(2*sigma_log^2)));

% cost with penalty forcing coincidence at gtrans_ref
cost_fix = @(plog) ...
    ( sum( wvec .* ( log10( bi_model_fix(plog, gamma) + epsv ) - log10(eta_HB + epsv) ).^2 ) ...
      + penalty_strength * ( log10( bi_model_fix(plog, gtrans_ref) + epsv ) - log10( eta_HB_at_gtrans + epsv ) )^2 ...
      + 1e-6 * ( max(0, 10.^(plog(1)) - max(eta_HB)*1e2 ) + max(0, min(eta_HB)/1e2 - 10.^(plog(2))) ) );

opts = optimset('TolX',1e-8,'TolFun',1e-8,'MaxIter',opts_fmin.MaxIter,'MaxFunEvals',opts_fmin.MaxFunEvals,'Display','off');

[plog_opt_fix, fval_opt_fix, exitflag_fix, output_fix] = fminsearch(cost_fix, p0_fix, opts);

% extract optimized parameters
eta_uny_opt = 10.^(plog_opt_fix(1));
eta_yd_opt  = 10.^(plog_opt_fix(2));
alpha_opt   = plog_opt_fix(3);
gtrans_opt  = gtrans_ref;   % fixed

% build pre-scale model
eta_model_pre = bi_model_fix(plog_opt_fix, gamma);

fprintf('\nOptimization with fixed gtrans completed:\n');
fprintf('eta_uny_opt (pre-scale) = %.6e Pa.s, eta_yd_opt (pre-scale) = %.6e Pa.s\n', eta_uny_opt, eta_yd_opt);
fprintf('alpha_opt = %.6e\n', alpha_opt);

%% ------------------ Force exact coincidence at gtrans_ref (scaling) ----
% evaluate model at gtrans_ref and compute scale factor to match exactly
eta_model_at_gtrans_pre = bi_model_fix(plog_opt_fix, gtrans_ref);
if eta_model_at_gtrans_pre <= 0
  f_scale = 1.0;
else
  f_scale = eta_HB_at_gtrans / (eta_model_at_gtrans_pre + epsv);
  % limit scale to avoid extreme values
  f_scale = max(min(f_scale, 10), 0.1);
end

% apply scale to plateaus (keeping their ratio)
eta_uny_opt = eta_uny_opt * f_scale;
eta_yd_opt  = eta_yd_opt  * f_scale;

% rebuild final model with scaled parameters
plog_final = [ log10(eta_uny_opt), log10(eta_yd_opt), alpha_opt ];
eta_model = bi_model_fix(plog_final, gamma);

% confirm numerical coincidence at gtrans_ref
eta_model_at_gtrans = bi_model_fix(plog_final, gtrans_ref);
diff_at_gtrans = abs(eta_model_at_gtrans - eta_HB_at_gtrans);
fprintf('After scaling: eta_model(gtrans_ref)=%.12e, eta_HB(gtrans_ref)=%.12e, diff=%.12e\n', ...
        eta_model_at_gtrans, eta_HB_at_gtrans, diff_at_gtrans);

%% ------------------ Metrics and diagnostics ------------------------------
% RMSE log before/after (diagnostic)
eta_model_init = 10.^(p0_fix(1)) + (10.^(p0_fix(2)) - 10.^(p0_fix(1))) .* (0.5*(1 + tanh(alpha_init*(log10(gamma+epsv) - log10(gtrans_ref+epsv)))));
rmse_init = sqrt(mean( (log10(eta_model_init+epsv) - log10(eta_HB+epsv)).^2 ));
rmse_opt  = sqrt(mean( (log10(eta_model+epsv) - log10(eta_HB+epsv)).^2 ));

% Bi multiplier and reference viscosity
eta_at_gtrans = eta_model_at_gtrans;
Bi_multi = (eta_at_gtrans * gtrans_ref) / (tau_y + epsv);
visco_value_ref = eta_at_gtrans / rho;

fprintf('RMSE log10 before = %.6e, after = %.6e\n', rmse_init, rmse_opt);

%% ------------------ Save results ----------------------------------
fid = fopen(out_csv,'w');
if fid < 0, error('Could not open %s for writing', out_csv); end
fprintf(fid,'ID,Rho,tau_y,K,n,gmin,gmax,Npts,ggeom,gtrans_ref,eta_uny_opt,eta_yd_opt,alpha_opt,eta_gtrans_Pa_s,visco_value_m2s,Bi_multi,rmse_init,rmse_opt\n');
fprintf(fid,'%s,%.6f,%.12e,%.12e,%.12e,%.12e,%.12e,%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n', ...
  ID, rho, tau_y, K, n, gmin, gmax, Npts, ggeom, gtrans_ref, eta_uny_opt, eta_yd_opt, alpha_opt, eta_at_gtrans, visco_value_ref, Bi_multi, rmse_init, rmse_opt);
fclose(fid);
fprintf('Summary saved to %s\n', out_csv);

% detailed curve (original grid)
fid2 = fopen(out_curve,'w');
if fid2 < 0, error('Could not open %s for writing', out_curve); end
fprintf(fid2,'gamma,eta_HB,eta_model\n');
for j=1:length(gamma)
  fprintf(fid2,'%.12e,%.12e,%.12e\n', gamma(j), eta_HB(j), eta_model(j));
end
fclose(fid2);
fprintf('Detailed curve saved to %s\n', out_curve);

%% ------------------ Diagnostic plots (Octave compatible) -------------
figure('Color','w','Position',[100 100 1000 700]);

% main panel: log-log scale (HB and final model)
subplot(2,1,1);
loglog(gamma, eta_HB, 'ko-','LineWidth',1,'MarkerSize',4,'MarkerFaceColor','k'); hold on;
loglog(gamma, eta_model, 'b--','LineWidth',1.4);
% mark gtrans_ref using the same function of the final model
eta_mark = eta_model_at_gtrans;
loglog(gtrans_ref, eta_mark, 'ro', 'MarkerFaceColor','r', 'MarkerSize',8);
xlabel('\gammȧ (s^{-1})'); ylabel('\mu_{app} (Pa·s)');
title(sprintf('HB and bi-viscosity fit (fixed gtrans) - ID=%s', char(ID)));
legend({'Herschel-Bulkley model','Bi-viscosity model','\gammȧ_{trans} Yield vs Power Law'}, 'location', 'northeast');
grid on; box on;
xlim([gmin gmax]);

% diagnostic panel: RMSE before/after
subplot(2,1,2);
bar([rmse_init rmse_opt]);
set(gca,'XTickLabel',{'RMSE init','RMSE opt'});
ylabel('RMSE (log10)');
title('Fit diagnostics');
grid on; box on;

print('-dpng','-r150', out_png);
fprintf('Figure saved to %s\n', out_png);

%% ------------------ Final message -------------------------------------
fprintf('\n=== FINAL RESULT ===\n');
fprintf('gtrans_ref (fixed) = %.12e s^-1\n', gtrans_ref);
fprintf('eta_uny_opt = %.12e Pa.s, eta_yd_opt = %.12e Pa.s, alpha_opt = %.12e\n', eta_uny_opt, eta_yd_opt, alpha_opt);
fprintf('eta(gtrans_ref) = %.12e Pa.s (final model)\n', eta_at_gtrans);
fprintf('Bi_multi = %.12e\n', Bi_multi);
fprintf('<visco_value>%.8e</visco_value>\n', eta_HB_at_gtrans / rho);
fprintf('RMSE log10 before = %.6e, after = %.6e\n', rmse_init, rmse_opt);
fprintf('Processing completed for ID = %s\n', char(ID));

