function pval = ks_boot(xdata, cdf_fun, param_est_fun, B)
  if nargin < 4; B = 500; end
  Nloc = numel(xdata);
  params0 = param_est_fun(xdata);
  F_emp = @(z) sum(xdata <= z)/Nloc;
  zvals = sort(xdata);
  D_emp = max(abs(arrayfun(@(z) F_emp(z) - cdf_fun(z, params0), zvals)));
  D_boot = zeros(B,1);
  for bb=1:B
    idxb = randi(Nloc, Nloc, 1);
    xb = xdata(idxb);
    params_b = param_est_fun(xb);
    sim = simulate_from_params(Nloc, params_b);
    params_sim = param_est_fun(sim);
    F_sim = @(z) sum(sim <= z)/Nloc;
    zsim = sort(sim);
    D_boot(bb) = max(abs(arrayfun(@(z) F_sim(z) - cdf_fun(z, params_sim), zsim)));
  end
  pval = mean(D_boot >= D_emp);
end
