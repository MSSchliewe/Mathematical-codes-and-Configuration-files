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
