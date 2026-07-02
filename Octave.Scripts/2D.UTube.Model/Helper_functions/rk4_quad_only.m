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
