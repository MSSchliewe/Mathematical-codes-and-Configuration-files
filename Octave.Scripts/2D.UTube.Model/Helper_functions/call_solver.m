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
