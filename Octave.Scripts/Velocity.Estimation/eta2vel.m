function eta2vel(zbottom, g, start_time)
% ETA2VEL - generate inlet files from the measured signal LabSignalAdjusted.csv
% Usage:
%   eta2vel()                       % uses zbottom=0.05, g=9.81, start_time=5.0
%   eta2vel(0.05, 9.81, 5.0)        % specifying inputs
%
% Requirements:
%   - File LabSignalAdjusted.csv in the current directory
%     format: time_seconds, eta_cm (no header)
%
% Outputs:
%   - velocity_components.csv
%   - velocitytimes3.xml

% -------------------- Mandatory inputs with defaults --------------------
if nargin < 1 || isempty(zbottom), zbottom = 0.05; end
if nargin < 2 || isempty(g), g = 9.81; end
if nargin < 3 || isempty(start_time), start_time = 5.0; end

% -------------------- Control flags (adjust if needed) --------------------
do_diagnostics = true;       % print detailed diagnostics
do_mask_test_disable = false;% true to test without mask (diagnostic)
do_band_test = false;        % true to test a larger band (1.0 Hz)
use_corr_factor = false;     % true to apply correction A_time/A_spec_sum (limited)
% mask parameters (adjustable)
delta = 0.01;                % transition ramp thickness (m)
min_depth = 1e-4;            % minimum depth to consider "water present" (m)
% default band around the peak (Hz)
band_default = 0.5;

% -------------------- Read measured signal --------------------
fname = 'LabSignalAdjusted3.csv';
if exist(fname,'file') ~= 2
    error('Arquivo %s nao encontrado no diretorio atual (%s).', fname, pwd);
end

% MATLAB/Octave compatible read (two columns)
fid = fopen(fname,'r');
C = textscan(fid, '%f%f', 'Delimiter', ',', 'CollectOutput', true);
fclose(fid);
M = C{1};
if isempty(M) || size(M,2) < 2
    error('Formato inesperado em %s. Esperado: time(s), eta(cm).', fname);
end
time_lab = M(:,1);
eta_lab_cm = M(:,2);
eta_lab = eta_lab_cm / 100;   % cm -> m

% -------------------- Determine fs, relative time and interpolate --------------------
% use median of dt from file for robustness
dt_file = median(diff(time_lab));
if dt_file <= 0
    error('Vetor de tempo em %s nao eh valido.', fname);
end
fs = 1 / dt_file;

% build relative time (starting at zero) and regular vector
t_rel = (0:dt_file:(time_lab(end) - time_lab(1)))';
Nt = length(t_rel);

% interpolate measured signal to t_rel (adjusting reference)
eta_recon = interp1(time_lab - time_lab(1), eta_lab, t_rel, 'linear', 'extrap');

% define t_out and Nt_out used for export
t_out = t_rel;
Nt_out = Nt;

% compute mean of the signal (used for h)
mean_eta = mean(eta_recon);
h = mean_eta - zbottom;
if h <= 0
    warning('Profundidade média h <= 0 (%.6g m). Verifique zbottom e o sinal.', h);
end

% -------------------- Normalized FFT and initial diagnostics --------------------
Nfft = 2^nextpow2(Nt);
freqs = (0:Nfft-1)*(fs/Nfft);
Npos = floor(Nfft/2)+1;
fpos = freqs(1:Npos);

eta0 = eta_recon - mean(eta_recon);        % remove offset
X = fft(eta0, Nfft) / Nfft;                % correct normalization
Xpos = X(1:Npos);
A_spec = 2 * abs(Xpos);
A_spec(1) = abs(Xpos(1));
if rem(Nfft,2) == 0, A_spec(end) = abs(Xpos(end)); end
phases = angle(Xpos);

% temporal amplitude (peak) and spectral peak (ignore DC)
A_time = max(abs(eta0));
[~, ib_peak] = max(A_spec(2:end)); ib_peak = ib_peak + 1;
f_peak = fpos(ib_peak);
A_spec_peak = A_spec(ib_peak);

% print initial summary
if do_diagnostics
    fprintf('Leitura: %s | fs = %.6f Hz | Nt = %d | Nfft = %d\n', fname, fs, Nt, Nfft);
    fprintf('mean_eta = %.6g m ; h = %.6g m\n', mean_eta, h);
    fprintf('Pico FFT: f_peak = %.6g Hz ; A_spec_peak = %.6g m ; A_time (pico) = %.6g m\n', f_peak, A_spec_peak, A_time);
end

% -------------------- Select bins for velocity reconstruction --------------------
if do_band_test
    band = min(1.0, fs/2 - 1e-6);
else
    band = min(band_default, fs/2 - 1e-6);
end

bins_for_vel = find(fpos >= max(0.01, f_peak - band) & fpos <= f_peak + band);
if isempty(bins_for_vel)
    warning('Nenhum bin selecionado para velocidades; ampliando banda automaticamente.');
    bins_for_vel = find(fpos >= 0.01 & fpos <= min(2.0, fs/2));
end
freqs_vel = fpos(bins_for_vel);
amps_vel = A_spec(bins_for_vel);
phases_vel = phases(bins_for_vel);

% -------------------- Compute k (dispersion relation) --------------------
omega = 2*pi*freqs_vel;
k_vals = zeros(size(freqs_vel));
for i = 1:length(freqs_vel)
    om = omega(i);
    % initial guess shallow-water
    k = max(om^2 / g, 1e-6);
    for it = 1:200
        kh = k * h;
        tkh = tanh(kh);
        fval = g * k * tkh - om^2;
        dfk = g * tkh + g * k * (1 - tkh^2) * h;
        dk = -fval / dfk;
        k = k + dk;
        if abs(dk) < 1e-12, break; end
    end
    if ~isfinite(k) || k <= 0, k = max(om^2/g, 1e-6); end
    k_vals(i) = k;
end

% -------------------- Levels and vertical factors --------------------
z_levels_frac = [1/6, 3/6, 5/6];           % fractions (adjustable)
z_targets = zbottom + h * z_levels_frac;   % absolute elevations
z_rel = z_targets - zbottom;               % relative to the bottom

vx = zeros(Nt_out, 3);
vz = zeros(Nt_out, 3);

% build vx, vz by summing selected components
for ib = 1:length(freqs_vel)
    om = omega(ib);
    A = amps_vel(ib);
    phi = phases_vel(ib);
    k = k_vals(ib);
    denom = sinh(k * h);
    if denom == 0, denom = 1e-12; end
    fac_u = cosh(k * z_rel) ./ denom;
    fac_w = sinh(k * z_rel) ./ denom;
    cos_term = cos(om * t_out + phi);
    sin_term = sin(om * t_out + phi);
    for iz = 1:3
        vx(:,iz) = vx(:,iz) + (A * om * fac_u(iz)) .* cos_term;
        vz(:,iz) = vz(:,iz) + (A * om * fac_w(iz)) .* sin_term;
    end
end

% -------------------- Mask to avoid imposing velocity "in air" --------------------
mask_levels = zeros(Nt_out, 3);
for iz = 1:3
    depth_level = z_targets(iz);
    rel = eta_recon - depth_level;   % positive when there is column above the level
    m = zeros(Nt_out,1);
    idx1 = find(rel >= delta); m(idx1) = 1;
    idx0 = find(rel <= -delta); m(idx0) = 0;
    idxt = find(rel > -delta & rel < delta);
    if ~isempty(idxt)
        m(idxt) = 0.5 * (1 + rel(idxt) / delta);
    end
    m(rel <= min_depth) = 0;
    mask_levels(:,iz) = m;
    vx(:,iz) = vx(:,iz) .* m;
    vz(:,iz) = vz(:,iz) .* m;
end

mag = sqrt(vx.^2 + vz.^2);

% -------------------- Automatic diagnostics --------------------
if do_diagnostics
    % theoretical estimate of u_peak
    om_peak = 2*pi*f_peak;
    [~, idx_kpeak] = min(abs(freqs_vel - f_peak));
    if isempty(idx_kpeak), idx_kpeak = 1; end
    k_peak = k_vals(idx_kpeak);
    fac_surface = cosh(k_peak * (z_targets(1)-zbottom)) / max(sinh(k_peak*h),1e-12);
    u_peak_est = A_spec_peak * om_peak * fac_surface;
    fprintf('\n--- QUICK DIAGNOSTIC ---\n');
    fprintf('f_peak = %.6f Hz ; A_spec_peak = %.6e m ; omega = %.6e rad/s\n', f_peak, A_spec_peak, om_peak);
    fprintf('Theoretical estimate u_peak = A_spec_peak * omega * fac_surface = %.6e m/s\n', u_peak_est);
    fprintf('max(|vx|) = %.6e m/s ; max(mag) = %.6e m/s\n', max(abs(vx(:))), max(mag(:)));
    for iz=1:3
        frac_active = mean(mask_levels(:,iz) > 0.5);
        idx_active = find(mask_levels(:,iz) > 0.5);
        if isempty(idx_active)
            mean_v_when_active = 0;
        else
            mean_v_when_active = mean(abs(vx(idx_active, iz)));
        end
        fprintf('Level %d (z=%.4g m): frac_active=%.3f ; mean(|vx| when active)=%.6e m/s\n', iz, z_targets(iz), frac_active, mean_v_when_active);
    end
    nshow = min(6, Nt_out);
    fprintf('\nSample of times and magnitudes (first %d lines):\n', nshow);
    for i=1:nshow
        fprintf('time_xml=%.6f ; mag = [%.6e, %.6e, %.6e]\n', t_out(i)+start_time, mag(i,1), mag(i,2), mag(i,3));
    end
    fprintf('--- END DIAGNOSTIC ---\n\n');
end

% -------------------- Adjustment options (diagnostic/safety) --------------------
% apply limited correction based on temporal vs spectral energy
if use_corr_factor
    A_spec_sum = sum(amps_vel);
    if A_spec_sum > 0
        corr_factor = A_time / max(A_spec_sum, eps);
    else
        corr_factor = 1.0;
    end
    corr_factor = min(max(corr_factor, 0.8), 1.5); % limit correction
    vx = vx * corr_factor;
    vz = vz * corr_factor;
    mag = sqrt(vx.^2 + vz.^2);
    if do_diagnostics
        fprintf('Aplicado corr_factor = %.3f para ajustar energia temporal vs espectral\n', corr_factor);
    end
end

% test without mask (diagnostic only)
if do_mask_test_disable
    vx_out = vx ./ (mask_levels + eps); % restore approximate raw values
    mag_out = sqrt(vx_out.^2 + vz.^2);
    if do_diagnostics
        fprintf('Teste sem mascara ativado: max(mag_out) = %.6e m/s\n', max(mag_out(:)));
    end
else
    vx_out = vx;
    mag_out = mag;
end

% -------------------- Export CSV --------------------
fidc = fopen('velocity_components.csv','w');
if fidc < 0, error('Nao foi possivel criar velocity_components.csv'); end
fprintf(fidc, 'time,vx1,vz1,vx2,vz2,vx3,vz3,v1,v2,v3\n');
for i = 1:Nt_out
    fprintf(fidc, '%.6f,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n', ...
        t_out(i), vx_out(i,1), vz(i,1), vx_out(i,2), vz(i,2), vx_out(i,3), vz(i,3), ...
        mag_out(i,1), mag_out(i,2), mag_out(i,3));
end
fclose(fidc);
fprintf('CSV de velocidades salvo em velocity_components.csv\n');

% -------------------- Export XML (solver expects magnitudes in m/s) --------------------
out_xml = 'velocitytimes3.xml';
fidx = fopen(out_xml,'w');
if fidx < 0, error('Nao foi possivel criar %s', out_xml); end
for i = 1:Nt_out
  time_xml = t_out(i) + start_time;
  fprintf(fidx, '<timevalue time="%.6f" v="%.9f" v2="%.9f" v3="%.9f" z="%.6f" z2="%.6f" z3="%.6f" />\n', ...
    time_xml, mag_out(i,1), mag_out(i,2), mag_out(i,3), z_targets(1), z_targets(2), z_targets(3));
end
fclose(fidx);
fprintf('XML timevalue salvo em %s (start_time = %.3f s)\n', out_xml, start_time);

% -------------------- Final --------------------
fprintf('Processamento concluido. Verifique velocity_components.csv e %s\n', out_xml);
end

