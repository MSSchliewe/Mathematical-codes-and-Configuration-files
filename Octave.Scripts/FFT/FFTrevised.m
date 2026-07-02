% FFTrevised.m (corrected version as a script, consistent units)
% Expected input: FileZsurf.csv (columns: time [s]; elevation [cm])

% load signal package if available (comment out if not present)
try
  pkg load signal;
catch
  % without package, continue with custom implementations
end

% --- Robust Octave-compatible reading ---
% --- BEGIN: small override block (safe to remove later) ---
% If fname already exists in workspace (set by a wrapper), keep it.
if ~exist('fname','var') || isempty(fname)
    fname = 'SPH3D8.csv';  % default filename (keeps original behavior)
end
% --- END: small override block ---

fid = fopen(fname,'r');
if fid == -1
  error('Could not open file %s', fname);
end

% read first raw line
firstline = fgetl(fid);
frewind(fid);

% detect delimiter
if ~isempty(strfind(firstline, ';'))
  delim = ';';
elseif ~isempty(strfind(firstline, ','))
  delim = ',';
else
  delim = ','; % fallback
end

% attempt reading with textscan (robust)
frewind(fid);
firstline = fgetl(fid);
% detect if first line is a textual header (non-numeric)
isHeader = isempty(regexp(firstline, '^[\s]*[+\-]?\d+(\.\d+)?', 'once'));
frewind(fid);

if isHeader
  C = textscan(fid, '%f%f', 'Delimiter', delim, 'HeaderLines', 1, 'CollectOutput', true);
else
  C = textscan(fid, '%f%f', 'Delimiter', delim, 'CollectOutput', true);
end
fclose(fid);

if isempty(C) || isempty(C{1})
  % fallback: read as text and split manually (handles BOM and comma decimals)
  txt = fileread(fname);
  % remove BOM if present
  if length(txt) >= 3 && double(txt(1))==239 && double(txt(2))==187 && double(txt(3))==191
    txt = txt(4:end);
  end
  lines = strsplit(strtrim(txt), {'\r\n','\n','\r'});
  N = numel(lines);
  data = zeros(N,2);
  for k=1:N
    parts = strsplit(strtrim(lines{k}), delim);
    if numel(parts) < 2
      error('Line %d does not have two columns: "%s"', k, lines{k});
    end
    s1 = strrep(parts{1}, ',', '.');
    s2 = strrep(parts{2}, ',', '.');
    data(k,1) = str2double(s1);
    data(k,2) = str2double(s2);
  end
else
  data = C{1};
end

% check dimensions and extract columns
[nr, nc] = size(data);
if nc < 2
  error('File read with %d columns. Expected 2 columns (t, eta).', nc);
end

% consistent names used in the script
t = data(:,1);
eta = data(:,2);
fprintf('File %s read: %d rows, %d columns. mean dt = %.6f s\n', fname, nr, nc, mean(diff(t)));

%% 2) Basic parameters (corrected)
dt = mean(diff(t));
Fs = 1/dt;
N = length(eta);
T_total = N / Fs;
df = 1 / T_total;
faxis = 0:df:(Fs/2);

%% 3) Pre-processing
% --- UNITS: input in cm. Convert to meters for all calculations ---
eta_cm = eta(:);            % keep a copy in cm
eta_m = eta_cm ./ 100;     % convert to meters (internal consistent unit)

% remove mean (in meters)
eta_m = eta_m - mean(eta_m);

% Tukey window: use package function if available, otherwise implement simple version
if exist('tukeywin','file') == 2
  TukeyAlpha = 0.5;
  w = tukeywin(N, TukeyAlpha);
else
  % simple Tukey implementation (cosine tapered edges)
  TukeyAlpha = 0.5;
  w = ones(N,1);
  L = floor(TukeyAlpha*(N-1)/2);
  if L>0
    n = (0:L-1)';
    w(1:L) = 0.5*(1 + cos(pi*(2*n/(TukeyAlpha*(N-1)) - 1)));
    w(end-L+1:end) = flipud(w(1:L));
  end
end
etaWindowed_m = eta_m .* w;   % window applied in meters

%% 4) Normalized FFT
X = fft(etaWindowed_m);
X = X / N;
Npos = floor(N/2)+1;
X_pos = X(1:Npos);

% peak amplitude (corrected for single-sided component) in meters
rawAmp = abs(X_pos);
principalAmp = rawAmp;
if rem(N,2) == 0
  principalAmp(2:end-1) = 2*rawAmp(2:end-1);
else
  principalAmp(2:end) = 2*rawAmp(2:end);
end

% locate peak (use faxis)
[peakVal, idxPeak] = max(principalAmp);
principalAmplitude_m = peakVal;            % in meters (CORRECT)
principalFrequency = faxis(idxPeak);
principalPhase = angle(X_pos(idxPeak));

fprintf('Principal Frequency: %.6f Hz\n', principalFrequency);
fprintf('Principal Amplitude: %.6e m\n', principalAmplitude_m);
fprintf('Principal Phase: %.6f radians\n', principalPhase);

%% 5) Spectrum (energy per bin) in dB
Sxx = zeros(size(X_pos));
Sxx(1) = abs(X_pos(1)).^2;
if rem(N,2) == 0
  Sxx(2:end-1) = 2 * abs(X_pos(2:end-1)).^2;
  Sxx(end) = abs(X_pos(end)).^2;
else
  Sxx(2:end) = 2 * abs(X_pos(2:end)).^2;
end
% spectral density in m^2/Hz (correct by df)
Sxx = Sxx ./ df;
Sxx_dB = 10 * log10(Sxx + eps);

%% 6) Main plot (time + power) with peak marker
fig1 = figure('Name','Processed Signal and Power Spectrum','NumberTitle','off','Position',[100 100 900 700]);

subplot(3,1,1);
% SIGNAL PLOT: display in cm (multiply by 100), label in cm
plot(t, etaWindowed_m*100, 'b-');
xlabel('Time (s)');
ylabel('Elevation (cm)');
title('Processed Signal (Detrended + Tukey Window)');
grid on;

subplot(3,1,2);
% Power spectrum in dB (physical unit: m^2/Hz)
plot(faxis, Sxx_dB, 'r-','LineWidth',1.2);
hold on;
% mark the peak in dB
peak_dB = Sxx_dB(idxPeak);
plot(principalFrequency, peak_dB, 'ko','MarkerFaceColor','y','MarkerSize',8);
% vertical line at the peak
yl = ylim;
plot([principalFrequency principalFrequency], yl, '--k');
text(principalFrequency, peak_dB, sprintf('  f_p = %.4f Hz\\n  A = %.6e m', principalFrequency, principalAmplitude_m), ...
  'VerticalAlignment','bottom','FontWeight','bold','BackgroundColor','white','EdgeColor','k');
xlabel('Frequency (Hz)');
ylabel('Power (dB re m^2/Hz)');
title('Power Spectrum (dB) with Dominant Peak');
grid on;
hold off;

%% 7) Amplitude spectrum plot (linear) and zoom around the peak
% principalAmp is in meters; convert to cm for display
amp_single_sided_m = principalAmp; % vector in meters
amp_single_sided_cm = amp_single_sided_m * 100; % for display

fig2 = figure('Name','Amplitude Spectrum','NumberTitle','off','Position',[200 150 900 400]);
plot(faxis, amp_single_sided_cm, '-b','LineWidth',1.2);
hold on;
plot(principalFrequency, principalAmplitude_m*100, 'ro','MarkerFaceColor','r','MarkerSize',8);
xlabel('Frequency (Hz)');
ylabel('Amplitude (cm)');
title('Single-Sided Amplitude Spectrum (linear)');
grid on;

% Zoom: +/- 0.2 Hz around the peak (adjust if necessary)
zoom_bw = max(0.2, principalFrequency*0.1);
xmin = max(0, principalFrequency - zoom_bw);
xmax = principalFrequency + zoom_bw;
xlim([xmin xmax]);
legend('Amplitude','Dominant Peak','Location','northeast');
hold off;

%% 8) Save figures (optional)
try
  saveas(fig1, 'processed_signal_and_spectrum.png');
  saveas(fig2, 'amplitude_spectrum_zoom.png');
catch
  warning('Could not save figures (check permissions).');
end

%% 9) Export main results to text file
T_total = N / Fs;
fid = fopen('fft_summary.txt','w');
fprintf(fid, 'Principal Frequency (Hz): %.6f\n', principalFrequency);
fprintf(fid, 'Principal Amplitude (m): %.6e\n', principalAmplitude_m);
fprintf(fid, 'Principal Amplitude (cm): %.6e\n', principalAmplitude_m*100);
fprintf(fid, 'Principal Phase (rad): %.6f\n', principalPhase);
fprintf(fid, 'Sampling frequency (Hz): %.6f\n', Fs);
fprintf(fid, 'Total duration (s): %.6f\n', T_total);
fclose(fid);
% --- BEGIN: export block for automated workflows (safe to remove later) ---
% Save numeric outputs to a .mat file so wrapper/other scripts can load them
try
    % sanitize base name (remove path and extension)
    [p, base, ext] = fileparts(fname);
    outmat = sprintf('FFTrevised_%s.mat', base);

    % variables to save: frequency axis, PSD (Sxx), amplitude spectrum, principal values, Fs, duration
    % ensure variables exist
    if exist('faxis','var') && exist('Sxx','var')
        % Sxx is single-sided energy per bin (m^2/Hz)
        save(outmat, 'faxis', 'Sxx', 'Sxx_dB', 'amp_single_sided_m', 'principalFrequency', 'principalAmplitude_m', 'principalPhase', 'Fs', 'T_total');
    else
        % fallback: try to compute minimal outputs if names differ
        % do nothing if not available
    end
    fprintf('FFTrevised: results exported to %s\n', outmat);
catch ME
    warning('FFTrevised: failed to export .mat (%s)', ME.message);
end
% --- END: export block ---

%% 10) Final message
fprintf('Analysis completed. Files generated (if possible): processed_signal_and_spectrum.png, amplitude_spectrum_zoom.png, fft_summary.txt\n');

