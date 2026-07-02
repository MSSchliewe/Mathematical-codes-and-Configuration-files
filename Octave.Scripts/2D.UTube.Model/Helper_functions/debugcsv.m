% debug_compare_csvs.m
files = {'ResultsSheet.csv','ResultsSheet1.csv'};

for f=1:numel(files)
  fname = files{f};
  fprintf('\n--- Checando %s ---\n', fname);
  % 1) file size, first bytes (detectar BOM)
  info = dir(fname);
  fprintf('size = %d bytes\n', info.bytes);
  fid = fopen(fname,'rb');
  head = fread(fid, 512, '*uint8');
  fclose(fid);
  if numel(head)>=3 && head(1)==239 && head(2)==187 && head(3)==191
    fprintf('BOM UTF-8 detectado\n');
  end
  % 2) tentativa robusta de leitura textual para inspecao
  fid = fopen(fname,'r','n','UTF-8');
  if fid < 0, error('Nao abriu %s', fname); end
  lines = {};
  k = 0;
  while ~feof(fid) && k < 1000
    k = k+1;
    ln = fgetl(fid);
    lines{k} = ln;
  end
  fclose(fid);
  % show first 8 lines
  for i=1:min(8,numel(lines))
    s = lines{i};
    if isempty(s), s='(empty)'; end
    fprintf('L%2d: %s\n', i, s(1:min(200,end)));
  end
  % 3) try readtable/textscan with common delimiters
  % try comma
  try
    C = dlmread(fname, ',');
    fprintf('dlmread with comma succeeded: size %dx%d\n', size(C));
  catch
    fprintf('dlmread comma failed\n');
    C = [];
  end
  if isempty(C)
    try
      C = dlmread(fname, ';');
      fprintf('dlmread with semicolon succeeded: size %dx%d\n', size(C));
    catch
      fprintf('dlmread semicolon failed\n');
      C = [];
    end
  end
  % 4) fallback: parse numeric tokens per line (more forgiving)
  if isempty(C)
    fid = fopen(fname,'r');
    nums = [];
    while ~feof(fid)
      ln = fgetl(fid);
      if isempty(ln), continue; end
      % replace comma decimal -> dot if appears (european)
      ln2 = regexprep(ln,'(?<=\d)[,](?=\d)','.');
      vals = sscanf(ln2, '%f');
      if numel(vals)>=2
        nums = [nums; vals(1:2)'];
      end
    end
    fclose(fid);
    C = nums;
    fprintf('Parsed by tokenizing lines: size %dx%d\n', size(C));
  end
  if isempty(C)
    error('Nao foi possivel extrair numeros de %s', fname);
  end
  t = C(:,1); z = C(:,2);
  % store diagnostics
  files_info(f).t = t;
  files_info(f).z = z;
  % 5) basic checks
  fprintf('n_points=%d  t range=%.6g .. %.6g  z range=%.6g .. %.6g\n', ...
    numel(t), min(t), max(t), min(z), max(z));
  % monotonicity
  dt = diff(t);
  fprintf('dt: min=%.6g mean=%.6g max=%.6g\n', min(dt), mean(dt), max(dt));
  if any(dt<=0)
    fprintf('WARNING: tempos nao monotônicos ou repetidos (dt <= 0) detectados\n');
  end
  % NaN/Inf
  nNan = sum(isnan(t)) + sum(isnan(z));
  nInf = sum(isinf(t)) + sum(isinf(z));
  fprintf('NaN count=%d  Inf count=%d\n', nNan, nInf);
  % check outliers / magnitude
  fprintf('z median=%.6g  z std=%.6g\n', median(z), std(z));
  % quick velocity estimate
  if numel(t) >= 2
    v_est = diff(z)./diff(t);
    fprintf('v_est: min=%.6g  mean=%.6g  max=%.6g\n', min(v_est), mean(v_est), max(v_est));
    if any(abs(v_est) > 1e3)
      fprintf('WARNING: velocities huge (>1e3) detected\n');
    end
  end
end

% --- Compare the two files pairwise (shapes, numeric differences) ---
fprintf('\n--- Comparacao entre arquivos ---\n');
t1 = files_info(1).t; z1 = files_info(1).z;
t2 = files_info(2).t; z2 = files_info(2).z;
if numel(t1) ~= numel(t2) || numel(z1) ~= numel(z2)
  fprintf('Diferenca de tamanho: %d vs %d\n', numel(t1), numel(t2));
else
  dd_t = t1 - t2;
  dd_z = z1 - z2;
  fprintf('t differences: max abs=%.6g  any nonzero=%d\n', max(abs(dd_t)), any(abs(dd_t)>0));
  fprintf('z differences: max abs=%.6g  any nonzero=%d\n', max(abs(dd_z)), any(abs(dd_z)>0));
  % show indices where differ
  idx_t = find(abs(dd_t) > 0);
  idx_z = find(abs(dd_z) > 0);
  if ~isempty(idx_t)
    fprintf('First differing t at idx %d: %g vs %g\n', idx_t(1), t1(idx_t(1)), t2(idx_t(1)));
  end
  if ~isempty(idx_z)
    fprintf('First differing z at idx %d: %g vs %g\n', idx_z(1), z1(idx_z(1)), z2(idx_z(1)));
  end
end

% If you find non-monotonic times or NaNs in ResultsSheet1.csv, create sanitized file:
% Example sanitizer (run if needed)
sanitize = true;
if sanitize
  bad = (isnan(t2) | isnan(z2) | isinf(t2) | isinf(z2));
  t2(bad) = []; z2(bad) = [];
  % enforce monotonic increasing times: sort by t and remove duplicates
  [t2s, I] = sort(t2);
  z2s = z2(I);
  % remove exact duplicate times (keep first)
  [tu, iu] = unique(t2s,'stable');
  t2s = t2s(iu); z2s = z2s(iu);
  % if small nonstrict monotonic issues (tiny negative dt), make strictly increasing
  for k=2:numel(t2s)
    if t2s(k) <= t2s(k-1)
      t2s(k) = t2s(k-1) + eps + 1e-12;
    end
  end
  csvwrite('ResultsSheet1_sanitized.csv',[t2s(:), z2s(:)]);
  fprintf('Sanitized file written: ResultsSheet1_sanitized.csv (n=%d)\n', numel(t2s));
end
