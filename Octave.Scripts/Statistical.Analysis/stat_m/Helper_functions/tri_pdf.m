function y = tri_pdf(z, a, m, b)
  % vectorized triangular density
  z = z(:);
  y = zeros(size(z));
  idx1 = (z >= a & z <= m);
  idx2 = (z > m & z <= b);
  if any(idx1)
    y(idx1) = 2 .* (z(idx1) - a) ./ ((b - a) .* (m - a));
  end
  if any(idx2)
    y(idx2) = 2 .* (b - z(idx2)) ./ ((b - a) .* (b - m));
  end
end
