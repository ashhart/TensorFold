
  const int g = int(thread_position_in_grid.x);
  if (g > EE[0]) return;
  int lo = 0, hi = MM[0];
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (int(IDX[mid]) < g) lo = mid + 1; else hi = mid;
  }
  OFF[g] = lo;
