
  // attention output [R, H, D] (rows of heads) times sigmoid(gate) (bf16 ops); gate from the [q | gate] pairs
  const uint i = thread_position_in_grid.x;
  const int r = int(i) / (NQ * HD), c = int(i) % (NQ * HD);
  const int head = c / HD, d = c % HD;
  const float g = float(P[r * PW + head * 2 * HD + HD + d]);
  OUT[i] = bfloat(float(A[i]) * bsig(g));
