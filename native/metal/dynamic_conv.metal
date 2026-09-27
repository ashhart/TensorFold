const uint index = thread_position_in_grid.x;
const int row = int(index) / N, channel = int(index) % N;
if (row >= dims[0]) return;
bfloat acc = bfloat(0);
for (int tap = 0; tap < 2; ++tap) {
  const bfloat value = row >= tap ? H[(row - tap) * N + channel] : bfloat(0);
  const bfloat base = bfloat(BASE[(PART * 2 + tap) * N + channel]);
  const bfloat dyn = DYNAMIC[(row * 4 + PART * 2 + tap) * (N / 16) + channel / 16];
  acc = bfloat(float(acc) + float(bfloat(float(base) * float(value))));
  acc = bfloat(float(acc) + float(bfloat(float(dyn) * float(value))));
}
OUT[index] = acc;
