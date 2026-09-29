
inline void load16(const device bfloat* p, thread float* v) {
  const device uint4* u = (const device uint4*)p;
  const uint4 a = u[0], b = u[1];
  const uint w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  for (int i = 0; i < 8; i++) { v[2 * i] = as_type<float>(w[i] << 16); v[2 * i + 1] = as_type<float>(w[i] & 0xffff0000u); }
}
