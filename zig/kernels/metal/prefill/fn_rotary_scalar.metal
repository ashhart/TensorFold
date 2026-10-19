struct TF_RotaryScalar { float frequency, angle, cosine, sine, co, si; };
inline TF_RotaryScalar tf_rotary_scalar(int i, int position, const constant float* LOG2BASE) {
  const float freq = LOG2BASE[1] != 0.0f ? LOG2BASE[3 + i] : metal::precise::exp2(-(float(i) / 32.0f) * LOG2BASE[0]);
  const float angle = float(position) * freq;
  const float ys = LOG2BASE[1] != 0.0f ? LOG2BASE[2] : 1.0f;
  const float co = metal::precise::cos(angle), si = metal::precise::sin(angle);
  return {freq, angle, co, si, co * ys, si * ys};
}
