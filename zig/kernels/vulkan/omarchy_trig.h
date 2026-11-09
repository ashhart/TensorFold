// From omarchy-mlx overlay/mlx/backend/omarchy/shaders/omarchy_trig.h: code unchanged, comments condensed.

// Copyright © 2026 Joshua Warren; MIT License, full text in LICENSES/mlx-omarchy-MIT.txt.

// Shared trig range reduction for omarchy compute shaders.

// The built-in sin/cos/tan lose upstream accuracy above ~1e3 and collapse past ~1e6 (llvmpipe past ~1e7, measured).

// Every path that evaluates a trigonometric function of an argument not bounded by construction uses these wrappers.

// The wrappers stay accurate to OMARCHY_TRIG_LIMIT and return NaN above it; a finite wrong value is never produced.

// Below OMARCHY_TRIG_BUILTIN_MAX the raw built-in stays bit-identical.

// The reduction is Cody-Waite with C1 = 6.28125 = 201 * 2^-5 (8 significant bits).

// So k * C1 is exact in float32 for every |k| <= 83601, that is for every argument up to 525371.9.

// The remaining 2*pi pieces are a dyadic 2^-9 (k times it is exact for all k) plus two f32 correction terms.

// The three-subtract chain is marked precise, so a driver may not reassociate it (Honeykrisp has a documented defect).

// Above the limit the argument leaves the exact-product zone, so the wrappers return NaN, not a wrong finite value.

// Do not copy this logic into another shader; include this header.

const float OMARCHY_TRIG_LIMIT = 500000.0;
const float OMARCHY_TRIG_BUILTIN_MAX = 10000.0;

const float OMARCHY_TRIG_C1 = 6.28125;                     // 201 * 2^-5
const float OMARCHY_TRIG_C2HI = 0.001953125;               // 2^-9
const float OMARCHY_TRIG_C2LO = -1.781781975296326e-05;    // 2*pi - C1 - C2HI
const float OMARCHY_TRIG_C3 = -6.608047442568932e-13;      // 2*pi - C1 - C2HI - C2LO
const float OMARCHY_TRIG_INV_2PI = 0.15915494309189535;    // f32(1 / 2*pi)

float omarchy_trig_nan() { return uintBitsToFloat(0x7fc00000u); }

precise float omarchy_trig_reduce(float x) {
  float k = floor(x * OMARCHY_TRIG_INV_2PI);
  float r = x - k * OMARCHY_TRIG_C1;
  r -= k * OMARCHY_TRIG_C2HI;
  r -= k * OMARCHY_TRIG_C2LO;
  r -= k * OMARCHY_TRIG_C3;
  return r;
}

float omarchy_trig_sin(float x) {
  float ax = abs(x);
  if (ax <= OMARCHY_TRIG_BUILTIN_MAX) return sin(x);
  if (ax > OMARCHY_TRIG_LIMIT) return omarchy_trig_nan();
  return sin(omarchy_trig_reduce(x));
}

float omarchy_trig_cos(float x) {
  float ax = abs(x);
  if (ax <= OMARCHY_TRIG_BUILTIN_MAX) return cos(x);
  if (ax > OMARCHY_TRIG_LIMIT) return omarchy_trig_nan();
  return cos(omarchy_trig_reduce(ax));
}

// tan reduces through the 2*pi-periodic sin/cos forms; the error grows near the poles as at the built-in.

float omarchy_trig_tan(float x) {
  float ax = abs(x);
  if (ax <= OMARCHY_TRIG_BUILTIN_MAX) return tan(x);
  if (ax > OMARCHY_TRIG_LIMIT) return omarchy_trig_nan();
  return sin(omarchy_trig_reduce(x)) / cos(omarchy_trig_reduce(ax));
}
