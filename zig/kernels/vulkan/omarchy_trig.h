// Shared trig range reduction for omarchy compute shaders.
// Copied from mlx-omarchy (joshuaswarren/omarchy-mlx overlay,
// mlx/backend/omarchy/shaders/omarchy_trig.h), MIT, unmodified:
// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// The Honeykrisp built-in sin/cos/tan lose upstream-grade accuracy above
// ~1e3 and collapse past ~1e6 (measured, docs/known-defects.md); llvmpipe
// collapses above ~1e7. Every path that evaluates a trigonometric
// function of an argument that is not bounded by construction goes
// through this header's omarchy_trig_* wrappers instead of the raw
// built-ins: accurate to kTrigArgumentLimit, NaN above it. A finite
// wrong value is never produced. The band below kTrigBuiltInMax keeps
// the raw built-in bit-identically.
//
// The reduction is Cody-Waite with C1 = 6.28125 = 201 * 2^-5 (8
// significant bits), so k * C1 is exact in float32 for every |k| <=
// 2^24 / 201 = 83601 - i.e. for every argument up to 525371.9. The
// remaining 2*pi pieces are a dyadic 2^-9 (k times it is exact for all
// k) plus two f32 correction terms; the three-subtract chain is marked
// precise so a driver may not reassociate it (Honeykrisp has a
// documented inexact-constant-reassociation defect). Above the limit
// the argument magnitude exceeds the exact-product zone or the floor
// of the method, and the wrappers return NaN rather than a wrong
// finite value.
//
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

// tan is reduced through the 2*pi-periodic forms: tan(x) = sin(x)/cos(x)
// and both numerator and denominator are exact period shifts, so the
// reduced argument serves. Relative error grows near the poles exactly
// as it does for the built-in at small arguments.
float omarchy_trig_tan(float x) {
  float ax = abs(x);
  if (ax <= OMARCHY_TRIG_BUILTIN_MAX) return tan(x);
  if (ax > OMARCHY_TRIG_LIMIT) return omarchy_trig_nan();
  return sin(omarchy_trig_reduce(x)) / cos(omarchy_trig_reduce(ax));
}
