// bfloat16 for the host emulation: the type and its two conversions.
#pragma once
#include <cstdint>
#include <cstring>
struct __nv_bfloat16 { uint16_t x; };
inline float __bfloat162float(__nv_bfloat16 v) {
  const uint32_t b = (uint32_t)v.x << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}
inline __nv_bfloat16 __float2bfloat16_rn(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return {(uint16_t)((b + 0x7FFF + ((b >> 16) & 1)) >> 16)};
}
