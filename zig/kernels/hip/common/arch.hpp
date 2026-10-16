#pragma once

// RDNA2 (dot2), RDNA3 / 3.5 (gfx11 WMMA), RDNA4 (gfx12 WMMA) from the build's TF_* cap macros; RDNA1 is refused.

#if !defined(TF_WAVE) || !defined(TF_MATRIX)
#error "the caps macros come from the build"
#endif

namespace tf {
namespace rocm {

enum class Gen { rdna2, rdna3, rdna4 };

// Device code only: host dispatch reads TF_MATRIX from the build.
#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12_generic__)
inline constexpr Gen kGen = Gen::rdna4;
inline constexpr bool kWmma = true;
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) \
    || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__) \
    || defined(__gfx11_generic__)
inline constexpr Gen kGen = Gen::rdna3;
inline constexpr bool kWmma = true;
#elif defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) \
    || defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__)
inline constexpr Gen kGen = Gen::rdna2;
inline constexpr bool kWmma = false;
#elif defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__)
static_assert(false, "RDNA1 (gfx101x) is not a TensorFold target");
#else
inline constexpr Gen kGen = Gen::rdna2;
inline constexpr bool kWmma = false;
#endif

}  // namespace rocm

// v_dot2_f32_bf16 (gfx11 and gfx12 device code): the BF16 dot2 tiles of the RDNA3 experts.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) \
    || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__) \
    || defined(__gfx11_generic__) || defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12_generic__)
#define TF_DEVICE_BF16_DOT2 1
#else
#define TF_DEVICE_BF16_DOT2 0
#endif

// gfx11 device code: v_wmma_f32_16x16x16_bf16 with its wave32 layouts (gfx12's differs).
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) \
    || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__) \
    || defined(__gfx11_generic__)
#define TF_DEVICE_WMMA_GFX11 1
#else
#define TF_DEVICE_WMMA_GFX11 0
#endif
}  // namespace tf
