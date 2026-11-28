// Extern "C" entry points for Hy4-flash MLA query absorb (`W_UK_T`) and gated value expand (`W_UV`).
#include "mla_c2.cuh"

using namespace hy4c2;

//                         K   MT NT WARPS KC STAGES D  PLANES STREAM
#define C2_AD(NT) Cfg<ND, 1, NT, 4, ND, 2, 6, 1, true>
#define C2_AP Cfg<ND, 2, 8, 8, 64, 3, 2, 1, false>
#define C2_EDF(NT) Cfg<KL, 1, NT, 4, KL, 2, 8, 3, true>
#define C2_EPF Cfg<KL, 2, 4, 8, 64, 3, 2, 3, false>

typedef C2_AD(1) AD1;
typedef C2_AD(2) AD2;
typedef C2_AP AP;
typedef C2_EDF(1) EDF1;
typedef C2_EDF(2) EDF2;
typedef Cfg<KL, 1, 4, 4, 64, 2, 2, 3, true> EDF4_K64_S2;
typedef C2_EPF EPF;

#define C2_ABSORB_X(NAME, CFG, MINB)                                                                                   \
    extern "C" __global__ void __launch_bounds__(CFG::THREADS, MINB)                                                   \
        NAME(const bf16* QB, int ldq, int R, int Hl, const bf16* WUKT, void* QA, long long QLO) {                      \
        absorb_body<CFG, false, false>(QB, ldq, R, Hl, WUKT, nullptr, nullptr, QA, QLO);                               \
    }
C2_ABSORB_X(hy4c2_absorb_d1_x, AD1, 4)
C2_ABSORB_X(hy4c2_absorb_d2_x, AD2, 4)
C2_ABSORB_X(hy4c2_absorb_p_x, AP, 2)

#define C2_EXPAND(NAME, CFG, LT, MINB)                                                                                 \
    extern "C" __global__ void __launch_bounds__(CFG::THREADS, MINB)                                                   \
        NAME(const LT* LAT, int R, int Hl, const bf16* WUV, const bf16* GATE, int ldg, bf16* ATT) {                    \
        expand_body<CFG, LT>(LAT, R, Hl, WUV, GATE, ldg, ATT);                                                         \
    }
C2_EXPAND(hy4c2_expand_d1_f32lat, EDF1, float, 4)
C2_EXPAND(hy4c2_expand_d2_f32lat, EDF2, float, 2)
C2_EXPAND(hy4c2_expand_d4k64s2_f32lat, EDF4_K64_S2, float, 3)
C2_EXPAND(hy4c2_expand_p_f32lat, EPF, float, 2)
