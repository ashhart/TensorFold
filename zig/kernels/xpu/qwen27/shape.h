// Qwen3.8-27B sizes for qwen27 kernels; TP>1 shards heads/MLP by Q27_TP; other qwen3_5 models pass each with -D.
#ifndef Q27_TP
#define Q27_TP 1
#endif
#ifndef Q27_D
#define Q27_D 5120         // hidden size, a multiple of 512
#endif
#ifndef Q27_D512
#define Q27_D512 10        // Q27_D / 512
#endif
#ifndef Q27_FF
#define Q27_FF (17408 / Q27_TP) // dense MLP width per rank (17408, 8704, 4352)
#endif
#ifndef Q27_GK
#define Q27_GK (16 / Q27_TP)    // GDN key heads (128 wide: 16, 8, 4)
#endif
#ifndef Q27_GV
#define Q27_GV (48 / Q27_TP)    // GDN value heads (128 wide: 48, 24, 12)
#endif
#ifndef Q27_GVK
#define Q27_GVK 3          // Q27_GV / Q27_GK
#endif
#ifndef Q27_NQK
#define Q27_NQK (Q27_GK * 128)
#endif
#ifndef Q27_VD
#define Q27_VD (Q27_GV * 128)
#endif
#ifndef Q27_NCONV
#define Q27_NCONV (2 * Q27_NQK + Q27_VD) // the convolved qkv channels (10240, 5120, 2560)
#endif
#ifndef Q27_GA
#define Q27_GA (Q27_NCONV + Q27_VD)      // column of a in the GDN input: Q27_NCONV + Q27_VD
#endif
#ifndef Q27_GB
#define Q27_GB (Q27_GA + Q27_GV)         // column of b: Q27_GA + Q27_GV
#endif
#ifndef Q27_GDN_IN
#define Q27_GDN_IN (((Q27_GB + Q27_GV) + 255) / 256 * 256) // qkv | z | a | b, padded to 256 (16640, 8448, 4352)
#endif
#ifndef Q27_AQ
#define Q27_AQ (24 / Q27_TP)    // attention query heads (256 wide: 24, 12, 6)
#endif
#ifndef Q27_AKV
#define Q27_AKV (4 / Q27_TP)    // attention K/V heads (4, 2, 1)
#endif
#ifndef Q27_AH
#define Q27_AH (Q27_AQ + 2 * Q27_AKV)
#endif
#ifndef Q27_AV0
#define Q27_AV0 (Q27_AQ + Q27_AKV) // the first v head
#endif
#ifndef Q27_GQA
#define Q27_GQA 6          // Q27_AQ / Q27_AKV
#endif
#ifndef Q27_QPG
#define Q27_QPG 6          // query heads one attn_decode subgroup holds (divides Q27_GQA)
#endif
#ifndef Q27_QPARTS
#define Q27_QPARTS 1       // Q27_GQA / Q27_QPG
#endif
#ifndef Q27_AK0
#define Q27_AK0 (Q27_AQ * 512)           // column of k: Q27_AQ * 512
#endif
#ifndef Q27_AV0C
#define Q27_AV0C (Q27_AK0 + Q27_AKV * 256) // column of v: Q27_AK0 + Q27_AKV * 256
#endif
#ifndef Q27_ATT_IN
#define Q27_ATT_IN (Q27_AV0C + Q27_AKV * 256) // (q | gate) per query head, then k, then v (14336, 7168, 3584)
#endif
#ifndef Q27_OD
#define Q27_OD (Q27_AQ * 256)            // attention output width: Q27_AQ * 256
#endif
#ifndef Q27_OB
#define Q27_OB (Q27_OD / 128)            // Q27_OD / 128
#endif
#ifndef Q27_KVROW
#define Q27_KVROW (Q27_AKV * 256)        // int8 cache bytes a position: Q27_AKV * 256
#endif
#ifndef Q27_GSTRIDE
#define Q27_GSTRIDE(g) (max(32, ((int)(g) + 7) & ~7)) // Xe2 LSC 2D block pitch/width >= 64B and 16B-aligned
#endif
