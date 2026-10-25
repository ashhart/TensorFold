// Model sizes the qwen27 kernels are compiled for. The defaults are Qwen3.8-27B; another qwen3_5 model (Sage-A1.5)
// passes every value with -D, computed from its config.json by zig/build/xpu.zig. Keep the defaults literal: then the
// 27B's preprocessed source does not change.
#ifndef Q27_D
#define Q27_D 5120         // hidden size, a multiple of 512
#endif
#ifndef Q27_D512
#define Q27_D512 10        // Q27_D / 512
#endif
#ifndef Q27_FF
#define Q27_FF 17408       // dense MLP width
#endif
#ifndef Q27_GK
#define Q27_GK 16          // GDN key heads (128 wide)
#endif
#ifndef Q27_GV
#define Q27_GV 48          // GDN value heads (128 wide)
#endif
#ifndef Q27_GVK
#define Q27_GVK 3          // Q27_GV / Q27_GK
#endif
#ifndef Q27_NQK
#define Q27_NQK 2048       // Q27_GK * 128
#endif
#ifndef Q27_NCONV
#define Q27_NCONV 10240    // 2 * Q27_NQK + Q27_VD: the convolved qkv channels
#endif
#ifndef Q27_VD
#define Q27_VD 6144        // Q27_GV * 128
#endif
#ifndef Q27_GDN_IN
#define Q27_GDN_IN 16640   // qkv | z | a | b, padded to 256
#endif
#ifndef Q27_GA
#define Q27_GA 16384       // column of a in the GDN input: Q27_NCONV + Q27_VD
#endif
#ifndef Q27_GB
#define Q27_GB 16432       // column of b: Q27_GA + Q27_GV
#endif
#ifndef Q27_AQ
#define Q27_AQ 24          // attention query heads (256 wide)
#endif
#ifndef Q27_AKV
#define Q27_AKV 4          // attention K/V heads
#endif
#ifndef Q27_AH
#define Q27_AH 32          // Q27_AQ + 2 * Q27_AKV
#endif
#ifndef Q27_AV0
#define Q27_AV0 28         // Q27_AQ + Q27_AKV: the first v head
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
#ifndef Q27_ATT_IN
#define Q27_ATT_IN 14336   // (q | gate) per query head, then k, then v
#endif
#ifndef Q27_AK0
#define Q27_AK0 12288      // column of k: Q27_AQ * 512
#endif
#ifndef Q27_AV0C
#define Q27_AV0C 13312     // column of v: Q27_AK0 + Q27_AKV * 256
#endif
#ifndef Q27_OD
#define Q27_OD 6144        // attention output width: Q27_AQ * 256
#endif
#ifndef Q27_OB
#define Q27_OB 48          // Q27_OD / 128
#endif
#ifndef Q27_KVROW
#define Q27_KVROW 1024     // int8 cache bytes a position: Q27_AKV * 256
#endif
