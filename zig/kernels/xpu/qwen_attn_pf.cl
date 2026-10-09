#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
// Qwen3.8 full-attention for prompt windows (many query rows): causal flash attention on the matrix engine. One sub-group takes 8 consecutive rows of one
// query head (M = 8 of the DPAS), streams the keys of the kv head in tiles of 16 (lane = key): scores S = Q K^T (16 DPAS over the 256 dims, Q from local memory),
// fp32 online softmax per row, P (bf16) times V (16 DPAS, V read with sub-group block loads), 128 fp32 accumulators a lane (needs the 256-GRF build).
// The key tiles are absolute (0, 16, 32, ...) and rows are independent, so a row's bits depend only on its own position: any window width or chunking gives the
// same result. Output: out[row][head][256] = bf16(bf16(o / l) * sigmoid(gate)) with the gate from qg (as attn_merge).
#define HD 256
#define NH 24
#define NKV 4
#define GQ 6
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float bfr(float f) { return bf(to_bf(f)); }

// Grid (ceil(rows / 16), 24 heads); 64 items = 4 sub-groups: two row groups of 8 rows, each split in two output halves that share the scores: per block of 64 keys
// each half computes the scores of two of the four 16-key tiles, the halves exchange the row maxima and the bf16 P tiles through local memory (two barriers a block).
// q [row][24][256] bf16, qg [row][24][512], out [row][24][256]; row z is at position pos0 + z.
__attribute__((reqd_work_group_size(64, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void attn_prefill(__global const ushort *q, __global const ushort *kc, __global const ushort *vc, __global const ushort *qg, __global ushort *out,
                           uint pos0, uint rows, float scale) {
    __local short qa[4][16 * 16 * 8]; // per sub-group: [dim block][lane][row]
    __local float mxs[4][8];
    __local float lls[4][8];
    __local short pbuf[4][2][16 * 8]; // per sub-group: its two P tiles, [lane][row]
    const uint head = get_group_id(1);
    const uint hk = head / GQ;
    const uint sg = get_sub_group_id();
    const uint pr = sg ^ 1; // the partner (the other output half of the same rows)
    const uint lane = get_sub_group_local_id();
    const uint r0 = get_group_id(0) * 16 + (sg >> 1) * 8;
    const uint hlf = sg & 1;
    for (uint i = lane; i < 16 * 16 * 8; i += 16) {
        const uint t = i >> 7, l = (i >> 3) & 15, r = i & 7;
        qa[sg][i] = (r0 + r < rows) ? (short)q[((ulong)(r0 + r) * NH + head) * HD + t * 16 + l] : (short)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint kmax = pos0 + min((uint)get_group_id(0) * 16 + 16, rows); // keys any row of the work-group can see (uniform: every barrier is reached by all)
    const uint ntiles = (kmax + 15) / 16;
    float m[8], ll[8];
    float8 acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        m[i] = -INFINITY;
        ll[i] = 0.0f;
    }
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = (float8)(0.0f);
    for (uint kb = 0; kb < ntiles; kb += 4) {
        float8 cc[2];
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const uint tile = kb + 2 * hlf + u;
            const uint key = tile * 16 + lane;
            const uint kl = min(key, kmax - 1);
            const __global uint *krow = (const __global uint *)(kc + ((ulong)kl * NKV + hk) * HD);
            float8 c = (float8)(0.0f);
#pragma unroll
            for (int t = 0; t < 16; t++) {
                const int8 b = as_int8(vload8(0, krow + t * 8));
                const short8 a = vload8(t * 16 + lane, qa[sg]);
                c = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c);
            }
#pragma unroll
            for (int i = 0; i < 8; i++) c[i] = (key <= pos0 + r0 + i && tile < ntiles) ? c[i] * scale : -INFINITY;
            cc[u] = c;
        }
        float8 lm8;
#pragma unroll
        for (int i = 0; i < 8; i++) lm8[i] = sub_group_reduce_max(fmax(cc[0][i], cc[1][i]));
        if (lane == 0) vstore8(lm8, 0, mxs[sg]);
        barrier(CLK_LOCAL_MEM_FENCE);
        const float8 pm = vload8(0, mxs[pr]);
        float8 al;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float mn = fmax(m[i], fmax(lm8[i], pm[i]));
            al[i] = (m[i] == -INFINITY) ? 0.0f : exp(m[i] - mn);
            m[i] = mn;
        }
        short8 pown[2], ppart[2];
#pragma unroll
        for (int u = 0; u < 2; u++) {
            short8 own;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float s = cc[u][i];
                const float pv = (s == -INFINITY) ? 0.0f : bfr(exp(s - m[i]));
                ll[i] = (u == 0 ? ll[i] * al[i] : ll[i]) + pv;
                own[i] = (short)to_bf(pv);
            }
            vstore8(own, lane, pbuf[sg][u]);
            pown[u] = own;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
        for (int u = 0; u < 2; u++) ppart[u] = vload8(lane, pbuf[pr][u]);
#pragma unroll
        for (int j = 0; j < 8; j++) acc[j] *= al;
#pragma unroll
        for (int u = 0; u < 4; u++) {
            if (kb + u >= ntiles) break;
            const short8 pa = ((u >> 1) == (int)hlf) ? pown[u & 1] : ppart[u & 1];
#pragma unroll
            for (int hb = 0; hb < 2; hb++) {
                ushort4 vv[16];
#pragma unroll
                for (int k = 0; k < 16; k++) {
                    const uint vk = min((kb + u) * 16 + k, kmax - 1);
                    vv[k] = intel_sub_group_block_read_us4((const __global ushort *)(vc + ((ulong)vk * NKV + hk) * HD + (hlf * 2 + hb) * 64));
                }
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int8 b;
#pragma unroll
                    for (int d = 0; d < 8; d++) b[d] = (int)vv[2 * d][e] | ((int)vv[2 * d + 1][e] << 16);
                    acc[hb * 4 + e] = intel_sub_group_bf16_bf16_matrix_mad_k16(pa, b, acc[hb * 4 + e]);
                }
            }
        }
    }
    float8 own_l;
#pragma unroll
    for (int i = 0; i < 8; i++) own_l[i] = sub_group_reduce_add(ll[i]);
    if (lane == 0) vstore8(own_l, 0, lls[sg]);
    barrier(CLK_LOCAL_MEM_FENCE);
    const float8 pl = vload8(0, lls[pr]);
    float inv[8];
#pragma unroll
    for (int i = 0; i < 8; i++) inv[i] = 1.0f / (hlf == 0 ? own_l[i] + pl[i] : pl[i] + own_l[i]);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint row = r0 + i;
        if (row >= rows) continue;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const uint d = (hlf * 8 + j) * 16 + lane;
            const float gate = bf(qg[((ulong)row * NH + head) * 2 * HD + HD + d]);
            out[((ulong)row * NH + head) * HD + d] = to_bf(bfr(acc[j][i] * inv[i]) * (1.0f / (1.0f + exp(-gate))));
        }
    }
}
