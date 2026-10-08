// RMSNorm over bf16 rows and the 4-bit affine embedding lookup. Activations are bf16, math is fp32.
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

// One work-group (64 items, 4 sub-groups of 16) per row. A work-item adds its elements lid, lid + 64, ... in turn, then the sub-group, then the 4 parts. The plain loop waits for each
// load in turn (latency bound), so for n == 64 * HK the loads of a work-item are issued together (static trip count) and the sums run on registers; same values, same order.
#define HK 42
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void rmsnorm(__global const ushort *x, __global const ushort *w, __global ushort *y, uint n, float eps) {
    __local float part[4];
    const uint row = get_group_id(0);
    const uint lid = get_local_id(0);
    float ss = 0.0f;
    ushort xa[HK], wa[HK];
    if (n == 64 * HK) {
#pragma unroll
        for (uint k = 0; k < HK; k++) {
            xa[k] = x[row * n + lid + 64 * k];
            wa[k] = w[lid + 64 * k];
        }
#pragma unroll
        for (uint k = 0; k < HK; k++) {
            const float v = bf(xa[k]);
            ss += v * v;
        }
    } else {
        for (uint i = lid; i < n; i += 64) {
            const float v = bf(x[row * n + i]);
            ss += v * v;
        }
    }
    ss = sub_group_reduce_add(ss);
    if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float total = part[0] + part[1] + part[2] + part[3];
    const float scale = rsqrt(total / (float)n + eps);
    if (n == 64 * HK) {
#pragma unroll
        for (uint k = 0; k < HK; k++) y[row * n + lid + 64 * k] = to_bf(bf(xa[k]) * scale * bf(wa[k]));
    } else {
        for (uint i = lid; i < n; i += 64) y[row * n + i] = to_bf(bf(x[row * n + i]) * scale * bf(w[i]));
    }
}

// Gathers rows of a 4-bit table by token id: one work-group (64 items) per id, one word (8 values) per item per step.
__kernel void embed4(__global const uint *w, __global const ushort *scales, __global const ushort *biases,
                     __global const uint *ids, __global ushort *y, uint dim) {
    const uint t = get_group_id(0);
    const uint row = ids[t];
    const uint words = dim / 8;
    const uint groups = dim / 64;
    for (uint wi = get_local_id(0); wi < words; wi += 64) {
        const uint g = wi / 8;
        const float sc = bf(scales[row * groups + g]);
        const float bi = bf(biases[row * groups + g]);
        const uint pk = w[row * words + wi];
        for (uint j = 0; j < 8; j++) y[t * dim + wi * 8 + j] = to_bf(sc * (float)((pk >> (4 * j)) & 15u) + bi);
    }
}

// One token id as a kernel argument (captured when the launch is queued: no host memory is read when the kernel runs).
__kernel void embed4_tok(__global const uint *w, __global const ushort *scales, __global const ushort *biases, uint row, __global ushort *y, uint dim) {
    const uint words = dim / 8;
    const uint groups = dim / 64;
    for (uint wi = get_local_id(0); wi < words; wi += 64) {
        const uint g = wi / 8;
        const float sc = bf(scales[row * groups + g]);
        const float bi = bf(biases[row * groups + g]);
        const uint pk = w[row * words + wi];
        for (uint j = 0; j < 8; j++) y[wi * 8 + j] = to_bf(sc * (float)((pk >> (4 * j)) & 15u) + bi);
    }
}

// Residual add and the next RMSNorm in one launch: x = bf16(x + d) in place (add_bf16), y = rmsnorm(x). The sums and their order are those of rmsnorm above (a work-item adds its
// elements lid, lid + 64, ... in turn, then the sub-group and the 4 parts). The loop of rmsnorm waits for each load in turn (the kernel is latency bound), so for n == 64 * HK the
// loads of a work-item are issued together (static trip count) and the sums run on registers; any other n takes the plain loop.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void add_rmsnorm(__global ushort *x, __global const ushort *d, __global const ushort *w, __global ushort *y, uint n, float eps) {
    __local float part[4];
    const uint row = get_group_id(0);
    const uint lid = get_local_id(0);
    float ss = 0.0f;
    ushort xa[HK], wa[HK];
    if (n == 64 * HK) {
        ushort da[HK];
#pragma unroll
        for (uint k = 0; k < HK; k++) {
            xa[k] = x[row * n + lid + 64 * k];
            da[k] = d[row * n + lid + 64 * k];
            wa[k] = w[lid + 64 * k];
        }
#pragma unroll
        for (uint k = 0; k < HK; k++) {
            const ushort s = to_bf(bf(xa[k]) + bf(da[k]));
            xa[k] = s;
            x[row * n + lid + 64 * k] = s;
            const float v = bf(s);
            ss += v * v;
        }
    } else {
        for (uint i = lid; i < n; i += 64) {
            const ushort s = to_bf(bf(x[row * n + i]) + bf(d[row * n + i]));
            x[row * n + i] = s;
            const float v = bf(s);
            ss += v * v;
        }
    }
    ss = sub_group_reduce_add(ss);
    if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float total = part[0] + part[1] + part[2] + part[3];
    const float scale = rsqrt(total / (float)n + eps);
    if (n == 64 * HK) {
#pragma unroll
        for (uint k = 0; k < HK; k++) y[row * n + lid + 64 * k] = to_bf(bf(xa[k]) * scale * bf(wa[k]));
    } else {
        for (uint i = lid; i < n; i += 64) y[row * n + i] = to_bf(bf(x[row * n + i]) * scale * bf(w[i]));
    }
}
