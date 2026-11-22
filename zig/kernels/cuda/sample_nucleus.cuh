#pragma once

// Probability mass past `b`, limited by `top`, `lo` and `cut`.
__device__ float mass(const Row& r, float m, float norm, float lo, Mark b, Mark top, Mark cut, bool strict, Shared& s) {
    float x = 0.0f;
    for (unsigned i = r.t; i < r.V; i += TG) {
        const float v = r.v(i);
        if (!(v >= lo)) continue;
        const unsigned k = key(v);
        const bool inside = strict ? (k > cut.key || (k == cut.key && i < cut.id)) : at_or_before(k, i, cut);
        if (after(k, i, b) && at_or_before(k, i, top) && inside) x += expf(v - m) / norm;
    }
    return group_sum(x, r, s);
}

// Each probe keeps the original per-thread addition and warp reduction order.
template <bool BY_ID>
__device__ unsigned probe_digit(const Row& r, float m, float norm, float lo, float top_p, float cum, Mark b, Mark top,
                                unsigned prefix, unsigned shift, unsigned key_cut, Shared& s) {
    float local[16];
#pragma unroll
    for (unsigned d = 0; d < 16u; d++) local[d] = 0.0f;
    for (unsigned i = r.t; i < r.V; i += TG) {
        const float v = r.v(i);
        if (!(v >= lo)) continue;
        const unsigned k = key(v);
        if (!after(k, i, b) || !at_or_before(k, i, top)) continue;
        const float p = expf(v - m) / norm;
#pragma unroll
        for (unsigned d = 0; d < 16u; d++) {
            const unsigned candidate = prefix | (d << shift);
            if (BY_ID && candidate >= r.V) continue;
            const bool inside = BY_ID ? (k > key_cut || (k == key_cut && i < candidate)) : k >= candidate;
            if (inside) local[d] += p;
        }
    }
#pragma unroll
    for (unsigned d = 0; d < 16u; d++) {
        const float part = warp_sum(local[d]);
        if (r.lane == 0) s.probe[d][r.sg] = part;
    }
    __syncthreads();
    if (r.t == 0) {
        unsigned chosen = BY_ID ? 0u : FULL;
        for (unsigned d = 0; d < 16u; d++) {
            const unsigned candidate = prefix | (d << shift);
            if (BY_ID && candidate >= r.V) continue;
            float total = 0.0f;
            for (unsigned g = 0; g < NSG; g++) total += s.probe[d][g];
            const float mass_at_cut = cum + total;
            if (BY_ID ? mass_at_cut < top_p : mass_at_cut >= top_p) chosen = d;
        }
        s.st[0] = chosen;
    }
    __syncthreads();
    return s.st[0];
}

// Four-bit radix probes find the boundary while retaining the reference reduction order.
__device__ Mark extend(const Row& r, float m, float norm, float lo, float top_p, float cum, Mark b, Mark top,
                       Shared& s) {
    unsigned k = 0u;
    for (int shift = 28; shift >= 0; shift -= 4) {
        const unsigned digit = probe_digit<false>(r, m, norm, lo, top_p, cum, b, top, k, shift, 0u, s);
        if (digit == FULL) return end_mark();
        k |= digit << shift;
    }
    unsigned id = 0u;
    const int high_id_bit = 31 - __clz(r.V - 1u);
    for (int shift = high_id_bit >= 0 ? (high_id_bit / 4) * 4 : -1; shift >= 0; shift -= 4) {
        const unsigned digit = probe_digit<true>(r, m, norm, lo, top_p, cum, b, top, id, shift, k, s);
        id |= digit << shift;
    }
    return Mark{k, id};
}

// The nucleus cut extends the sorted block only when its sequential sum falls short.
__device__ Mark nucleus(const Row& r, float m, float lo, float top_p, float near, unsigned kc, bool whole, Mark top,
                        Shared& s) {
    float z = 0.0f;
    const unsigned n = first_block(r, m, top_p, near, whole ? 0u : kc, z, s);
    const float norm = whole ? z : mass(r, m, 1.0f, -INFINITY, Mark{0xFFFFFFFFu, 0xFFFFFFFFu}, top,
                                        end_mark(), false, s);
    if (r.t == 0) {
        float cum = 0.0f;
        unsigned keep = 0u;
        for (unsigned j = 0; j < n; j++) {
            cum += expf(val(s.ck[j]) - m) / norm;
            if (cum >= top_p) {
                keep = j + 1;
                break;
            }
        }
        s.st[0] = keep;
        s.fsh2[0] = cum;
    }
    __syncthreads();
    const unsigned keep = s.st[0];
    const float cum = s.fsh2[0];
    const Mark last = keep > 0u ? Mark{s.ck[keep - 1u], s.ci[keep - 1u]} : Mark{s.ck[n - 1u], s.ci[n - 1u]};
    __syncthreads();
    return keep > 0u ? last : extend(r, m, norm, lo, top_p, cum, last, top, s);
}
