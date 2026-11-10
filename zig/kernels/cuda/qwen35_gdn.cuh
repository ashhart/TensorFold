// gdn_pre, conv_commit and gdn_chain. qwen35.cu includes this twice: TF_Q35_TABLE 0 (main's), 1 (_table, TF_Q35_K)

// Conv, silu, q/k norms, the linear heads' decay and write: a warp a (row, head) over q, k, v; taps from the parents.
extern "C" __global__ void __launch_bounds__(32) TF_Q35_K(tf_q35_gdn_pre)(Geo geo, const __nv_bfloat16* __restrict__ QKV, const Seg* __restrict__ segs,
                                                                  const int* __restrict__ rowseg, const int* __restrict__ rowpar, int layer,
                                                                  const __nv_bfloat16* __restrict__ CW,
                                                                  const __nv_bfloat16* __restrict__ Ain, const __nv_bfloat16* __restrict__ Bin,
                                                                  const float* __restrict__ ALOG, const __nv_bfloat16* __restrict__ DT,
                                                                  __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K,
                                                                  __nv_bfloat16* __restrict__ V, float* __restrict__ G, __nv_bfloat16* __restrict__ BETA,
                                                                  const int* __restrict__ P) {
    const int lane = threadIdx.x;
    const int head = blockIdx.y;
    const int w = blockIdx.z;
    if (past_rows(P, w)) return;
    const int NK = geo.nk, NV = geo.nv, C = geo.conv();
#if TF_Q35_TABLE
    const int s = rowseg[w];
    const Seg sg = segs[s];
    const unsigned long long* T = forward_states(geo, segs, s);
    const __nv_bfloat16* CS = conv_of(geo, T, layer);
#else
    const Seg sg = segs[rowseg[w]];
    const __nv_bfloat16* CS = reinterpret_cast<const __nv_bfloat16*>(sg.conv) + (size_t)layer * geo.conv_rows();
#endif
    const bool isq = head < NK, isk = !isq && head < 2 * NK;
    const int c0 = head * DK;  // q, k and v channels are consecutive heads of DK
    int src[TAPS];  // tap t reads the row TAPS - 1 - t tokens back: a forward row, or -1 - (tail row)
#pragma unroll
    for (int tap = 0; tap < TAPS; tap++) src[tap] = back(rowpar, w, TAPS - 1 - tap);
    constexpr int PER = DK / 32;
    // a recorded window's row logs the tail after it (taps 1..TAPS-1) here, so conv_commit only moves the stream's
#if TF_Q35_TABLE
    __nv_bfloat16* tail = recorded(P) ? log_row(geo, T, layer, w - sg.row0).conv : nullptr;
#else
    __nv_bfloat16* tail = recorded(P) ? log_row(geo, sg, layer, w - sg.row0).conv : nullptr;
#endif
    // a lane's PER channels in 8-byte loads: each tap's row, the weights (PER x TAPS in a row), the logged tail
    static_assert(PER == 4 && TAPS == 4, "a tap's channels and a channel's taps are one uint2");
    const int cl = c0 + lane * PER;
    uint2 xw[TAPS], ww[TAPS];
#pragma unroll
    for (int tap = 0; tap < TAPS; tap++) {
        const int at = src[tap];
        xw[tap] = *reinterpret_cast<const uint2*>(at < 0 ? CS + (size_t)(-1 - at) * C + cl : QKV + (size_t)at * C + cl);
        ww[tap] = *reinterpret_cast<const uint2*>(CW + (size_t)cl * TAPS + tap * PER);
        if (tail != nullptr && tap > 0) *reinterpret_cast<uint2*>(tail + (size_t)(tap - 1) * C + cl) = xw[tap];
    }
    const __nv_bfloat16* cw = reinterpret_cast<const __nv_bfloat16*>(ww);
    float vals[PER];
#pragma unroll
    for (int j = 0; j < PER; j++) {
        float acc = 0.0f;
#pragma unroll
        for (int tap = 0; tap < TAPS; tap++) acc += bf(cw[j * TAPS + tap]) * bf(reinterpret_cast<const __nv_bfloat16*>(&xw[tap])[j]);
        const float conv = rbf(acc);
        const float sig = rbf(1.0f / (1.0f + expf(-conv)));
        vals[j] = rbf(conv * sig);
    }
    if (isq || isk) {
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; j++) ss += vals[j] * vals[j];
        ss = warp_sum(ss);
        const float inv = rsqrtf(ss / (float)DK + (1e-6f / (float)DK));
        const float scale = isq ? rbf(1.0f / (float)DK) : rbf(rsqrtf((float)DK));
        __nv_bfloat16* out = isq ? Q + ((size_t)w * NK + head) * DK : K + ((size_t)w * NK + head - NK) * DK;
#pragma unroll
        for (int j = 0; j < PER; j++) out[lane * PER + j] = __float2bfloat16_rn(scale * rbf(vals[j] * inv));
    } else {
        const int hv = head - 2 * NK;
#pragma unroll
        for (int j = 0; j < PER; j++) V[((size_t)w * NV + hv) * DK + lane * PER + j] = __float2bfloat16_rn(vals[j]);
        if (lane == 0) {
            const float s = rbf(bf(Ain[w * NV + hv]) + bf(DT[hv]));
            const float sp = rbf(fmaxf(s, 0.0f) + logf(1.0f + expf(-fabsf(s))));
            G[w * NV + hv] = expf(-expf(ALOG[hv]) * sp);
            BETA[w * NV + hv] = __float2bfloat16_rn(1.0f / (1.0f + expf(-bf(Bin[w * NV + hv]))));
        }
    }
}

// A segment's conv tail: a recorded window leaves the stream's (gdn_pre logged its rows'); others update it in place.
extern "C" __global__ void TF_Q35_K(tf_q35_conv_commit)(Geo geo, const __nv_bfloat16* __restrict__ QKV, const Seg* __restrict__ segs,
                                                const int* __restrict__ rowpar, int layer, const int* __restrict__ P) {
    const int C = geo.conv();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || past_windows(P, blockIdx.y)) return;
    const Seg sg = segs[blockIdx.y];
#if TF_Q35_TABLE
    const unsigned long long* T = forward_states(geo, segs, blockIdx.y);
    __nv_bfloat16* CS = conv_of(geo, T, layer);
    if (recorded(P)) {
#else
    __nv_bfloat16* CS = reinterpret_cast<__nv_bfloat16*>(sg.conv) + (size_t)layer * geo.conv_rows();
    if (recorded(P)) {
#endif
        // gdn_pre logged each row's tail
        if (sg.rows != 1) return;
#if TF_Q35_TABLE
        __nv_bfloat16* before = before_conv(geo, T, layer);
#else
        __nv_bfloat16* before = reinterpret_cast<__nv_bfloat16*>(sg.before + geo.nl * geo.rec() * sizeof(float)) + (size_t)layer * geo.conv_rows();
#endif
        for (int r = 0; r < TAPS - 1; ++r) before[(size_t)r * C + c] = CS[(size_t)r * C + c];
    }
    // ascending: slot r takes [r + rows], which is either a later state slot (not yet written) or a row
    for (int r = 0; r < TAPS - 1; r++) {
        const int at = r + sg.rows;
        CS[(size_t)r * C + c] = at < TAPS - 1 ? CS[(size_t)at * C + c] : QKV[(size_t)(sg.row0 + at - (TAPS - 1)) * C + c];
    }
}

// Gated delta rule, each row's state its parent's: a chain keeps it in registers, a branch rebuilds it from the log.
extern "C" __global__ void __launch_bounds__(128) TF_Q35_K(tf_q35_gdn_chain)(Geo geo, const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                                                                     const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                                                                     const __nv_bfloat16* __restrict__ beta, const Seg* __restrict__ segs,
                                                                     const int* __restrict__ rowpar, int layer, __nv_bfloat16* __restrict__ y,
                                                                     const int* __restrict__ P) {
    const int NK = geo.nk, NV = geo.nv;
    if (past_windows(P, blockIdx.z / NV)) return;
    const int hv = blockIdx.z % NV;
    const int hk = hv / (NV / NK);
    const Seg sg = segs[blockIdx.z / NV];
#if TF_Q35_TABLE
    const unsigned long long* T = forward_states(geo, segs, blockIdx.z / NV);
#endif
    const int dk = threadIdx.x;
    const int dv = blockIdx.y * 4 + threadIdx.y;
    constexpr int PT = DK / 32;
    const size_t at = ((size_t)hv * DK + dv) * DK + PT * dk;
#if TF_Q35_TABLE
    float* st = rec_of(T, layer) + at;
#else
    float* st = reinterpret_cast<float*>(sg.rec) + (size_t)layer * geo.rec() + at;
#endif
    const bool record = recorded(P);
    float state[PT];
#pragma unroll
    for (int i = 0; i < PT; i++) state[i] = st[i];
    int prev = -1;  // the row the registers' state follows
    // row r's inputs, loaded while row r - 1 computes
    float kn[PT], qn[PT], vn = 0.0f, gn = 0.0f, bn = 0.0f;
    int pn = -1;
    auto load = [&](int row) {
        const uint2 kw = *reinterpret_cast<const uint2*>(k + ((size_t)row * NK + hk) * DK + PT * dk);
        const uint2 qw = *reinterpret_cast<const uint2*>(q + ((size_t)row * NK + hk) * DK + PT * dk);
        const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kw);
        const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(&qw);
#pragma unroll
        for (int i = 0; i < PT; i++) {
            kn[i] = bf(kb[i]);
            qn[i] = bf(qb[i]);
        }
        vn = bf(v[((size_t)row * NV + hv) * DK + dv]);
        gn = g[row * NV + hv];
        bn = bf(beta[row * NV + hv]);
        pn = rowpar[row];
    };
    if (sg.rows > 0) load(sg.row0);
    for (int r = 0; r < sg.rows; r++) {
        const int row = sg.row0 + r;
        float kc[PT], qc[PT];
#pragma unroll
        for (int i = 0; i < PT; i++) {
            kc[i] = kn[i];
            qc[i] = qn[i];
        }
        const float vc = vn, gc = gn, bc = bn;
        const int parent = pn;
        if (r + 1 < sg.rows) load(row + 1);
        if (parent != prev) {
            // a branch: its parent's state from the stream's and the ancestors' logged updates, oldest first
            int anc[MAXW];
            int n = 0;
            for (int a = parent; a >= 0; a = rowpar[a]) anc[n++] = a;
#pragma unroll
            for (int i = 0; i < PT; i++) state[i] = st[i];
            for (int j = n - 1; j >= 0; j--) {
                const int a = anc[j];
#if TF_Q35_TABLE
                const float ga = g[a * NV + hv], da = log_row(geo, T, layer, a - sg.row0).delta[hv * DK + dv];
#else
                const float ga = g[a * NV + hv], da = log_row(geo, sg, layer, a - sg.row0).delta[hv * DK + dv];
#endif
#pragma unroll
                for (int i = 0; i < PT; i++) {
                    state[i] = state[i] * ga;
                    state[i] = fmaf(bf(k[((size_t)a * NK + hk) * DK + PT * dk + i]), da, state[i]);
                }
            }
        }
        prev = row;
        float kv_mem = 0.0f;
#pragma unroll
        for (int i = 0; i < PT; i++) {
            state[i] = state[i] * gc;
            kv_mem += state[i] * kc[i];
        }
        kv_mem = warp_sum(kv_mem);
        const float delta = (vc - kv_mem) * bc;
        float out = 0.0f;
#pragma unroll
        for (int i = 0; i < PT; i++) {
            state[i] = fmaf(kc[i], delta, state[i]);
            out += state[i] * qc[i];
        }
        out = warp_sum(out);
        if (dk == 0) y[((size_t)row * NV + hv) * DK + dv] = __float2bfloat16_rn(out);
        if (record) {
#if TF_Q35_TABLE
            const LogRow l = log_row(geo, T, layer, r);
#else
            const LogRow l = log_row(geo, sg, layer, r);
#endif
            if (dk == 0) {
                l.delta[hv * DK + dv] = delta;
                if (dv == 0) l.g[hv] = gc;
            }
            if (dv == 0 && hv % (NV / NK) == 0)
#pragma unroll
                for (int i = 0; i < PT; i++) l.k[hk * DK + PT * dk + i] = k[((size_t)row * NK + hk) * DK + PT * dk + i];
        }
    }
    if (record && sg.rows != 1) return;
#pragma unroll
    for (int i = 0; i < PT; i++) st[i] = state[i];
}
