
    // One threadgroup per (lane, value head); thread dv owns output row dv.
    const uint dv = thread_position_in_threadgroup.x;
    const uint group = threadgroup_position_in_grid.x;
    const uint n = group / HV;
    const uint h = group % HV;
    const uint hk = h / (HV / HK);
    const uint sg = dv / 32;
    const uint sl = dv % 32;
    const int t = tlen[0];

    threadgroup float ks[DK];
    threadgroup float qs[DK];
    threadgroup float aw[CAP];
    threadgroup float cw[CAP];
    threadgroup float red[DV / 32];

    ks[dv] = static_cast<float>(k[(n * HK + hk) * DK + dv]);
    qs[dv] = static_cast<float>(q[(n * HK + hk) * DK + dv]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S0 k and S0 q, from one batched matmul over all lanes outside the kernel.
    const float s0k = s0kq[((n * HV + h) * 2 + 0) * DV + dv];
    const float s0q = s0kq[((n * HV + h) * 2 + 1) * DV + dv];

    const float lg = log_prev[n * HV + h] + log_g[n * HV + h];
    const float decay = metal::precise::exp(lg);

    // Past keys against this key and this query; one simdgroup per stride of i.
    for (int i = int(sg); i < t; i += int(DV / 32)) {
        const device HistT* krow = k_hist + ((n * HK + hk) * CAP + i) * DK;
        float a = 0.0f;
        float c = 0.0f;
        for (int j = int(sl); j < DK; j += 32) {
            float kv = static_cast<float>(krow[j]);
            a += kv * ks[j];
            c += kv * qs[j];
        }
        a = simd_sum(a);
        c = simd_sum(c);
        if (sl == 0) {
            float w = metal::precise::exp(lg - lg_hist[(n * HV + h) * CAP + i]);
            aw[i] = w * a;
            cw[i] = w * c;
        }
    }
    float kq = simd_sum(ks[dv] * qs[dv]);
    if (sl == 0) {
        red[sg] = kq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kq = 0.0f;
    for (int r = 0; r < int(DV / 32); ++r) {
        kq += red[r];
    }

    const device HistT* dcol = d_hist + (n * HV + h) * CAP * DV + dv;
    float memory = decay * s0k;
    float yv = decay * s0q;
    for (int i = 0; i < t; ++i) {
        float dval = static_cast<float>(dcol[i * DV]);
        memory += aw[i] * dval;
        yv += cw[i] * dval;
    }
    const float delta = (static_cast<float>(v[(n * HV + h) * DV + dv]) - memory) * beta[n * HV + h];
    yv += kq * delta;
    y[(n * HV + h) * DV + dv] = static_cast<OutT>(yv);
    delta_out[(n * HV + h) * DV + dv] = static_cast<HistT>(delta);
    if (dv == 0) {
        log_out[n * HV + h] = lg;
    }
