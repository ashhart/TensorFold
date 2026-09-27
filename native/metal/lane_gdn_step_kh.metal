
    const uint tid = thread_position_in_threadgroup.x;
    const uint r = tid / DV;
    const uint dv = tid % DV;
    const uint group = threadgroup_position_in_grid.x;
    const uint n = group / HK;
    const uint hk = group % HK;
    const uint h = hk * R + r;
    const uint sg = tid / 32;
    const uint sl = tid % 32;
    const int t = tlen[0];
    const uint lanes = uint(tlen[1]);
    constexpr int NSG = (R * DV) / 32;

    threadgroup float ks[DK];
    threadgroup float qs[DK];
    threadgroup float dk[CAP];
    threadgroup float dq[CAP];
    threadgroup float aw[R * CAP];
    threadgroup float cw[R * CAP];
    threadgroup float red[NSG];

    if (tid < DK) {
        ks[tid] = static_cast<float>(k[(n * HK + hk) * DK + tid]);
        qs[tid] = static_cast<float>(q[(n * HK + hk) * DK + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float kqp = (tid < DK) ? ks[tid] * qs[tid] : 0.0f;
    kqp = simd_sum(kqp);
    if (sl == 0) {
        red[sg] = kqp;
    }
    // Past keys against this key and this query, once per key head.
    for (int i = int(sg); i < t; i += NSG) {
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
            dk[i] = a;
            dq[i] = c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Decay weights per value head.
    for (int e = int(tid); e < R * t; e += R * DV) {
        const int rr = e / t;
        const int i = e % t;
        const uint hh = hk * R + uint(rr);
        const float lgr = log_prev[n * HV + hh] + log_g[n * HV + hh];
        const float w = metal::exp(lgr - lg_hist[(n * HV + hh) * CAP + i]);
        aw[rr * CAP + i] = w * dk[i];
        cw[rr * CAP + i] = w * dq[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float kq = 0.0f;
    for (int s2 = 0; s2 < DK / 32; ++s2) {
        kq += red[s2];
    }

    const float lg = log_prev[n * HV + h] + log_g[n * HV + h];
    const float decay = metal::exp(lg);
    const float s0k = s0kq[((h * lanes + n) * 2 + 0) * DV + dv];
    const float s0q = s0kq[((h * lanes + n) * 2 + 1) * DV + dv];
    const device HistT* dcol = d_hist + (n * HV + h) * CAP * DV + dv;
    float memory = decay * s0k;
    float yv = decay * s0q;
    for (int i = 0; i < t; ++i) {
        float dval = static_cast<float>(dcol[i * DV]);
        memory += aw[r * CAP + i] * dval;
        yv += cw[r * CAP + i] * dval;
    }
    const float delta = (static_cast<float>(v[(n * HV + h) * DV + dv]) - memory) * beta[n * HV + h];
    yv += kq * delta;
    y[(n * HV + h) * DV + dv] = static_cast<OutT>(yv);
    delta_out[(n * HV + h) * DV + dv] = static_cast<HistT>(delta);
    if (dv == 0) {
        log_out[n * HV + h] = lg;
    }
