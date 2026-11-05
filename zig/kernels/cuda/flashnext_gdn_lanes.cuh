// A shared pass's lanes, one launch a DeltaNet step (blockIdx.z the lane, each its lone launch's program).
namespace fn_lanes {
constexpr int MAXL = 16;
struct Lane {
    const __nv_bfloat16* proj;  // the lane's projection rows in the pass
    __nv_bfloat16* conv;        // its conv window
    float* state;               // its DeltaNet state (the chain's start)
    float* state_out;           // the window's end state (a spare); null: in place
    float* log;                 // its pending-row log (fn_late); null: none
    const unsigned* ctl;
    __nv_bfloat16* keep_proj;   // where its rollback keeps the rows and the conv window before them; null: none
    __nv_bfloat16* keep_hist;
    int row0, rows;             // its first row in the pass, its rows
};
struct Lanes {
    Lane l[MAXL];
    int n;
};
}  // namespace fn_lanes

// Each lane's rows and conv window into its keep (the lone pass's two copies), 16 bytes a thread.
extern "C" __global__ void fn_gdn_lanes_keep(const fn_lanes::Lanes t) {
    using namespace fn_gdn;
    constexpr int C = G<NK, NV>::C, PW = G<NK, NV>::PW;
    const fn_lanes::Lane& l = t.l[blockIdx.y];
    const long long nproj = (long long)l.rows * PW * 2 / 16, nhist = (long long)(TAPS - 1) * C * 2 / 16;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < nproj + nhist; i += (long long)gridDim.x * blockDim.x) {
        if (i < nproj) {
            if (l.keep_proj) reinterpret_cast<uint4*>(l.keep_proj)[i] = reinterpret_cast<const uint4*>(l.proj)[i];
        } else if (l.keep_hist) reinterpret_cast<uint4*>(l.keep_hist)[i - nproj] = reinterpret_cast<const uint4*>(l.conv)[i - nproj];
    }
}

// fn_gdn_front (or, with a log, fn_gdn_front_log) of every lane: grid (the widest lane's rows, NK + NV, lanes).
extern "C" __global__ void __launch_bounds__(256) fn_gdn_lanes_front(
        const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        float* __restrict__ qn, float* __restrict__ kn, float* __restrict__ v, float* __restrict__ gb, int log_rows,
        const fn_lanes::Lanes t) {
    using namespace fn_gdn;
    const fn_lanes::Lane& l = t.l[blockIdx.z];
    if ((int)blockIdx.x >= l.rows) return;
    float* q = qn + (size_t)l.row0 * NK * DK;
    if (l.log) {
        float* vl = l.log + (size_t)log_rows * NK * DK;
        gdn_front<NK, NV, true>(l.proj, l.conv, cw, a_log, dt_bias, q, l.log, vl, vl + (size_t)log_rows * NV * DV, l.ctl, log_rows);
    } else {
        gdn_front<NK, NV, false>(l.proj, l.conv, cw, a_log, dt_bias, q, kn + (size_t)l.row0 * NK * DK, v + (size_t)l.row0 * NV * DV,
                         gb + (size_t)l.row0 * NV * 2, nullptr, 1);
    }
}

// Every lane's chain (with a log fn_gdn_rows_late, else fn_gdn_rows_to or in place): grid (NV, DV / 16, lanes).
extern "C" __global__ void __launch_bounds__(128) fn_gdn_lanes_chain(
        const float* __restrict__ qn, const float* __restrict__ kn, const float* __restrict__ v, const float* __restrict__ gb,
        float* __restrict__ ys, int log_rows, int store_above, const fn_lanes::Lanes t) {
    using namespace fn_gdn;
    const fn_lanes::Lane& l = t.l[blockIdx.z];
    const float* q = qn + (size_t)l.row0 * NK * DK;
    float* y = ys + (size_t)l.row0 * NV * DV;
    if (l.log) gdn_rows_late(q, l.log, l.ctl, log_rows, store_above, l.state, l.state_out, l.rows, y);
    else gdn_rows<NK, NV>(q, kn + (size_t)l.row0 * NK * DK, v + (size_t)l.row0 * NV * DV, gb + (size_t)l.row0 * NV * 2, l.state,
                  l.state_out ? l.state_out : l.state, l.rows, y);
}

// Every lane's conv window moved on past its rows: grid (C / 256 rounded up, lanes).
extern "C" __global__ void fn_gdn_lanes_shift(const fn_lanes::Lanes t) {
    using namespace fn_gdn;
    constexpr int C = G<NK, NV>::C, PW = G<NK, NV>::PW;
    const fn_lanes::Lane& l = t.l[blockIdx.y];
    shift_window(l.conv, l.proj, PW, l.rows, TAPS - 1, C);
}
