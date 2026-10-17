// Latency-bound decode kernels (CSA2 attention core, mHC boundary) at 1-16 rows on 48 emulated SMs, cold L2 a launch.
#include "gb10emu.cuh"
#include "../attn_cuda.cu"
#include "../mhc_cuda.cu"
#include "attn_phases.cuh"
#include "attn_fastmerge.cuh"
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace {

__global__ void fill_bytes(uint8_t* p, size_t n, uint32_t seed, uint8_t mask) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 0x9E3779B1u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
        p[i] = (uint8_t)(x & mask);
    }
}
__global__ void fill_f32(float* p, size_t n, uint32_t seed, float lo, float hi) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 0x9E3779B1u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
        p[i] = lo + (hi - lo) * (x & 0xffffff) / 16777216.f;
    }
}
__global__ void flush(int4* p, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = make_int4((int)i, 0, 0, 0);
}

template <typename T> T* dalloc(size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T) + 256)); CK(cudaMemset(p, 0, n * sizeof(T) + 256)); return p; }
uint8_t* rbytes(size_t n, uint32_t seed, uint8_t mask) { auto* p = dalloc<uint8_t>(n); fill_bytes<<<512, 256>>>(p, n, seed, mask); return p; }
float* rf32(size_t n, uint32_t seed, float lo, float hi) { auto* p = dalloc<float>(n); fill_f32<<<512, 256>>>(p, n, seed, lo, hi); return p; }

// times fn() `reps` times, each after a cold-L2 write; returns the median in us
template <typename F>
double time_us(gb10::Emu& emu, cudaStream_t st, int reps, int4* fl, size_t fln, F fn) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<double> v;
    fn();                                                       // warm (first-launch attributes)
    CK(cudaStreamSynchronize(st));
    emu.begin();
    for (int i = 0; i < reps; ++i) {
        flush<<<emu.grid_sms() * 4, 512, 0, st>>>(fl, fln);
        CK(cudaEventRecord(a, st));
        fn();
        CK(cudaEventRecord(b, st));
        CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        v.push_back(ms * 1e3);
    }
    emu.end();
    CK(cudaGetLastError());
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {  // glue_bench [--mode full|sms] [--reps 40]
    gb10::eager();
    std::string mode = "sms"; int reps = 40;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--mode") mode = argv[i + 1]; else if (k == "--reps") reps = atoi(argv[i + 1]);
    }
    gb10::Emu emu; emu.init(mode == "full" ? gb10::Mode::Full : gb10::Mode::Sms);
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    const size_t fln = (256u << 20) / 16; int4* fl = dalloc<int4>(fln);
    constexpr int RMAX = 16, H = 32, NCOMP_ROWS = 16384, RING = 128, POS = 32768;
    using namespace dsv41_attn;
    // ---- attention: 512 selected compressed rows + the 128-row window a row, 32 heads (one rank) ----
    uint8_t* cv = rbytes((size_t)NCOMP_ROWS * VB, 1, 0x3b);
    uint8_t* csc = dalloc<uint8_t>((size_t)NCOMP_ROWS * SB); CK(cudaMemset(csc, 127, (size_t)NCOMP_ROWS * SB));
    uint8_t* sv = rbytes((size_t)RMAX * RING * VB, 2, 0x3b);
    uint8_t* ssc = dalloc<uint8_t>((size_t)RMAX * RING * SB); CK(cudaMemset(ssc, 127, (size_t)RMAX * RING * SB));
    std::vector<int> tok((size_t)RMAX * 512), cnt(RMAX, 512), lo(RMAX, 0);
    for (size_t i = 0; i < tok.size(); ++i) tok[i] = (int)((i * 2654435761u) % NCOMP_ROWS);
    int *dtok = dalloc<int>(tok.size()), *dcnt = dalloc<int>(RMAX), *dlo = dalloc<int>(RMAX), *dpos = dalloc<int>(1);
    CK(cudaMemcpy(dtok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcnt, cnt.data(), RMAX * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dlo, lo.data(), RMAX * 4, cudaMemcpyHostToDevice));
    const int pos = POS; CK(cudaMemcpy(dpos, &pos, 4, cudaMemcpyHostToDevice));
    uint8_t* q = rbytes((size_t)RMAX * H * D * 2, 3, 0x3b);
    float* cs = rf32((size_t)(POS + RMAX + 8) * 64, 4, -1.f, 1.f);
    float* sink = rf32(H, 5, -1.f, 1.f);
    float* po = dalloc<float>((size_t)NCH * RMAX * H * D);
    float *pm = dalloc<float>((size_t)NCH * RMAX * H), *pl = dalloc<float>((size_t)NCH * RMAX * H);
    uint16_t* out = dalloc<uint16_t>((size_t)RMAX * H * D);
    int* ticket = dalloc<int>(RMAX * H / BMQ);
    // ---- mHC boundary (mode 0: post of 2 ranks' partials, collapse, 24 mixes, finish, norm) ----
    constexpr int DM = dsv41_mhc::D, NBM = dsv41_mhc::NB;
    uint8_t* x = rbytes((size_t)RMAX * 4 * DM * 2, 6, 0x3b);
    uint16_t* xout = dalloc<uint16_t>((size_t)RMAX * 4 * DM);
    uint8_t* gp = rbytes((size_t)2 * RMAX * DM * 2, 7, 0x3b);
    float *post = rf32(RMAX * 4, 8, 0.1f, 1.f), *comb = rf32(RMAX * 16, 9, 0.f, 0.5f), *pre = rf32(RMAX * 4, 10, 0.f, 1.f);
    uint8_t* fn = rbytes((size_t)24 * 4 * DM * 2, 11, 0x3b);
    float *base = rf32(24, 12, -0.1f, 0.1f), *scale = rf32(3, 13, 0.5f, 1.f);
    float* part = dalloc<float>((size_t)RMAX * 4 * NBM * 32);
    uint16_t *cc = dalloc<uint16_t>((size_t)RMAX * DM), *mout = dalloc<uint16_t>((size_t)RMAX * DM);
    uint8_t* nw = rbytes((size_t)DM * 2, 14, 0x3b);
    float *opre = dalloc<float>(RMAX * 4), *opost = dalloc<float>(RMAX * 4), *ocomb = dalloc<float>(RMAX * 16);
    int* mcnt = dalloc<int>(1);
    CK(cudaDeviceSynchronize());

    // a CTA's time against its key count: chunk 0 holds `kc` selected keys (chunks 1-3 exit at once), the window kc
    if (getenv("ATTN_KEYS")) {
        for (int kc : {128, 64, 32, 16}) {
            std::vector<int> c1(RMAX, kc), l1(RMAX);
            for (int r = 0; r < RMAX; ++r) l1[r] = POS + r - (kc - 1);
            CK(cudaMemcpy(dcnt, c1.data(), RMAX * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dlo, l1.data(), RMAX * 4, cudaMemcpyHostToDevice));
            Args a{};
            a.q = reinterpret_cast<const uint16_t*>(q); a.cv = cv; a.csc = csc; a.cvs = VB; a.css = SB;
            a.tok = dtok; a.ts = 512; a.cnt = dcnt; a.sv = sv; a.ssc = ssc; a.svs = VB; a.sss = SB;
            a.lo = dlo; a.pos = dpos; a.sink = sink; a.cs = cs; a.csst = 64; a.po = po; a.pm = pm; a.pl = pl;
            a.out = out; a.ticket = ticket; a.R = 1; a.H = H; a.ring = RING; a.has_comp = 1;
            const double t = time_us(emu, st, reps, fl, fln, [&] { CK(dispatch(a, 4, false, false, st)); });
            printf("KEYS %3d a chunk (2 chunks x 2 head tiles working): %.1f us\n", kc, t);
        }
        CK(cudaMemcpy(dcnt, cnt.data(), RMAX * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dlo, lo.data(), RMAX * 4, cudaMemcpyHostToDevice));
    }
    if (getenv("ATTN_PHASES")) {                                // per-phase times of the R = 1 launch's CTAs
        dsv41_attn_ph::Args a{};
        a.q = reinterpret_cast<const uint16_t*>(q); a.cv = cv; a.csc = csc; a.cvs = VB; a.css = SB;
        a.tok = dtok; a.ts = 512; a.cnt = dcnt; a.sv = sv; a.ssc = ssc; a.svs = VB; a.sss = SB;
        a.lo = dlo; a.pos = dpos; a.sink = sink; a.cs = cs; a.csst = 64; a.po = po; a.pm = pm; a.pl = pl;
        a.out = out; a.ticket = ticket; a.R = 1; a.H = H; a.ring = RING; a.has_comp = 1;
        const double t = time_us(emu, st, reps, fl, fln, [&] { CK(dsv41_attn_ph::dispatch(a, 4, false, false, st)); });
        unsigned long long hs[256][10];
        CK(cudaMemcpyFromSymbol(hs, dsv41_attn_ph::g_stamp, sizeof(hs)));
        unsigned long long t0 = ~0ull;
        for (int i = 0; i < 10; ++i) t0 = std::min(t0, hs[i][0]);
        printf("PHASES launch %.1f us; per CTA (us from the first CTA's start): start rows+q keys S softmax PV ticket merge\n", t);
        for (int i = 0; i < 10; ++i) {
            printf("PHASES cta %d:", i);
            for (int k = 0; k < 8; ++k) printf(" %6.2f", hs[i][k] ? (hs[i][k] - t0) / 1e3 : -1.0);
            printf("\n");
        }
    }
    {                                                            // fast merge: same bits, time at 1-16 rows
        std::vector<uint16_t> ref((size_t)RMAX * H * D), got((size_t)RMAX * H * D);
        for (int R : {1, 4, 16}) {
            Args a{};
            a.q = reinterpret_cast<const uint16_t*>(q); a.cv = cv; a.csc = csc; a.cvs = VB; a.css = SB;
            a.tok = dtok; a.ts = 512; a.cnt = dcnt; a.sv = sv; a.ssc = ssc; a.svs = VB; a.sss = SB;
            a.lo = dlo; a.pos = dpos; a.sink = sink; a.cs = cs; a.csst = 64; a.po = po; a.pm = pm; a.pl = pl;
            a.out = out; a.ticket = ticket; a.R = R; a.H = H; a.ring = RING; a.has_comp = 1;
            CK(cudaMemset(out, 0, (size_t)RMAX * H * D * 2));
            const double t0 = time_us(emu, st, reps, fl, fln, [&] { CK(dispatch(a, 4, false, false, st)); });
            CK(cudaMemcpy(ref.data(), out, ref.size() * 2, cudaMemcpyDeviceToHost));
            dsv41_attn_fm::Args b{};
            static_assert(sizeof(b) == sizeof(a), "same Args");
            memcpy(&b, &a, sizeof(a));
            CK(cudaMemset(out, 0, (size_t)RMAX * H * D * 2));
            const double t1 = time_us(emu, st, reps, fl, fln, [&] { CK(dsv41_attn_fm::dispatch(b, 4, false, false, st)); });
            CK(cudaMemcpy(got.data(), out, got.size() * 2, cudaMemcpyDeviceToHost));
            size_t diff = 0, nz = 0;
            for (size_t i = 0; i < (size_t)R * H * D; ++i) { diff += ref[i] != got[i]; nz += ref[i] != 0; }
            printf("FASTMERGE R %2d: attn_cuda %.1f us, fast merge %.1f us, %zu of %zu outputs differ (%zu nonzero)\n",
                   R, t0, t1, diff, (size_t)R * H * D, nz);
        }
    }
    printf("mode %s: per launch, median of %d cold launches (us)\n rows | attn_kernel | mhc boundary\n", mode.c_str(), reps);
    for (int R : {1, 2, 4, 8, 16}) {
        Args a{};
        a.q = reinterpret_cast<const uint16_t*>(q); a.cv = cv; a.csc = csc; a.cvs = VB; a.css = SB;
        a.tok = dtok; a.ts = 512; a.cnt = dcnt; a.sv = sv; a.ssc = ssc; a.svs = VB; a.sss = SB;
        a.lo = dlo; a.hi = nullptr; a.pos = dpos; a.sl = nullptr; a.pt = nullptr; a.pts = 0; a.psh = 0;
        a.sink = sink; a.cs = cs; a.csst = 64; a.po = po; a.pm = pm; a.pl = pl; a.out = out; a.ticket = ticket;
        a.R = R; a.H = H; a.ring = RING; a.has_comp = 1;
        const double ta = time_us(emu, st, reps, fl, fln, [&] { CK(dispatch(a, 4, false, false, st)); });
        dsv41_mhc::Args m{};
        m.x = reinterpret_cast<const uint16_t*>(x); m.xs = 4 * DM; m.xout = xout; m.g = gp; m.gr = (long long)R * DM;
        m.world = 2; m.gbf16 = 1; m.post = post; m.comb = comb; m.pre = pre; m.fn = fn; m.base = base; m.scale = scale;
        m.part = part; m.c = cc; m.tap = nullptr; m.ts = 0; m.nw = nw; m.nwbf16 = 1; m.out = mout;
        m.opre = opre; m.opost = opost; m.ocomb = ocomb; m.cnt = mcnt; m.R = R;
        m.eps = 1e-20f; m.hc_eps = 1e-6f; m.post_alpha = 2.f; m.iters = 20;
        const double tm = time_us(emu, st, reps, fl, fln, [&] { CK(dsv41_mhc::dispatch(m, 0, false, st)); });
        printf("RESULT %4d | %9.1f | %9.1f\n", R, ta, tm);
    }
    return 0;
}
