// Times x3gm rotation, gate/up and down at one DeepSeek-V4.1 layer's per-rank shapes under gb10emu.cuh modes.
#include "gb10emu.cuh"
#include "../x3gm.cu"
#include "x3gm_ablate.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <string>

using namespace dsv41_x3gm;

namespace {

constexpr int E = 384, S = 6, D = 5120, I = 1152;

__global__ void fill_words(uint32_t* p, size_t n, uint32_t seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 0x9E3779B1u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        p[i] = x;
    }
}
__global__ void fill_half(half* p, size_t n, uint32_t seed, float lo, float hi) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 0x9E3779B1u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
        p[i] = __float2half(lo + (hi - lo) * (x & 0xffffff) / 16777216.f);
    }
}

using Kern = void (*)(const Args);
struct Launch { Kern k; int threads; size_t smem; int bm; };

template <int MATS, int XM, int K2, int MTL, int NG, int KS, int NSA, int NGR>
Launch make() {
    using C = Cfg<MATS, XM, K2, MTL, NG, KS, NSA, NGR>;
    Kern k = gm_kernel<MATS, XM, K2, MTL, NG, KS, NSA, NGR>;
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)C::SMEM));
    CK(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    return {k, C::THREADS, C::SMEM, C::BM};
}
template <int ABL, int MATS, int XM, int K2, int MTL, int NG, int KS, int NSA, int NGR>
Launch make_abl() {
    using C = Cfg<MATS, XM, K2, MTL, NG, KS, NSA, NGR>;
    Kern k = gm_ablate<ABL, MATS, XM, K2, MTL, NG, KS, NSA, NGR>;
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)C::SMEM));
    CK(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    return {k, C::THREADS, C::SMEM, C::BM};
}
// ablations (x3gm_ablate.cuh) at the default configurations, one rotated input, 2 and 3 bits
#define ABL_K(A, K2) if (abl == A && k2 == K2) return down ? make_abl<A, 1, 1, K2, GM_DN0>() : make_abl<A, 2, 1, K2, GM_GU0>();
#define ABL_A(A) ABL_K(A, 4) ABL_K(A, 6)
Launch abl_launch(int abl, int k2, bool down) {
    ABL_A(1) ABL_A(2) ABL_A(3) ABL_A(4) ABL_A(5) ABL_A(6) ABL_A(7)
    fprintf(stderr, "no ablation %d at k2 %d\n", abl, k2); exit(2);
}

// default instances (2,048 rows: gate/up 0 with one rotated input, else 1; down 0) + alternatives, K2 4-10 = 2-5 bits
#define GU_CASE(XM, K2, CFG, ID) if (xm == XM && k2 == K2 && id == ID) return make<2, XM, K2, CFG>();
#define DN_CASE(K2, CFG, ID) if (k2 == K2 && id == ID) return make<1, 1, K2, CFG>();
#define GU_W(K2) GU_CASE(1, K2, GM_GU0, 0) GU_CASE(1, K2, GM_GU1, 1) GU_CASE(1, K2, GM_GU5, 5) GU_CASE(1, K2, GM_GU2, 2) GU_CASE(1, K2, GM_GU3, 3) \
                 GU_CASE(2, K2, GM_GU0, 0) GU_CASE(2, K2, GM_GU1, 1) GU_CASE(2, K2, GM_GU5, 5)
#define DN_W(K2) DN_CASE(K2, GM_DN0, 0) DN_CASE(K2, GM_DN1, 1) DN_CASE(K2, GM_DN2, 2) DN_CASE(K2, GM_DN3, 3)
Launch gu_launch(int xm, int k2, int id) {
    GU_W(4) GU_W(6) GU_W(8) GU_W(10)
    fprintf(stderr, "no gate/up instance xm %d k2 %d cfg %d\n", xm, k2, id); exit(2);
}
Launch dn_launch(int k2, int id) {
    DN_W(4) DN_W(6) DN_W(8) DN_W(10)
    fprintf(stderr, "no down instance k2 %d cfg %d\n", k2, id); exit(2);
}

struct Plan { std::vector<int> pick, order, pe, poff, pcnt; int npass = 0; };

// routing: each row picks 6 distinct experts; zipf > 0 skews the experts' popularity (rank^-zipf)
Plan route(int rows, double zipf, int bm, uint32_t seed) {
    Plan p; std::mt19937 rng(seed);
    std::vector<double> w(E);
    for (int e = 0; e < E; ++e) w[e] = zipf > 0 ? std::pow(e + 1.0, -zipf) : 1.0;
    std::shuffle(w.begin(), w.end(), rng);
    std::discrete_distribution<int> pickd(w.begin(), w.end());
    p.pick.resize((size_t)rows * S);
    for (int r = 0; r < rows; ++r) {
        int got = 0;
        while (got < S) {
            const int e = pickd(rng);
            bool dup = false;
            for (int j = 0; j < got; ++j) dup |= p.pick[(size_t)r * S + j] == e;
            if (!dup) p.pick[(size_t)r * S + got++] = e;
        }
    }
    const int P = rows * S;
    p.order.resize(P); std::iota(p.order.begin(), p.order.end(), 0);
    std::stable_sort(p.order.begin(), p.order.end(), [&](int a, int b) { return p.pick[a] < p.pick[b]; });
    std::vector<int> cnt(E, 0);
    for (int x : p.pick) ++cnt[x];
    int off = 0;
    for (int e = 0; e < E; ++e) {
        for (int j = 0; j < cnt[e]; j += bm) {
            p.pe.push_back(e); p.poff.push_back(off + j); p.pcnt.push_back(std::min(bm, cnt[e] - j));
        }
        off += cnt[e];
    }
    p.npass = (int)p.pe.size();
    return p;
}

template <typename T> T* dev(const std::vector<T>& v) {
    T* d; CK(cudaMalloc(&d, v.size() * sizeof(T) + 16));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

}  // namespace

int main(int argc, char** argv) {  // flags: --k2 --rows --mode --gu --dn --xm --zipf --reps --target --abl
    int k2 = 4, rows = 2048, gu = 0, dn = 0, xm = 1, reps = 5, abl = 0;
    double zipf = 0, target = 240;
    std::string mode = "full";
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--k2") k2 = atoi(argv[i + 1]);
        else if (a == "--rows") rows = atoi(argv[i + 1]);
        else if (a == "--gu") gu = atoi(argv[i + 1]);
        else if (a == "--dn") dn = atoi(argv[i + 1]);
        else if (a == "--xm") xm = atoi(argv[i + 1]);
        else if (a == "--abl") abl = atoi(argv[i + 1]);
        else if (a == "--reps") reps = atoi(argv[i + 1]);
        else if (a == "--zipf") zipf = atof(argv[i + 1]);
        else if (a == "--target") target = atof(argv[i + 1]);
        else if (a == "--mode") mode = argv[i + 1];
        else { fprintf(stderr, "unknown %s\n", a.c_str()); return 2; }
    }
    gb10::eager();
    gb10::Emu emu;
    emu.init(mode == "gb10" ? gb10::Mode::Gb10 : mode == "sms" ? gb10::Mode::Sms : gb10::Mode::Full);
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

    const Launch G = abl ? abl_launch(abl, k2, false) : gu_launch(xm, k2, gu);
    const Launch Dn = abl ? abl_launch(abl, k2, true) : dn_launch(k2, dn);
    const Plan pl = route(rows, zipf, G.bm, 1234);
    const Plan pd = G.bm == Dn.bm ? pl : route(rows, zipf, Dn.bm, 1234);
    const int P = rows * S;
    const size_t mw = (size_t)E * (D / 16) * (I / 16) * 4 * k2;          // words of one stacked matrix
    uint32_t *tg, *tu, *td;
    CK(cudaMalloc(&tg, mw * 4)); CK(cudaMalloc(&tu, mw * 4)); CK(cudaMalloc(&td, mw * 4));
    fill_words<<<1024, 256>>>(tg, mw, 1); fill_words<<<1024, 256>>>(tu, mw, 2); fill_words<<<1024, 256>>>(td, mw, 3);
    half *x, *suh_g, *suh_u, *svg, *svu, *sud, *svd, *xg, *xu, *xd; float* y;
    CK(cudaMalloc(&x, (size_t)rows * D * 2));
    CK(cudaMalloc(&suh_g, (size_t)E * D * 2)); CK(cudaMalloc(&suh_u, (size_t)E * D * 2));
    CK(cudaMalloc(&svg, (size_t)E * I * 2)); CK(cudaMalloc(&svu, (size_t)E * I * 2));
    CK(cudaMalloc(&sud, (size_t)E * I * 2)); CK(cudaMalloc(&svd, (size_t)E * D * 2));
    CK(cudaMalloc(&xg, (size_t)P * D * 2)); CK(cudaMalloc(&xu, (size_t)P * D * 2)); CK(cudaMalloc(&xd, (size_t)P * I * 2));
    CK(cudaMalloc(&y, (size_t)P * D * 4));
    fill_half<<<1024, 256>>>(x, (size_t)rows * D, 4, -1.f, 1.f);
    for (half* p : {suh_g, suh_u}) fill_half<<<1024, 256>>>(p, (size_t)E * D, 5, -1.f, 1.f);
    for (half* p : {svg, svu, sud}) fill_half<<<1024, 256>>>(p, (size_t)E * I, 6, 0.5f, 1.5f);
    fill_half<<<1024, 256>>>(svd, (size_t)E * D, 7, 0.5f, 1.5f);
    int *pick = dev(pl.pick), *order = dev(pl.order), *pe = dev(pl.pe), *poff = dev(pl.poff), *pcnt = dev(pl.pcnt);
    int *orderd = dev(pd.order), *ped = dev(pd.pe), *poffd = dev(pd.poff), *pcntd = dev(pd.pcnt);
    int *np = dev(std::vector<int>{pl.npass}), *npd = dev(std::vector<int>{pd.npass});
    int* ticket; CK(cudaMalloc(&ticket, 8));
    CK(cudaDeviceSynchronize());

    int per_g = 0, per_d = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_g, G.k, G.threads, G.smem));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_d, Dn.k, Dn.threads, Dn.smem));
    emu.calibrate(tg, mw * 4, st, target);
    const int gg = emu.grid_sms() * per_g, gd = emu.grid_sms() * per_d;

    Args ag{}; ag.x0 = xg; ag.x1 = xm == 2 ? xu : nullptr; ag.t0 = tg; ag.t1 = tu; ag.order = order; ag.pe = pe;
    ag.poff = poff; ag.pcnt = pcnt; ag.npass = np; ag.ticket = ticket; ag.sv0 = svg; ag.sv1 = svu; ag.sd = sud;
    ag.xd = xd; ag.K = D; ag.N = I; ag.limit = 10.f;
    Args ad{}; ad.x0 = xd; ad.t0 = td; ad.order = orderd; ad.pe = ped; ad.poff = poffd; ad.pcnt = pcntd; ad.npass = npd;
    ad.ticket = ticket + 1; ad.sv0 = svd; ad.y = y; ad.K = I; ad.N = D;
    const int rot_items = P * (D / 128);

    cudaEvent_t ev[4];
    for (auto& e : ev) CK(cudaEventCreate(&e));
    std::vector<double> t_rot, t_gu, t_dn;
    for (int r = 0; r < reps + 1; ++r) {
        CK(cudaMemsetAsync(ticket, 0, 8, st));
        emu.begin();
        const bool dbg = getenv("X3GM_DEBUG") != nullptr;
        auto step = [&](const char* w) { if (dbg) { CK(cudaStreamSynchronize(st)); fprintf(stderr, "  %s done\n", w); } };
        step("begin");
        CK(cudaEventRecord(ev[0], st));
        if (xm == 2) rot_kernel<half, 2><<<(rot_items + 7) / 8, 256, 0, st>>>(x, D, pick, suh_g, suh_u, xg, xu, D, S, rot_items);
        else rot_kernel<half, 1><<<(rot_items + 7) / 8, 256, 0, st>>>(x, D, pick, suh_g, nullptr, xg, nullptr, D, S, rot_items);
        step("rot");
        CK(cudaEventRecord(ev[1], st));
        G.k<<<gg, G.threads, G.smem, st>>>(ag);
        step("gateup");
        CK(cudaEventRecord(ev[2], st));
        Dn.k<<<gd, Dn.threads, Dn.smem, st>>>(ad);
        CK(cudaEventRecord(ev[3], st));
        CK(cudaGetLastError());
        CK(cudaEventSynchronize(ev[3]));
        emu.end();
        float a, b, c;
        CK(cudaEventElapsedTime(&a, ev[0], ev[1])); CK(cudaEventElapsedTime(&b, ev[1], ev[2])); CK(cudaEventElapsedTime(&c, ev[2], ev[3]));
        if (r) { t_rot.push_back(a); t_gu.push_back(b); t_dn.push_back(c); }
    }
    const double bits = k2 / 2.0, wbytes = 3.0 * E * D * I * bits / 8;
    const double flop = 2.0 * P * 3.0 * D * I;
    const double tr = median(t_rot), tg_ = median(t_gu), td_ = median(t_dn), tot = tr + tg_ + td_;
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
    int clk = 0; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
    printf("RESULT abl=%d mode=%s k2=%d rows=%d xm=%d gu=%d dn=%d zipf=%.2f passes=%d/%d grid=%d/%d (per_sm %d/%d) sms=%d clk=%dMHz hogs=%d\n",
           abl, mode.c_str(), k2, rows, xm, gu, dn, zipf, pl.npass, pd.npass, gg, gd, per_g, per_d, emu.grid_sms(), clk / 1000, emu.hogs);
    printf("RESULT ms rot %.3f gateup %.3f down %.3f total %.3f | weights %.3f GB -> %.1f GB/s | %.1f TFLOP/s\n",
           tr, tg_, td_, tot, wbytes / 1e9, wbytes / (tot * 1e6), flop / (tot * 1e9));
    return 0;
}
