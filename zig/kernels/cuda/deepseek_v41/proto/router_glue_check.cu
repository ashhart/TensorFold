// Oracle and timing for router_glue.cu against the pinned upstream code; usage: router_glue_check [--reps N].

// Keep fmad on: the oracle reproduces upstream's FMA tree bit for bit (SASS-audited on sm_120/sm_121).
#include "../router_glue.cu"
#include "../exl3_experts.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(call) do { const cudaError_t err = (call); if (err != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
    std::exit(2); } } while (0)

template <typename T> struct Device {
    T* p;
    size_t n;
    explicit Device(size_t count) : n(count) { CK(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void set(const std::vector<T>& v) { CK(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    void sentinel() { CK(cudaMemset(p, 0xa5, n * sizeof(T))); }
    std::vector<T> get() const {
        std::vector<T> v(n); CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
    }
};

template <typename T> void same(const Device<T>& ref, const Device<T>& got, const char* what, int R, int mode, int edges) {
    const auto a = ref.get(), b = got.get();
    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0) return;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(a.data() + i, b.data() + i, sizeof(T)) != 0) {
            std::fprintf(stderr, "FAIL %s R=%d mode=%d edges=%d element=%zu\n", what, R, mode, edges, i);
            std::exit(1);
        }
}

uint32_t mix(uint32_t v) { v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; return v ^ (v >> 16); }

template <typename F> float time_us(int reps, F launch) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    launch(); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int q = 0; q < reps; ++q) launch();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms * 1000.0f / reps;
}

template <typename F> float time_graph_us(int reps, F launch) {
    cudaStream_t stream; CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph; cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch(stream);
    CK(cudaStreamEndCapture(stream, &graph));
    CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    const float us = time_us(reps, [&] { CK(cudaGraphLaunch(exec, nullptr)); });
    CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph)); CK(cudaStreamDestroy(stream));
    return us;
}

void check_case(int R, int mode, bool edges, int reps) {
    constexpr int K = 5120;
    const int slots = mode == 4 ? 1 : mode == 6 ? 4 : (mode == 5 || mode == 7) ? 7 : 9;
    const int E = mode == 4 ? 1 : mode == 5 ? 512 : mode == 6 ? 17 : mode == 7 ? 385 : 257;
    const int maxm = mode == 2 ? 3 : R;
    const int stride = K + 8, P = R * slots;
    const size_t out_n = (size_t)P * K + 32;
    Device<uint16_t> x((size_t)R * stride), suh0((size_t)E * K), suh1((size_t)E * K);
    Device<int> pick(P), ids0(E + 32), ids1(E + 32), ids2(E + 32), count0(1), count1(1), count2(1);
    Device<int> mem0((size_t)E * maxm + 32), mem1((size_t)E * maxm + 32), mem2((size_t)E * maxm + 32);
    Device<uint16_t> out00(out_n), out01(out_n), out10(out_n), out11(out_n);
    std::vector<uint16_t> xv(x.n), s0(suh0.n), s1(suh1.n);
    // bf16/fp16 edge bits (signed zeros, subnormals, maxima, infinities, NaNs) plus finite values of both signs
    const uint16_t bf_edge[] = {0, 0x8000, 1, 0x8001, 0x007f, 0x0080, 0x3f80, 0xbf80,
                               0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0xffc2};
    const uint16_t hf_edge[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00,
                               0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e01, 0xfe02};
    for (size_t i = 0; i < xv.size(); ++i) {
        const auto h = mix((uint32_t)i + 77);
        xv[i] = edges ? bf_edge[h % 14] : (uint16_t)((h & 0x8000u) | (0x3e00u + h % 512));
    }
    for (size_t i = 0; i < s0.size(); ++i) {
        const auto h = mix((uint32_t)i + 41), j = mix((uint32_t)i + 43);
        s0[i] = edges ? hf_edge[h % 14] : (uint16_t)((h & 0x8000u) | (0x3400u + h % 2048));
        s1[i] = edges ? hf_edge[j % 14] : (uint16_t)((j & 0x8000u) | (0x3400u + j % 2048));
    }
    std::vector<int> pv(P);
    for (int i = 0; i < P; ++i) {
        const int slot = i % slots;
        if (mode == 0 || mode == 5 || mode == 6 || mode == 7)
            pv[i] = slot == slots - 1 ? E - 1 : (i * 13 + slot * 7) % (E - 1);
        else if (mode == 1) pv[i] = i % 5 == 0 ? E : i % 7 == 0 ? -1 : (i * 11) % E;
        else if (mode == 2) pv[i] = i % 3;  // duplicates and truncation to maxm=3
        else if (mode == 3) pv[i] = i % 2 ? E : -1;  // no valid experts
        else pv[i] = 0;
    }
    x.set(xv); suh0.set(s0); suh1.set(s1); pick.set(pv);
    ids0.sentinel(); ids1.sentinel(); ids2.sentinel(); mem0.sentinel(); mem1.sentinel(); mem2.sentinel();
    out00.sentinel(); out01.sentinel(); out10.sentinel(); out11.sentinel();
    const auto reference_group = [&](cudaStream_t stream = nullptr) {
        tf_exl3_experts::group_kernel<<<1, 1024, P * sizeof(int), stream>>>(pick.p, ids0.p, count0.p, mem0.p, R, slots, E, maxm);
        CK(cudaGetLastError());
    };
    const auto reference = [&](cudaStream_t stream = nullptr) {
        reference_group(stream);
        tf_exl3_experts::rot_in_kernel<__nv_bfloat16><<<dim3(P, K / 128, 2), 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.p), stride, pick.p,
            reinterpret_cast<const half*>(suh0.p), reinterpret_cast<const half*>(suh1.p),
            reinterpret_cast<half*>(out00.p), reinterpret_cast<half*>(out01.p), K, slots, E);
        CK(cudaGetLastError());
    };
    const auto group = [&](cudaStream_t stream = nullptr) {
        router_group<<<1, 1024, 0, stream>>>(pick.p, ids2.p, count2.p, mem2.p, R, slots, E, maxm);
        CK(cudaGetLastError());
    };
    const auto fused = [&](cudaStream_t stream = nullptr) {
        router_group_rot<<<1 + (P * (K / 128) * 2 + 31) / 32, 1024, 0, stream>>>(x.p, stride, pick.p,
            reinterpret_cast<const half*>(suh0.p), reinterpret_cast<const half*>(suh1.p),
            reinterpret_cast<half*>(out10.p), reinterpret_cast<half*>(out11.p),
            ids1.p, count1.p, mem1.p, R, K, slots, E, maxm);
        CK(cudaGetLastError());
    };
    reference(); group(); fused(); CK(cudaDeviceSynchronize());
    same(count0, count1, "fused count", R, mode, edges); same(count0, count2, "group count", R, mode, edges);
    same(ids0, ids1, "fused ids", R, mode, edges); same(ids0, ids2, "group ids", R, mode, edges);
    same(mem0, mem1, "fused members", R, mode, edges); same(mem0, mem2, "group members", R, mode, edges);
    same(out00, out10, "gate rotation", R, mode, edges); same(out01, out11, "up rotation", R, mode, edges);
    // Repeated launches require no ticket reset; check both integer and fp bits.
    for (int repeat = 0; repeat < 3; ++repeat) fused();
    CK(cudaDeviceSynchronize());
    same(mem0, mem1, "replayed members", R, mode, edges);
    same(out00, out10, "replayed gate", R, mode, edges);
    if ((mode == 0 || mode == 6 || mode == 7) && !edges) {
        const float old_group = time_us(reps, reference_group), new_group = time_us(reps, group);
        const float old_pair = time_us(reps, reference), new_pair = time_us(reps, fused);
        std::printf("PASS R=%d slots=%d E=%d group_us %.3f -> %.3f pair_us %.3f -> %.3f (warm eager)\n",
                    R, slots, E, old_group, new_group, old_pair, new_pair);
        const float graph_old = time_graph_us(reps, reference), graph_new = time_graph_us(reps, fused);
        same(count0, count1, "graphed count", R, mode, edges);
        same(ids0, ids1, "graphed ids", R, mode, edges);
        same(mem0, mem1, "graphed members", R, mode, edges);
        same(out00, out10, "graphed gate", R, mode, edges);
        same(out01, out11, "graphed up", R, mode, edges);
        std::printf("PASS R=%d slots=%d E=%d graph_pair_us %.3f -> %.3f nodes 2 -> 1 (warm graph replay)\n",
                    R, slots, E, graph_old, graph_new);
    } else std::printf("PASS R=%d slots=%d E=%d maxm=%d mode=%d edges=%d\n", R, slots, E, maxm, mode, edges);
}

int main(int argc, char** argv) {
    int reps = 100;
    if (argc == 3 && std::strcmp(argv[1], "--reps") == 0) reps = std::max(1, std::atoi(argv[2]));
    else if (argc != 1) { std::fprintf(stderr, "usage: %s [--reps N]\n", argv[0]); return 2; }
    int count = 0; CK(cudaGetDeviceCount(&count));
    if (count == 0) { std::fprintf(stderr, "No CUDA GPU: oracle not run\n"); return 2; }
    for (int R : {17, 20, 24, 32, 48, 64})
        for (int mode = 0; mode < 8; ++mode)
            for (bool edges : {false, true}) check_case(R, mode, edges, reps);
    std::printf("96/96 cases bit-exact, replay and sentinel guards included\n");
    return 0;
}
