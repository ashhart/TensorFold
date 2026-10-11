// mma.sync issue rates (FLOP/SM/clock) on sm_120/sm_121, 8 accumulator chains a warp; usage: mma_rate [warps_per_sm]
#include "gb10emu.cuh"
#include <cuda_fp16.h>

namespace {

constexpr int ITERS = 4096, CH = 8;

template <int V>
__global__ void __launch_bounds__(512) mma_loop(float* out, uint32_t seed) {
    uint32_t a[4], b[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = 0x3c003c00u ^ (seed * (threadIdx.x + i));
    b[0] = a[0] ^ 1u; b[1] = a[1] ^ 3u;
    float d[CH][4] = {};
    uint32_t h[CH][2] = {};
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            if constexpr (V == 0)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            else if constexpr (V == 1)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            else if constexpr (V == 2)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                             : "+r"(h[c][0]), "+r"(h[c][1])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            else if constexpr (V == 3)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            else
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        }
    }
    float s = 0;
#pragma unroll
    for (int c = 0; c < CH; ++c) s += d[c][0] + d[c][3] + __half2float(__ushort_as_half((unsigned short)h[c][0]));
    if (s == 1234.5f) out[threadIdx.x] = s;
}

template <int V>
void run(const char* name, gb10::Emu& emu, int warps, double flop_per_mma, float* out) {
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    const int grid = emu.grid_sms();
    mma_loop<V><<<grid, warps * 32, 0, st>>>(out, 7);       // warm
    emu.begin();
    CK(cudaEventRecord(a, st));
    mma_loop<V><<<grid, warps * 32, 0, st>>>(out, 7);
    CK(cudaEventRecord(b, st)); CK(cudaEventSynchronize(b));
    emu.end();
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    int clk; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
    const double flop = (double)grid * warps * ITERS * CH * flop_per_mma;
    const double per = flop / (ms * 1e-3) / grid / (clk * 1e3);
    printf("RATE %-28s %2d warps/SM: %8.1f TFLOP/s on %d SMs, %6.0f FLOP/SM/clk (at %d MHz) -> GB10 48 SMs @2.5 GHz %.1f TFLOP/s\n",
           name, warps, flop / (ms * 1e9), grid, per, clk / 1000, per * 48 * 2.5e9 / 1e12);
}

}  // namespace

int main(int argc, char** argv) {
    gb10::eager();
    const int warps = argc > 1 ? atoi(argv[1]) : 8;
    gb10::Emu emu; emu.init(gb10::Mode::Sms);
    float* out; CK(cudaMalloc(&out, 4096));
    run<0>("f16 x f16 -> f32 k16", emu, warps, 2.0 * 16 * 8 * 16, out);
    run<1>("bf16 x bf16 -> f32 k16", emu, warps, 2.0 * 16 * 8 * 16, out);
    run<2>("f16 x f16 -> f16 k16", emu, warps, 2.0 * 16 * 8 * 16, out);
    run<3>("e4m3 kind::f8f6f4 -> f32 k32", emu, warps, 2.0 * 16 * 8 * 32, out);
    run<4>("e4m3 -> f32 k32", emu, warps, 2.0 * 16 * 8 * 32, out);
    return 0;
}
