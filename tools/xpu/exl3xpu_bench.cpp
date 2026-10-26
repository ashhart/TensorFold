// SPDX-License-Identifier: Apache-2.0
// This host sequence is adapted from exl3xpu by 0xSero and is MIT licensed.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Run exl3xpu's M <= 8 decode path (HadIn, GemvKernel, HadOut from csrc/exl3_esimd.h, 0xSero, MIT) on our data.
// Host logic copies exl3_gemm_small (exl3_ops.sycl @15ded2f3): MR, NT = nt_of, split-K P for 1024 threads, local 8.
// Usage: exl3xpu_bench <dir> <K> <N> <bits> <groups> <M> <reps>
//   dir holds tr.bin (int16, groups x trellis), suh.bin, svh.bin (fp16, groups x K / N), x.bin (bf16, M x K).
//   Writes out.bin (fp32, groups x M x N). Prints the median, over reps, of the summed kernel time of all groups.
#include "exl3_esimd.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <vector>
using namespace exl3;
using bf16 = sycl::ext::oneapi::bfloat16;

static std::vector<char> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}
static int round_up(int a, int b) { return (a + b - 1) / b * b; }
constexpr int nt_of(int K, int MR) { return MR == 8 ? 2 : (K <= 4 ? (MR == 1 ? 8 : 4) : 4); }

template <int KB, int MR>
static double run(sycl::queue& q, const std::string& dir, int Kd, int N, int G, int M, int reps) {
    constexpr int NT = nt_of(KB, MR);
    auto tr_h = slurp(dir + "/tr.bin"), suh_h = slurp(dir + "/suh.bin"), svh_h = slurp(dir + "/svh.bin"), x_h = slurp(dir + "/x.bin");
    size_t trb = (size_t)Kd / 16 * (N / 16) * 16 * KB * 2;
    auto* tr = sycl::malloc_device<char>(tr_h.size(), q);
    auto* suh = sycl::malloc_device<fp16>(suh_h.size() / 2, q);
    auto* svh = sycl::malloc_device<fp16>(svh_h.size() / 2, q);
    auto* x = sycl::malloc_device<bf16>(x_h.size() / 2, q);
    q.memcpy(tr, tr_h.data(), tr_h.size()); q.memcpy(suh, suh_h.data(), suh_h.size());
    q.memcpy(svh, svh_h.data(), svh_h.size()); q.memcpy(x, x_h.data(), x_h.size()).wait();
    int tk = Kd / 16, n_strips = (N / 16) / NT;
    int P = std::max(1, std::min(tk, (1024 + n_strips - 1) / n_strips));
    int rps = (tk + P - 1) / P;
    P = (tk + rps - 1) / rps;
    auto* xh = sycl::malloc_device<fp16>((size_t)G * M * Kd, q);
    auto* part = sycl::malloc_device<float>((size_t)G * P * M * N, q);
    auto* out = sycl::malloc_device<float>((size_t)G * M * N, q);
    auto* shard = sycl::malloc_device<int>(N / 128, q);
    q.memset(shard, 0, N / 128 * 4).wait();
    std::vector<double> times;
    for (int it = 0; it <= reps; ++it) {
        std::vector<sycl::event> ev;
        for (int g = 0; g < G; ++g) {
            fp16* xg = xh + (size_t)g * M * Kd;
            float* pg = part + (size_t)g * P * M * N;
            HadInKernel<bf16> hi{x, suh + (size_t)g * Kd, xg, M, Kd, 1, Kd, M, 0};
            int n1 = M * (Kd / 128);
            ev.push_back(q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n1, 8), 8), hi); }));
            GemvKernel<KB, 2, MR, NT> gk{xg, reinterpret_cast<const uint32_t*>(tr + g * trb), shard, pg, M, Kd, N, N / 16, rps, n_strips};
            int n2 = n_strips * P;
            ev.push_back(q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n2, 8), 8), gk); }));
            HadOutKernel<float> ho{pg, svh + (size_t)g * N, out + (size_t)g * M * N, M, N, P, N};
            int n3 = M * (N / 128);
            ev.push_back(q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n3, 8), 8), ho); }));
        }
        q.wait();
        double t = 0;
        for (auto& e : ev) t += (e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                 e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-9;
        if (it) times.push_back(t);
    }
    std::vector<float> o((size_t)G * M * N);
    q.memcpy(o.data(), out, o.size() * 4).wait();
    std::ofstream(dir + "/out.bin", std::ios::binary).write((char*)o.data(), o.size() * 4);
    std::sort(times.begin(), times.end());
    std::printf("P=%d NT=%d MR=%d\n", P, NT, MR);
    return times[times.size() / 2];
}

int main(int argc, char** argv) {
    std::string dir = argv[1];
    int Kd = atoi(argv[2]), N = atoi(argv[3]), bits = atoi(argv[4]), G = atoi(argv[5]), M = atoi(argv[6]), reps = atoi(argv[7]);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    int MR = M <= 1 ? 1 : (M <= 2 ? 2 : (M <= 4 ? 4 : 8));
    double t = -1;
#define C(KB, R) if (bits == KB && MR == R) t = run<KB, R>(q, dir, Kd, N, G, M, reps);
    C(4, 1) C(4, 8) C(6, 1) C(6, 8)
    std::printf("exl3xpu K=%d N=%d bits=%d groups=%d M=%d: %.1f us\n", Kd, N, bits, G, M, t * 1e6);
}
