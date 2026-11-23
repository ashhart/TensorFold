// Compare the deployed and candidate GLM prompt kernels on a single GB10, one projection per process.
// Build with g++ -O2 -std=c++17 -I/usr/local/cuda/include exl3_prompt_gb10.cpp -lcuda -lcrypto -o bench.
// Run ./bench base.fatbin candidate.fatbin gu and ./bench base.fatbin candidate.fatbin down.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <openssl/sha.h>
#include <random>
#include <vector>

static constexpr int D = 4096, W = 1024, E = 288, S = 9, max_rows = 4096;
static constexpr size_t projection_bytes = size_t(D) * W / 2;
static constexpr uint32_t sentinel = 0x55b755b7;
static void check(CUresult status, int line) {
    if (status == CUDA_SUCCESS) {
        return;
    }
    const char *error = nullptr;
    cuGetErrorString(status, &error);
    fprintf(stderr, "CUDA line %d: %s\n", line, error);
    exit(2);
}
#define CUDA(x) check((x), __LINE__)
static CUdeviceptr make(size_t bytes) {
    CUdeviceptr p;
    CUDA(cuMemAlloc(&p, bytes));
    return p;
}
template <typename T> static CUdeviceptr upload(const std::vector<T> &values) {
    const size_t bytes = values.size() * sizeof(T);
    CUdeviceptr p = make(bytes);
    CUDA(cuMemcpyHtoD(p, values.data(), bytes));
    return p;
}
struct Model {
    CUdeviceptr g{}, u{}, d{}, sg{}, su{}, sd{}, so{}, x{}, xd{}, y{};
    explicit Model(std::mt19937 &rng, bool gu) {
        CUdeviceptr *ptrs[] = {&g, &u, &d};
        std::vector<uint32_t> trellis(projection_bytes / 4);
        for (int proj = gu ? 0 : 2; proj < (gu ? 2 : 3); ++proj) {
            auto table = ptrs[proj];
            CUdeviceptr bank = make(projection_bytes * E);
            std::vector<CUdeviceptr> addresses(E);
            for (int i = 0; i < E; ++i) {
                for (auto &v : trellis) {
                    v = rng();
                }
                addresses[i] = bank + projection_bytes * i;
                CUDA(cuMemcpyHtoD(addresses[i], trellis.data(), projection_bytes));
            }
            *table = upload(addresses);
        }
        auto halves = [&](size_t n, uint16_t exponent) {
            std::vector<uint16_t> data(n);
            for (auto &v : data) {
                const uint32_t b = rng();
                v = uint16_t(exponent | (b & 1023) | ((b >> 16) & 0x8000));
            }
            return upload(data);
        };
        if (gu) {
            sg = halves(E * W, 0x3800);
            su = halves(E * W, 0x3800);
            sd = halves(E * W, 0x3800);
            x = halves(size_t(max_rows) * D, 0x2800);
            xd = make(size_t(max_rows) * S * W * 2);
        } else {
            so = halves(E * D, 0x3800);
            xd = halves(size_t(max_rows) * S * W, 0x2800);
            y = make(size_t(max_rows) * S * D * 4);
        }
    }
};
struct Routes {
    CUdeviceptr items, count, members;
    int blocks;
    Routes(int rows, bool skewed) {
        std::mt19937 rng(25613);
        std::array<std::vector<int>, E + 1> groups;
        std::vector<int> table;
        for (int e = 0; e < E; ++e) {
            int weight = skewed ? (e < 24 ? 8 : (e >= E - 4 ? 0 : 1)) : 1;
            for (int j = 0; j < weight; ++j) {
                table.push_back(e);
            }
        }
        for (int r = 0; r < rows; ++r) {
            std::array<bool, E> used{};
            for (int slot = 0; slot < 8; ++slot) {
                int e;
                do {
                    e = table[rng() % table.size()];
                } while (used[e]);
                used[e] = true;
                groups[e].push_back(r * S + slot);
            }
            groups[E].push_back(r * S + 8);
        }
        std::vector<int> jobs, pairs;
        int empty = 0, partial = 0;
        std::array<int, 8> tail{};
        for (int e = 0; e <= E; ++e) {
            if (groups[e].empty()) {
                ++empty;
            }
            for (size_t at = 0; at < groups[e].size(); at += 64) {
                int n = int(std::min(size_t(64), groups[e].size() - at));
                if (e != E) {
                    ++tail[(n - 1) / 8];
                    if (n < 64) {
                        ++partial;
                    }
                }
                jobs.insert(jobs.end(), {e, int(pairs.size()), n});
                pairs.insert(pairs.end(), groups[e].begin() + at, groups[e].begin() + at + n);
            }
        }
        int actual = int(jobs.size() / 3);
        blocks = (rows * S) / 64 + (E + 1); // same conservative rectangular launch as the resident router
        printf("rows=%d skew=%d items=%d padded=%d empty_experts=%d partial_tiles=%d classes=", rows, skewed,
               actual, blocks - actual, empty, partial);
        for (int n : tail) {
            printf("%d,", n);
        }
        puts("");
        items = upload(jobs);
        members = upload(pairs);
        count = upload(std::vector<int>{actual});
    }
    ~Routes() {
        CUDA(cuMemFree(items));
        CUDA(cuMemFree(count));
        CUDA(cuMemFree(members));
    }
};
struct Functions {
    CUfunction gu, down;
};
static float run(const Functions &f, Model &m, Routes &r, bool gate_up, CUstream stream, CUevent *t) {
    int dims = D, width = W, slots = S, mode = 0, skip = E;
    float limit = 10;
    void *gu[] = {&m.x,  &m.x,  &m.g,  &m.u,   &r.items, &r.count, &r.members, &m.sg, &m.su,
                  &m.sd, &m.xd, &dims, &width, &slots,   &limit,   &mode,      &skip};
    void *dn[] = {&m.xd, &m.d, &r.items, &r.count, &r.members, &m.so, &m.y, &width, &dims, &skip};
    CUDA(cuEventRecord(t[0], stream));
    if (gate_up) {
        CUDA(cuLaunchKernel(f.gu, W / 128, r.blocks, 1, 256, 1, 1, 0, stream, gu, nullptr));
    } else {
        CUDA(cuLaunchKernel(f.down, D / 256, r.blocks, 1, 256, 1, 1, 0, stream, dn, nullptr));
    }
    CUDA(cuEventRecord(t[1], stream));
    CUDA(cuEventSynchronize(t[1]));
    float ms;
    CUDA(cuEventElapsedTime(&ms, t[0], t[1]));
    return ms;
}
using Digest = std::array<unsigned char, SHA256_DIGEST_LENGTH>;
static std::vector<Digest> inspect(CUdeviceptr a, int rows, int words_per_slot, bool floats) {
    const size_t total = size_t(rows) * S * words_per_slot, batch = 1 << 20;
    std::vector<uint32_t> data(batch);
    std::vector<Digest> digests;
    size_t nonzero = 0, untouched = 0;
    for (size_t at = 0; at < total; at += batch) {
        size_t n = std::min(batch, total - at);
        CUDA(cuMemcpyDtoH(data.data(), a + at * 4, n * 4));
        Digest digest;
        SHA256(reinterpret_cast<unsigned char *>(data.data()), n * 4, digest.data());
        digests.push_back(digest);
        for (size_t i = 0; i < n; ++i) {
            uint32_t x = data[i];
            bool skip = ((at + i) / words_per_slot) % S == 8;
            if ((x == sentinel) != skip) {
                fprintf(
                    stderr,
                    "incorrect shared slot or blank output word %zu value=%08x expected_skip=%d first nine slots:",
                    at + i, x, skip);
                for (int s = 0; s < S; ++s) {
                    fprintf(stderr, " %08x", data[s * words_per_slot]);
                }
                fprintf(stderr, "\n");
                exit(3);
            }
            if (skip) {
                ++untouched;
                continue;
            }
            if (floats && (x & 0x7f800000) == 0x7f800000) {
                fprintf(stderr, "nonfinite output at word %zu\n", at + i);
                exit(3);
            }
            if (x) {
                ++nonzero;
            }
        }
    }
    if (!nonzero) {
        exit(3);
    }
    printf("raw_bytes_hashed=%zu digests=%zu protected_bytes=%zu nonzero_routed_words=%zu\n", total * 4,
           digests.size(), untouched * 4, nonzero);
    return digests;
}
int main(int argc, char **argv) {
    if (argc != 4 || (strcmp(argv[3], "gu") && strcmp(argv[3], "down"))) {
        fprintf(stderr, "Usage: %s BASE_FATBIN CANDIDATE_FATBIN {gu|down}\n", argv[0]);
        return 1;
    }
    bool gate_up = !strcmp(argv[3], "gu");
    setvbuf(stdout, nullptr, _IOLBF, 0);
    CUDA(cuInit(0));
    CUdevice device;
    CUcontext ctx;
    CUDA(cuDeviceGet(&device, 0));
    CUDA(cuDevicePrimaryCtxRetain(&ctx, device));
    CUDA(cuCtxSetCurrent(ctx));
    // Order the uploads, output sentinel fill, launches and reads on the default CUDA stream.
    CUstream stream = nullptr;
    CUevent t[2];
    for (auto &e : t) {
        CUDA(cuEventCreate(&e, 0));
    }
    Functions f[2];
    CUmodule modules[2];
    for (int j = 0; j < 2; ++j) {
        CUDA(cuModuleLoad(&modules[j], argv[j + 1]));
        CUDA(cuModuleGetFunction(&f[j].gu, modules[j], "tf_exl3_prompt_gu"));
        CUDA(cuModuleGetFunction(&f[j].down, modules[j], "tf_exl3_prompt_down"));
    }
    size_t free_before, free_after, total;
    CUDA(cuMemGetInfo(&free_before, &total));
    std::mt19937 rng(19091);
    Model model(rng, gate_up);
    CUDA(cuMemGetInfo(&free_after, &total));
    printf("projection=%s observed_free_delta_MiB=%.1f\n", argv[3],
           (double(free_before) - double(free_after)) / (1 << 20));
    for (int rows : {2048, 4096}) {
        for (bool skew : {false, true}) {
            Routes r(rows, skew);
            std::vector<Digest> reference;
            int words = gate_up ? W / 2 : D;
            CUdeviceptr output = gate_up ? model.xd : model.y;
            for (int j = 0; j < 2; ++j) {
                CUDA(cuMemsetD32(output, sentinel, size_t(rows) * S * words));
                run(f[j], model, r, gate_up, stream, t);
                auto hashes = inspect(output, rows, words, !gate_up);
                if (!j) {
                    reference = hashes;
                } else if (reference != hashes) {
                    fprintf(stderr, "raw output SHA256 mismatch\n");
                    return 3;
                }
            }
            std::vector<float> times[2];
            for (int pair = 0; pair < 5; ++pair) {
                for (int k = 0; k < 2; ++k) {
                    int j = (pair + k) % 2;
                    float ms = run(f[j], model, r, gate_up, stream, t);
                    times[j].push_back(ms);
                    printf("pair=%d candidate=%d ms=%.3f\n", pair, j, ms);
                }
            }
            for (int j = 0; j < 2; ++j) {
                std::sort(times[j].begin(), times[j].end());
                printf("rows=%d skew=%d projection=%s candidate=%d median_ms=%.3f\n", rows, skew, argv[3], j,
                       times[j][2]);
            }
        }
    }
}
