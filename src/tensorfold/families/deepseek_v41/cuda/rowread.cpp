// Many small positioned reads at once (Engram table rows): pread from a thread pool, the GIL released.
#include <torch/extension.h>
#include <unistd.h>

#include <atomic>
#include <thread>
#include <vector>

// fds int32 [n], offsets int64 [n], sizes int32 [n], dest int64 [n] (byte offsets into out), out uint8 (any size)
static int64_t read_many(torch::Tensor fds, torch::Tensor offsets, torch::Tensor sizes, torch::Tensor dest,
                         torch::Tensor out, int64_t threads) {
    TORCH_CHECK(!out.is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kUInt8, "out: CPU uint8");
    const int64_t n = offsets.numel();
    const int32_t* fd = fds.data_ptr<int32_t>();
    const int64_t* off = offsets.data_ptr<int64_t>();
    const int32_t* sz = sizes.data_ptr<int32_t>();
    const int64_t* dst = dest.data_ptr<int64_t>();
    uint8_t* base = out.data_ptr<uint8_t>();
    std::atomic<int64_t> next{0}, failed{0};
    auto work = [&]() {
        for (int64_t i = next++; i < n; i = next++) {
            int64_t done = 0;
            while (done < sz[i]) {
                ssize_t got = pread(fd[i], base + dst[i] + done, sz[i] - done, off[i] + done);
                if (got <= 0) { failed++; break; }
                done += got;
            }
        }
    };
    {
        pybind11::gil_scoped_release release;
        const int64_t t = std::max<int64_t>(1, std::min<int64_t>(threads, n));
        std::vector<std::thread> pool;
        for (int64_t k = 1; k < t; ++k) pool.emplace_back(work);
        work();
        for (auto& th : pool) th.join();
    }
    return failed.load();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("read_many", &read_many); }
