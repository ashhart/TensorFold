// Many small positioned reads at once (Engram table rows): pread from a persistent thread pool, the GIL released.
//
// The pool's threads live across calls: spawning threads per call cost ~20x the reads themselves when the rows are in
// the page cache (12 rows: 342 us with 16 new threads vs 20 us on one). A call wakes up to ``threads - 1`` helpers,
// works itself, and returns once every read is done. Idle helpers spin briefly (a decode step's next call comes
// within milliseconds), then sleep.
#include <torch/extension.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct Job {
    const int32_t* fd;
    const int64_t* off;
    const int32_t* sz;
    const int64_t* dst;
    uint8_t* base;
    int64_t n;
    std::atomic<int64_t> next{0}, done{0}, failed{0};
};

void run(Job& j) {
    for (int64_t i = j.next++; i < j.n; i = j.next++) {
        int64_t got_all = 0;
        while (got_all < j.sz[i]) {
            ssize_t got = pread(j.fd[i], j.base + j.dst[i] + got_all, j.sz[i] - got_all, j.off[i] + got_all);
            if (got <= 0) { j.failed++; break; }
            got_all += got;
        }
        j.done++;
    }
}

class Pool {
   public:
    static Pool& get() {
        static Pool* pool = new Pool;                        // never destroyed: helpers sleep on it through exit
        return *pool;
    }

    // every read of ``job`` done when this returns; ``helpers`` pool threads join the caller
    void execute(Job& job, int helpers) {
        std::lock_guard<std::mutex> serial(call_);           // one job at a time (the decode loop is one caller)
        grow(helpers);
        {
            std::lock_guard<std::mutex> lk(mu_);
            job_ = &job;
            active_ = helpers;
            gen_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all();
        run(job);
        while (job.done.load(std::memory_order_acquire) < job.n) std::this_thread::yield();
        // withdraw the job first (helpers take it, and count themselves busy, under mu_), then wait for the helpers
        // that took it to leave run() before it goes out of scope: they hold no work left (next >= n)
        {
            std::lock_guard<std::mutex> lk(mu_);
            job_ = nullptr;
        }
        while (busy_.load(std::memory_order_acquire) > 0) std::this_thread::yield();
    }

   private:
    void grow(int want) {
        while ((int)threads_.size() < want) {
            const int id = (int)threads_.size();
            threads_.emplace_back([this, id] { loop(id); });
            threads_.back().detach();
        }
    }

    void loop(int id) {
        uint64_t seen = gen_.load(std::memory_order_acquire);
        for (;;) {
            // spin ~200 us for the next call, then sleep until notified
            const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
            while (gen_.load(std::memory_order_acquire) == seen && std::chrono::steady_clock::now() < until)
                std::this_thread::yield();
            if (gen_.load(std::memory_order_acquire) == seen) {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return gen_.load(std::memory_order_acquire) != seen; });
            }
            Job* job;
            int active;
            {
                std::lock_guard<std::mutex> lk(mu_);
                seen = gen_.load(std::memory_order_acquire);
                job = job_;
                active = active_;
                if (job && id < active) busy_.fetch_add(1, std::memory_order_acq_rel);
            }
            if (job && id < active) {
                run(*job);
                busy_.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }

    std::mutex call_, mu_;
    std::condition_variable cv_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> busy_{0};
    Job* job_ = nullptr;
    int active_ = 0;
    std::vector<std::thread> threads_;
};

}  // namespace

// fds int32 [n], offsets int64 [n], sizes int32 [n], dest int64 [n] (byte offsets into out), out uint8 (any size)
static int64_t read_many(torch::Tensor fds, torch::Tensor offsets, torch::Tensor sizes, torch::Tensor dest,
                         torch::Tensor out, int64_t threads) {
    TORCH_CHECK(!out.is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kUInt8, "out: CPU uint8");
    Job job;
    job.fd = fds.data_ptr<int32_t>();
    job.off = offsets.data_ptr<int64_t>();
    job.sz = sizes.data_ptr<int32_t>();
    job.dst = dest.data_ptr<int64_t>();
    job.base = out.data_ptr<uint8_t>();
    job.n = offsets.numel();
    {
        pybind11::gil_scoped_release release;
        const int64_t t = std::max<int64_t>(1, std::min<int64_t>(threads, job.n));
        if (t == 1) {
            run(job);
        } else {
            Pool::get().execute(job, (int)(t - 1));
        }
    }
    return job.failed.load();
}

// flag int64 [1] in pinned memory <- value, after everything this thread has seen (the rows read above, joined):
// the decode graph's wait kernel polls the flag, then reads the rows from the same pinned buffer
static void publish(torch::Tensor flag, int64_t value) {
    TORCH_CHECK(!flag.is_cuda() && flag.scalar_type() == torch::kInt64, "flag: CPU int64");
    std::atomic_thread_fence(std::memory_order_seq_cst);
    __atomic_store_n(flag.data_ptr<int64_t>(), value, __ATOMIC_RELEASE);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("read_many", &read_many);
    m.def("publish", &publish);
}
