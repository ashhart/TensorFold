// Grammar compiles on their own threads: a deadline on each request's wait, a cap on compiles at once, one per input.
#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <process.h>
#else
#include <pthread.h>
#endif

namespace tfc {
constexpr int time_limit_ms = 2000;      // the longest a request waits for its grammar, a free slot included
constexpr size_t active_limit = 4;       // compiles running at once; an abandoned one keeps its slot until it returns
constexpr size_t cache_limit = 8;        // grammars finished in time, served without a slot or a thread
constexpr size_t stack_bytes = 8 << 20;  // the server's request threads' stack, where compiles ran before

inline std::runtime_error TimeError(int milliseconds) {
    return std::runtime_error("grammar compilation exceeds the " + std::to_string(milliseconds) + " ms time limit");
}

inline std::runtime_error SlotError(int milliseconds) {
    return std::runtime_error("grammar compilation exceeds the " + std::to_string(milliseconds) +
                              " ms time limit waiting for one of " + std::to_string(active_limit) + " compile slots");
}

// Compiles keyed by format and exact text: hits under the lock, misses on a detached thread, published under it.
template <class Result> class Cache {
public:
    using Work = std::function<Result(const std::string&)>;

    // `work` runs once per pending input, outside every lock; a caller past `milliseconds` stops waiting for it.
    Result Compile(int kind, std::string text, Work work, int milliseconds = time_limit_ms) {
        const int budget = std::clamp(milliseconds, 1, time_limit_ms);
        const auto deadline = Clock::now() + std::chrono::milliseconds(budget);
        const auto state = state_;
        std::unique_lock<std::mutex> lock(state->mutex);
        std::shared_ptr<Job> job;
        for (;;) {
            for (auto it = state->cache.begin(); it != state->cache.end(); ++it) {
                if (it->kind != kind || it->text != text) continue;
                state->cache.splice(state->cache.begin(), state->cache, it);
                return state->cache.front().result;
            }
            for (const auto& pending : state->jobs) {
                if (!pending->done && pending->kind == kind && pending->text == text) job = pending;
            }
            if (job) break;
            if (state->jobs.size() < active_limit) {
                job = Start(state, kind, std::move(text), deadline, budget, work);
                break;
            }
            const bool late = state->freed.wait_until(lock, deadline) == std::cv_status::timeout;
            if (late || Clock::now() >= deadline) throw SlotError(budget);
        }
        if (!job->changed.wait_until(lock, std::min(deadline, job->deadline), [&] { return job->done; })) {
            job->abandoned = true;
            throw TimeError(std::min(budget, job->milliseconds));
        }
        if (job->abandoned) throw TimeError(std::min(budget, job->milliseconds));
        if (job->failure) std::rethrow_exception(job->failure);
        if (!job->result) throw std::runtime_error("grammar compilation failed");
        return *job->result;
    }

    // Compiles holding a slot, abandoned ones included (tests wait for them to return).
    size_t Active() const {
        std::lock_guard<std::mutex> hold(state_->mutex);
        return state_->jobs.size();
    }

private:
    using Clock = std::chrono::steady_clock;
    struct Job {
        int kind;
        std::string text;
        Clock::time_point deadline;
        int milliseconds;
        std::condition_variable changed;
        bool done = false;
        bool abandoned = false;
        std::optional<Result> result;
        std::exception_ptr failure;
        Job(int k, std::string t, Clock::time_point d, int ms) : kind(k), text(std::move(t)), deadline(d), milliseconds(ms) {}
    };
    struct Entry {
        int kind;
        std::string text;
        Result result;
    };
    struct State {
        std::mutex mutex;
        std::condition_variable freed;  // a slot came free
        std::vector<std::shared_ptr<Job>> jobs;
        std::list<Entry> cache;
    };
    struct Run {
        std::shared_ptr<State> state;
        std::shared_ptr<Job> job;
        Work work;
    };
    std::shared_ptr<State> state_ = std::make_shared<State>();

    // Called holding the lock: the job's slot, its thread started; nothing is left behind when either fails.
    static std::shared_ptr<Job> Start(const std::shared_ptr<State>& state, int kind, std::string text,
                                      Clock::time_point deadline, int budget, Work& work) {
        auto job = std::make_shared<Job>(kind, std::move(text), deadline, budget);
        auto run = std::make_unique<Run>(Run{state, job, nullptr});
        run->work.swap(work);  // swapped, never moved: the caller's copy must not keep the compiler alive
        state->jobs.push_back(job);
        try {
            Detach(run.get());
        } catch (...) {
            state->jobs.pop_back();
            throw;
        }
        run.release();
        return job;
    }

#if defined(_WIN32)
    static void Detach(Run* run) {
        auto start = [](void* p) { Main(p); };
        if (_beginthread(start, (unsigned)stack_bytes, run) == (uintptr_t)-1) {
            throw std::runtime_error("grammar compilation could not start a thread");
        }
    }
#else
    static void Detach(Run* run) {
        pthread_attr_t attr;
        if (pthread_attr_init(&attr) != 0) throw std::runtime_error("grammar compilation could not start a thread");
        pthread_attr_setstacksize(&attr, stack_bytes);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        pthread_t id;
        const int failed = pthread_create(&id, &attr, Main, run);
        pthread_attr_destroy(&attr);
        if (failed) throw std::runtime_error("grammar compilation could not start a thread");
    }
#endif

    // The compile thread: work, then publish (in time) or discard (late), release the slot after cleanup.
    static void* Main(void* p) {
        std::unique_ptr<Run> run(static_cast<Run*>(p));
        const auto& state = run->state;
        Job& job = *run->job;
        std::optional<Result> result;
        std::exception_ptr failure;
        try {
            result.emplace(run->work(job.text));
        } catch (...) {
            failure = std::current_exception();
        }
        std::optional<Entry> evicted;
        {
            std::lock_guard<std::mutex> hold(state->mutex);
            if (!job.abandoned && Clock::now() <= job.deadline) {
                job.failure = failure;
                if (result && !failure) {
                    try {
                        state->cache.push_front({job.kind, job.text, *result});
                        job.result.emplace(*result);
                        if (state->cache.size() > cache_limit) {
                            evicted.emplace(std::move(state->cache.back()));
                            state->cache.pop_back();
                        }
                    } catch (...) {
                        job.failure = std::current_exception();
                    }
                }
            } else {
                job.abandoned = true;
            }
            job.done = true;
            job.changed.notify_all();
        }
        // freed outside the lock, the slot still held: the last reference may hold the whole compiler
        result.reset();
        evicted.reset();
        run->work = nullptr;
        std::lock_guard<std::mutex> hold(state->mutex);
        state->jobs.erase(std::remove(state->jobs.begin(), state->jobs.end(), run->job), state->jobs.end());
        state->freed.notify_all();
        return nullptr;
    }
};
}  // namespace tfc
