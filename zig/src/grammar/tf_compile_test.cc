// tfc::Cache, the compile runner tf_grammar.cc uses, with fake compiles in place of xgrammar's.
#include "tf_compile.h"

#include <atomic>
#include <cstdio>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;
using Cache = tfc::Cache<std::string>;

void require(bool value, const char* why) {
    if (!value) throw std::runtime_error(why);
}

// A fake compile blocks here until Release (or 5 s, so a broken test cannot hang the run).
struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool released = false;
    std::atomic<int> entered{0};
    std::atomic<int> exited{0};
    void Wait() {
        entered++;
        std::unique_lock<std::mutex> hold(mutex);
        changed.wait_for(hold, std::chrono::seconds(5), [&] { return released; });
        exited++;
    }
    void Release() {
        std::lock_guard<std::mutex> hold(mutex);
        released = true;
        changed.notify_all();
    }
};

// Releases its gate on every way out of a test.
struct Release {
    std::shared_ptr<Gate> gate;
    ~Release() { gate->Release(); }
};

template <class Ready> void until(Ready ready) {
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    while (!ready()) {
        require(Clock::now() < deadline, "a fake compile did not reach its gate");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

template <class Work> void refuses(Work work, const char* part) {
    try {
        work();
    } catch (const std::runtime_error& e) {
        require(std::string(e.what()).find(part) != std::string::npos, e.what());
        return;
    }
    throw std::runtime_error("the compile should have been refused");
}

std::string Bytes(const std::string& text) { return "grammar of " + text; }

void cached() {
    Cache cache;
    std::atomic<int> calls{0};
    auto work = [&](const std::string& t) { calls++; return Bytes(t); };
    const auto first = cache.Compile(1, "schema", work);
    const auto hit = cache.Compile(1, "schema", [](const std::string&) { return std::string("changed"); });
    const auto other = cache.Compile(2, "schema", work);
    require(first == "grammar of schema" && hit == first && other == first && calls == 2, "cache bytes or key changed");
    until([&] { return cache.Active() == 0; });
}

void deadline() {
    Cache cache;
    auto gate = std::make_shared<Gate>();
    Release release{gate};
    auto calls = std::make_shared<std::atomic<int>>(0);
    auto slow = [gate, calls](const std::string&) { (*calls)++; gate->Wait(); return std::string("late"); };
    const auto start = Clock::now();
    refuses([&] { cache.Compile(1, "slow", slow, 40); }, "40 ms time limit");
    require(Clock::now() - start < std::chrono::seconds(1), "the caller waited for abandoned work");
    refuses([&] { cache.Compile(1, "slow", slow, 40); }, "time limit");
    require(*calls == 1 && cache.Active() == 1, "an abandoned input compiled twice or lost its slot");
    gate->Release();
    until([&] { return cache.Active() == 0; });
    require(cache.Compile(1, "slow", Bytes) == "grammar of slow", "a late result entered the cache");
}

void independent() {
    Cache cache;
    auto gate = std::make_shared<Gate>();
    Release release{gate};
    std::thread big([&] {
        try {
            cache.Compile(1, "large", [gate](const std::string&) { gate->Wait(); return std::string("large"); });
        } catch (...) {
        }
    });
    std::string small;
    Clock::duration elapsed{};
    try {
        until([&] { return gate->entered.load() == 1; });
        const auto start = Clock::now();
        small = cache.Compile(1, "small", Bytes);
        elapsed = Clock::now() - start;
    } catch (...) {
        gate->Release();
        big.join();
        throw;
    }
    gate->Release();
    big.join();
    require(small == "grammar of small" && elapsed < std::chrono::seconds(1), "a small grammar waited for a large one");
    until([&] { return cache.Active() == 0; });
}

void shared() {
    Cache cache;
    auto gate = std::make_shared<Gate>();
    Release release{gate};
    auto calls = std::make_shared<std::atomic<int>>(0);
    auto work = [gate, calls](const std::string& t) { (*calls)++; gate->Wait(); return Bytes(t); };
    std::string got[2];
    std::thread first([&] { got[0] = cache.Compile(1, "same", work); });
    try {
        until([&] { return gate->entered.load() == 1; });
    } catch (...) {
        gate->Release();
        first.join();
        throw;
    }
    std::thread second([&] { got[1] = cache.Compile(1, "same", work); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    gate->Release();
    first.join();
    second.join();
    require(*calls == 1 && got[0] == "grammar of same" && got[1] == got[0], "two callers compiled one input twice");
    until([&] { return cache.Active() == 0; });
}

void capacity() {
    Cache cache;
    cache.Compile(0, "warm", Bytes);
    auto gate = std::make_shared<Gate>();
    Release release{gate};
    auto slow = [gate](const std::string&) { gate->Wait(); return std::string("late"); };
    for (size_t i = 0; i < tfc::active_limit; i++) {
        refuses([&] { cache.Compile(1, std::to_string(i), slow, 20); }, "time limit");
    }
    const auto start = Clock::now();
    refuses([&] { cache.Compile(1, "fifth", Bytes, 60); }, "60 ms time limit waiting for one of 4 compile slots");
    require(Clock::now() - start >= std::chrono::milliseconds(50), "a full runner refused without waiting");
    require(cache.Compile(0, "warm", Bytes) == "grammar of warm", "busy compiles blocked a cached grammar");
    std::thread opener([gate] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        gate->Release();
    });
    std::string waited;
    try {
        waited = cache.Compile(1, "sixth", Bytes);
    } catch (...) {
        opener.join();
        throw;
    }
    opener.join();
    require(waited == "grammar of sixth", "a caller with time left did not get the slot that came free");
    until([&] { return cache.Active() == 0; });
}

void ownership() {
    struct Owner {
        std::shared_ptr<std::atomic<bool>> destroyed;
        ~Owner() { *destroyed = true; }
    };
    auto destroyed = std::make_shared<std::atomic<bool>>(false);
    auto owner = std::make_shared<Owner>();
    owner->destroyed = destroyed;
    auto cache = std::make_unique<Cache>();
    auto gate = std::make_shared<Gate>();
    Release release{gate};
    refuses([&] {
        cache->Compile(1, "owned", [hold = owner, gate](const std::string&) { gate->Wait(); return std::string("late"); }, 20);
    }, "time limit");
    owner.reset();
    cache.reset();
    require(!*destroyed, "an abandoned compile lost the state it owns");
    gate->Release();
    until([&] { return destroyed->load(); });
}

void errors() {
    Cache cache;
    auto gate = std::make_shared<Gate>();
    Release release{gate};
    struct Cleanup {
        std::shared_ptr<Gate> gate;
        ~Cleanup() { gate->Wait(); }
    };
    auto cleanup = std::make_shared<Cleanup>();
    cleanup->gate = gate;
    const auto start = Clock::now();
    auto bad = [hold = std::move(cleanup)](const std::string&) -> std::string { throw std::runtime_error("bad syntax"); };
    refuses([&] { cache.Compile(1, "bad", std::move(bad)); }, "bad syntax");
    require(Clock::now() - start < std::chrono::seconds(1), "the caller waited for a failed compile's cleanup");
    until([&] { return gate->entered.load() == 1; });
    require(cache.Compile(1, "bad", Bytes) == "grammar of bad", "a failure was cached or kept its input");
    gate->Release();
    until([&] { return cache.Active() == 0; });
}

// Touches about 4 MiB of stack, a page at a time, far past a default 512 KiB thread stack.
int Deep(int n) {
    volatile char frame[64 << 10];
    for (size_t i = 0; i < sizeof frame; i += 1024) frame[i] = (char)n;
    return n == 0 ? frame[0] : Deep(n - 1) + frame[sizeof frame - 1024];
}

void stack() {
    Cache cache;
    require(cache.Compile(1, "deep", [](const std::string&) { return std::to_string(Deep(64)); }) == "2080",
            "a compile thread's stack is smaller than the request threads'");
    until([&] { return cache.Active() == 0; });
}
}  // namespace

extern "C" int tf_compile_host_test(int which) {
    try {
        switch (which) {
            case 0: cached(); break;
            case 1: deadline(); break;
            case 2: independent(); break;
            case 3: shared(); break;
            case 4: capacity(); break;
            case 5: ownership(); break;
            case 6: errors(); break;
            case 7: stack(); break;
            default: return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "grammar compile test %d: %s\n", which, e.what());
        return 1;
    }
}
