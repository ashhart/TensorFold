#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

#define __device__
#define __global__
#define __shared__
#define __forceinline__ inline
#define __launch_bounds__(value)
struct TestDim { int x; };
static TestDim threadIdx{0}, blockIdx{0}, blockDim{1}, gridDim{1};
struct uint4 { unsigned x, y, z, w; };
using cudaGraphConditionalHandle = unsigned long long;
static void __syncthreads() {}
static int __ffs(unsigned value) { return __builtin_ffs(value); }
static int __float_as_int(float value) {
    int bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
static void cudaGraphSetConditional(cudaGraphConditionalHandle, unsigned) {}

#include "../../../../../kernels/cuda/flashnext_exl3/flashnext_exl3_dplan.cu"

static constexpr unsigned pool_words = 1024 * 1024 / 4;
static constexpr unsigned sentinel = 0xfeedface;

// The planner's line geometry: three segments a line at these offsets, lines from test_line_base, and the shared scratch
// (one line-sized slot for each staged expert of a round) from test_scratch_base.
static constexpr unsigned long long test_line_base = 1024, test_line_bytes = 1024, test_scratch_base = 1ull << 20;
static constexpr unsigned long long test_offsets[3] = {0, 256, 512};

static std::vector<unsigned> make_trace(unsigned lines, unsigned rows) {
    std::vector<unsigned> trace(pool_words + 1, 0);
    const unsigned seed_words = 2 * lines + 3 * DP_EXPERTS + DP_RECENT_WORDS + 10;
    const unsigned stride = 15 + rows * 11;
    auto* header = reinterpret_cast<DpTraceHeader*>(trace.data());
    *header = {0x44505431, 5, 0, 0, lines, rows, seed_words, stride,
               (pool_words - 16 - 2 * seed_words) / stride, 0, 0, 0, 0, 0, 0, 0};
    trace[pool_words] = sentinel;
    return trace;
}

static std::vector<unsigned> make_state(unsigned lines) {
    std::vector<unsigned> state(2 * lines + 3 * DP_EXPERTS + DP_RECENT_WORDS + 10, 0);
    for (unsigned line = 0; line < lines; ++line) state[line] = 0xffffffffu;
    for (unsigned expert = 0; expert < DP_EXPERTS; ++expert) state[2 * lines + expert] = 0xffffffffu;
    return state;
}

struct PlanOutput {
    long long tabs[3][513]{};
    int widths[3][513]{};
    long long work[512 * 9]{};
    int work_n = 0, staged[512]{}, staged_n = 0;
};

static void run_plan(std::vector<unsigned>& state, unsigned lines, const std::vector<int>& picks,
                     unsigned* trace, int recent, PlanOutput& output) {
    long long src[512 * 3], lens[512 * 3];
    int widths[512 * 3];
    for (int index = 0; index < 512 * 3; ++index) {
        src[index] = 0x100000 + index * 256;
        lens[index] = 256;
        widths[index] = 3;
    }
    fn_exl3_dplan(picks.data(), static_cast<int>(picks.size()), state.data(), lines, recent,
        output.tabs[0], output.tabs[1], output.tabs[2],
        output.widths[0], output.widths[1], output.widths[2],
        src, lens, widths, test_line_base, test_line_bytes, test_offsets[0], test_offsets[1], test_offsets[2], test_scratch_base,
        output.work, &output.work_n, nullptr, nullptr, output.staged, &output.staged_n, trace);
}

static void assert_same(const PlanOutput& first, const PlanOutput& second) {
    assert(std::memcmp(first.tabs, second.tabs, sizeof(first.tabs)) == 0);
    assert(std::memcmp(first.widths, second.widths, sizeof(first.widths)) == 0);
    assert(std::memcmp(first.work, second.work, sizeof(first.work)) == 0);
    assert(std::memcmp(first.staged, second.staged, sizeof(first.staged)) == 0);
    assert(first.work_n == second.work_n && first.staged_n == second.staged_n);
}

static void test_bounds_and_flags() {
    auto trace = make_trace(2, 47);
    auto* header = reinterpret_cast<DpTraceHeader*>(trace.data());
    auto state = make_state(2);
    std::vector<int> picks(47 * 11, 512);
    for (int index = 0; index < 512; ++index) picks[index] = 511 - index;
    picks[512] = -1;
    picks[513] = 511;
    PlanOutput output;
    const auto seed = state;
    run_plan(state, 2, picks, trace.data(), 1, output);
    const unsigned offset = 16 + 2 * header->seed_words;
    const unsigned* round = trace.data() + offset;
    assert(round[0] == 47 && round[1] == 513 && round[2] == 512 && round[3] == 1);
    assert(round[10] == 1 && round[8] == 0 && round[14] == 0);
    assert(output.work_n == 0 && output.staged_n == 0);
    assert(std::memcmp(trace.data() + 16, seed.data(), seed.size() * 4) == 0);
    assert(std::memcmp(round + 15, picks.data(), picks.size() * 4) == 0);
    assert(std::memcmp(trace.data() + 16 + seed.size(), state.data(), state.size() * 4) == 0);
    const auto first = std::vector<unsigned>(round, round + header->record_words);
    const unsigned capacity = header->capacity;
    for (unsigned index = 1; index < capacity + 3; ++index)
        run_plan(state, 2, picks, trace.data(), 1, output);
    assert(header->recorded == capacity && header->dropped_lo == 3 && header->dropped_hi == 0);
    assert(std::memcmp(trace.data() + offset, first.data(), first.size() * 4) == 0);
    assert(trace[pool_words] == sentinel);
    const auto terminal = std::vector<unsigned>(trace.data() + 16 + seed.size(), trace.data() + offset);
    header->dropped_lo = 0xffffffffu;
    header->dropped_hi = 4;
    run_plan(state, 2, picks, trace.data(), 0, output);
    assert(header->dropped_lo == 0 && header->dropped_hi == 5);
    header->dropped_lo = header->dropped_hi = 0xffffffffu;
    run_plan(state, 2, picks, trace.data(), 0, output);
    assert(header->dropped_lo == 0xffffffffu && header->dropped_hi == 0xffffffffu);
    assert(std::memcmp(trace.data() + 16 + seed.size(), terminal.data(), terminal.size() * 4) == 0);
    assert(dp_trace_begin(nullptr, 0, nullptr, nullptr, 0, 2) == nullptr);
    auto invalid = make_trace(2, 1);
    auto* invalid_header = reinterpret_cast<DpTraceHeader*>(invalid.data());
    assert(dp_trace_begin(picks.data(), 12, invalid.data(), state.data(), 0, 2) == nullptr);
    assert(invalid_header->invalid_launch == 1 && invalid_header->recorded == 0);
    auto broken = make_trace(2, 47);
    run_plan(state, 2, picks, broken.data(), 0, output);
    state.back() ^= 1;
    run_plan(state, 2, picks, broken.data(), 0, output);
    assert(reinterpret_cast<DpTraceHeader*>(broken.data())->discontinuity == 1);

    for (unsigned field : {0u, 1u, 4u, 5u, 6u, 7u, 8u, 9u, 12u, 13u, 14u}) {
        auto corrupt = make_trace(2, 1);
        corrupt[field] = 0xffffffffu;
        assert(dp_trace_begin(nullptr, 11, corrupt.data(), nullptr, 0, 2) == nullptr);
        assert(corrupt[13] == 1 && corrupt[pool_words] == sentinel);
        corrupt[field] = 0xffffffffu;
        fn_exl3_dcheckpoint(nullptr, 2, corrupt.data(), 1);
        assert(corrupt[pool_words] == sentinel);
    }
    auto full = make_trace(2, 16);
    auto* full_header = reinterpret_cast<DpTraceHeader*>(full.data());
    const unsigned checkpoint_words = 4 + 2 * full_header->seed_words;
    const unsigned checkpoint_capacity = (pool_words - 16 - 2 * full_header->seed_words) / checkpoint_words;
    for (unsigned index = 0; index < checkpoint_capacity + 1; ++index)
        fn_exl3_dcheckpoint(state.data(), 2, full.data(), 1);
    assert(full_header->checkpoints == checkpoint_capacity && full_header->dropped_checkpoints == 1);
    full_header->dropped_checkpoints = 0xffffffffu;
    fn_exl3_dcheckpoint(state.data(), 2, full.data(), 1);
    assert(full_header->dropped_checkpoints == 0xffffffffu && full[pool_words] == sentinel);
    run_plan(state, 2, std::vector<int>(11, -1), full.data(), 0, output);
    assert(full_header->recorded == 0 && full_header->dropped_lo == 1);

    auto sealed = make_trace(2, 16);
    fn_exl3_dcheckpoint(state.data(), 2, sealed.data(), 1);
    state.back() ^= 1;
    fn_exl3_dcheckpoint(state.data(), 2, sealed.data(), 0);
    assert(reinterpret_cast<DpTraceHeader*>(sealed.data())->discontinuity == 1);
}

static std::vector<unsigned> test_checkpoint_capture() {
    const unsigned lines = 10;
    auto trace = make_trace(lines, 16);
    auto* header = reinterpret_cast<DpTraceHeader*>(trace.data());
    auto state = make_state(lines);
    auto untraced = state;
    PlanOutput captured, disabled;
    fn_exl3_dcheckpoint(state.data(), lines, trace.data(), 1);
    for (unsigned round = 0; round < 1024; ++round) {
        if (round == 512 || round == 768) {
            fn_exl3_dcheckpoint(state.data(), lines, trace.data(), 0);
            for (unsigned host_round = 0; host_round < 3; ++host_round) {
                std::vector<int> host_picks(11, static_cast<int>((round + host_round * 13) % 512));
                run_plan(state, lines, host_picks, nullptr, 0, captured);
                run_plan(untraced, lines, host_picks, nullptr, 0, disabled);
            }
            const unsigned misc = 2 * lines + 3 * DP_EXPERTS + DP_RECENT_WORDS;
            for (unsigned counter = 1; counter < 8; ++counter)
                state[misc + counter] = untraced[misc + counter] = 0;
            fn_exl3_dcheckpoint(state.data(), lines, trace.data(), 1);
        }
        std::vector<int> picks(11, static_cast<int>((round * 7) % 512));
        picks.back() = 512;
        run_plan(state, lines, picks, trace.data(), round % 3 != 0, captured);
        run_plan(untraced, lines, picks, nullptr, round % 3 != 0, disabled);
        assert(state == untraced);
        assert_same(captured, disabled);
    }
    fn_exl3_dcheckpoint(state.data(), lines, trace.data(), 0);
    auto* misc = reinterpret_cast<DpMisc*>(state.data() + 2 * lines + 3 * DP_EXPERTS + DP_RECENT_WORDS);
    misc->now += 100;
    fn_exl3_dcheckpoint(state.data(), lines, trace.data(), 1);
    assert(header->recorded == 1024 && header->checkpoints == 4);
    assert(header->dropped_lo == 0 && header->dropped_hi == 0 && header->dropped_checkpoints == 0);
    assert(header->discontinuity == 0 && header->invalid_launch == 0);
    assert(std::memcmp(trace.data() + 16 + header->seed_words, state.data(), state.size() * 4) == 0);
    assert(trace[pool_words] == sentinel);
    return trace;
}

static std::vector<unsigned> test_long_capture() {
    const unsigned lines = 10;
    auto trace = make_trace(lines, 16);
    auto* header = reinterpret_cast<DpTraceHeader*>(trace.data());
    assert(header->capacity >= 1200);
    assert(16 + 2 * header->seed_words + header->capacity * header->record_words <= pool_words);
    auto state = make_state(lines);
    const unsigned misc_at = 2 * lines + 3 * DP_EXPERTS + DP_RECENT_WORDS;
    for (unsigned index = 0; index < 8; ++index) state[misc_at + index] = 0xfffffffeu;
    state[2 * lines + DP_EXPERTS + 7] = 0xffffffffu;
    auto untraced = state;
    const auto initial = state;
    PlanOutput captured, disabled;
    for (int index = 0; index < 1200; ++index) {
        std::vector<int> picks(16 * 11);
        for (unsigned slot = 0; slot < picks.size(); ++slot) {
            switch (slot % 11) {
                case 0: picks[slot] = -1; break;
                case 1: picks[slot] = 512; break;
                case 2: picks[slot] = -99; break;
                case 3: picks[slot] = 800; break;
                default: picks[slot] = (index * 7 + slot * 3) % 512; break;
            }
        }
        const int recent = index % 3 != 0;
        DpMisc before;
        std::memcpy(&before, state.data() + misc_at, sizeof(before));
        run_plan(state, lines, picks, trace.data(), recent, captured);
        run_plan(untraced, lines, picks, nullptr, recent, disabled);
        assert(state == untraced);
        assert_same(captured, disabled);
        assert(header->recorded == static_cast<unsigned>(index + 1));
        const unsigned* round = trace.data() + 16 + 2 * header->seed_words + index * header->record_words;
        assert(std::memcmp(round + 4, state.data() + misc_at, sizeof(DpMisc)) == 0);
        DpMisc after;
        std::memcpy(&after, state.data() + misc_at, sizeof(after));
        // misses with no free line are staged (counted apart from the resident copies) and copied with them
        const unsigned copied = after.copies - before.copies, staged = after.staged - before.staged;
        assert(captured.staged_n == static_cast<int>(staged));
        assert(captured.work_n == static_cast<int>(copied + staged));
        unsigned staged_picks = 0;
        for (int pick : picks)
            for (int slot = 0; slot < captured.staged_n; ++slot)
                if (pick == captured.staged[slot]) ++staged_picks;
        assert(round[14] == staged_picks);
        for (int slot = 0; slot < captured.staged_n; ++slot) {
            const int expert = captured.staged[slot];
            for (int part = 0; part < 3; ++part) {
                const long long scratch = static_cast<long long>(test_scratch_base + slot * test_line_bytes + test_offsets[part]);
                assert(captured.tabs[part][expert] == scratch && captured.widths[part][expert] == 3);
            }
        }
        int scratch_segments = 0;
        for (int segment = 0; segment < captured.work_n * 3; ++segment) {
            const unsigned long long destination = static_cast<unsigned long long>(captured.work[segment * 3 + 1]);
            if (destination >= test_scratch_base) {
                assert(destination < test_scratch_base + staged * test_line_bytes);
                ++scratch_segments;
            } else {
                assert(destination >= test_line_base && destination < test_line_base + lines * test_line_bytes);
            }
        }
        assert(scratch_segments == captured.staged_n * 3);
        assert(staged > 0 && copied <= lines);
        assert(std::memcmp(round + 15, picks.data(), picks.size() * 4) == 0);
    }
    assert(header->discontinuity == 0 && header->invalid_launch == 0);
    assert(std::memcmp(trace.data() + 16, initial.data(), initial.size() * 4) == 0);
    assert(std::memcmp(trace.data() + 16 + header->seed_words, state.data(), state.size() * 4) == 0);
    assert(trace[pool_words] == sentinel);
    return trace;
}

static void write_trace(const std::vector<unsigned>& trace, const char* path) {
    const auto* header = reinterpret_cast<const DpTraceHeader*>(trace.data());
    const unsigned used = header->checkpoints ? pool_words : 16 + 2 * header->seed_words + header->recorded * header->record_words;
    FILE* output = std::fopen(path, "wb");
    assert(output != nullptr);
    for (unsigned index = 0; index < used; ++index) {
        const unsigned word = trace[index];
        const unsigned char bytes[4] = {
            static_cast<unsigned char>(word), static_cast<unsigned char>(word >> 8),
            static_cast<unsigned char>(word >> 16), static_cast<unsigned char>(word >> 24)};
        assert(std::fwrite(bytes, 1, 4, output) == 4);
    }
    assert(std::fclose(output) == 0);
}

int main(int argc, char** argv) {
    test_bounds_and_flags();
    const auto trace = test_long_capture();
    const auto checkpoint_trace = test_checkpoint_capture();
    if (argc >= 2) write_trace(trace, argv[1]);
    if (argc >= 3) write_trace(checkpoint_trace, argv[2]);
}
