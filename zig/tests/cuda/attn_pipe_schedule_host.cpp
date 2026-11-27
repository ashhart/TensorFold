// The actual device scheduling helper on an asynchronous two-slot model, including missing-fence/wait controls.
#define TF_ATTN_PIPE_FN inline
#include "common/attn_pipe_schedule.cuh"
#include <deque>
#include <stdexcept>
#include <cstdio>

struct Model {
    struct Group { int item; };
    std::deque<Group> pending;
    int slot[2] = {-2, -2};
    bool active[2] = {false, false};
    bool queries = false;
    int barriers = 0, read_count = 0, staged = 0;
    bool omit_barrier = false, short_wait = false;
    void stage(int item) {
        if (item >= 0 && active[item % 2]) throw std::logic_error("overwrite before all warps consumed");
        pending.push_back({item});
        if (item >= 0) ++staged;
    }
    void wait(bool first) {
        const unsigned keep = first || short_wait ? 1 : 0;
        while (pending.size() > keep) {
            const int item = pending.front().item;
            pending.pop_front();
            if (item == -1) queries = true;
            else if (item >= 0) slot[item % 2] = item;
        }
    }
    void barrier() {
        ++barriers;
        if (!omit_barrier) active[0] = active[1] = false;
    }
    void read(int item) {
        if (!queries || slot[item % 2] != item) throw std::logic_error("read before asynchronous copy completed");
        if (item != read_count++) throw std::logic_error("changed QK/PV consumer order");
        active[item % 2] = true;
    }
    void prologue() { stage(-1); stage(0); stage(1); }
};

static Model candidate(int tiles, bool omit_barrier = false, bool short_wait = false) {
    Model m;
    m.omit_barrier = omit_barrier;
    m.short_wait = short_wait;
    m.prologue();
    for (int item = 0; item < 4 * tiles; ++item) {
        tf_attn_pipe::advance(item, 4 * tiles,
            [&](bool first) { m.wait(first); }, [&] { m.barrier(); }, [&](int next) { m.stage(next); });
        m.read(item);
    }
    m.wait(false);
    return m;
}

static Model original(int tiles) {
    Model m;
    m.prologue();
    for (int item = 0; item < 4 * tiles; ++item) {
        m.wait(true);
        m.barrier();
        m.read(item);
        m.barrier();
        m.stage(item + 2 < 4 * tiles ? item + 2 : -2); // the original commits empty tail groups too
    }
    m.wait(false);
    return m;
}

int main() {
    for (const int tiles : {1, 2, 3, 4, 17, 448, 1985, 4096}) {
        const Model old = original(tiles), next = candidate(tiles);
        if (old.barriers != 8 * tiles || next.barriers != 4 * tiles ||
            old.read_count != next.read_count || next.staged != 4 * tiles || !next.pending.empty()) return 1;
    }
    int red = 0;
    try { (void)candidate(3, true); } catch (const std::logic_error&) { ++red; }
    try { (void)candidate(3, false, true); } catch (const std::logic_error&) { ++red; }
    if (red != 2) return 2;
    std::puts("PASS actual attention schedule: 8 tile counts, 8->4 barriers/tile, two slots, queries ready, all consumers ordered; 2 controls red");
}
