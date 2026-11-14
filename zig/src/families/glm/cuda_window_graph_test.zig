//! TF_GLM_GRAPHS' keys on host values alone: what moves a held graph in place and what takes another one.
const std = @import("std");
const t = std.testing;
const fw = @import("cuda_forward.zig");
const weights = @import("cuda_weights.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const wg = @import("cuda_window_graph.zig");
const kg = @import("cuda").keyed_graph;

const dense_limit = 2048;

/// A model of one KDA layer then one latent layer, as `Key.init` reads it.
const Model = struct {
    layers: [2]weights.Layer,
    w: weights.Weights,
    b: fw.Buffers,

    fn init(m: *Model) void {
        var mla: dsa_block.Layer = undefined;
        mla.dense_limit = dense_limit;
        m.layers[0].mixer = .{ .kda = undefined };
        m.layers[1].mixer = .{ .mla = mla };
        m.w = undefined;
        m.w.layers = &m.layers;
        m.b = undefined;
        m.b.nch = 64;
    }
};

/// A sequence at `pos` whose buffers start at `base` (another sequence: another base).
const Seq = struct {
    st: fw.State,
    cur: [1]u1 = .{0},
    caches: [1]dsa_block.Caches,

    fn init(s: *Seq, base: u64, pos: usize) void {
        s.caches = .{.{ .lc = base + 0x100, .ik = base + 0x200, .ig = base + 0x300, .pk = base + 0x400, .pk_rows = 16386 }};
        s.st = undefined;
        s.st.pos = pos;
        s.st.capacity = 32768;
        s.st.base = 0;
        s.st.table = 0;
        inline for (.{ "pos_dev", "conv", "rec", "proj", "save_k", "save_v", "save_g", "save_b" }, 0..) |name, i| @field(s.st, name) = base + 0x10 * (i + 1);
        s.st.cur = &s.cur;
        s.st.caches = &s.caches;
    }
};

fn key(m: *Model, s: *Seq, slot: u32, rows: usize) wg.Key {
    var one: [1]fw.Rank = .{.{ .w = &m.w, .st = &s.st, .b = &m.b }};
    var f: fw.Forward = undefined;
    f.kda_wide = true;
    f.split = true;
    f.ring = false;
    f.exchange = .local;
    f.ranks = &one;
    return .init(&f, slot, rows);
}

fn sameSlot(a: wg.Key, b: wg.Key) bool {
    return std.meta.eql(kg.slotOf(a), kg.slotOf(b));
}

test "GLM graphs: the next tokens replay, a chunk or score grid moves the graph in place" {
    var m: Model = undefined;
    m.init();
    var s: Seq = undefined;
    s.init(0x10000, 100);
    const at100 = key(&m, &s, 0, 3);
    s.st.pos = 101;
    try t.expectEqual(at100, key(&m, &s, 0, 3)); // the same chunk, dense: nothing on the host changed
    s.st.pos = 509; // rows 509-511: one 512-key chunk
    const before = key(&m, &s, 0, 3);
    s.st.pos = 510; // rows 510-512: a second
    const past = key(&m, &s, 0, 3);
    try t.expect(!std.meta.eql(before, past));
    try t.expect(sameSlot(before, past));
    s.st.pos = 9000;
    const sparse = key(&m, &s, 0, 3);
    s.st.pos = 9000 + 256; // another score grid size, the same kernels
    const later = key(&m, &s, 0, 3);
    try t.expect(!std.meta.eql(sparse, later));
    try t.expect(sameSlot(sparse, later));
}

test "GLM graphs: a KDA half, the sparse branch, rows or a stream take another graph" {
    var m: Model = undefined;
    m.init();
    var s: Seq = undefined;
    s.init(0x10000, 100);
    const base = key(&m, &s, 0, 3);
    s.cur[0] = 1;
    try t.expect(!sameSlot(base, key(&m, &s, 0, 3))); // the other state half: other pointers every round
    s.cur[0] = 0;
    s.st.pos = dense_limit - 2; // its last row past the dense limit: the sparse selection launches
    try t.expect(!sameSlot(base, key(&m, &s, 0, 3)));
    s.st.pos = 100;
    try t.expect(!sameSlot(base, key(&m, &s, 0, 4)));
    try t.expect(!sameSlot(base, key(&m, &s, 1, 3)));
    var other: Seq = undefined;
    other.init(0x90000, 100); // the next request in the same slot, on other buffers: moves in place
    const next = key(&m, &other, 0, 3);
    try t.expect(!std.meta.eql(base, next));
    try t.expect(sameSlot(base, next));
}

test "GLM graphs: a shared window's key follows its segments' order and rows" {
    var m: Model = undefined;
    m.init();
    var a: Seq = undefined;
    var b: Seq = undefined;
    a.init(0x10000, 100);
    b.init(0x90000, 700);
    const segs = [_]fw.Seg{ .{ .st = &a.st, .row0 = 0, .rows = 2 }, .{ .st = &b.st, .row0 = 2, .rows = 3 } };
    const swapped = [_]fw.Seg{ .{ .st = &b.st, .row0 = 0, .rows = 3 }, .{ .st = &a.st, .row0 = 3, .rows = 2 } };
    const k = shared(&m, &segs, &.{ 0, 1 });
    try t.expectEqual(k, shared(&m, &segs, &.{ 0, 1 }));
    try t.expect(!std.meta.eql(kg.slotOf(k), kg.slotOf(shared(&m, &swapped, &.{ 1, 0 }))));
    b.st.pos = 1100; // the second stream past a chunk: in place
    const moved = shared(&m, &segs, &.{ 0, 1 });
    try t.expect(!std.meta.eql(k, moved));
    try t.expect(std.meta.eql(kg.slotOf(k), kg.slotOf(moved)));
}

fn shared(m: *Model, segs: []const fw.Seg, slots: []const u32) wg.SharedKey {
    var one: [1]fw.Rank = .{.{ .w = &m.w, .st = segs[0].st, .b = &m.b, .segs = segs }};
    var f: fw.Forward = undefined;
    f.kda_wide = true;
    f.split = true;
    f.ring = false;
    f.exchange = .local;
    f.ranks = &one;
    var rows: usize = 0;
    for (segs) |g| rows += g.rows;
    return .init(&f, slots, rows);
}
