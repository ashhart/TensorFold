//! A stream's state on a rank: sparse layers' keys, DeltaNet states, the n-gram tail; a window's per-row snapshots.
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");

const Dims = model.Dims;

/// Most rows a verified window holds: the pending token and up to three drafts.
pub const max_window = 4;

/// Where the bytes of one sparse layer's caches start, and their strides.
pub const SparseView = struct { k: u64, v: u64, raw: u64, pooled: u64 };

/// Where one DeltaNet layer's conv tail (fp32 (taps - 1, channels)) and recurrent state (fp32) live.
pub const LinearView = struct { conv: u64, state: u64 };

pub const Layout = struct {
    sparse: usize,
    linear: usize,
    act: usize,
    cap: usize,
    kv_bytes: usize,
    raw_bytes: usize,
    pooled_bytes: usize,
    conv_bytes: usize,
    state_bytes: usize,
    tail_rows: usize,
    tail_bytes: usize,
    kv_heads: usize,
    head_dim: usize,
    index_dim: usize,
    ratio: usize,

    pub fn of(d: Dims, kinds: []const bool, cap: usize, act: model.Kind) Layout {
        var sparse: usize = 0;
        for (kinds) |linear| sparse += @intFromBool(!linear);
        const a = act.size();
        const blocks = cap / d.index_ratio + 1;
        return .{
            .sparse = sparse,
            .linear = kinds.len - sparse,
            .act = a,
            .cap = cap,
            .kv_bytes = d.kv_heads * cap * d.head_dim * a,
            .raw_bytes = cap * d.index_dim * a,
            .pooled_bytes = blocks * d.index_dim * a,
            .conv_bytes = (d.conv - 1) * d.convWidth() * 4,
            .state_bytes = d.value_heads * d.key_dim * d.value_dim * 4,
            .tail_rows = d.ngram * (d.ple_taps - 1),
            .tail_bytes = d.ngram * (d.ple_taps - 1) * d.wide() * a,
            .kv_heads = d.kv_heads,
            .head_dim = d.head_dim,
            .index_dim = d.index_dim,
            .ratio = d.index_ratio,
        };
    }

    fn sparseBytes(l: Layout) usize {
        return 2 * l.kv_bytes + l.raw_bytes + l.pooled_bytes;
    }

    fn linearBytes(l: Layout) usize {
        return l.conv_bytes + l.state_bytes;
    }

    /// Device bytes of a stream's caches at `cap` positions.
    pub fn bytes(l: Layout) usize {
        return l.sparse * l.sparseBytes() + l.linear * l.linearBytes() + l.tail_bytes;
    }

    /// Bytes of the state after `at` positions: the sparse layers' first `at` rows, the rest whole.
    pub fn stateBytes(l: Layout, at: usize) usize {
        const sparse_rows = 2 * l.kv_heads * at * l.head_dim * l.act + at * l.index_dim * l.act +
            at / l.ratio * l.index_dim * l.act;
        return l.sparse * sparse_rows + l.linear * l.linearBytes() + l.tail_bytes;
    }

    /// Device bytes of a window's snapshots: every DeltaNet layer's conv tail and state after each row.
    pub fn windowBytes(l: Layout) usize {
        return max_window * l.linear * l.linearBytes();
    }

    /// Device bytes of the n-gram conv's input over a window: the tail, then the window's rows.
    pub fn cinBytes(l: Layout) usize {
        return l.tail_bytes + max_window * (l.tail_bytes / l.tail_rows);
    }
};

pub const Caches = struct {
    layout: Layout,
    /// Each layer's place among its kind (sparse or linear).
    slot: []usize,
    linear: []bool,
    mem: hip.DeviceBuffer,
    /// Snapshots of the last window: every linear layer's conv tails (a row each), then their states (a row each).
    window: hip.DeviceBuffer,
    /// The n-gram conv's input over the last window (the tail, then its rows): a keep takes its new tail from it.
    cin: hip.DeviceBuffer,
    /// Positions kept, and complete blocks pooled in every sparse layer.
    len: usize = 0,
    pooled: usize = 0,
    /// The n-gram history: the last tokens before the next row (eos before any).
    history: [2]i64,
    /// The slot these caches fill in the engine's pool (every rank's the same): a round's graph is keyed by it.
    id: u32 = 0,

    pub fn init(
        gpa: std.mem.Allocator,
        d: *const hip.Runtime,
        dims: Dims,
        kinds: []const bool,
        cap: usize,
        act: model.Kind,
        eos: i64,
    ) !Caches {
        const layout = Layout.of(dims, kinds, cap, act);
        const slot = try gpa.alloc(usize, kinds.len);
        errdefer gpa.free(slot);
        var counts: [2]usize = .{ 0, 0 };
        for (kinds, slot) |linear, *s| {
            s.* = counts[@intFromBool(linear)];
            counts[@intFromBool(linear)] += 1;
        }
        const linear = try gpa.dupe(bool, kinds);
        errdefer gpa.free(linear);
        var mem = try hip.DeviceBuffer.alloc(d, layout.bytes());
        errdefer mem.free();
        try mem.fill8(0);
        var window = try hip.DeviceBuffer.alloc(d, @max(layout.windowBytes(), 4));
        errdefer window.free();
        var cin = try hip.DeviceBuffer.alloc(d, layout.cinBytes());
        errdefer cin.free();
        return .{
            .layout = layout,
            .slot = slot,
            .linear = linear,
            .mem = mem,
            .window = window,
            .cin = cin,
            .history = .{ eos, eos },
        };
    }

    /// The state of fresh caches again, the memory kept: zeros, no positions, eos before any row.
    pub fn reset(c: *Caches, eos: i64) !void {
        try c.mem.fill8(0);
        c.len = 0;
        c.pooled = 0;
        c.history = .{ eos, eos };
    }

    pub fn deinit(c: *Caches, gpa: std.mem.Allocator) void {
        c.cin.free();
        c.window.free();
        c.mem.free();
        gpa.free(c.slot);
        gpa.free(c.linear);
        c.* = undefined;
    }

    pub fn sparse(c: *const Caches, layer: usize) SparseView {
        const l = c.layout;
        const at = c.mem.base() + c.slot[layer] * l.sparseBytes();
        return .{
            .k = at,
            .v = at + l.kv_bytes,
            .raw = at + 2 * l.kv_bytes,
            .pooled = at + 2 * l.kv_bytes + l.raw_bytes,
        };
    }

    pub fn linearAt(c: *const Caches, layer: usize) LinearView {
        const l = c.layout;
        const at = c.mem.base() + l.sparse * l.sparseBytes() + c.slot[layer] * l.linearBytes();
        return .{ .conv = at, .state = at + l.conv_bytes };
    }

    /// The n-gram conv's tail rows (act dtype, (tail_rows, W)).
    pub fn tail(c: *const Caches) u64 {
        const l = c.layout;
        return c.mem.base() + l.sparse * l.sparseBytes() + l.linear * l.linearBytes();
    }

    /// Row `row` of the window's snapshots of linear layer `layer`; a layer's rows consecutive, as the kernels write.
    pub fn snapshot(c: *const Caches, layer: usize, row: usize) LinearView {
        const l = c.layout;
        const n = c.slot[layer] * max_window + row;
        const states = c.window.base() + l.linear * max_window * l.conv_bytes;
        return .{ .conv = c.window.base() + n * l.conv_bytes, .state = states + n * l.state_bytes };
    }
};

/// Streams' caches by slot, made once and kept: a slot's buffers never change, so the graphs naming them stay valid.
pub const Pool = struct {
    slots: std.ArrayList(?Caches) = .empty,

    /// Slot `slot`'s caches, fresh (`maker.caches(cap)` makes them the first time); a failed reset keeps them there.
    pub fn lease(
        p: *Pool,
        gpa: std.mem.Allocator,
        slot: u32,
        s: hip.Stream,
        eos: i64,
        cap: usize,
        maker: anytype,
    ) !Caches {
        while (p.slots.items.len <= slot) try p.slots.append(gpa, null);
        const held = &p.slots.items[slot];
        if (held.* == null) {
            held.* = try maker.caches(cap);
            held.*.?.id = slot;
        }
        try s.synchronize();
        try held.*.?.reset(eos);
        const c = held.*.?;
        held.* = null;
        return c;
    }

    /// A finished stream's caches back to their slot.
    pub fn unlease(p: *Pool, c: Caches) void {
        p.slots.items[c.id] = c;
    }

    pub fn deinit(p: *Pool, gpa: std.mem.Allocator) void {
        for (p.slots.items) |*c| if (c.*) |*x| x.deinit(gpa);
        p.slots.deinit(gpa);
    }
};

test "a stream's bytes at tp 4: 12 sparse layers' caches, 36 DeltaNet states and the n-gram tail" {
    const config = @import("../config.zig");
    var c = try config.parse(std.testing.allocator, @embedFile("../fixtures/config.json"));
    defer c.deinit();
    const d = try Dims.of(&c, 4);
    var kinds: [48]bool = undefined;
    for (&kinds, 0..) |*k, i| k.* = (try c.kind(i)) == .linear_attention;
    const l = Layout.of(d, &kinds, 4096, .f16);
    try std.testing.expectEqual(@as(usize, 12), l.sparse);
    try std.testing.expectEqual(@as(usize, 36), l.linear);
    const per_layer = 2 * 4096 * 256 * 2 + 4096 * 128 * 2 + 1025 * 128 * 2;
    const linear = 3 * 2560 * 4 + 12 * 128 * 128 * 4;
    try std.testing.expectEqual(@as(usize, 12 * per_layer + 36 * linear + 9 * 10240 * 2), l.bytes());
}
