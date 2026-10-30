//! A stream's state on a rank: sparse keys, DeltaNet states, n-gram tail, a window's conv snapshots and DeltaNet inputs
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");

const Dims = model.Dims;

/// Most rows a verified window holds: the pending token and up to seven drafts.
pub const max_window = 8;

/// Where the bytes of one sparse layer's caches start, and their strides.
pub const SparseView = struct { k: u64, v: u64, raw: u64, pooled: u64 };

/// Where one DeltaNet layer's conv tail (fp32 (taps - 1, channels)) and recurrent state (fp32) live.
pub const LinearView = struct { conv: u64, state: u64 };

/// One DeltaNet layer's window inputs, as the window's conv and split wrote them and its keep reads them.
pub const WindowInputs = struct { k: u64, v: u64, gate: u64, beta: u64 };

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
    /// fp32 values of one window row's DeltaNet inputs: k (key heads x Dk), v (value heads x Dv), gate and beta.
    in_row: usize,
    key_heads: usize,
    key_dim: usize,
    value_heads: usize,
    value_dim: usize,
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
            .in_row = d.key_heads * d.key_dim + d.value_heads * d.value_dim + 2 * d.value_heads,
            .key_heads = d.key_heads,
            .key_dim = d.key_dim,
            .value_heads = d.value_heads,
            .value_dim = d.value_dim,
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

    /// Device bytes of one DeltaNet layer's window inputs: max_window rows of k, then of v, then of gate, then of beta.
    pub fn inBytes(l: Layout) usize {
        return max_window * l.in_row * 4;
    }

    /// Device bytes of a window's keep material: every DeltaNet layer's conv tail after each row, then its inputs.
    pub fn windowBytes(l: Layout) usize {
        return max_window * l.linear * l.conv_bytes + l.linear * l.inBytes();
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
    /// The last window's keep material: every linear layer's conv tails (a row each), then its DeltaNet inputs.
    window: hip.DeviceBuffer,
    /// The n-gram conv's input over the last window (the tail, then its rows): a keep takes its new tail from it.
    cin: hip.DeviceBuffer,
    /// The snapshot row the conv tails are at (null: their places), as `start()` says: a dropping keep names it.
    pending: ?usize = null,
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
        // the snapshots, then the i32 naming the row a window starts from (`start`)
        var window = try hip.DeviceBuffer.alloc(d, layout.windowBytes() + 16);
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
        // the start word -1 (no snapshot row): bytes 0xff by the same blocking fill as the memory's
        try hip.runtime.check(c.window.r.api.hipMemset(@ptrFromInt(c.start()), 0xff, 4));
        try hip.runtime.check(c.window.r.api.hipStreamSynchronize(null));
        c.pending = null;
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

    /// The device i32 the window kernels read: the snapshot row their conv tails start from, -1 for their places.
    pub fn start(c: *const Caches) u64 {
        return c.window.base() + c.layout.windowBytes();
    }

    /// The window starts from snapshot row `row` (null: the states' own places), after the work on `s` so far.
    pub fn startAt(c: *Caches, row: ?usize, s: hip.Stream) !void {
        const same = if (row) |r| (if (c.pending) |q| q == r else false) else c.pending == null;
        if (same) return;
        const word: c_int = if (row) |r| @intCast(r) else -1;
        try hip.runtime.check(c.window.r.api.hipMemsetD32Async(@ptrFromInt(c.start()), word, 1, s.handle));
        c.pending = row;
    }

    /// The conv tails copied back from the snapshot row a keep named, for readers other than a window's kernels.
    pub fn settle(c: *Caches, s: hip.Stream) !void {
        const row = c.pending orelse return;
        const l = c.layout;
        for (c.linear, 0..) |linear, i| {
            if (!linear) continue;
            try hip.runtime.check(c.window.r.api.hipMemcpyDtoDAsync(@ptrFromInt(c.linearAt(i).conv), @ptrFromInt(c.snapshot(i, row)), l.conv_bytes, s.handle));
        }
        try c.startAt(null, s);
    }

    /// Every DeltaNet state run through the last window's first `kept` rows in one launch (the window's own scan).
    pub fn keepStates(c: *const Caches, l: *const hip.Launcher, kept: usize, s: hip.Stream) !void {
        const lay = c.layout;
        if (lay.linear == 0) return;
        const first = for (c.linear, 0..) |linear, i| {
            if (linear and c.slot[i] == 0) break i;
        } else return error.NoLinearLayer;
        try l.tf_gdn_keep(
            @ptrFromInt(c.linearAt(first).state),
            @intCast(lay.linearBytes() / 4),
            @ptrFromInt(c.inputs(first).k),
            @intCast(lay.inBytes() / 4),
            max_window,
            @intCast(kept),
            @intCast(lay.linear),
            @intCast(lay.key_heads),
            @intCast(lay.value_heads),
            @intCast(lay.key_dim),
            @intCast(lay.value_dim),
            s.handle,
        );
    }

    /// The n-gram conv's tail rows (act dtype, (tail_rows, W)).
    pub fn tail(c: *const Caches) u64 {
        const l = c.layout;
        return c.mem.base() + l.sparse * l.sparseBytes() + l.linear * l.linearBytes();
    }

    /// Row `row` of linear layer `layer`'s window conv-tail snapshots, a layer's rows consecutive (the kernels' order).
    pub fn snapshot(c: *const Caches, layer: usize, row: usize) u64 {
        const n = c.slot[layer] * max_window + row;
        return c.window.base() + n * c.layout.conv_bytes;
    }

    /// Where linear layer `layer`'s window inputs live: max_window rows of k, of v, of gate and of beta (fp32).
    pub fn inputs(c: *const Caches, layer: usize) WindowInputs {
        const l = c.layout;
        const k = c.window.base() + l.linear * max_window * l.conv_bytes + c.slot[layer] * l.inBytes();
        const v = k + max_window * l.key_heads * l.key_dim * 4;
        const gate = v + max_window * l.value_heads * l.value_dim * 4;
        return .{ .k = k, .v = v, .gate = gate, .beta = gate + max_window * l.value_heads * 4 };
    }
};

/// Streams' caches by slot, made once and kept: a slot's buffers never change, so the graphs naming them stay valid.
pub const Pool = struct {
    slots: std.ArrayList(?Caches) = .empty,
    /// Each slot's buffers as made (zeros: none yet), leased or not: caches made elsewhere never take a slot's graphs.
    made: std.ArrayList([3]u64) = .empty,

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
        while (p.made.items.len <= slot) try p.made.append(gpa, .{ 0, 0, 0 });
        const held = &p.slots.items[slot];
        if (held.* == null) {
            held.* = try maker.caches(cap);
            held.*.?.id = slot;
            p.made.items[slot] = buffersOf(&held.*.?);
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

    /// Whether `c` holds the buffers its slot was made with, the ones a graph keyed by that slot names.
    pub fn holds(p: *const Pool, c: *const Caches) bool {
        if (c.id >= p.made.items.len) return false;
        return std.mem.eql(u64, &p.made.items[c.id], &buffersOf(c));
    }

    fn buffersOf(c: *const Caches) [3]u64 {
        return .{ c.mem.base(), c.window.base(), c.cin.base() };
    }

    pub fn deinit(p: *Pool, gpa: std.mem.Allocator) void {
        for (p.slots.items) |*c| if (c.*) |*x| x.deinit(gpa);
        p.slots.deinit(gpa);
        p.made.deinit(gpa);
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
