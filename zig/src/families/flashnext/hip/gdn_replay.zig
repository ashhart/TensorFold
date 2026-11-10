//! A verify window's DeltaNet without a snapshot a row: the round runs from a plan, the keep replays it.
const std = @import("std");
const hip = @import("hip");

const pops = hip.plan_ops;
const cache = @import("cache.zig");
const forward = @import("forward.zig");
const model = @import("model.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;
const take = forward.take;

/// Rows one window holds at most: the pending token and its drafts.
pub const window_rows = cache.max_window;

/// Slots a window's plan runs: the window, then an empty slot that stops its wave writing the state back.
const slots = 2;

/// Whether this model's shapes are the plan kernels': key dim 128, eight value rows a wave, gated heads.
pub fn supported(d: model.Dims, layers: usize) bool {
    if (d.key_dim != 128) return false;
    if (d.value_dim % 8 != 0 or d.key_heads < 1 or d.value_heads % d.key_heads != 0) return false;
    if (d.conv < 1 or d.conv > 8) return false;
    return layers > 0;
}

/// The plan's words in four-byte units: a slot's first row, its rows, its kept rows, its descriptor pointer.
pub const Layout = struct {
    first: usize,
    count: usize,
    keep: usize,
    ptr: usize,
    snaps: usize,
    words: usize,

    pub fn of(layers: usize) Layout {
        var l: Layout = undefined;
        l.first = 0;
        l.count = slots;
        l.keep = 2 * slots;
        l.ptr = std.mem.alignForward(usize, 3 * slots, 4); // eight-byte pointers start on an even word
        l.snaps = l.ptr + 2 * slots;
        l.words = l.snaps + 4 * layers;
        return l;
    }

    /// Words a slot's descriptor takes: cached positions, the last kept final row, two addresses a layer.
    pub fn desc(layers: usize) usize {
        return 2 * (2 + 2 * layers);
    }

    /// Where the descriptors start (eight-byte aligned).
    fn start(layers: usize) usize {
        return std.mem.alignForward(usize, of(layers).words, 2);
    }

    /// Every word of the plan: its control words, then a descriptor a slot.
    pub fn total(layers: usize) usize {
        return start(layers) + slots * desc(layers);
    }
};

/// One stream's window plan: the round's words, its descriptors, and the rows its keep reads again.
pub const Plan = struct {
    r: *const hip.Runtime,
    layers: usize,
    /// Rows the buffers hold at most: a window.
    room: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv: usize,
    channels: usize,
    kept_stride: usize,
    samples_stride: usize,
    /// The words and descriptors, and the device copy a round and a keep read.
    host: hip.HostBuffer,
    dev: hip.DeviceBuffer,
    /// A layer's k, v, gate and beta over the window's rows, section-major: what a keep replays from.
    kept: hip.DeviceBuffer,
    /// A layer's conv input rows over the window, section-major: where a keep takes its conv tail.
    samples: hip.DeviceBuffer,
    /// Rows the window last ran.
    rows: usize = 0,

    pub fn init(d: *const hip.Runtime, dims: model.Dims) !Plan {
        if (!supported(dims, dims.layers)) return error.UnsupportedPlan;
        var host = try hip.HostBuffer.alloc(d, 4 * Layout.total(dims.layers));
        errdefer host.free();
        var dev = try hip.DeviceBuffer.alloc(d, host.bytes.len);
        errdefer dev.free();
        const width = dims.key_heads * dims.key_dim + dims.value_heads * dims.value_dim + 2 * dims.value_heads;
        var kept = try hip.DeviceBuffer.alloc(d, 4 * window_rows * width * dims.layers);
        errdefer kept.free();
        var samples = try hip.DeviceBuffer.alloc(d, 4 * window_rows * dims.convWidth() * dims.layers);
        errdefer samples.free();
        @memset(host.bytes, 0);
        return .{
            .r = d,
            .layers = dims.layers,
            .room = window_rows,
            .key_heads = dims.key_heads,
            .value_heads = dims.value_heads,
            .key_dim = dims.key_dim,
            .value_dim = dims.value_dim,
            .conv = dims.conv,
            .channels = dims.convWidth(),
            .kept_stride = 4 * window_rows * width,
            .samples_stride = 4 * window_rows * dims.convWidth(),
            .host = host,
            .dev = dev,
            .kept = kept,
            .samples = samples,
        };
    }

    pub fn deinit(p: *Plan) void {
        p.samples.free();
        p.kept.free();
        p.dev.free();
        p.host.free();
    }

    /// Device bytes a stream's plan takes, whether or not one is made: the admission counts it the same either way.
    pub fn bytes(dims: model.Dims) usize {
        const width = dims.key_heads * dims.key_dim + dims.value_heads * dims.value_dim + 2 * dims.value_heads;
        const words = 4 * Layout.total(dims.layers);
        return words + 4 * window_rows * width * dims.layers + 4 * window_rows * dims.convWidth() * dims.layers;
    }

    pub fn layout(p: *const Plan) Layout {
        return Layout.of(p.layers);
    }

    /// A slot's descriptor words: [0] cached positions, [1] the last kept final row (0: none), then a layer's pair.
    fn descriptor(p: *Plan, slot: usize) []u64 {
        const at = Layout.start(p.layers) + slot * Layout.desc(p.layers);
        return p.host.slice(u64)[at / 2 ..][0 .. Layout.desc(p.layers) / 2];
    }

    fn descAt(p: *const Plan, slot: usize) u64 {
        return p.dev.base() + 4 * (Layout.start(p.layers) + slot * Layout.desc(p.layers));
    }

    /// Linear layer `layer`'s section-major k, v, gate and beta over the window's row `at`.
    pub fn keptOf(p: *const Plan, layer: usize, at: usize) struct { k: u64, v: u64, gate: u64, beta: u64 } {
        const kw = p.key_heads * p.key_dim;
        const vw = p.value_heads * p.value_dim;
        const rows = p.rows;
        const base = p.kept.base() + layer * p.kept_stride;
        const k = base + 4 * at * kw;
        const v = base + 4 * rows * kw + 4 * at * vw;
        const gate = base + 4 * rows * (kw + vw) + 4 * at * p.value_heads;
        return .{ .k = k, .v = v, .gate = gate, .beta = gate + 4 * rows * p.value_heads };
    }

    /// The conv samples of layer `layer` (the window's `channels` a row).
    pub fn samplesOf(p: *const Plan, layer: usize) u64 {
        return p.samples.base() + layer * p.samples_stride;
    }

    /// The kept rows a round and a keep read.
    pub fn keepWords(p: *const Plan) u64 {
        return p.dev.base() + 4 * p.layout().keep;
    }

    /// The window's words and descriptor for caches `c` over `n` rows, on the host: the copy goes up with `send`.
    pub fn fill(p: *Plan, c: *const cache.Caches, n: usize) !void {
        if (n == 0 or n > p.room) return error.TooManyRows;
        const l = p.layout();
        const w = p.host.slice(u32);
        const ptr = p.host.slice(u64);
        const desc = p.descriptor(0);
        for (0..p.layers) |i| {
            // a sparse layer keeps its descriptor pair, conv samples and replay inputs zero: the kernels skip it
            desc[2 + 2 * i ..][0..2].* = if (c.linear[i]) .{ c.linearAt(i).conv, c.linearAt(i).state } else .{ 0, 0 };
            ptr[l.snaps / 2 + 2 * i] = if (c.linear[i]) p.samplesOf(i) else 0;
            ptr[l.snaps / 2 + 2 * i + 1] = if (c.linear[i]) p.kept.base() + i * p.kept_stride else 0;
        }
        // a round keeps no final row: a keep takes the conv tail and the state alone
        desc[0] = 0;
        desc[1] = 0;
        @memset(p.descriptor(1), 0);
        w[l.first] = 0;
        w[l.count] = @intCast(n);
        w[l.keep] = std.math.maxInt(u32);
        ptr[l.ptr / 2] = p.descAt(0);
        // the empty slot runs nothing and keeps nothing: the kernels stop at its zero row count
        w[l.first + 1] = 0;
        w[l.count + 1] = 0;
        w[l.keep + 1] = std.math.maxInt(u32);
        ptr[l.ptr / 2 + 1] = p.descAt(1);
        p.rows = n;
    }

    /// The words up to the device: a graph holds the copy, so its every replay sends the window's own words.
    pub fn send(p: *Plan, s: hip.abi.Stream) !void {
        try hip.raw.upload(p.dev, 0, p.host.bytes, s);
    }

    /// The window's first `n` rows kept: the conv tail from their samples, the state replayed through them.
    pub fn keep(p: *Plan, o: Ops, n: usize) !void {
        // a window of one row left the state and its conv tail as the round wrote them
        if (p.rows <= 1 or n == 0) return;
        p.host.slice(u32)[p.layout().keep] = @bitCast(@as(i32, @intCast(n)));
        try p.send(o.stream);
        try pops.keep(o, p.ref(), .{
            .keep = p.keepWords(),
            .hidden = 0,
            .hidden_words = 0,
            .channels = p.channels,
            .taps = p.conv - 1,
            .layers = p.layers,
        });
        try pops.gdnReplay(o, p.keepWords(), p.ref(), p.key_heads, p.value_heads, p.key_dim, p.value_dim, p.layers);
    }

    /// The words as launches take them: the window's rows from its first.
    pub fn ref(p: *const Plan) pops.Plan {
        const l = p.layout();
        const at = p.dev.base();
        return .{
            .args = .{
                .first = at + 4 * l.first,
                .count = at + 4 * l.count,
                .desc = at + 4 * l.ptr,
                .snaps = at + 4 * l.snaps,
            },
            .rows = p.rows,
            .slots = slots,
        };
    }
};

/// Every cache slot's window plan: a slot's keep reads its own rows, which the next round over it has not yet written.
pub const Slots = struct {
    gpa: std.mem.Allocator,
    plans: []Plan,

    pub fn init(gpa: std.mem.Allocator, d: *const hip.Runtime, dims: model.Dims, streams: usize) !Slots {
        const plans = try gpa.alloc(Plan, @max(streams, 1));
        var made: usize = 0;
        errdefer {
            for (plans[0..made]) |*p| p.deinit();
            gpa.free(plans);
        }
        for (plans) |*p| {
            p.* = try Plan.init(d, dims);
            made += 1;
        }
        return .{ .gpa = gpa, .plans = plans };
    }

    pub fn deinit(s: *Slots) void {
        for (s.plans) |*p| p.deinit();
        s.gpa.free(s.plans);
    }

    /// Cache slot `id`'s plan (null: a slot this rank never leased, so nothing can round or keep it).
    pub fn planFor(s: *Slots, id: u32) ?*Plan {
        if (id >= s.plans.len) return null;
        return &s.plans[id];
    }

    /// Device bytes the slots take.
    pub fn bytes(dims: model.Dims, streams: usize) usize {
        return Plan.bytes(dims) * @max(streams, 1);
    }
};
