//! DFlash 2 Metal scratch arenas, per-stream private K/V ring buffers, and speculate-early state matching.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const w = @import("dflash2_weights.zig");
const k = @import("dflash2_kernels.zig");

pub const absorb_rows: usize = 256;
pub const max_chunks: usize = (w.window_size + w.block_size + 127) / 128;
pub const max_batch_streams: usize = k.max_batch;
pub const max_window_rows: usize = 16;
pub const max_tree_nodes: usize = max_window_rows - 1;

pub const scratch_fields = [_][]const u8{
    "ids",  "resid", "x",  "xc",   "dyn", "q_raw", "k_raw",  "v_raw", "q",    "kb",  "vb",  "part",
    "attn", "y",     "yc", "gate", "up",  "act",   "logits", "hid",   "tval", "tid", "seq", "taps",
    "fcx",  "hctx",  "ck", "cv",
};

pub fn scratchSizes(max_streams: usize) [scratch_fields.len]usize {
    const streams = @max(@as(usize, 1), @min(max_streams, max_batch_streams));
    const n = streams * w.block_size;
    const a = absorb_rows;
    const bf = 2;
    const tap_rows = if (streams > 1) a else max_window_rows;
    return .{
        n * 4,
        n * w.hidden * 4,
        n * w.hidden * bf,
        n * w.hidden * bf,
        n * w.dyn_width * bf,
        n * w.query_width * bf,
        n * w.kv_width * bf,
        n * w.kv_width * bf,
        n * w.query_width * bf,
        n * w.kv_width * bf,
        n * w.kv_width * bf,
        n * w.q_heads * max_chunks * 132 * 4,
        n * w.query_width * bf,
        n * w.hidden * bf,
        n * w.hidden * bf,
        n * w.intermediate * bf,
        n * w.intermediate * bf,
        n * w.intermediate * bf,
        n * w.padded_vocab * bf,
        n * w.rank * bf,
        n * w.top_k * 4,
        n * w.top_k * 4,
        n * 4,
        tap_rows * w.taps_n * w.hidden * bf,
        a * w.hidden * bf,
        a * w.hidden * bf,
        a * w.kv_width * bf,
        a * w.kv_width * bf,
    };
}

pub fn scratchBytes(max_streams: usize) u64 {
    var total: u64 = 0;
    for (scratchSizes(max_streams)) |s| total += s;
    return total;
}

pub fn streamBytes() u64 {
    return @as(u64, w.layers_n) * 2 * w.ring_size * w.kv_width * 2;
}

pub const Scratch = struct {
    ids: mtl.Buffer,
    resid: mtl.Buffer,
    x: mtl.Buffer,
    xc: mtl.Buffer,
    dyn: mtl.Buffer,
    q_raw: mtl.Buffer,
    k_raw: mtl.Buffer,
    v_raw: mtl.Buffer,
    q: mtl.Buffer,
    kb: mtl.Buffer,
    vb: mtl.Buffer,
    part: mtl.Buffer,
    attn: mtl.Buffer,
    y: mtl.Buffer,
    yc: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    logits: mtl.Buffer,
    hid: mtl.Buffer,
    tval: mtl.Buffer,
    tid: mtl.Buffer,
    seq: mtl.Buffer,
    taps: mtl.Buffer,
    fcx: mtl.Buffer,
    hctx: mtl.Buffer,
    ck: mtl.Buffer,
    cv: mtl.Buffer,

    pub fn init(device: mtl.Device, max_streams: usize) !Scratch {
        @setEvalBranchQuota(10_000);
        const ns = scratchSizes(max_streams);
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (scratch_fields, 0..) |f, i| if (i < made) @field(s, f).deinit();
        inline for (scratch_fields, 0..) |f, i| {
            @field(s, f) = try device.buffer(@max(ns[i], 16), mtl.ResourceOptions.shared);
            made += 1;
        }
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (scratch_fields) |f| @field(s, f).deinit();
        s.* = undefined;
    }
};

pub const Lane = struct {
    k: [w.layers_n]mtl.Buffer,
    v: [w.layers_n]mtl.Buffer,
    from: u64 = 0,
    end: u64 = 0,
    held: [max_tree_nodes]u32 = @splat(0),
    parents: [max_tree_nodes]i32 = @splat(-1),
    own: [max_tree_nodes]f64 = @splat(0),
    grown: u32 = 0,
    held_len: u32 = 0,
    zero_streak: u32 = 0,
    plain_wait: u32 = 0,

    pub fn init(device: mtl.Device) !Lane {
        var l: Lane = .{ .k = undefined, .v = undefined };
        const bytes = w.ring_size * w.kv_width * 2;
        var made: usize = 0;
        errdefer for (0..made) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        };
        for (0..w.layers_n) |i| {
            l.k[i] = try device.buffer(bytes, mtl.ResourceOptions.private);
            l.v[i] = device.buffer(bytes, mtl.ResourceOptions.private) catch |err| {
                l.k[i].deinit();
                return err;
            };
            made += 1;
        }
        return l;
    }

    pub fn deinit(l: *Lane) void {
        for (0..w.layers_n) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        }
        l.* = undefined;
    }
};

pub const AbsorbSeg = struct { lane: *const Lane, src: mtl.Buffer, src_off: usize, pos0: u64, rows: usize, rows_map: ?[]const u32 = null };
pub const AbsorbWork = struct { lane: *Lane, prev_end: u64, start: u64, keep_from: u64, end: u64, src: mtl.Buffer, offset: usize, rows_map: ?[]const u32 = null };
pub const Block = struct { lane: *const Lane, pending: u32, anchor: u64, from: u64 };

pub const SpecAbsorb = struct {
    stream: *lanes.Stream,
    start: u64,
    count: u32,
    has_rows: bool,
    rows: [max_window_rows]u32 = @splat(0),
};

pub const SpecHold = struct {
    stream: *lanes.Stream,
    pending: u32,
    anchor: u64,
};

/// Tracks an in-flight asynchronous speculate_early absorb+hold pass for exact hit verification.
pub const SpecState = struct {
    cb: ?mtl.CommandBuffer = null,
    n_absorbs: usize = 0,
    n_holds: usize = 0,
    absorbs: [max_batch_streams]SpecAbsorb = undefined,
    holds: [max_batch_streams]SpecHold = undefined,

    pub fn record(sp: *SpecState, cb: ?mtl.CommandBuffer, absorbs: []const lanes.drafter.Absorb, holds: []const lanes.drafter.Hold) void {
        std.debug.assert(absorbs.len <= max_batch_streams and holds.len <= max_batch_streams);
        sp.cb = cb;
        sp.n_absorbs = absorbs.len;
        sp.n_holds = holds.len;
        for (absorbs, 0..) |it, i| {
            const count: usize = if (it.rows) |rs| rs.len else it.features.rows;
            var rec: SpecAbsorb = .{ .stream = it.stream, .start = it.start, .count = @intCast(count), .has_rows = it.rows != null };
            if (it.rows) |rs| @memcpy(rec.rows[0..rs.len], rs);
            sp.absorbs[i] = rec;
        }
        for (holds, 0..) |it, i| {
            sp.holds[i] = .{ .stream = it.stream, .pending = it.pending, .anchor = it.position - 1 };
        }
    }

    pub fn matches(sp: *const SpecState, absorbs: []const lanes.drafter.Absorb, holds: []const lanes.drafter.Hold) bool {
        if (absorbs.len != sp.n_absorbs or holds.len != sp.n_holds) return false;
        for (absorbs, sp.absorbs[0..sp.n_absorbs]) |it, want| {
            if (it.stream != want.stream or it.start != want.start or it.prompt) return false;
            const count: usize = if (it.rows) |rs| rs.len else it.features.rows;
            if (count != want.count) return false;
            if (it.rows) |rs| {
                if (want.has_rows) {
                    if (!std.mem.eql(u32, rs, want.rows[0..want.count])) return false;
                } else {
                    for (rs, 0..) |rv, j| if (rv != j) return false;
                }
            } else if (want.has_rows) {
                for (want.rows[0..want.count], 0..) |rv, j| if (rv != j) return false;
            }
        }
        for (holds, sp.holds[0..sp.n_holds]) |it, want| {
            if (it.stream != want.stream or it.pending != want.pending or it.position == 0 or it.position - 1 != want.anchor or it.lanes != null) return false;
        }
        return true;
    }

    pub fn clear(sp: *SpecState) void {
        sp.cb = null;
        sp.n_absorbs = 0;
        sp.n_holds = 0;
    }
};

test "SpecState matches exact prefix and non-prefix tree passes and rejects cuts or token shifts" {
    const gpa = std.testing.allocator;
    var s = try lanes.Stream.init(gpa, .{ .id = "sp", .prompt = &.{ 1, 2, 3 }, .max_new = 16 });
    defer s.deinit(gpa);
    const feat: lanes.backend.Features = .{ .buffer = 0x1000, .rows = 6, .row_bytes = 64, .space = .device, .dtype = .bf16 };
    const rows = [_]u32{ 0, 2, 5 };
    const abs = [_]lanes.drafter.Absorb{.{ .stream = &s, .features = feat, .start = 10, .follow = &.{ 4, 5, 6 }, .rows = &rows, .prompt = false }};
    const hld = [_]lanes.drafter.Hold{.{ .stream = &s, .pending = 6, .position = 14, .depth = 15 }};
    var sp: SpecState = .{};
    sp.record(null, &abs, &hld);
    try std.testing.expect(sp.matches(&abs, &hld));

    // Different bonus token -> miss.
    var bad_hld = hld;
    bad_hld[0].pending = 7;
    try std.testing.expect(!sp.matches(&abs, &bad_hld));

    // Shorter cut path -> miss.
    var cut_abs = abs;
    cut_abs[0].rows = rows[0..2];
    try std.testing.expect(!sp.matches(&cut_abs, &hld));

    // Different non-prefix branch of same length -> miss.
    const other_rows = [_]u32{ 0, 1, 4 };
    var br_abs = abs;
    br_abs[0].rows = &other_rows;
    try std.testing.expect(!sp.matches(&br_abs, &hld));

    // Empty holds (e.g. depth=0) -> miss.
    try std.testing.expect(!sp.matches(&abs, &.{}));
}
