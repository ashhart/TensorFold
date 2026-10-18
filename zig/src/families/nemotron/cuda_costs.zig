//! Time verify windows and draft levels over consecutive tokens of real text.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const Engine = @import("cuda_engine.zig").Engine;
const mtp = @import("cuda_mtp.zig");
const Head = mtp.Head;
const Vocab = mtp.Vocab;
const state = @import("cuda_state.zig");
const rule = lanes.cost_rule;
const profile_window = @import("cuda_profile_window.zig");
const regime = cuda.launch.few_rows; // windows past it take the router's and the lanes' wider kernels
const prefix = 16; // text tokens before the timed windows, at any window width: a window's ms depends on its tokens

// Windows hold this prefix's pending token, then the text's next tokens as drafts, after the head drafts as in a round.
const text = "The river had been rising for three days, and by the time the ferry stopped running the town had moved its " ++
    "market up the hill. Children carried baskets of apples past the church while their parents argued about " ++
    "whether the old bridge would hold. In the workshop behind the bakery, a carpenter measured each plank " ++
    "twice, wrote the numbers on the wall, and cut slowly.\n\ndef mean(values):\n    total = 0\n    for v in " ++
    "values:\n        total += v\n    return total / len(values)\n";

const Timer = struct {
    a: cuda.Event,
    b: cuda.Event,

    fn ms(t: Timer, e: *Engine) !f64 {
        try t.b.record(e.stream);
        try t.b.synchronize();
        return @floatCast(try cuda.Event.elapsedMs(t.a, t.b));
    }
};

/// Cache identity: GPU and driver, model shape, window and glue kernels and the draft head, alongside the build.
fn keyParts(e: *Engine, h: *Head, v: Vocab, name: []u8, shape: []u8) ![3][]const u8 {
    const c = e.c;
    const gpu = e.ctx.name(name) catch "gpu";
    const at = try std.fmt.bufPrint(shape, "sm{d} cuda{d} {d}/{d}/{d}/{d}/{d} draft{d} {s}{s}", .{ try e.ctx.capability(), try e.ctx.d.version(), c.layers, c.hidden, c.vocab, c.experts, e.max_len, h.width(v), if (e.k.triton != null) "captured" else "own", if (c.format == .modelopt) " modelopt" else "" });
    return .{ "nemotron-cuda", gpu, at };
}

/// True when any width or the level differs from `ref` by more than `drift`.
fn drifted(c: core.draft_depth.Costs, ref: core.draft_depth.Costs) bool {
    if (ref.rows != c.rows or rule.changed(c.level, ref.level)) return true;
    for (1..c.rows + 1) |r| if (rule.changed(c.verify[r], ref.verify[r])) return true;
    return false;
}

/// Each head's costs: the draft head's, and the whole head's where the checkpoint drafts with both.
pub const Measured = struct { subset: core.draft_depth.Costs, full: ?core.draft_depth.Costs };

/// Fastest-of-seven costs by width and each head's level, cached per head; one pass times the windows for every head.
pub fn measure(gpa: std.mem.Allocator, io: std.Io, e: *Engine, h: *Head, model_dir: []const u8) !Measured {
    const all = [_]Vocab{ .subset, .full };
    const heads = all[0..if (e.w.draft_coverage != null) 2 else 1];
    var name: [2][256]u8 = undefined;
    var shape: [2][192]u8 = undefined;
    var parts: [2][3][]const u8 = undefined;
    var keys: [2]?[32]u8 = undefined;
    var kept: [2]?core.draft_depth.Costs = @splat(null);
    for (heads, 0..) |v, i| {
        parts[i] = try keyParts(e, h, v, &name[i], &shape[i]);
        keys[i] = lanes.cost_cache.key(gpa, io, &parts[i]) catch null;
        if (keys[i]) |key| kept[i] = lanes.cost_cache.load(core.draft_depth.Costs, gpa, io, key);
    }
    for (kept[0..heads.len]) |k| {
        if (k == null) break;
    } else return .{ .subset = kept[0].?, .full = kept[1] };
    var raw: Raw = undefined;
    try timeAll(gpa, io, e, h, model_dir, heads, &raw, false);
    var costs: [2]core.draft_depth.Costs = undefined;
    tables(e, raw, heads, &costs);
    for (0..heads.len) |i| {
        const ref = lanes.cost_cache.load(core.draft_depth.Costs, gpa, io, lanes.cost_cache.cudaReferenceKey(&parts[i])) orelse continue;
        if (!drifted(costs[i], ref)) continue;
        try timeAll(gpa, io, e, h, model_dir, heads, &raw, true);
        tables(e, raw, heads, &costs);
        break;
    }
    for (0..heads.len) |i| {
        if (keys[i]) |key| lanes.cost_cache.save(core.draft_depth.Costs, gpa, io, key, costs[i]);
        lanes.cost_cache.save(core.draft_depth.Costs, gpa, io, lanes.cost_cache.cudaReferenceKey(&parts[i]), costs[i]);
    }
    return .{ .subset = costs[0], .full = if (heads.len > 1) costs[1] else null };
}

/// Each head's table: the windows' widths with its level.
fn tables(e: *Engine, raw: Raw, heads: []const Vocab, out: []core.draft_depth.Costs) void {
    const widths = state.windowRows(e.c) + 1;
    for (heads, out[0..heads.len]) |v, *c| c.* = core.draft_depth.Costs.measured(raw.verify[0..widths], raw.level[@backingInt(v)], regime);
}

/// Fastest-run window ms by width (index 0 unused; widths to the engine's window) and a level's ms by head.
const Raw = struct { verify: [state.max_rows + 1]f64, level: [2]f64 };

/// Time widths with each head's levels interleaved, retime dips, and retain faster values on a retry.
fn timeAll(gpa: std.mem.Allocator, io: std.Io, e: *Engine, h: *Head, model_dir: []const u8, heads: []const Vocab, raw: *Raw, again: bool) !void {
    const path = try std.fs.path.join(gpa, &.{ model_dir, "tokenizer.json" });
    defer gpa.free(path);
    var tok = try core.tokenizer.loadTokenizer(io, gpa, path);
    defer tok.deinit();
    const ids = try tok.encode(gpa, text);
    defer gpa.free(ids);
    const rows = state.windowRows(e.c);
    if (ids.len < prefix + rows) return error.CostTextTooShort;
    const pending = try e.prefill(ids[0..prefix], null, h);
    const last_hidden = e.b.p_hidden + (prefix - 1) * @as(u64, e.c.hidden) * 2;
    var saved = try e.b.snapshot(e.ops());
    defer saved.free();
    var head_saved = try h.snapshot();
    defer head_saved.free();
    const head_pos = h.pos;
    var last_saved = try cuda.DeviceBuffer.alloc(e.ctx.d, e.c.hidden * 2);
    defer last_saved.free();
    try e.ops().copy(last_saved.ptr, last_hidden, e.c.hidden * 2);
    const cont = ids[prefix..];
    var next = try cuda.DeviceBuffer.alloc(e.ctx.d, rows * 4); // the text's own next tokens: the window's drafts
    defer next.free();
    try e.ops().upload(next.ptr, std.mem.sliceAsBytes(cont[0..rows]));
    var t: Timer = .{ .a = try cuda.Event.init(e.ctx.d, true), .b = try cuda.Event.init(e.ctx.d, true) };
    defer t.a.deinit();
    defer t.b.deinit();
    for (0..32) |_| _ = try e.step(cont[0], null); // the GPU at its working clocks before any window is timed
    const W = struct {
        e: *Engine,
        h: *Head,
        t: Timer,
        saved: cuda.DeviceBuffer,
        head_saved: cuda.DeviceBuffer,
        head_pos: usize,
        last: u64,
        pending: u32,
        continuation: u64,

        pub fn restore(w: *@This()) !void {
            try w.e.b.restore(w.e.ops(), w.saved);
            w.e.pos = prefix;
            w.e.parity = 0;
            w.e.prev_keep = 0;
        }
        pub fn restoreHead(w: *@This()) !void {
            try w.h.restore(w.head_saved, w.head_pos);
            try w.e.ops().copy(w.e.b.hidden, w.last, w.e.c.hidden * 2);
            try w.e.ops().fill32(w.e.b.sampled, w.pending, 1);
        }
        pub fn draft(w: *@This(), n: usize) !void {
            try w.h.begin(1);
            for (1..n + 1) |j| try w.h.launch(@intCast(j), .subset);
        }
        pub fn follow(w: *@This(), n: usize) !void {
            try w.e.ops().copy(w.e.b.ids + 4, w.continuation, n * 4);
        }
        pub fn sync(w: *@This()) !void {
            try w.e.stream.synchronize();
        }
        pub fn start(w: *@This()) !void {
            try w.t.a.record(w.e.stream);
        }
        pub fn verify(w: *@This(), width: usize) !void {
            try w.e.verify(&.{w.pending}, @intCast(width), null);
        }
        pub fn elapsed(w: *@This()) !f64 {
            return w.t.ms(w.e);
        }
        pub fn time(w: @This(), i: usize) !f64 {
            var window = w;
            return profile_window.time(&window, i + 1);
        }
    };
    const L = struct {
        e: *Engine,
        h: *Head,
        t: Timer,
        saved: cuda.DeviceBuffer,
        pos: usize,
        last: u64,
        pending: u32,
        vocab: Vocab,

        pub fn time(l: @This(), levels: usize) !f64 {
            try l.h.restore(l.saved, l.pos);
            try l.e.ops().copy(l.e.b.hidden, l.last, l.e.c.hidden * 2);
            try l.e.ops().fill32(l.e.b.sampled, l.pending, 1);
            try l.e.stream.synchronize();
            try l.t.a.record(l.e.stream);
            try l.h.begin(1);
            for (1..levels + 1) |j| try l.h.launch(@intCast(j), l.vocab);
            return l.t.ms(l.e);
        }
    };
    // each head samples its chains between the same widths, as one head alone would
    const Heads = struct {
        chains: []rule.ChainLevel(L),

        pub fn afterWidth(x: *@This(), i: usize) !void {
            for (x.chains) |*c| try c.afterWidth(i);
        }
    };
    const windows: W = .{ .e = e, .h = h, .t = t, .saved = saved, .head_saved = head_saved, .head_pos = head_pos, .last = last_saved.ptr, .pending = pending, .continuation = next.ptr };
    var chains: [2]rule.ChainLevel(L) = undefined;
    for (heads, chains[0..heads.len]) |v, *c| c.* = .{ .timer = .{ .e = e, .h = h, .t = t, .saved = head_saved, .pos = head_pos, .last = last_hidden, .pending = pending, .vocab = v } };
    var observer: Heads = .{ .chains = chains[0..heads.len] };
    try rule.measureWidths(windows, &observer, raw.verify[1 .. rows + 1], again);
    if (!again) raw.verify[0] = 0;
    try rule.smooth(windows, raw.verify[1 .. rows + 1]);
    for (heads, chains[0..heads.len]) |v, c| {
        const level = &raw.level[@backingInt(v)];
        level.* = if (again) @min(level.*, c.level()) else c.level();
    }
    try e.reset();
    try h.reset();
}
