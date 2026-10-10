//! Time verify windows and draft levels over consecutive tokens of real text.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const Engine = @import("cuda_engine.zig").Engine;
const Head = @import("cuda_mtp.zig").Head;
const state = @import("cuda_state.zig");
const rule = lanes.cost_rule;

const text = "The river had been rising for three days, and by the time the ferry stopped running the town had moved its " ++
    "market up the hill. Children carried baskets of apples past the church while their parents argued about " ++
    "whether the old bridge would hold. In the workshop behind the bakery, a carpenter measured each plank " ++
    "twice, wrote the numbers on the wall, and cut slowly.\n\ndef mean(values):\n    total = 0\n    for v in " ++
    "values:\n        total += v\n    return total / len(values)\n";

const max_rows = state.max_rows;

const Timer = struct {
    a: cuda.Event,
    b: cuda.Event,

    fn ms(t: Timer, e: *Engine) !f64 {
        try t.b.record(e.stream);
        try t.b.synchronize();
        return @floatCast(try cuda.Event.elapsedMs(t.a, t.b));
    }
};

/// Cache identity: GPU and driver, model shape, window and glue kernels, alongside the build.
fn keyParts(e: *Engine, name: []u8, shape: []u8) ![3][]const u8 {
    const c = e.c;
    const gpu = e.ctx.name(name) catch "gpu";
    const at = try std.fmt.bufPrint(shape, "sm{d} cuda{d} {d}/{d}/{d}/{d}/{d} draft{d} {s}{s}", .{ try e.ctx.capability(), try e.ctx.d.version(), c.layers, c.hidden, c.vocab, c.experts, e.max_len, e.w.draft_count, if (e.k.triton != null) "captured" else "own", if (c.format == .modelopt) " modelopt" else "" });
    return .{ "nemotron-cuda", gpu, at };
}

/// True when any width or the level differs from `ref` by more than `drift`.
fn drifted(c: core.draft_depth.Costs, ref: core.draft_depth.Costs) bool {
    if (ref.rows != c.rows or rule.changed(c.level, ref.level)) return true;
    for (1..c.rows + 1) |r| if (rule.changed(c.verify[r], ref.verify[r])) return true;
    return false;
}

/// Fastest-of-seven costs by width and head level, cached per build, GPU and model shape.
pub fn measure(gpa: std.mem.Allocator, io: std.Io, e: *Engine, h: *Head, model_dir: []const u8) !core.draft_depth.Costs {
    var name: [256]u8 = undefined;
    var shape: [192]u8 = undefined;
    const parts = try keyParts(e, &name, &shape);
    const k = lanes.cost_cache.key(gpa, io, &parts) catch null;
    if (k) |key| if (lanes.cost_cache.load(core.draft_depth.Costs, gpa, io, key)) |kept| return kept;
    const rk = lanes.cost_cache.cudaReferenceKey(&parts);
    var raw: Raw = undefined;
    try timeAll(gpa, io, e, h, model_dir, &raw, false);
    var costs = core.draft_depth.Costs.measured(&raw.verify, raw.level);
    if (lanes.cost_cache.load(core.draft_depth.Costs, gpa, io, rk)) |ref| if (drifted(costs, ref)) {
        try timeAll(gpa, io, e, h, model_dir, &raw, true);
        costs = core.draft_depth.Costs.measured(&raw.verify, raw.level);
    };
    if (k) |key| lanes.cost_cache.save(core.draft_depth.Costs, gpa, io, key, costs);
    lanes.cost_cache.save(core.draft_depth.Costs, gpa, io, rk, costs);
    return costs;
}

/// Fastest-run window ms by width (index 0 unused) and a head level's ms.
const Raw = struct { verify: [max_rows + 1]f64, level: f64 };

/// Time widths with interleaved levels, retime dips, and retain faster values on a retry.
fn timeAll(gpa: std.mem.Allocator, io: std.Io, e: *Engine, h: *Head, model_dir: []const u8, raw: *Raw, again: bool) !void {
    const path = try std.fs.path.join(gpa, &.{ model_dir, "tokenizer.json" });
    defer gpa.free(path);
    var tok = try core.tokenizer.loadTokenizer(io, gpa, path);
    defer tok.deinit();
    const ids = try tok.encode(gpa, text);
    defer gpa.free(ids);
    const rows = max_rows;
    if (ids.len < 2 * rows + 2) return error.CostTextTooShort;
    const pending = try e.prefill(ids[0..rows], null, h);
    const last_hidden = e.b.p_hidden + (rows - 1) * @as(u64, e.c.hidden) * 2;
    var saved = try e.b.snapshot(e.ops());
    defer saved.free();
    var head_saved = try h.snapshot();
    defer head_saved.free();
    const head_pos = h.pos;
    var t: Timer = .{ .a = try cuda.Event.init(e.ctx.d, true), .b = try cuda.Event.init(e.ctx.d, true) };
    defer t.a.deinit();
    defer t.b.deinit();
    const cont = ids[rows..];
    for (0..32) |_| _ = try e.step(cont[0], null); // the GPU at its working clocks before any window is timed
    const W = struct {
        e: *Engine,
        t: Timer,
        saved: cuda.DeviceBuffer,
        cont: []const u32,

        pub fn time(w: @This(), i: usize) !f64 {
            const r = i + 1;
            try w.e.b.restore(w.e.ops(), w.saved);
            w.e.pos = max_rows;
            w.e.parity = 0;
            w.e.prev_keep = 0;
            _ = try w.e.step(w.cont[0], null);
            try w.e.stream.synchronize();
            try w.t.a.record(w.e.stream);
            try w.e.verify(w.cont[1..][0..r], @intCast(r), null);
            return w.t.ms(w.e);
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

        pub fn time(l: @This(), levels: usize) !f64 {
            try l.h.restore(l.saved, l.pos);
            try l.e.ops().copy(l.e.b.hidden, l.last, l.e.c.hidden * 2);
            try l.e.ops().fill32(l.e.b.sampled, l.pending, 1);
            try l.e.stream.synchronize();
            try l.t.a.record(l.e.stream);
            try l.h.begin(1);
            for (1..levels + 1) |j| try l.h.launch(@intCast(j));
            return l.t.ms(l.e);
        }
    };
    const windows: W = .{ .e = e, .t = t, .saved = saved, .cont = cont };
    var chains: rule.ChainLevel(L) = .{ .timer = .{ .e = e, .h = h, .t = t, .saved = head_saved, .pos = head_pos, .last = last_hidden, .pending = pending } };
    try rule.measureWidths(windows, &chains, raw.verify[1..], again);
    if (!again) raw.verify[0] = 0;
    try rule.smooth(windows, raw.verify[1..]);
    const level = chains.level();
    raw.level = if (again) @min(raw.level, level) else level;
    try e.reset();
    try h.reset();
}
