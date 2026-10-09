//! costs.measure: what a verify window and a draft level cost on this GPU, timed over consecutive tokens of real text.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const Engine = @import("cuda_engine.zig").Engine;
const Head = @import("cuda_mtp.zig").Head;
const state = @import("cuda_state.zig");

const text = "The river had been rising for three days, and by the time the ferry stopped running the town had moved its " ++
    "market up the hill. Children carried baskets of apples past the church while their parents argued about " ++
    "whether the old bridge would hold. In the workshop behind the bakery, a carpenter measured each plank " ++
    "twice, wrote the numbers on the wall, and cut slowly.\n\ndef mean(values):\n    total = 0\n    for v in " ++
    "values:\n        total += v\n    return total / len(values)\n";

/// Timed runs a width after one untimed: noise only adds time, so each width keeps its fastest.
const reps = 7;
/// A width timed this far under the one before it is noise: both are timed again.
const dip = 0.03;
/// A table this far from the last one measured on this GPU and shape (any build) is timed a second time.
const drift = 0.15;
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

/// What the timings depend on beside the build: this GPU and driver, the model's shape, the window and the glue kernels.
fn keyParts(e: *Engine, name: []u8, shape: []u8) ![3][]const u8 {
    const c = e.c;
    const gpu = e.ctx.name(name) catch "gpu";
    const at = try std.fmt.bufPrint(shape, "sm{d} cuda{d} {d}/{d}/{d}/{d}/{d} draft{d} {s}", .{ try e.ctx.capability(), try e.ctx.d.version(), c.layers, c.hidden, c.vocab, c.experts, e.max_len, e.w.draft_count, if (e.k.triton != null) "captured" else "own" });
    return .{ "nemotron-cuda", gpu, at };
}

/// The cache key of the last table measured for this GPU and shape by any build: the drift check's reference.
fn referenceKey(parts: []const []const u8) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("window costs reference\n");
    for (parts) |p| {
        hash.update(p);
        hash.update("\n");
    }
    var out: [32]u8 = undefined;
    hash.final(&out);
    return out;
}

/// True when any width or the level differs from `ref` by more than `drift`.
fn drifted(c: core.draft_depth.Costs, ref: core.draft_depth.Costs) bool {
    if (ref.rows != c.rows or @abs(c.level - ref.level) > drift * ref.level) return true;
    for (1..c.rows + 1) |r| if (@abs(c.verify[r] - ref.verify[r]) > drift * ref.verify[r]) return true;
    return false;
}

/// Windows of 1..16 rows and a head level's ms (fastest of 7), timed once per build, GPU and model shape: widths, never bits.
pub fn measure(gpa: std.mem.Allocator, io: std.Io, e: *Engine, h: *Head, model_dir: []const u8) !core.draft_depth.Costs {
    var name: [256]u8 = undefined;
    var shape: [192]u8 = undefined;
    const parts = try keyParts(e, &name, &shape);
    const k = lanes.cost_cache.key(gpa, io, &parts) catch null;
    if (k) |key| if (lanes.cost_cache.load(core.draft_depth.Costs, gpa, io, key)) |kept| return kept;
    const rk = referenceKey(&parts);
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

/// Times every width and the level into `raw` (`again`: keep the faster of this pass and the last), widths dipping under the one before timed again.
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
        fn time(e_: *Engine, t_: Timer, saved_: cuda.DeviceBuffer, cont_: []const u32, r: usize) !f64 {
            var best: f64 = std.math.inf(f64);
            for (0..reps + 1) |i| {
                try e_.b.restore(e_.ops(), saved_);
                e_.pos = max_rows;
                e_.parity = 0;
                e_.prev_keep = 0;
                _ = try e_.step(cont_[0], null);
                try e_.stream.synchronize();
                try t_.a.record(e_.stream);
                try e_.verify(cont_[1..][0..r], @intCast(r), null);
                const ms = try t_.ms(e_);
                if (i > 0) best = @min(best, ms);
            }
            return best;
        }
    };
    for (1..rows + 1) |r| {
        const ms = try W.time(e, t, saved, cont, r);
        raw.verify[r] = if (again) @min(raw.verify[r], ms) else ms;
    }
    if (!again) raw.verify[0] = 0;
    for (0..2) |_| {
        var steady = true;
        for (2..rows + 1) |r| if (raw.verify[r] < raw.verify[r - 1] * (1 - dip)) {
            steady = false;
            raw.verify[r - 1] = @min(raw.verify[r - 1], try W.time(e, t, saved, cont, r - 1));
            raw.verify[r] = @min(raw.verify[r], try W.time(e, t, saved, cont, r));
        };
        if (steady) break;
    }
    var chain: [2]f64 = undefined;
    for ([_]usize{ 1, 8 }, &chain) |levels, *out| {
        var best: f64 = std.math.inf(f64);
        for (0..reps + 1) |i| {
            try h.restore(head_saved, head_pos);
            try e.ops().copy(e.b.hidden, last_hidden, e.c.hidden * 2);
            try e.ops().fill32(e.b.sampled, pending, 1);
            try e.stream.synchronize();
            try t.a.record(e.stream);
            try h.begin(1);
            for (1..levels + 1) |j| try h.launch(@intCast(j));
            const ms = try t.ms(e);
            if (i > 0) best = @min(best, ms);
        }
        out.* = best;
    }
    const level = (chain[1] - chain[0]) / 7;
    raw.level = if (again) @min(raw.level, level) else level;
    try e.reset();
    try h.reset();
}
