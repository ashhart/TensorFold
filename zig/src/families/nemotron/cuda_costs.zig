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

const reps = 5;

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return if (xs.len % 2 == 1) xs[xs.len / 2] else (xs[xs.len / 2 - 1] + xs[xs.len / 2]) / 2;
}

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

/// Windows of 1..16 rows and a head level's ms (median of 5), timed once per build, GPU and model shape: widths, never bits.
pub fn measure(gpa: std.mem.Allocator, io: std.Io, e: *Engine, h: *Head, model_dir: []const u8) !core.draft_depth.Costs {
    var name: [256]u8 = undefined;
    var shape: [192]u8 = undefined;
    const parts = try keyParts(e, &name, &shape);
    const k = lanes.cost_cache.key(gpa, io, &parts) catch null;
    if (k) |key| if (lanes.cost_cache.load(core.draft_depth.Costs, gpa, io, key)) |kept| return kept;
    const path = try std.fs.path.join(gpa, &.{ model_dir, "tokenizer.json" });
    defer gpa.free(path);
    var tok = try core.tokenizer.loadTokenizer(io, gpa, path);
    defer tok.deinit();
    const ids = try tok.encode(gpa, text);
    defer gpa.free(ids);
    const rows = state.max_rows;
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
    var verify: [rows + 1]f64 = @splat(0);
    var times: [reps + 1]f64 = undefined;
    for (1..rows + 1) |r| {
        for (&times) |*x| {
            try e.b.restore(e.ops(), saved);
            e.pos = rows;
            e.parity = 0;
            e.prev_keep = 0;
            _ = try e.step(cont[0], null);
            try e.stream.synchronize();
            try t.a.record(e.stream);
            try e.verify(cont[1..][0..r], @intCast(r), null);
            x.* = try t.ms(e);
        }
        verify[r] = median(times[1..]);
    }
    var chain: [2]f64 = undefined;
    for ([_]usize{ 1, 8 }, &chain) |levels, *out| {
        for (&times) |*x| {
            try h.restore(head_saved, head_pos);
            try e.ops().copy(e.b.hidden, last_hidden, e.c.hidden * 2);
            try e.ops().fill32(e.b.sampled, pending, 1);
            try e.stream.synchronize();
            try t.a.record(e.stream);
            try h.begin(1);
            for (1..levels + 1) |j| try h.launch(@intCast(j));
            x.* = try t.ms(e);
        }
        out.* = median(times[1..]);
    }
    try e.reset();
    try h.reset();
    const costs = core.draft_depth.Costs.measured(&verify, (chain[1] - chain[0]) / 7);
    if (k) |key| lanes.cost_cache.save(core.draft_depth.Costs, gpa, io, key, costs);
    return costs;
}
