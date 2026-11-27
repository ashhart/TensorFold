//! GLM profiler lifetime operation used by unwind and close.
const std = @import("std");
const Engine = @import("cuda_engine.zig").Engine;
pub fn freeProfs(e: *Engine) void {
    for (&e.profs) |*p| {
        if (p.*) |v| v.deinit();
        p.* = null;
    }
    for (&e.prompt_profs) |*p| {
        if (p.*) |v| v.deinit();
        p.* = null;
    }
}

pub fn heatRoot() []const u8 {
    const s = (@import("sys").environment.Process{}).get("TF_HEAT_ROOT") orelse return @import("heat").default_root;
    return if (s.len == 0) @import("heat").default_root else s;
}

pub fn verifyLabel(rows: usize) []const u8 {
    const l = [_][]const u8{ "verify 1 row", "verify 2 rows", "verify 3 rows", "verify 4 rows", "verify 5 rows", "verify 6 rows", "verify 7 rows", "verify 8 rows" };
    return l[@min(rows, l.len) - 1];
}

pub fn setup(e: *Engine, layers: usize) !void {
    const prof_mod = @import("cuda_prof.zig");
    const d = e.d;
    const io = e.io;
    const profiled = try prof_mod.Prof.wanted();
    if (profiled) for (&e.profs) |*p| {
        p.* = try prof_mod.Prof.init(d, io);
    };
    if (profiled) for (&e.prompt_profs, 0..) |*p, bucket| {
        p.* = try prof_mod.Prof.init(d, io);
        p.*.?.every = 4;
        const started = std.Io.Clock.awake.now(io).toNanoseconds();
        try p.*.?.initPrompt(layers, e.pieces.count);
        const create_ms = prof_mod.Prof.elapsedMs(io, started);
        const pairs = prof_mod.Prof.promptCapacity(layers, e.pieces.count);
        const bucket_rows = ([_]usize{ 64, 256, 1024, 2048 })[bucket];
        std.log.info("glm prof startup rank={d} bucket_rows={d} layers={d} pieces={d} pairs={d} events={d} create_ms={d:.4}", .{ e.share.rank, bucket_rows, layers, e.pieces.count, pairs, 2 * pairs, create_ms });
    };
    if (profiled) if (e.heat_gate.bands) |bands| {
        std.log.info("glm prof startup rank={d} heat=on high_c={d:.1} low_c={d:.1} root={s}", .{ e.share.rank, bands.high_c, bands.low_c, e.heat_gate.root });
    } else {
        std.log.info("glm prof startup rank={d} heat=off", .{e.share.rank});
    };
}

pub fn ledger(e: *Engine) !void {
    const prefix = e.settings.prefill_trace orelse return;
    const plan = @import("cuda_prefill_plan.zig");
    const kinds = try e.gpa.alloc(plan.Layer, e.w.layers.len);
    defer e.gpa.free(kinds);
    for (e.w.layers, kinds) |l, *k| k.* = .{
        .kind = if (l.mixer == .kda) .kda else if (e.c.dsa) .dsa else .mla,
        .ffn = if (l.ffn == .dense) .dense else .moe,
    };
    e.prefill_trace = try @import("cuda_prefill_trace.zig").Trace.init(e.d, e.io, prefix, e.share.rank, e.share.world, kinds);
}
