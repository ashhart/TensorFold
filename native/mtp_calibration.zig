//! Measure native target windows and one MTP step without changing request state.
const std = @import("std");
const mx = @import("mlx.zig");
const Stopwatch = @import("vendor/io_util.zig").Stopwatch;
const Adaptive = @import("draft_depth.zig").Adaptive;

pub fn measure(comptime M: type, m: *M, io: std.Io, policy: *Adaptive, settings: @import("sampling.zig").Sampling) !void {
    defer m.reset();
    var cache = M.DraftCache{};
    defer cache.deinit();
    var hidden = mx.empty;
    defer mx.free(hidden);
    var ids: [16]i32 = undefined;
    for (&ids, 0..) |*id, j| id.* = @intCast(1000 + j * 37);
    for (0..3) |_| {
        var p = try m.forward(&ids);
        defer p.deinit();
        for (ids, 0..) |id, j| {
            if (hidden.ctx != null) _ = try m.draftStep(&p.scope, hidden, id, &cache);
            try mx.replace(&hidden, try p.scope.slice(p.hidden, 0, @intCast(j), @intCast(j + 1)));
        }
        try m.commit(&p, 16);
    }
    for (1..policy.budget + 2) |width| {
        var best = std.math.inf(f64);
        // First launch compiles the width; only subsequent completed passes are timed.
        for (0..4) |attempt| {
            const timer = Stopwatch.init(io);
            var p = try m.forward(ids[0..width]);
            defer p.deinit();
            if (attempt > 0) best = @min(best, @as(f64, @floatFromInt(timer.read())) / 1e6);
        }
        policy.forward_ms[width] = best;
    }
    var best = std.math.inf(f64);
    for (0..6) |attempt| {
        var s = mx.Scope{};
        defer s.deinit();
        const timer = Stopwatch.init(io);
        var next = try cache.clone();
        defer next.deinit();
        const h = try m.draftStepArray(&s, hidden, try s.ints(&.{77}), &next, true);
        const picked = try @import("sampling.zig").rowsMapped(&m.kernels, &s, try m.draftHead(&s, h), &.{m.position + 1}, settings, m.weights.arrays.get("draft_ids"));
        defer mx.allocator.free(picked);
        if (attempt > 0) best = @min(best, @as(f64, @floatFromInt(timer.read())) / 1e6);
    }
    policy.mtp_ms = best;
    std.debug.print("Calibrated target widths 1..{d}, MTP step {d:.3}ms\n", .{ policy.budget + 1, best });
}
