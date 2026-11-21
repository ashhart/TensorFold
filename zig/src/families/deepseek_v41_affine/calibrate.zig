//! Time the actual affine verifier widths for the lane scheduler.
const std = @import("std");
const c = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const backend = @import("backend.zig");
const Metal = backend.Metal;
const max_streams = backend.max_streams;
const shared_rows = backend.shared_rows;

pub fn measure(self: *Metal) !void {
    var cache = try st.Cache.initWith(self.gpa, self.model.device, 512, true);
    defer cache.deinit();
    var ids: [256]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast((i * 7919 + 13) % c.vocab);
    try fwd.run(self.model, &self.scratch, &cache, &ids, .last, null);
    const base = cache.len;
    var held: [c.draft_block]u32 = undefined;
    for (&self.costs, 1..) |*cost, w| {
        var best: f64 = std.math.floatMax(f64);
        for (0..3) |_| {
            const t0 = now();
            try fwd.run(self.model, &self.scratch, &cache, ids[0..w], .all, null);
            best = @min(best, now() - t0);
            cache.len = base;
        }
        cost.* = .{ .width = @intCast(w), .ms = best };
    }
    var best: f64 = std.math.floatMax(f64);
    for (0..3) |_| {
        const t0 = now();
        try fwd.draft(self.model, &self.drafts.?, &cache, ids[0], base - 1, &held);
        best = @min(best, now() - t0);
    }
    self.draft_ms = best;
    var more: [max_streams - 1]st.Cache = undefined;
    var made: usize = 0;
    defer for (more[0..made]) |*m| m.deinit();
    for (&more) |*m| {
        m.* = try st.Cache.initWith(self.gpa, self.model.device, 512, true);
        made += 1;
        try fwd.run(self.model, &self.scratch, m, &ids, .last, null);
    }
    const caches = [max_streams]*st.Cache{ &cache, &more[0], &more[1], &more[2] };
    for (&self.shared, shared_rows) |*cost, total| {
        const each = total / max_streams;
        var segs: [max_streams]fwd.Seg = undefined;
        for (&segs, caches, 0..) |*g, cc, k| g.* = .{ .cache = cc, .tokens = ids[0..each], .r0 = k * each };
        best = std.math.floatMax(f64);
        for (0..3) |_| {
            const t0 = now();
            try fwd.many(self.model, &self.scratch, &segs, .all);
            best = @min(best, now() - t0);
            for (caches) |cc| cc.len = base;
        }
        cost.* = .{ .width = @intCast(total), .ms = best };
    }
    std.log.info("deepseek_v41: window ms by width 1 {d:.1}, 2 {d:.1}, 4 {d:.1}, 6 {d:.1}; shared over 4 streams 8 rows {d:.1}, 16 {d:.1}, 24 {d:.1}; a draft round {d:.1} ms", .{ self.costs[0].ms, self.costs[1].ms, self.costs[3].ms, self.costs[5].ms, self.shared[0].ms, self.shared[2].ms, self.shared[3].ms, self.draft_ms });
}

fn now() f64 {
    var t: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t);
    return @as(f64, @floatFromInt(t.sec)) * 1e3 + @as(f64, @floatFromInt(t.nsec)) / 1e6;
}
