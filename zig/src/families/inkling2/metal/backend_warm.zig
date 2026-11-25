//! Startup warmup and batch cost measurement helpers for Inkling 2 Mini Metal backend.
const std = @import("std");
const builtin = @import("builtin");
const mtl = @import("metal");
const st = @import("state.zig");
const fwd = @import("forward.zig");

pub const shared_widths = [_]usize{ 24, 32, 48, 64 };

fn now() f64 {
    if (comptime !builtin.os.tag.isDarwin()) return 0;
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1000.0 + @as(f64, @floatFromInt(ts.nsec)) * 1e-6;
}

pub fn warm(self: anytype) !void {
    try self.settleCommit();
    var cache = try st.Cache.init(self.model.device, st.prompt_rows + st.window_rows + 4, &self.model.weight_plan);
    defer cache.deinit();
    var ids: [st.prompt_rows]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i * 37 % 20000);
    try self.feed(&cache, &ids, null, null);
    const tree_parents = [_]i32{ -1, 0, 0 };
    try cache.beginTree(3, &tree_parents);
    errdefer cache.rollback() catch {};
    _ = self.nextVerifyGeneration();
    try fwd.runShared(self.model, &self.scratch, &.{.{ .cache = &cache, .rows = 3, .parents = &tree_parents }}, ids[0..3], .{ .all = true });
    try cache.commitPath(self.model.queue, &.{ 0, 2 }, &self.model.tree);
    _ = self.nextVerifyGeneration();
    try fwd.run(self.model, &self.scratch, &cache, ids[0..st.window_rows], .{ .all = true });
    // A window of each width after that prompt, the second of two runs timed.
    for (1..st.window_rows + 1) |rows| {
        var ms: f64 = 0;
        for (0..2) |attempt| {
            const t0 = now();
            try fwd.run(self.model, &self.scratch, &cache, ids[0..rows], .{ .all = true });
            if (attempt == 1) ms = now() - t0;
        }
        self.costs[rows - 1] = .{ .width = @intCast(rows), .ms = ms };
    }
    self.timed = st.window_rows;
}

/// Times 1..window_rows if not timed yet and 4-stream shared batches up to batch_rows.
pub fn measure(self: anytype) !void {
    try self.settleCommit();
    _ = self.nextVerifyGeneration();
    var ids: [st.batch_rows]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(1000 + i * 7919 % 50000);
    const cap: usize = if (self.options.streams > 1) 4 else 1;
    var caches: [4]st.Cache = undefined;
    var made: usize = 0;
    defer for (caches[0..made]) |*c| c.deinit();
    for (caches[0..cap]) |*c| {
        c.* = try st.Cache.init(self.model.device, 64, &self.model.weight_plan);
        made += 1;
    }
    if (self.timed == 0) {
        for (1..st.window_rows + 1) |rows| {
            var ms: f64 = 0;
            for (0..2) |attempt| {
                caches[0].len = 0;
                const t0 = now();
                try fwd.run(self.model, &self.scratch, &caches[0], ids[0..rows], .{ .all = true });
                if (attempt == 1) ms = now() - t0;
            }
            self.costs[rows - 1] = .{ .width = @intCast(rows), .ms = ms };
        }
        self.timed = st.window_rows;
    }
    self.shared_timed = 0;
    if (self.options.streams > 1) {
        for (shared_widths) |rows| {
            if (rows > st.batch_rows) continue;
            const seg_rows = rows / 4;
            const segs = [_]fwd.Segment{
                .{ .cache = &caches[0], .rows = seg_rows },
                .{ .cache = &caches[1], .rows = seg_rows },
                .{ .cache = &caches[2], .rows = seg_rows },
                .{ .cache = &caches[3], .rows = seg_rows },
            };
            var ms: f64 = 0;
            for (0..2) |attempt| {
                for (caches[0..4]) |*c| c.len = 0;
                const t0 = now();
                try fwd.runShared(self.model, &self.scratch, &segs, ids[0..rows], .{ .all = true });
                if (attempt == 1) ms = now() - t0;
            }
            self.shared[self.shared_timed] = .{ .width = @intCast(rows), .ms = ms };
            self.shared_timed += 1;
        }
    }
}
