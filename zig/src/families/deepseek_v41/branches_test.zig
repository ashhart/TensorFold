//! TF_DSV41_BRANCHES / TF_DSV41_MHC_DEFER on the real decode emitter: same launches, order, marks and shared roles.

const std = @import("std");
const calls = @import("block_calls.zig");
const br = @import("branches.zig");

const testing = std.testing;
const block = @import("block.zig");
const check = @import("calls_check.zig");
const rowmode = @import("rowmode.zig");
const Config = @import("config.zig").Config;

test "TF_DSV41_BRANCHES on the decode emitter: the same launches in the same order, the plan's marks, no role shared across the streams but read-only inputs" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    var seen: std.StringArrayHashMapUnmanaged(void) = .empty;
    var dseen: std.StringArrayHashMapUnmanaged(void) = .empty;
    for ([_]bool{ false, true }) |r1| for ([_]i64{ 1, 5, 16, 24, 25 }) |n| {
        const o: block.Options = .{ .limit = 8192, .r1 = r1, .expert_topp = 0.85 };
        var ob = o;
        ob.branches = true;
        ob.mhc_defer = r1;
        ob.branch_rows = 24; // the 25-row window stays on one stream
        const off = try block.emit(a, &cfg, &w, o, &backbone, n, 4000, true);
        const on = try block.emit(a, &cfg, &w, ob, &backbone, n, 4000, true);
        try testing.expectEqual(off.len, on.len);
        var sides: usize = 0;
        var forks: usize = 0;
        var joins: usize = 0;
        for (off, on) |x, y| {
            try testing.expectEqualStrings(x.name, y.name);
            try testing.expectEqual(x.args.len, y.args.len);
            try testing.expect(!x.side and !x.fork and !x.join and !x.defer_side and !x.defer_join);
            sides += @intFromBool(y.side);
            forks += @intFromBool(y.fork);
            joins += @intFromBool(y.join);
        }
        // MHC_DEFER: every coefficient launch deferred and joined by the next mHC call, at any window size
        var deferred: usize = 0;
        for (on) |y| {
            if (y.defer_side) {
                deferred += 1;
                try testing.expectEqualStrings("tf_dsv41_mhc_cuda_v1.coef", y.name);
            }
        }
        // the coefficient launch is mhc_cuda's (<= 16 rows; wider windows take Triton's _site, block_wide.zig)
        if (r1 and n <= 16) try testing.expect(deferred >= 2 * backbone.len) else try testing.expectEqual(@as(usize, 0), deferred);
        for (try br.sharedDeferred(a, on)) |role| try dseen.put(a, role, {});
        if (n > ob.branch_rows) {
            try testing.expectEqual(@as(usize, 0), sides + forks + joins);
            continue;
        }
        // a join a layer; index layers fork twice, the others once
        try testing.expectEqual(@as(usize, backbone.len), joins);
        try testing.expect(forks > joins and sides > 2 * joins);
        for (try br.shared(a, on)) |role| try seen.put(a, role, {});
    };
    // read only inside the region (rope, positions, mHC input, q low-rank rows); a new role here fails until reviewed
    const read_only = [_][]const u8{ "s.rope.main", "s.rope.comp", "w.pos", "w.pos64", "w.out", "L.qa", "L.qn" };
    for (seen.keys()) |k| {
        const ok = for (read_only) |r| {
            if (std.mem.eql(u8, r, k)) break true;
        } else false;
        if (!ok) std.debug.print("branches: role {s} touched by both streams in a fork region\n", .{k});
        try testing.expect(ok);
    }
    try testing.expectEqual(@as(usize, 0), dseen.count()); // the deferred coefficients share nothing before their join
}
