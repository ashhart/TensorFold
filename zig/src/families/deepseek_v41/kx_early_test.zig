//! TF_DSV41_KV_SPLIT_EARLY on the decode emitter: the same calls, each exchange on the side before the q group.

const std = @import("std");
const testing = std.testing;
const block = @import("block.zig");
const br = @import("branches.zig");
const calls = @import("block_calls.zig");
const check = @import("calls_check.zig");
const Config = @import("config.zig").Config;

fn key(a: std.mem.Allocator, c: calls.Call) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, c.name);
    for (c.args) |x| switch (x.arg) {
        .t => |t| try out.print(a, " {s}@{d}", .{ switch (t.role) {
            .weight, .buf => |r| r,
            .empty => "-",
        }, t.offset }),
        else => try out.print(a, " {any}", .{x.arg}),
    };
    return out.items;
}

test "TF_DSV41_KV_SPLIT_EARLY: the same calls, each index layer's exchange on the side before the q group" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 8192;
    const pages = @divExact(limit, block.Pool.page);
    for ([_]bool{ false, true }) |rows| for ([_]i64{ 1, 6, 16, 24 }) |n| {
        var o: block.Options = .{ .limit = limit, .r1 = true, .expert_topp = 0.85, .branches = true, .mhc_defer = true, .rows = rows };
        o.pool = .{ .comp_pages = @divExact(pages, 2) + 2, .ik_pages = pages + 1, .pts = pages, .split = true };
        var oe = o;
        oe.pool.?.early = true;
        const off = try block.emit(a, &cfg, &w, o, &backbone, n, 4000, true);
        const on = try block.emit(a, &cfg, &w, oe, &backbone, n, 4000, true);
        try testing.expectEqual(off.len, on.len);
        // the same calls (name, roles, offsets, scalars) as a multiset
        var want: std.StringHashMapUnmanaged(i32) = .empty;
        for (off) |c| (try want.getOrPutValue(a, try key(a, c), 0)).value_ptr.* += 1;
        for (on) |c| (try want.getOrPutValue(a, try key(a, c), 0)).value_ptr.* -= 1;
        var it = want.valueIterator();
        while (it.next()) |v| try testing.expectEqual(@as(i32, 0), v.*);
        // each exchange on the side, before the join that precedes its attention
        var moved: usize = 0;
        var open = false;
        for (on) |c| {
            if (c.join) open = false;
            if (std.mem.eql(u8, c.name, "glue.kx_dense")) {
                try testing.expect(c.side);
                open = true;
                moved += 1;
            }
            if (std.mem.eql(u8, c.name, "tf_dsv41_attn_cuda_v1.attn")) try testing.expect(!open);
        }
        // a union window (past 16 rows, not row mode) keeps its exchange after the join
        const dense = n <= 16 or rows;
        try testing.expectEqual(if (dense) cfg.index_sources.items().len else 0, moved);
        if (!dense) for (off, on) |x, y| try testing.expectEqualStrings(x.name, y.name);
        // reviewed read-only roles (branches_test's); the exchange's own roles stay on the side until the join
        const read_only = [_][]const u8{ "s.rope.main", "s.rope.comp", "w.pos", "w.pos64", "w.out", "L.qa", "L.qn", "w.rows" };
        for (try br.shared(a, on)) |role| {
            const ok = for (read_only) |r| {
                if (std.mem.eql(u8, r, role)) break true;
            } else false;
            if (!ok) std.debug.print("kx early: role {s} touched by both streams in a fork region\n", .{role});
            try testing.expect(ok);
        }
    };
}
