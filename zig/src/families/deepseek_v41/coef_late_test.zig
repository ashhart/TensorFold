//! TF_DSV41_COEF_LATE on the real emitters: deferred coef launches move before their exchange, all else unchanged.

const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const check = @import("calls_check.zig");
const rowmode = @import("rowmode.zig");
const branches = @import("branches.zig");
const dspark_emit = @import("dspark_emit.zig");
const Config = @import("config.zig").Config;
const Arg = calls.Arg;
const Call = calls.Call;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    return w;
}

fn servedOptions(limit: i64, pf: i64, late: bool) block.Options {
    var o: block.Options = .{ .limit = limit, .r1 = true, .mhc_defer = true, .coef_late = late, .expert_topp = 0.85, .index_budget = 64 << 20, .rows = true, .rope_rows = limit + 2048, .taps = true, .mhc_pf_rows = pf, .branches = true };
    const pages = @divExact(limit, block.Pool.page);
    o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages };
    return o;
}

fn eq(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

fn same(x: Arg, y: Arg) bool {
    if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
    return switch (x) {
        .t, .opaque_table => |t| blk: {
            const u = if (y == .t) y.t else y.opaque_table;
            if (std.meta.activeTag(t.role) != std.meta.activeTag(u.role)) break :blk false;
            const rs = switch (t.role) {
                .weight => |r| eq(r, u.role.weight),
                .buf => |r| eq(r, u.role.buf),
                .empty => true,
            };
            break :blk rs and t.dt == u.dt and t.offset == u.offset and std.mem.eql(i64, t.shape, u.shape) and std.mem.eql(i64, t.stride, u.stride);
        },
        .i => |v| v == y.i,
        .f => |v| v == y.f,
        .b => |v| v == y.b,
        .none => true,
        .list => |l| l.len == y.list.len and for (l, y.list) |p, q| {
            if (!same(p, q)) break false;
        } else true,
    };
}

fn sameCall(x: Call, y: Call) bool {
    if (!eq(x.name, y.name) or x.triton != y.triton or x.glue != y.glue or x.begin != y.begin) return false;
    if (!std.mem.eql(i64, &x.grid, &y.grid) or x.args.len != y.args.len) return false;
    if (x.side != y.side or x.fork != y.fork or x.join != y.join) return false;
    if (x.defer_side != y.defer_side or x.defer_join != y.defer_join) return false;
    for (x.args, y.args) |p, q| if (!eq(p.name, q.name) or !same(p.arg, q.arg)) return false;
    return true;
}

/// Each deferred launch is joined before the next mHC call; returns the roles the deferred stream shares with main.
fn expectJoined(a: std.mem.Allocator, cs: []const Call) ![]const []const u8 {
    var pending = false;
    for (cs, 0..) |c, i| {
        errdefer std.debug.print("call {d} ({s}) reads coefficients still on the deferred stream\n", .{ i, c.name });
        if (c.defer_join) {
            try testing.expect(pending);
            pending = false;
        }
        const reads = eq(c.name, "tf_dsv41_mhc_pf_v1.run") or eq(c.name, "tf_dsv41_mhc_cuda_v1.run") or eq(c.name, "_site");
        if (reads) try testing.expect(!pending);
        if (c.defer_side) pending = true;
    }
    try testing.expect(!pending);
    return branches.sharedDeferred(a, cs);
}

/// `late` is `off` with some deferred launches moved to just before an exchange; returns how many moved.
fn expectMoved(a: std.mem.Allocator, off: []const Call, late: []const Call) !usize {
    try testing.expectEqual(off.len, late.len);
    var rest_off: std.ArrayList(Call) = .empty;
    var rest_late: std.ArrayList(Call) = .empty;
    var def_off: std.ArrayList(Call) = .empty;
    var def_late: std.ArrayList(Call) = .empty;
    for (off) |c| try (if (c.defer_side) &def_off else &rest_off).append(a, c);
    for (late) |c| try (if (c.defer_side) &def_late else &rest_late).append(a, c);
    try testing.expectEqual(def_off.items.len, def_late.items.len);
    for (rest_off.items, rest_late.items, 0..) |x, y, i| {
        errdefer std.debug.print("main call {d}: {s} vs {s}\n", .{ i, x.name, y.name });
        try testing.expect(sameCall(x, y));
    }
    for (def_off.items, def_late.items) |x, y| try testing.expect(sameCall(x, y));
    var moved: usize = 0;
    for (late, 0..) |c, i| {
        if (!c.defer_side) continue;
        if (sameCall(off[i], c)) continue; // left in place: a join before any exchange
        errdefer std.debug.print("moved deferred launch {d} not before an exchange: next {s}\n", .{ i, late[i + 1].name });
        try testing.expect(i + 1 < late.len and late[i + 1].glue and std.mem.startsWith(u8, late[i + 1].name, "glue.exchange"));
        moved += 1;
    }
    // no role newly shared between the deferred stream and main inside a fork region
    const so = try expectJoined(a, off);
    const sn = try expectJoined(a, late);
    for (sn) |r| {
        errdefer std.debug.print("new shared role {s}\n", .{r});
        for (so) |q| {
            if (eq(q, r)) break;
        } else return error.TestUnexpectedResult;
    }
    return moved;
}

test "TF_DSV41_COEF_LATE: decode windows (mhc_cuda <= 16 rows, PFDEC above), each coefficient launch before its exchange" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 1 << 20;
    for ([_]i64{ 0, 1, 17 }) |pf| for ([_]i64{ 1, 2, 5, 16, 17, 24, 48, 64 }) |n| {
        errdefer std.debug.print("pfdec {d}, {d} rows\n", .{ pf, n });
        const off = try block.emit(a, &cfg, &w, servedOptions(limit, pf, false), &backbone, n, 300_000, true);
        const late = try block.emit(a, &cfg, &w, servedOptions(limit, pf, true), &backbone, n, 300_000, true);
        const moved = try expectMoved(a, off, late);
        // a window with mixing sites under MHC_DEFER (mhc_cuda <= 16 rows, or PFDEC) moves every launch but none
        const deferred = n <= 16 or (pf > 0 and n >= pf);
        if (deferred) try testing.expect(moved > 0) else try testing.expectEqual(@as(usize, 0), moved);
        if (n > 1) {
            const rw = try rowmode.transform(a, late, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = servedOptions(limit, 0, false).pool.?.pts });
            try testing.expectEqual(late.len, rw.len);
        }
    };
}

test "TF_DSV41_COEF_LATE: DSpark passes (1 and 4 slots), the same rule" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    for ([_]i64{ 0, 17 }) |pf| for ([_]i64{ 1, 4 }) |slots| {
        errdefer std.debug.print("pfdec {d}, {d} slots\n", .{ pf, slots });
        const off = try dspark_emit.emitPassSlots(a, &cfg, &w, servedOptions(1 << 20, pf, false), cfg.dspark_block, slots, 4);
        const late = try dspark_emit.emitPassSlots(a, &cfg, &w, servedOptions(1 << 20, pf, true), cfg.dspark_block, slots, 4);
        _ = try expectMoved(a, off, late);
    };
}
