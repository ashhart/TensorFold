//! TF_DSV41_INDEX_BOUND on the real emitters, host only (`zig build test-dsv41-wide`): bounded score/top-k renames.

const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const check = @import("calls_check.zig");
const rowmode = @import("rowmode.zig");
const dspark_emit = @import("dspark_emit.zig");
const pk = @import("serve_knobs.zig");
const Config = @import("config.zig").Config;
const Call = calls.Call;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    return w;
}

fn servedOptions(limit: i64, bound: bool) block.Options {
    var o: block.Options = .{ .limit = limit, .r1 = true, .mhc_defer = true, .expert_topp = 0.85, .index_budget = 64 << 20, .rows = true, .rope_rows = limit + 2048, .taps = true, .index_bound = bound };
    const pages = @divExact(limit, block.Pool.page);
    o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages };
    return o;
}

fn eq(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

fn named(c: Call, name: []const u8) ?calls.Arg {
    for (c.args) |x| if (eq(x.name, name)) return x.arg;
    return null;
}

fn roleKey(r: calls.Role) []const u8 {
    return switch (r) {
        .weight => |s| s,
        .buf => |s| s,
        .empty => "",
    };
}

/// Two arguments with the same contents (tensors by role, dtype, shape, strides and offset; lists item by item).
fn argEql(p: calls.Arg, q: calls.Arg) bool {
    if (std.meta.activeTag(p) != std.meta.activeTag(q)) return false;
    return switch (p) {
        .t, .opaque_table => |t| blk: {
            const u = if (q == .t) q.t else q.opaque_table;
            break :blk std.meta.activeTag(t.role) == std.meta.activeTag(u.role) and eq(roleKey(t.role), roleKey(u.role)) and
                t.dt == u.dt and t.offset == u.offset and std.mem.eql(i64, t.shape, u.shape) and std.mem.eql(i64, t.stride, u.stride);
        },
        .list => |xs| blk: {
            if (xs.len != q.list.len) break :blk false;
            for (xs, q.list) |x, y| if (!argEql(x, y)) break :blk false;
            break :blk true;
        },
        .i => |v| v == q.i,
        .f => |v| v == q.f,
        .b => |v| v == q.b,
        .none => true,
    };
}

/// What each twin replaced, and what stayed.
const Seen = struct {
    scores_b: usize = 0,
    gather_kept: usize = 0,
    dtopk_b: [3]usize = @splat(0), // by MODE 0 / 1 / 2 (1: never)
    topk_b: usize = 0,
    topk_b_blocks: usize = 0, // twin launches with a mode-2 job
    topk_kept: usize = 0, // Reindex (mode 3)
};

/// attn_cuda's top-k: [s0, c0, o0, n0, nk0, k0, m0, e0, s1 .. e1, jobs, pos, ratio, bs, R, CL]
fn topkModes(c: Call) [2]i64 {
    return .{ c.args[6].arg.i, c.args[14].arg.i };
}

/// `on` is `off` with the knob's renames (`_scores_b`, `_dtopk_b`, `tf_dsv41_topk_b_v1.topk`) and nothing else.
fn expectRenamed(off: []const Call, on: []const Call) !Seen {
    try testing.expectEqual(off.len, on.len);
    var s: Seen = .{};
    for (off, on, 0..) |x, y, i| {
        errdefer std.debug.print("call {d}: {s} vs {s}\n", .{ i, x.name, y.name });
        if (eq(x.name, "_scores")) {
            const gather = named(x, "GATHER").?.b;
            try testing.expectEqualStrings(if (gather) "_scores" else "_scores_b", y.name);
            if (gather) s.gather_kept += 1 else s.scores_b += 1;
        } else if (eq(x.name, "_dtopk")) {
            try testing.expectEqualStrings("_dtopk_b", y.name);
            s.dtopk_b[@intCast(named(x, "MODE").?.i)] += 1;
        } else if (eq(x.name, "tf_dsv41_attn_cuda_v1.topk")) {
            const m = topkModes(x);
            const dense = (m[0] == 0 or m[0] == 2) and (m[1] == 0 or m[1] == 2);
            try testing.expect(dense or (m[0] == 3 and m[1] == 3));
            try testing.expectEqualStrings(if (dense) "tf_dsv41_topk_b_v1.topk" else "tf_dsv41_attn_cuda_v1.topk", y.name);
            if (dense) s.topk_b += 1 else s.topk_kept += 1;
            if (dense and (m[0] == 2 or m[1] == 2)) s.topk_b_blocks += 1;
        } else try testing.expectEqualStrings(x.name, y.name);
        try testing.expect(x.triton == y.triton and x.glue == y.glue and x.begin == y.begin and x.side == y.side and std.mem.eql(i64, &x.grid, &y.grid));
        try testing.expectEqual(x.args.len, y.args.len);
        for (x.args, y.args, 0..) |p, q, j| {
            errdefer std.debug.print("  arg {d} {s}\n", .{ j, p.name });
            try testing.expectEqualStrings(p.name, q.name);
            try testing.expect(argEql(p.arg, q.arg));
        }
    }
    return s;
}

test "TF_DSV41_INDEX_BOUND reader: unset / 0 off, 1 on, anything else refused" {
    const Fake = struct {
        var v: ?[]const u8 = null;
        fn get(name: []const u8) ?[]const u8 {
            return if (eq(name, "TF_DSV41_INDEX_BOUND")) v else null;
        }
    };
    Fake.v = null;
    try testing.expect(!try pk.indexBound(&Fake.get));
    Fake.v = "0";
    try testing.expect(!try pk.indexBound(&Fake.get));
    Fake.v = " 1 ";
    try testing.expect(try pk.indexBound(&Fake.get));
    for ([_][]const u8{ "2", "on", "" }) |s| {
        Fake.v = s;
        try testing.expectError(error.BadKnob, pk.indexBound(&Fake.get));
    }
}

test "TF_DSV41_INDEX_BOUND: decode windows' dense scores as _scores_b (materialised and row blocks), GATHER kept, row mode alike" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 1 << 20;
    // short and 131K-class positions; at 900K the 24 / 64-row windows pass the 64 MiB budget: backend._blocked
    for ([_]i64{ 4_000, 150_000, 900_000 }) |start| for ([_]i64{ 1, 5, 16, 24, 64 }) |n| {
        errdefer std.debug.print("start {d}, {d} rows\n", .{ start, n });
        const off = try block.emit(a, &cfg, &w, servedOptions(limit, false), &backbone, n, start, true);
        const on = try block.emit(a, &cfg, &w, servedOptions(limit, true), &backbone, n, start, true);
        const got = try expectRenamed(off, on);
        try testing.expect(got.scores_b > 0);
        if (n > 1) {
            const ro = try rowmode.transform(a, off, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = servedOptions(limit, false).pool.?.pts });
            const rn = try rowmode.transform(a, on, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = servedOptions(limit, false).pool.?.pts });
            _ = try expectRenamed(ro, rn);
            // row mode turns ROWS on and hands SL the read slots on the twin as on `_scores`
            for (rn) |c| if (eq(c.name, "_scores_b")) try testing.expect(named(c, "ROWS").?.b);
        }
    };
}

test "TF_DSV41_INDEX_BOUND: decode windows' top-k as _dtopk_b / tf_dsv41_topk_b_v1.topk (dense select and candidate blocks), Reindex kept, row mode alike" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 1 << 20;
    var all: Seen = .{};
    // 4K, 150K, 900K, 1,040,000: attn_cuda only, ratio splits, budget row blocks, source blocks on _dtopk mode 2
    for ([_]i64{ 4_000, 150_000, 900_000, 1_040_000 }) |start| for ([_]i64{ 1, 5, 16, 24, 64 }) |n| {
        if (start + n > limit) continue;
        errdefer std.debug.print("start {d}, {d} rows\n", .{ start, n });
        const off = try block.emit(a, &cfg, &w, servedOptions(limit, false), &backbone, n, start, true);
        const on = try block.emit(a, &cfg, &w, servedOptions(limit, true), &backbone, n, start, true);
        const got = try expectRenamed(off, on);
        // every window has index layers: the selection is a twin unless every index layer ran backend._blocked
        var blocked = false;
        for (off) |c| blocked = blocked or (c.glue and eq(c.name, "glue.topk"));
        try testing.expect(got.topk_b + got.dtopk_b[0] > 0 or blocked);
        try testing.expect(got.topk_kept > 0); // the Reindex layers
        inline for (.{ "scores_b", "gather_kept", "topk_b", "topk_b_blocks", "topk_kept" }) |f| @field(all, f) += @field(got, f);
        for (got.dtopk_b, &all.dtopk_b) |g, *t| t.* += g;
        if (n > 1) {
            const pts = servedOptions(limit, false).pool.?.pts;
            const ro = try rowmode.transform(a, off, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = pts });
            const rn = try rowmode.transform(a, on, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = pts });
            _ = try expectRenamed(ro, rn);
            // row mode: the twins read each row's own position (the table's int64 column, ROWS on), as the originals
            for (rn) |c| {
                if (eq(c.name, "_dtopk_b")) try testing.expect(named(c, "ROWS").?.b);
                if (eq(c.name, "tf_dsv41_topk_b_v1.topk")) try testing.expectEqualStrings("s.rows.tab", roleKey(c.args[17].arg.t.role));
            }
        }
    };
    errdefer std.debug.print("seen {any}\n", .{all});
    try testing.expect(all.topk_b > 0 and all.topk_b_blocks > 0 and all.topk_kept > 0);
    try testing.expect(all.dtopk_b[0] > 0 and all.dtopk_b[1] == 0 and all.dtopk_b[2] > 0);
}

test "TF_DSV41_INDEX_BOUND: DSpark passes (1 and 4 slots) do not change" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    for ([_]i64{ 1, 4 }) |slots| {
        errdefer std.debug.print("{d} slots\n", .{slots});
        const off = try dspark_emit.emitPassSlots(a, &cfg, &w, servedOptions(1 << 20, false), cfg.dspark_block, slots, 4);
        const on = try dspark_emit.emitPassSlots(a, &cfg, &w, servedOptions(1 << 20, true), cfg.dspark_block, slots, 4);
        const got = try expectRenamed(off, on);
        try testing.expect(got.topk_b == 0 and got.dtopk_b[0] == 0 and got.dtopk_b[2] == 0);
    }
}
