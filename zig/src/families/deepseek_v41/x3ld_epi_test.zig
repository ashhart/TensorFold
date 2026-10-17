//! TF_DSV41_X3LD_EPI on the real emitters, host only: each MoE's four x3ld calls become the two fused ones in place.

const std = @import("std");
const testing = std.testing;
const dk = @import("dsv41_kernels");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const check = @import("calls_check.zig");
const rowmode = @import("rowmode.zig");
const dspark_emit = @import("dspark_emit.zig");
const pk = @import("serve_knobs.zig");
const Config = @import("config.zig").Config;
const Arg = calls.Arg;
const Call = calls.Call;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    return w;
}

fn options(limit: i64, r1: bool, epi: bool) block.Options {
    var o: block.Options = .{ .limit = limit, .r1 = r1, .x3ld_epi = epi, .expert_topp = 0.85, .index_budget = 64 << 20, .rows = true, .rope_rows = limit + 2048, .taps = true };
    const pages = @divExact(limit, block.Pool.page);
    o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages };
    return o;
}

fn eq(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

fn same(x: Arg, y: Arg) !void {
    try testing.expectEqualDeep(x, y);
}

fn role(x: Arg) []const u8 {
    return switch (x) {
        .t => |t| switch (t.role) {
            .buf => |b| b,
            .weight => |w| w,
            .empty => "",
        },
        else => "",
    };
}

fn numel(x: Arg) i64 {
    var n: i64 = 1;
    for (x.t.shape) |d| n *= d;
    return n;
}

/// The fused gate/up call against x3ld (gate/up) + gateup_epilogue.
fn expectGateup(c: Call, ld: Call, ep: Call) !void {
    try testing.expectEqualStrings("tf_dsv41_x3ld_epi_v1.gateup", c.name);
    try testing.expectEqualStrings("tf_dsv41_x3ld_v1.grouped", ld.name);
    try testing.expectEqualStrings("tensorfold_exl3_experts_v1.gateup_epilogue", ep.name);
    try testing.expectEqual(@as(usize, 30), c.args.len);
    // x3ld's arguments through nt, pd (its probe 0 and PDL off), then lo, hi
    for (0..19) |i| try same(ld.args[i].arg, c.args[i].arg);
    try testing.expectEqual(@as(i64, 0), ld.args[19].arg.i);
    try testing.expect(!ld.args[22].arg.b);
    try same(ld.args[20].arg, c.args[19].arg);
    try same(ld.args[21].arg, c.args[20].arg);
    // the epilogue reads the Z x3ld wrote, at x3ld's P, N, SK
    try same(ld.args[9].arg, ep.args[0].arg);
    try testing.expectEqual(ld.args[13].arg.i, ep.args[7].arg.i);
    try testing.expectEqual(ld.args[12].arg.i, ep.args[8].arg.i);
    try testing.expectEqual(ld.args[14].arg.i, ep.args[9].arg.i);
    try testing.expectEqual(ld.args[15].arg.i, ep.args[10].arg.i);
    // its pick, svh_g, svh_u, suh_d, Xd, E, limit, act_mode
    for ([_]usize{ 1, 2, 3, 4, 5, 11, 12, 13 }, 21..) |i, j| try same(ep.args[i].arg, c.args[j].arg);
    try expectTicket(c.args[29].arg, ld, .gu);
}

/// The fused down call against x3ld (down) + down_combine.
fn expectDown(c: Call, ld: Call, dc: Call) !void {
    try testing.expectEqualStrings("tf_dsv41_x3ld_epi_v1.down", c.name);
    try testing.expectEqualStrings("tf_dsv41_x3ld_v1.grouped", ld.name);
    try testing.expectEqualStrings("tensorfold_exl3_experts_v1.down_combine", dc.name);
    try testing.expectEqual(@as(usize, 28), c.args.len);
    for (0..19) |i| try same(ld.args[i].arg, c.args[i].arg);
    try testing.expectEqual(@as(i64, 0), ld.args[19].arg.i);
    try testing.expect(!ld.args[22].arg.b);
    try same(ld.args[20].arg, c.args[19].arg);
    try same(ld.args[21].arg, c.args[20].arg);
    try same(ld.args[9].arg, dc.args[0].arg);
    try testing.expectEqual(ld.args[13].arg.i, dc.args[7].arg.i);
    try testing.expectEqual(ld.args[12].arg.i, dc.args[8].arg.i);
    try testing.expectEqual(@as(i64, 1), ld.args[14].arg.i);
    try testing.expectEqual(ld.args[14].arg.i, dc.args[9].arg.i);
    try testing.expectEqual(ld.args[15].arg.i, dc.args[10].arg.i);
    // its pick, svh_d, y, weights, out (L.moe), E
    for ([_]usize{ 1, 2, 3, 4, 5, 11 }, 21..) |i, j| try same(dc.args[i].arg, c.args[j].arg);
    try expectTicket(c.args[27].arg, ld, .dn);
}

/// The ticket: the scratch's own persistent int32 role, holding every word the launch counts on.
fn expectTicket(t: Arg, ld: Call, kind: dk.exl3.EpiKind) !void {
    const r = role(t);
    try testing.expect(eq(r, "s.ex.epi") or eq(r, "s.dx.epi"));
    try testing.expectEqual(calls.Dt.i32, t.t.dt);
    try testing.expectEqual(@as(i64, 0), t.t.offset);
    const a = ld.args;
    const g: dk.exl3.Grouped = .{
        .x0 = 0,
        .x1 = 0,
        .tp0 = 0,
        .tp1 = 0,
        .k2_0 = 0,
        .k2_1 = 0,
        .uids = 0,
        .ucount = 0,
        .members = 0,
        .z = 0,
        .mats = @intCast(a[10].arg.i),
        .K = @intCast(a[11].arg.i),
        .N = @intCast(a[12].arg.i),
        .P = @intCast(a[13].arg.i),
        .SK = @intCast(a[14].arg.i),
        .slots = @intCast(a[15].arg.i),
        .maxm = @intCast(a[8].arg.t.shape[1]),
        .nexp = @intCast(numel(a[6].arg)),
        .lo = @intCast(a[20].arg.i),
        .hi = @intCast(a[21].arg.i),
    };
    _ = try dk.exl3.x3ldEpiCheck(g, kind, @intCast(a[18].arg.i));
    try testing.expect(numel(t) >= @as(i64, @intCast(dk.exl3.epiTicketWords(g, kind))));
    // the role is its own (no other call names it with a different meaning): checked by the caller's scan
}

/// `on` against `off`: every MoE's four calls replaced by the two fused ones in place, the rest equal; the MoEs fused.
fn expectFused(off: []const Call, on: []const Call) !usize {
    var i: usize = 0;
    var j: usize = 0;
    var fused: usize = 0;
    while (i < off.len) {
        if (eq(off[i].name, "tf_dsv41_x3ld_v1.grouped")) {
            try testing.expect(i + 3 < off.len and j + 1 < on.len);
            try expectGateup(on[j], off[i], off[i + 1]);
            try expectDown(on[j + 1], off[i + 2], off[i + 3]);
            // one ticket role a launch pair (gate/up's and down's words from 0)
            try same(on[j].args[29].arg, on[j + 1].args[27].arg);
            i += 4;
            j += 2;
            fused += 1;
            continue;
        }
        try testing.expect(j < on.len);
        try testing.expectEqualDeep(off[i], on[j]);
        i += 1;
        j += 1;
    }
    try testing.expectEqual(on.len, j);
    for (on) |c| {
        try testing.expect(!eq(c.name, "tf_dsv41_x3ld_v1.grouped"));
        try testing.expect(!eq(c.name, "tensorfold_exl3_experts_v1.gateup_epilogue"));
        try testing.expect(!eq(c.name, "tensorfold_exl3_experts_v1.down_combine"));
        // the ticket words are named by the fused calls alone
        if (!std.mem.startsWith(u8, c.name, "tf_dsv41_x3ld_epi_v1.")) for (c.args) |x| {
            const r = role(x.arg);
            try testing.expect(!eq(r, "s.ex.epi") and !eq(r, "s.dx.epi"));
        };
    }
    return fused;
}

test "TF_DSV41_X3LD_EPI: decode windows (1-64 rows, R1 and not), each MoE's four calls as two fused ones" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 1 << 20;
    for ([_]bool{ true, false }) |r1| for ([_]i64{ 1, 2, 5, 16, 24, 64 }) |n| {
        errdefer std.debug.print("r1 {}, {d} rows\n", .{ r1, n });
        const off = try block.emit(a, &cfg, &w, options(limit, r1, false), &backbone, n, 300_000, true);
        const on = try block.emit(a, &cfg, &w, options(limit, r1, true), &backbone, n, 300_000, true);
        try testing.expectEqual(@as(usize, cfg.layers), try expectFused(off, on));
        try testing.expectEqual(off.len - 2 * cfg.layers, on.len);
        if (n > 1) {
            const rw = try rowmode.transform(a, on, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = options(limit, r1, false).pool.?.pts });
            try testing.expectEqual(on.len, rw.len);
            var k: usize = 0;
            for (rw) |c| k += @intFromBool(std.mem.startsWith(u8, c.name, "tf_dsv41_x3ld_epi_v1."));
            try testing.expectEqual(2 * @as(usize, cfg.layers), k);
        }
    };
}

test "TF_DSV41_X3LD_EPI: DSpark passes (1 and 4 slots), the same rule on the drafter's scratch" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    for ([_]i64{ 1, 4 }) |slots| {
        errdefer std.debug.print("{d} slots\n", .{slots});
        const off = try dspark_emit.emitPassSlots(a, &cfg, &w, options(1 << 20, true, false), cfg.dspark_block, slots, 4);
        const on = try dspark_emit.emitPassSlots(a, &cfg, &w, options(1 << 20, true, true), cfg.dspark_block, slots, 4);
        const fused = try expectFused(off, on);
        try testing.expect(fused > 0);
        for (on) |c| if (eq(c.name, "tf_dsv41_x3ld_epi_v1.gateup")) try testing.expectEqualStrings("s.dx.epi", role(c.args[29].arg));
    }
}

test "TF_DSV41_X3LD_EPI's reader: unset / 0 off, 1 on, anything else refused" {
    const Fake = struct {
        var v: ?[]const u8 = null;
        fn get(name: []const u8) ?[]const u8 {
            return if (eq(name, "TF_DSV41_X3LD_EPI")) v else null;
        }
    };
    Fake.v = null;
    try testing.expect(!try pk.x3ldEpi(&Fake.get));
    Fake.v = "0";
    try testing.expect(!try pk.x3ldEpi(&Fake.get));
    Fake.v = " 1 ";
    try testing.expect(try pk.x3ldEpi(&Fake.get));
    for ([_][]const u8{ "2", "on", "true", "yes" }) |bad| {
        Fake.v = bad;
        try testing.expectError(error.BadKnob, pk.x3ldEpi(&Fake.get));
    }
    Fake.v = null;
}
