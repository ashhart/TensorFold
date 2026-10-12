//! Host tests of x3gm v2 on prefill (`zig build test-dsv41-gm2pf`): knob reader, capture mode, emitter off/one/split.
const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const check = @import("calls_check.zig");
const gm2pf = @import("gm2pf.zig");
const knob_types = @import("knob_types.zig");
const Config = @import("config.zig").Config;

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "TF_DSV41_GM_V2: x3gm.config's modes (unset / empty / 0 off, 1 one, split), anything else refused" {
    const cases = [_]struct { v: ?[]const u8, m: knob_types.GmMode }{
        .{ .v = null, .m = .off }, .{ .v = "", .m = .off }, .{ .v = "0", .m = .off }, .{ .v = " 1 ", .m = .one }, .{ .v = "split", .m = .split },
    };
    for (cases) |c| {
        Fake.pairs = if (c.v) |v| &.{.{ "TF_DSV41_GM_V2", v }} else &.{};
        try testing.expectEqual(c.m, try gm2pf.mode(&Fake.get));
    }
    for ([_][]const u8{ "2", "on", "one" }) |v| {
        Fake.pairs = &.{.{ "TF_DSV41_GM_V2", v }};
        try testing.expectError(error.BadKnob, gm2pf.mode(&Fake.get));
    }
    Fake.pairs = &.{};
}

const Names = struct {
    list: []const []const u8,
    i: usize = 0,
    pub fn next(it: *Names) ?[]const u8 {
        if (it.i == it.list.len) return null;
        it.i += 1;
        return it.list[it.i - 1];
    }
};

test "a capture's mode from its launches: v1, one gateup2 a rotation, several" {
    const rot = "tf_dsv41_x3gm_v1.rot";
    const g1 = "tf_dsv41_x3gm_v1.gateup";
    const g2 = "tf_dsv41_x3gm_v1.gateup2";
    var v1: Names = .{ .list = &.{ rot, g1, g1, "tf_dsv41_x3gm_v1.down", rot, g1 } };
    try testing.expectEqual(knob_types.GmMode.off, gm2pf.ofLaunches(&v1));
    var one: Names = .{ .list = &.{ rot, g2, "tf_dsv41_x3gm_v1.down2", rot, g2, "tf_dsv41_x3gm_v1.down2" } };
    try testing.expectEqual(knob_types.GmMode.one, gm2pf.ofLaunches(&one));
    var split: Names = .{ .list = &.{ rot, g2, g2, "tf_dsv41_x3gm_v1.down2", "tf_dsv41_x3gm_v1.down2" } };
    try testing.expectEqual(knob_types.GmMode.split, gm2pf.ofLaunches(&split));
}

const gu_widths: u32 = (1 << 3) | (1 << 4) | (1 << 6);
const dn_widths: u32 = (1 << 4) | (1 << 10);

fn widths(a: std.mem.Allocator, gm: bool) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    if (gm) for (0..40) |L| try w.gm.put(a, @intCast(L), .{ gu_widths, dn_widths });
    return w;
}

/// The routed launches and their plans: what v2 replaces.
fn isRouted(c: calls.Call) bool {
    const n = c.name;
    for ([_][]const u8{ "glue.gm_plan", "tf_dsv41_x3gm_v1.gateup", "tf_dsv41_x3gm_v1.down", "tf_dsv41_x3gm_v1.gateup2", "tf_dsv41_x3gm_v1.down2" }) |r|
        if (std.mem.eql(u8, n, r)) return true;
    return false;
}

fn role(x: calls.Named) []const u8 {
    return switch (x.arg.t.role) {
        .buf => |b| b,
        .weight => |w| w,
        .empty => "",
    };
}

fn count(cs: []const calls.Call, name: []const u8) usize {
    var n: usize = 0;
    for (cs) |c| n += @intFromBool(std.mem.eql(u8, c.name, name));
    return n;
}

test "TF_DSV41_GM_V2 on the prefill emitter: v2's launches in place of v1's, every other call the same" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa, true);
    const bare = try widths(aa, false);
    const layers = [_]u32{ 1, 2, 3 };
    for ([_]i64{ 2048, 300, 12, 1 }) |n| {
        const off = try bp.emitPrefill(aa, &cfg, &w, .{}, &layers, n, 0, true);
        const moes = count(off, "tf_dsv41_x3gm_v1.rot");
        try testing.expect(moes > 0);
        try testing.expectEqual(moes * @popCount(gu_widths), count(off, "tf_dsv41_x3gm_v1.gateup"));
        for ([_]knob_types.GmMode{ .one, .split }) |m| {
            const on = try bp.emitPrefill(aa, &cfg, &w, .{ .gm_v2 = m }, &layers, n, 0, true);
            // every call that is not a routed launch or its plan: the same, in the same order
            var i: usize = 0;
            var j: usize = 0;
            while (true) {
                while (i < off.len and isRouted(off[i])) i += 1;
                while (j < on.len and isRouted(on[j])) j += 1;
                if (i == off.len or j == on.len) break;
                try testing.expectEqualStrings(off[i].name, on[j].name);
                try testing.expectEqual(off[i].args.len, on[j].args.len);
                i += 1;
                j += 1;
            }
            try testing.expectEqual(off.len, i);
            try testing.expectEqual(on.len, j);
            try testing.expectEqual(@as(usize, 0), count(on, "tf_dsv41_x3gm_v1.gateup") + count(on, "tf_dsv41_x3gm_v1.down"));
            const per: [2]usize = if (m == .one) .{ 1, 1 } else .{ @popCount(gu_widths), @popCount(dn_widths) };
            try testing.expectEqual(moes * per[0], count(on, "tf_dsv41_x3gm_v1.gateup2"));
            try testing.expectEqual(moes * per[1], count(on, "tf_dsv41_x3gm_v1.down2"));
            try testing.expectEqual(moes * (per[0] + per[1]) - (if (m == .one) moes else 0), count(on, "glue.gm_plan"));
            try launches(on, m);
        }
        // `one` needs no widths (each expert's is in its table); v1 and split do
        _ = try bp.emitPrefill(aa, &cfg, &bare, .{ .gm_v2 = .one }, &layers, n, 0, true);
        try testing.expectError(error.NoWidth, bp.emitPrefill(aa, &cfg, &bare, .{}, &layers, n, 0, true));
        try testing.expectError(error.NoWidth, bp.emitPrefill(aa, &cfg, &bare, .{ .gm_v2 = .split }, &layers, n, 0, true));
    }
    // x3gm.supported: a width gm2_kernel does not dispatch (2, 9) refuses the block
    var odd = try widths(aa, false);
    for (0..40) |L| try odd.gm.put(aa, @intCast(L), .{ (1 << 2) | (1 << 4), 1 << 4 });
    try testing.expectError(error.Unsupported, bp.emitPrefill(aa, &cfg, &odd, .{ .gm_v2 = .one }, &layers, 300, 0, true));
    for (0..40) |L| try odd.gm.put(aa, @intCast(L), .{ 1 << 4, 1 << 9 });
    try testing.expectError(error.Unsupported, bp.emitPrefill(aa, &cfg, &odd, .{ .gm_v2 = .split }, &layers, 300, 0, true));
}

/// Each gateup2 / down2 against x3gm.cpp's signatures: width tables, ticket halves, shared plan (one), widths (split).
fn launches(cs: []const calls.Call, m: knob_types.GmMode) !void {
    var last_plan: ?[]const calls.Named = null;
    var gu_plan: [5][]const u8 = undefined;
    var jg: i64 = 0;
    var jd: i64 = 0;
    for (cs) |c| {
        if (std.mem.eql(u8, c.name, "tf_dsv41_x3gm_v1.rot")) {
            jg = 0;
            jd = 0;
        }
        if (std.mem.eql(u8, c.name, "glue.gm_plan")) {
            try testing.expectEqual(@as(usize, 10), c.args.len);
            const k2 = c.args[6].arg.i;
            if (m == .one) try testing.expectEqual(@as(i64, 0), k2) else try testing.expect(k2 > 0);
            try testing.expectEqual(gm2pf.bm, c.args[8].arg.i);
            last_plan = c.args;
        }
        const gu = std.mem.eql(u8, c.name, "tf_dsv41_x3gm_v1.gateup2");
        const dn = std.mem.eql(u8, c.name, "tf_dsv41_x3gm_v1.down2");
        if (!gu and !dn) continue;
        const p = last_plan.?;
        const a = c.args;
        try testing.expectEqual(@as(usize, if (gu) 22 else 16), a.len);
        const k2e = a[if (gu) 4 else 2].arg.t;
        try testing.expectEqualStrings(if (gu) "k2g" else "k2d", role(a[if (gu) 4 else 2])[role(a[if (gu) 4 else 2]).len - 3 ..]);
        try testing.expectEqual(calls.Dt.i32, k2e.dt);
        try testing.expectEqual(a[if (gu) 2 else 1].arg.t.shape[0] + 1, k2e.shape[0]);
        const first: usize = if (gu) 5 else 3;
        for (0..5) |q| try testing.expectEqualStrings(role(p[1 + q]), role(a[first + q]));
        if (gu) {
            for (0..5) |q| gu_plan[q] = role(a[first + q]);
            try testing.expectEqual(gm2pf.cfg_gu, a[18].arg.i);
            try testing.expect(!a[17].arg.b and a[19].arg.b and a[21].arg.b);
        } else {
            try testing.expectEqual(gm2pf.cfg_dn, a[13].arg.i);
            try testing.expect(a[14].arg.b and a[15].arg.b);
            if (m == .one) for (0..5) |q| try testing.expectEqualStrings(gu_plan[q], role(a[first + q]));
        }
        // the ticket: gate/up from 0, down from MAX_WIDTHS, one a launch
        const t = a[if (gu) 10 else 8].arg.t;
        try testing.expectEqualStrings("L.gm.ticket", role(a[if (gu) 10 else 8]));
        if (gu) {
            try testing.expectEqual(jg * 4, t.offset);
            jg += 1;
            if (m == .split) try testing.expect(!p[7].arg.b);
        } else {
            try testing.expectEqual((gm2pf.max_widths + jd) * 4, t.offset);
            jd += 1;
            if (m == .split) try testing.expect(p[7].arg.b);
        }
    }
}
