//! The merged router, pick and plan launches against the launches they replace and against served plans.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const p = @import("decode.zig").p;

/// Router and pick in one launch vs tf_moe_router, tf_select_decode and (2 to 8 rows, tile 8) tf_moe_route: same bytes.
fn routerSelect(t: *Rig, rng: *rig.Rng, rows: usize, d: usize, e: usize, top_k: usize, ties: bool) !void {
    const slots = top_k + 1;
    const cap: usize = 16 * slots + 16 * slots / 8;
    const hx = try gpa.alloc(u16, rows * d);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    const h16 = try gpa.alloc(u16, e * d);
    defer gpa.free(h16);
    for (h16) |*v| v.* = rig.bf16Bits(rng.unit() / 8.0);
    if (ties) for ([_]usize{ 3, 9, 100, 511 }) |dst| @memcpy(h16[dst * d ..][0..d], h16[0..d]);
    const h32 = try gpa.alloc(f32, e * d);
    defer gpa.free(h32);
    for (h16, h32) |b, *v| v.* = @bitCast(@as(u32, b) << 16);
    var x = try t.upload(hx);
    defer x.free();
    var w16 = try t.upload(h16);
    defer w16.free();
    var w32 = try t.upload(h32);
    defer w32.free();
    var done = try t.upload(@as([]const u32, &.{0}));
    defer done.free();
    const sizes = [_]usize{ rows * e * 4, rows * slots * 4, rows * slots * 4, cap * 12, rows * slots * 4 };
    var got: [3][5][]u8 = undefined;
    // 0: the two launches; 1 and 2: the one launch twice (its block counter is zero again after each)
    for (0..3) |v| {
        var bufs: [5]@import("../memory.zig").DeviceBuffer = undefined;
        for (&bufs, sizes) |*b, n| {
            b.* = try t.alloc(n);
            const zero = try gpa.alloc(u8, n);
            defer gpa.free(zero);
            @memset(zero, 0);
            try b.upload(0, zero);
        }
        defer for (&bufs) |*b| b.free();
        const s = t.stream.handle;
        const planned = rows <= 8;
        const tile: usize = if (rows == 1) 0 else 8;
        const items: ?[*]i32 = if (planned) @ptrFromInt(at(bufs[3])) else null;
        const members: ?[*]i32 = if (planned) @ptrFromInt(at(bufs[4])) else null;
        if (v == 0) {
            try t.on.tf_moe_router(p(x), t.kind(), @ptrFromInt(at(w32)), @ptrFromInt(at(bufs[0])), @intCast(rows), @intCast(d), @intCast(e), s);
            const one = rows == 1;
            try t.on.tf_moe_select(@ptrFromInt(at(bufs[0])), @ptrFromInt(at(bufs[1])), @ptrFromInt(at(bufs[2])), if (one) items else null, if (one) members else null, @intCast(cap), @intCast(rows), @intCast(e - 1), @intCast(top_k), s);
            if (planned and !one) try t.on.tf_moe_route(@ptrFromInt(at(bufs[1])), @intCast(rows * slots), @intCast(e), @intCast(tile), members, items, @intCast(cap), s);
        } else {
            try std.testing.expect(try t.on.routerSelect(p(x), t.kind(), p(w16), @ptrFromInt(at(bufs[0])), @ptrFromInt(at(bufs[1])), @ptrFromInt(at(bufs[2])), items, members, @intCast(cap), @intCast(rows), @intCast(d), @intCast(e), @intCast(top_k), @intCast(tile), at(done), s));
        }
        try t.stream.synchronize();
        for (&got[v], bufs) |*g, b| g.* = try rig.download(u8, b);
    }
    defer for (&got) |*g| for (g) |b| gpa.free(b);
    for (1..3) |v| for (0..5) |k| {
        if (!std.mem.eql(u8, got[0][k], got[v][k])) {
            std.debug.print("router+pick rows={d} ties={} run {d}: buffer {d} differs\n", .{ rows, ties, v, k });
            return error.TestUnexpectedResult;
        }
    };
}

/// tf_moe_route's prompt plan puts every live item before planItems(pairs, experts, 128); `bound` routings fill it.
fn routeLive(t: *Rig, rng: *rig.Rng, rows: usize, bound: bool) !void {
    const moe = @import("../moe.zig");
    const experts = 513;
    const tile = 128;
    const pairs = rows * 11;
    const cap = moe.capacity(pairs, experts);
    const pick = try gpa.alloc(i32, pairs);
    defer gpa.free(pick);
    if (bound) {
        // 128 k + 1 pairs an expert, the pairs past them in whole tiles and one partial spare, in a shuffled order
        const per = (pairs - experts) / experts / tile;
        var n: usize = 0;
        for (0..experts) |e| for (0..tile * per + 1) |_| {
            pick[n] = @intCast(e);
            n += 1;
        };
        var e: usize = 0;
        while (pairs - n >= tile) : (e += 1) {
            for (0..tile) |_| {
                pick[n] = @intCast(e);
                n += 1;
            }
        }
        while (n < pairs) : (n += 1) pick[n] = experts - 1;
        var i: usize = pairs;
        while (i > 1) : (i -= 1) std.mem.swap(i32, &pick[i - 1], &pick[rng.next() % i]);
    } else for (pick) |*e| {
        const r = rng.next();
        e.* = @intCast(if (r % 2 == 0) r % 8 else r % experts);
    }
    var dev_pick = try t.upload(pick);
    defer dev_pick.free();
    var members = try t.alloc(pairs * 4);
    defer members.free();
    var items = try t.alloc(cap * 12);
    defer items.free();
    try items.fill8(0xFF);
    try t.on.tf_moe_route(@ptrFromInt(at(dev_pick)), @intCast(pairs), experts, tile, @ptrFromInt(at(members)), @ptrFromInt(at(items)), @intCast(cap), t.stream.handle);
    try t.stream.synchronize();
    const host = try rig.download(i32, items);
    defer gpa.free(host);
    const z = moe.planItems(pairs, experts, tile);
    var live: usize = 0;
    var past: usize = 0;
    var covered: usize = 0;
    for (0..cap) |slot| if (host[3 * slot + 2] > 0) {
        live += 1;
        covered += @intCast(host[3 * slot + 2]);
        if (slot >= z) past += 1;
    };
    std.debug.print("  prompt plan rows {d}{s}: {d} live items, z {d} (capacity {d})\n", .{ rows, if (bound) " at the bound" else "", live, z, cap });
    try std.testing.expectEqual(@as(usize, 0), past);
    try std.testing.expectEqual(pairs, covered);
    if (bound) try std.testing.expectEqual(z, live);
}

test "a prompt plan's live items come before planItems(pairs, experts, 128), which they fill at 128 k + 1 pairs an expert" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x2545F4914F6CDD1D };
    for ([_]usize{ 8192, 17715, 20480 }) |rows| for ([_]bool{ true, false }) |bound| try routeLive(t, &rng, rows, bound);
}

test "the router and the pick in one launch give the two launches' bytes (Qwen3.8 Flash Next: 513 x 2560, top 10)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 16 }) |rows| for ([_]bool{ false, true }) |ties| try routerSelect(t, &rng, rows, 2560, 513, 10, ties);
}

/// Served plans, 3 MoE layers a width of 1-8 rows: i32 layer, rows, slots, cap; picks (rows x slots); items (cap x 3).
const served_plans = @embedFile("served_plans.bin");

/// One launch on a served plan (one-hot x, served picks the top weights): the served bytes, as the launches replaced.
fn servedPlan(t: *Rig, rng: *rig.Rng, rows: usize, slots: usize, cap: usize, picks: []const i32, items: []const i32) !void {
    const d: usize = 2560;
    const e: usize = 513;
    const top_k = slots - 1;
    const hx = try gpa.alloc(u16, rows * d);
    defer gpa.free(hx);
    @memset(hx, t.bits(0));
    for (0..rows) |r| hx[r * d + r] = t.bits(1);
    const h16 = try gpa.alloc(u16, e * d);
    defer gpa.free(h16);
    @memset(h16, 0);
    for (0..rows) |r| {
        for (0..e) |ex| h16[ex * d + r] = rig.bf16Bits(rng.unit() / 8.0);
        for (0..top_k) |k| h16[@as(usize, @intCast(picks[r * slots + k])) * d + r] = rig.bf16Bits(4.0 - 0.25 * @as(f32, @floatFromInt(k)));
    }
    const h32 = try gpa.alloc(f32, e * d);
    defer gpa.free(h32);
    for (h16, h32) |b, *v| v.* = @bitCast(@as(u32, b) << 16);
    var x = try t.upload(hx);
    defer x.free();
    var w16 = try t.upload(h16);
    defer w16.free();
    var w32 = try t.upload(h32);
    defer w32.free();
    var done = try t.upload(@as([]const u32, &.{0}));
    defer done.free();
    const sizes = [_]usize{ rows * e * 4, rows * slots * 4, rows * slots * 4, cap * 12, rows * slots * 4 };
    var got: [2][5][]u8 = undefined;
    const tile: usize = if (rows == 1) 0 else 8;
    for (0..2) |v| {
        var bufs: [5]@import("../memory.zig").DeviceBuffer = undefined;
        for (&bufs, sizes) |*b, n| {
            b.* = try t.alloc(n);
            const zero = try gpa.alloc(u8, n);
            defer gpa.free(zero);
            @memset(zero, 0);
            try b.upload(0, zero);
        }
        defer for (&bufs) |*b| b.free();
        const s = t.stream.handle;
        const it: [*]i32 = @ptrFromInt(at(bufs[3]));
        const mem: [*]i32 = @ptrFromInt(at(bufs[4]));
        if (v == 0) {
            try t.on.tf_moe_router(p(x), t.kind(), @ptrFromInt(at(w32)), @ptrFromInt(at(bufs[0])), @intCast(rows), @intCast(d), @intCast(e), s);
            const one = rows == 1;
            try t.on.tf_moe_select(@ptrFromInt(at(bufs[0])), @ptrFromInt(at(bufs[1])), @ptrFromInt(at(bufs[2])), if (one) it else null, if (one) mem else null, @intCast(cap), @intCast(rows), @intCast(e - 1), @intCast(top_k), s);
            if (!one) try t.on.tf_moe_route(@ptrFromInt(at(bufs[1])), @intCast(rows * slots), @intCast(e), @intCast(tile), mem, it, @intCast(cap), s);
        } else {
            try std.testing.expect(try t.on.routerSelect(p(x), t.kind(), p(w16), @ptrFromInt(at(bufs[0])), @ptrFromInt(at(bufs[1])), @ptrFromInt(at(bufs[2])), it, mem, @intCast(cap), @intCast(rows), @intCast(d), @intCast(e), @intCast(top_k), @intCast(tile), at(done), s));
        }
        try t.stream.synchronize();
        for (&got[v], bufs) |*g, b| g.* = try rig.download(u8, b);
    }
    defer for (&got) |*g| for (g) |b| gpa.free(b);
    for (0..2) |v| {
        if (!std.mem.eql(u8, got[v][1], std.mem.sliceAsBytes(picks)) or !std.mem.eql(u8, got[v][3], std.mem.sliceAsBytes(items))) {
            std.debug.print("served plan rows={d} run {d}: picks or items differ from the served bytes\n", .{ rows, v });
            return error.TestUnexpectedResult;
        }
    }
    for (0..5) |k| if (!std.mem.eql(u8, got[0][k], got[1][k])) {
        std.debug.print("served plan rows={d}: buffer {d} differs between the launches\n", .{ rows, k });
        return error.TestUnexpectedResult;
    };
}

test "the router and the pick in one launch give a served plan's picks and items, byte for byte (rows 1 to 8)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x2545F4914F6CDD1D };
    var at_byte: usize = 0;
    var seen: [9]usize = @splat(0);
    while (at_byte + 16 <= served_plans.len) {
        const head = std.mem.bytesAsSlice(i32, served_plans[at_byte..][0..16]);
        const rows: usize = @intCast(head[1]);
        const slots: usize = @intCast(head[2]);
        const cap: usize = @intCast(head[3]);
        at_byte += 16;
        const picks = try gpa.alloc(i32, rows * slots);
        defer gpa.free(picks);
        @memcpy(std.mem.sliceAsBytes(picks), served_plans[at_byte..][0 .. rows * slots * 4]);
        at_byte += rows * slots * 4;
        const items = try gpa.alloc(i32, cap * 3);
        defer gpa.free(items);
        @memcpy(std.mem.sliceAsBytes(items), served_plans[at_byte..][0 .. cap * 12]);
        at_byte += cap * 12;
        try servedPlan(t, &rng, rows, slots, cap, picks, items);
        seen[rows] += 1;
    }
    for (seen[1..]) |n| try std.testing.expect(n == 3);
}

test "a prompt's picks a wave a row (tf_select_rows) are select_body's bytes, ties included" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x5E1EC7 };
    const top_k: usize = 10;
    const slots = top_k + 1;
    for ([_][2]usize{ .{ 4099, 512 }, .{ 37, 512 }, .{ 1029, 1024 }, .{ 6, 64 } }) |shape| {
        const rows = shape[0];
        const experts = shape[1];
        const h = try gpa.alloc(f32, rows * (experts + 1));
        defer gpa.free(h);
        for (h) |*v| v.* = rng.unit() * 4.0;
        // ties: three experts copy expert 0's logit in every row, and in odd rows expert 5 copies expert 1's
        for (0..rows) |r| {
            const row = h[r * (experts + 1) ..][0 .. experts + 1];
            for ([_]usize{ 3, 9, experts / 2 }) |c| row[c] = row[0];
            if (r % 2 == 1) row[5] = row[1];
        }
        var logits = try t.upload(h);
        defer logits.free();
        var got: [2][2][]u8 = undefined;
        for (0..2) |v| {
            var pick = try t.alloc(rows * slots * 4);
            defer pick.free();
            var wts = try t.alloc(rows * slots * 4);
            defer wts.free();
            var items = try t.alloc(slots * 12);
            defer items.free();
            var members = try t.alloc(slots * 4);
            defer members.free();
            // 0: select_body (tf_select_decode, taken where a plan is asked for); 1: tf_select_rows
            const it: ?[*]i32 = if (v == 0) @ptrFromInt(at(items)) else null;
            const mem: ?[*]i32 = if (v == 0) @ptrFromInt(at(members)) else null;
            try t.on.tf_moe_select(@ptrFromInt(at(logits)), @ptrFromInt(at(pick)), @ptrFromInt(at(wts)), it, mem, @intCast(slots), @intCast(rows), @intCast(experts), @intCast(top_k), t.stream.handle);
            try t.stream.synchronize();
            got[v] = .{ try rig.download(u8, pick), try rig.download(u8, wts) };
        }
        defer for (&got) |*g| for (g) |b| gpa.free(b);
        for (0..2) |k| if (!std.mem.eql(u8, got[0][k], got[1][k])) {
            std.debug.print("select rows={d} experts={d}: {s} differ\n", .{ rows, experts, if (k == 0) "picks" else "weights" });
            return error.TestUnexpectedResult;
        };
    }
}
