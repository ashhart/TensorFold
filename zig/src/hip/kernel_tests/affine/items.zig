//! The routed item-rows tile writes the bytes of the tile it replaces, and each pair's bytes as that pair alone.

const std = @import("std");
const base = @import("../rig.zig");
const affine = @import("../../launches/affine.zig");
const stream_pick = @import("../../launches/stream_pick.zig");
const Buffer = base.Buffer;
const gpa = base.gpa;
const data = @import("data.zig");
const Rig = @import("product.zig").Rig;
const Kernels = affine.Kernels;

const Kind = enum { plain, pair, relu2 };

/// A served routed projection: `n` rows an expert (a pair's gate and up together), `fc1` reads x by token.
const Case = struct { name: []const u8, kind: Kind, n: usize, k: usize, group: c_int, experts: usize, top_k: usize, fc1: bool };

const cases = [_]Case{
    .{ .name = "Lightning fc1", .kind = .relu2, .n = 1856, .k = 2688, .group = 64, .experts = 128, .top_k = 6, .fc1 = true },
    .{ .name = "Lightning fc2", .kind = .plain, .n = 2688, .k = 1856, .group = 64, .experts = 128, .top_k = 6, .fc1 = false },
    .{ .name = "Flash Next gate+up", .kind = .pair, .n = 1280, .k = 2560, .group = 32, .experts = 512, .top_k = 10, .fc1 = true },
    .{ .name = "Flash Next down", .kind = .plain, .n = 2560, .k = 640, .group = 32, .experts = 512, .top_k = 10, .fc1 = false },
    .{ .name = "Qwen3.6-35B-A3B gate+up", .kind = .pair, .n = 1024, .k = 2048, .group = 64, .experts = 256, .top_k = 8, .fc1 = true },
    .{ .name = "Qwen3.6-35B-A3B down", .kind = .plain, .n = 2048, .k = 512, .group = 64, .experts = 256, .top_k = 8, .fc1 = false },
};

const most_rows = 17;
const bits = 4;

/// A verify window: `tokens` rows with random picks, or with picks giving every item at most `cap` rows, one exactly.
const Window = struct { tokens: usize, cap: ?usize = null };

/// 1 to 8 random rows, the 16 and 17 of the long windows with items of 4 rows or fewer, and both sides of the rule.
const windows = [_]Window{
    .{ .tokens = 1 },           .{ .tokens = 2 },           .{ .tokens = 3 }, .{ .tokens = 4 },            .{ .tokens = 5 },
    .{ .tokens = 6 },           .{ .tokens = 7 },           .{ .tokens = 8 }, .{ .tokens = 16, .cap = 4 }, .{ .tokens = 17, .cap = 4 },
    .{ .tokens = 8, .cap = 4 }, .{ .tokens = 8, .cap = 5 },
};

/// The inputs' values: random, or with signed zeros, infinities and NaN planted in x and the group tables.
const Fill = enum { random, special };

fn launch(t: *Rig, k: *const Kernels, c: Case, arg: affine.Arg, items: c_int) !void {
    const r = &t.base.r;
    const s = t.base.stream.handle;
    switch (c.kind) {
        .plain => try k.routed(r, arg, items, s),
        .pair => if (!try k.pairRun(r, arg, 0, items, s)) return error.NoTile,
        .relu2 => if (!try k.routedRelu2(r, arg, items, s)) return error.NoTile,
    }
    try t.base.stream.synchronize();
}

/// -0 in every 7th x value of two rows, infinities and NaN in a few x values and in a column's tables an expert.
fn plant(t: *Rig, c: Case, hx: []u16, hs: []u16, hb: []u16) void {
    for (hx[0 .. 2 * c.k], 0..) |*v, i| if (i % 7 == 0) {
        v.* = 0x8000;
    };
    const inf: u16 = if (t.fp16) 0x7c00 else 0x7f80;
    const nan: u16 = if (t.fp16) 0x7e00 else 0x7fc0;
    hx[2 * c.k + 5] = nan;
    hx[3 * c.k + 9] = inf;
    hx[4 * c.k + 1] = inf | 0x8000;
    const groups = c.k / @as(usize, @intCast(c.group));
    for (0..c.experts) |e| {
        const at = (e * c.n + (7 * e) % c.n) * groups + e % groups;
        switch (e % 3) {
            0 => hs[at] = nan,
            1 => hb[at] = inf,
            else => hs[at] = inf | 0x8000,
        }
    }
}

/// The pairs' experts: random distinct ones a token, or `cap` tokens in turn on one expert of each slot's own set.
fn pickExperts(rng: *data.Rng, c: Case, w: Window, picks: *[most_rows][16]usize) void {
    if (w.cap) |cap| {
        var order: [512]usize = undefined;
        for (order[0..c.experts], 0..) |*e, i| e.* = i;
        var i: usize = c.experts;
        while (i > 1) : (i -= 1) std.mem.swap(usize, &order[i - 1], &order[rng.next() % i]);
        const sets = (w.tokens + cap - 1) / cap;
        for (picks[0..w.tokens], 0..) |*row, t| for (row[0..c.top_k], 0..) |*p, slot| {
            p.* = order[slot * sets + t / cap];
        };
        return;
    }
    for (picks[0..w.tokens]) |*row| for (row[0..c.top_k], 0..) |*p, slot| {
        pick: while (true) {
            p.* = @intCast(rng.next() % c.experts);
            for (row[0..slot]) |q| if (q == p.*) continue :pick;
            break;
        }
    };
}

/// One case at every window: the 32-bit words that differ from the tile's, and from each pair alone.
fn compare(t: *Rig, rng: *data.Rng, c: Case, fill: Fill) ![2]usize {
    var reference = t.kernels.*;
    reference.variants = .{};
    const words_row = c.k * bits / 32;
    const groups = c.k / @as(usize, @intCast(c.group));
    const pairs_most = most_rows * c.top_k;
    const xrows: usize = if (c.fc1) most_rows else pairs_most;
    const hx = try data.fill(gpa, u16, xrows * c.k, rng, if (t.fp16) data.makeX16 else data.makeXB);
    defer gpa.free(hx);
    const hw = try data.fill(gpa, u32, c.experts * c.n * words_row, rng, data.makeWord);
    defer gpa.free(hw);
    const hs = try data.fill(gpa, u16, c.experts * c.n * groups, rng, data.makeScale);
    defer gpa.free(hs);
    const hb = try data.fill(gpa, u16, c.experts * c.n * groups, rng, data.makeBias);
    defer gpa.free(hb);
    if (fill == .special) plant(t, c, hx, hs, hb);
    var x = try Buffer.fromHost(t.base, hx);
    defer x.free();
    var words = try Buffer.fromHost(t.base, hw);
    defer words.free();
    var scale = try Buffer.fromHost(t.base, hs);
    defer scale.free();
    var bias = try Buffer.fromHost(t.base, hb);
    defer bias.free();
    const cols = if (c.kind == .pair) c.n / 2 else c.n;
    const size: usize = if (c.kind == .plain) 4 else 2;
    const bytes = pairs_most * cols * size;
    var outs: [3]Buffer = undefined;
    var made: usize = 0;
    defer for (outs[0..made]) |*o| o.free();
    for (&outs) |*o| {
        o.* = try Buffer.alloc(t.base, bytes);
        made += 1;
    }
    var diff: [2]usize = .{ 0, 0 };
    for (windows) |w| {
        const m = w.tokens;
        // each token picks top_k distinct experts; a plan's items are its experts in order, their pairs in order
        var picks: [most_rows][16]usize = undefined;
        pickExperts(rng, c, w, &picks);
        const pairs = m * c.top_k;
        var items: [3 * most_rows * 16]c_int = undefined;
        var alone: [3 * most_rows * 16]c_int = undefined;
        var members: [most_rows * 16]c_int = undefined;
        var held: usize = 0;
        var at: usize = 0;
        var most: c_int = 0;
        for (0..c.experts) |e| {
            const first = at;
            for (0..pairs) |p| if (picks[p / c.top_k][p % c.top_k] == e) {
                members[at] = @intCast(p);
                alone[3 * at ..][0..3].* = .{ @intCast(e), @intCast(at), 1 };
                at += 1;
            };
            if (at == first) continue;
            items[3 * held ..][0..3].* = .{ @intCast(e), @intCast(first), @intCast(at - first) };
            most = @max(most, @as(c_int, @intCast(at - first)));
            held += 1;
        }
        var plan = try Buffer.fromHost(t.base, @as([]const c_int, items[0 .. 3 * held]));
        defer plan.free();
        var plan_alone = try Buffer.fromHost(t.base, @as([]const c_int, alone[0 .. 3 * pairs]));
        defer plan_alone.free();
        var mem = try Buffer.fromHost(t.base, @as([]const c_int, members[0..pairs]));
        defer mem.free();
        if (w.cap) |cap| {
            // only plain fp32-out K <= single_round_k plans with largest item 3 or 4 set item_rows
            var probe = std.mem.zeroInit(affine.Arg, .{
                .m = most,
                .n = @as(c_int, @intCast(c.n)),
                .k = @as(c_int, @intCast(c.k)),
                .bits = bits,
                .group = c.group,
                .out = @as(u64, if (c.kind == .plain) 1 else 0),
                .out16 = @as(u64, if (c.kind == .plain) 0 else 1),
                .route = .{ .items = 1, .members = 1 },
            });
            const pair_cols: c_int = if (c.kind == .pair) @intCast(c.n / 2) else 0;
            const tile = t.kernels.variants.pick(t.kernels.caps, &probe, 0, @intCast(c.n), @intCast(held), pair_cols, c.kind == .relu2);
            const want = cap <= 4 and c.kind == .plain and c.k <= stream_pick.single_round_k;
            try base.expect(
                most == cap and tile.item_rows == want and (probe.route.item_rows != 0) == want,
                "{s}: largest item {d} of cap {d}, item_rows {}",
                .{ c.name, most, cap, tile.item_rows },
            );
        }
        const len = pairs * cols * size;
        for (outs, 0..) |out, i| {
            try out.fill8(0xA5, null);
            const arg: affine.Arg = .{
                .x = x.ptr,
                .words = words.ptr,
                .scale = .{ .p = scale.ptr, .kind = 1 },
                .bias = .{ .p = bias.ptr, .kind = 1 },
                .out = if (c.kind == .plain) out.ptr else 0,
                .out16 = if (c.kind == .plain) 0 else out.ptr,
                .m = if (i == 2) 1 else most,
                .n = @intCast(c.n),
                .k = @intCast(c.k),
                .bits = bits,
                .group = c.group,
                .fp16 = @intFromBool(t.fp16),
                .route = .{ .items = if (i == 2) plan_alone.ptr else plan.ptr, .members = mem.ptr, .x_div = if (c.fc1) @intCast(c.top_k) else 1 },
            };
            // the served launcher, the launcher with no variants, and every pair an item of its own
            try launch(t, if (i == 1) &reference else t.kernels, c, arg, @intCast(if (i == 2) pairs else held));
        }
        for ([_]usize{ 1, 2 }, 0..) |o, d| {
            const n = try data.sameOnDevice(outs[0], outs[o], len);
            if (n != 0) std.debug.print("  item rows {s} {t}, {d} rows, largest item {d}: {d} words differ from {s}\n", .{ c.name, fill, m, most, n, if (d == 0) "the tile" else "each pair alone" });
            diff[d] += n;
        }
    }
    return diff;
}

test "the routed item-rows tile writes the tile's bytes and each pair's bytes alone, at 1 to 8, 16 and 17 rows of served plans" {
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    var rig = Rig.open(b);
    if (!rig.kernels.variants.enabled) return error.SkipZigTest; // picked on RDNA3.5 only
    var rng: data.Rng = .{ .state = 0x1D872B41C4A6F3E5 };
    var diff: [2]usize = .{ 0, 0 };
    for (cases) |c| for ([_]Fill{ .random, .special }) |fill| {
        const d = try compare(&rig, &rng, c, fill);
        diff[0] += d[0];
        diff[1] += d[1];
    };
    try base.expect(diff[0] == 0 and diff[1] == 0, "item rows: {d} words differ from the tile, {d} from each pair alone", .{ diff[0], diff[1] });
}
