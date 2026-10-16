//! Rows are the same bytes however a product is cut: decode's lane rounds, prefill's tiles and the router's kernels.

const std = @import("std");
const base = @import("../rig.zig");
const affine = @import("../../launches/affine.zig");
const registry = @import("core").registry;
const runtime = @import("../../runtime.zig");
const Buffer = base.Buffer;
const gpa = base.gpa;
const data = @import("data.zig");
const prod = @import("product.zig");
const shapes = @import("cases.zig");
const Rig = prod.Rig;
const Entry = affine.Entry;

const Case = struct { name: []const u8, n: usize, k: usize, bits: c_int = 4, group: c_int = 64 };

const cases = [_]Case{
    .{ .name = "vocab", .n = 248320, .k = 2048 },
    .{ .name = "qkv", .n = 8192, .k = 2048 },
    .{ .name = "down", .n = 2048, .k = 512 },
    .{ .name = "9b gate_up", .n = 24576, .k = 4096 },
    .{ .name = "ragged", .n = 1001, .k = 2112 },
    .{ .name = "2-bit", .n = 4096, .k = 2048, .bits = 2 },
    .{ .name = "3-bit", .n = 4096, .k = 2048, .bits = 3 },
    .{ .name = "5-bit", .n = 4096, .k = 2048, .bits = 5 },
    .{ .name = "6-bit group 32", .n = 4096, .k = 2048, .bits = 6, .group = 32 },
    .{ .name = "8-bit group 128", .n = 4096, .k = 2048, .bits = 8, .group = 128 },
};

/// Row counts launched together: every count of a lane round up to 16, then wider rounds.
const together = [_]usize{ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 32 };
const most_rows = 32;

/// One product at every row count: the number of rows whose bytes differ from the same row alone.
fn product(t: *Rig, rng: *data.Rng, c: Case, half: bool) !usize {
    const words_row = c.k * @as(usize, @intCast(c.bits)) / 32;
    const groups = c.k / @as(usize, @intCast(c.group));
    const hx = try data.fill(gpa, u16, most_rows * c.k, rng, if (t.fp16) data.makeX16 else data.makeXB);
    defer gpa.free(hx);
    const hw = try data.fill(gpa, u32, c.n * words_row, rng, data.makeWord);
    defer gpa.free(hw);
    const hs = try data.fill(gpa, u16, c.n * groups, rng, data.makeScale);
    defer gpa.free(hs);
    const hb = try data.fill(gpa, u16, c.n * groups, rng, data.makeBias);
    defer gpa.free(hb);
    var x = try Buffer.fromHost(t.base, hx);
    defer x.free();
    var words = try Buffer.fromHost(t.base, hw);
    defer words.free();
    var scale = try Buffer.fromHost(t.base, hs);
    defer scale.free();
    var bias = try Buffer.fromHost(t.base, hb);
    defer bias.free();
    const size: usize = if (half) 2 else 4;
    var out = try Buffer.alloc(t.base, most_rows * c.n * size);
    defer out.free();
    const alone = try gpa.alloc(u8, most_rows * c.n * size);
    defer gpa.free(alone);
    const got = try gpa.alloc(u8, most_rows * c.n * size);
    defer gpa.free(got);

    var arg: affine.Arg = .{
        .x = x.ptr,
        .words = words.ptr,
        .scale = .{ .p = scale.ptr, .kind = 1 },
        .bias = .{ .p = bias.ptr, .kind = 1 },
        .out = out.ptr,
        .m = 1,
        .n = @intCast(c.n),
        .k = @intCast(c.k),
        .bits = c.bits,
        .group = c.group,
        .fp16 = @intFromBool(t.fp16),
    };
    // each row alone
    for (0..most_rows) |i| {
        arg.x = x.ptr + i * c.k * 2;
        arg.m = 1;
        try out.fill8(0xA5, null);
        try t.kernels.run(&t.base.r, arg, 4, t.base.stream.handle, 0, 1, half);
        try t.base.stream.synchronize();
        try out.download(0, alone[i * c.n * size ..][0 .. c.n * size]);
    }
    // the rows together
    var bad: usize = 0;
    arg.x = x.ptr;
    for (together) |m| {
        arg.m = @intCast(m);
        try out.fill8(0xA5, null);
        try t.kernels.run(&t.base.r, arg, 4, t.base.stream.handle, 0, 1, half);
        try t.base.stream.synchronize();
        try out.download(0, got[0 .. m * c.n * size]);
        for (0..m) |i| {
            const row = i * c.n * size;
            if (!std.mem.eql(u8, got[row..][0 .. c.n * size], alone[row..][0 .. c.n * size])) {
                if (bad < 6) std.debug.print("  {s} {s}: row {d} of {d} differs from the row alone\n", .{ c.name, if (half) "rounded" else "fp32", i, m });
                bad += 1;
            }
        }
    }
    return bad;
}

const Plan = struct { counts: [4]c_int };
const plans = [_]Plan{ .{ .counts = .{ 5, 3, 6, 2 } }, .{ .counts = .{ 15, 9, 1, 4 } }, .{ .counts = .{ 2, 2, 2, 2 } }, .{ .counts = .{ 1, 4, 1, 3 } } };
const routed_cases = [_]Case{
    .{ .name = "moe gate_up", .n = 1024, .k = 2048 },
    .{ .name = "moe down", .n = 2048, .k = 512 },
    .{ .name = "ragged", .n = 1002, .k = 2112 },
    .{ .name = "3-bit", .n = 1024, .k = 2048, .bits = 3 },
    .{ .name = "8-bit", .n = 1024, .k = 2048, .bits = 8, .group = 128 },
};

fn launchRouted(t: *Rig, arg: affine.Arg, items: c_int, pair: bool) !void {
    if (pair) {
        if (!try t.kernels.pairRun(&t.base.r, arg, 0, items, t.base.stream.handle)) return error.NoTile;
    } else {
        try t.kernels.routed(&t.base.r, arg, items, t.base.stream.handle);
    }
    try t.base.stream.synchronize();
}

/// A routed plan's pairs together, against every pair as an item of its own: the number of output rows that differ.
fn routedProduct(t: *Rig, rng: *data.Rng, c: Case, pair: bool) !usize {
    const experts = 4;
    const words_row = c.k * @as(usize, @intCast(c.bits)) / 32;
    const groups = c.k / @as(usize, @intCast(c.group));
    const hx = try data.fill(gpa, u16, 32 * c.k, rng, if (t.fp16) data.makeX16 else data.makeXB);
    defer gpa.free(hx);
    const hw = try data.fill(gpa, u32, experts * c.n * words_row, rng, data.makeWord);
    defer gpa.free(hw);
    const hs = try data.fill(gpa, u16, experts * c.n * groups, rng, data.makeScale);
    defer gpa.free(hs);
    const hb = try data.fill(gpa, u16, experts * c.n * groups, rng, data.makeBias);
    defer gpa.free(hb);
    var x = try Buffer.fromHost(t.base, hx);
    defer x.free();
    var words = try Buffer.fromHost(t.base, hw);
    defer words.free();
    var scale = try Buffer.fromHost(t.base, hs);
    defer scale.free();
    var bias = try Buffer.fromHost(t.base, hb);
    defer bias.free();
    const cols = if (pair) c.n / 2 else c.n;
    const size: usize = if (pair) 2 else 4;
    var out = try Buffer.alloc(t.base, 32 * cols * size);
    defer out.free();
    const alone = try gpa.alloc(u8, 32 * cols * size);
    defer gpa.free(alone);
    const got = try gpa.alloc(u8, 32 * cols * size);
    defer gpa.free(got);
    var members: [32]c_int = undefined;
    for (&members, 0..) |*mb, i| mb.* = @intCast(i);
    var mem = try Buffer.fromHost(t.base, @as([]const c_int, &members));
    defer mem.free();

    var bad: usize = 0;
    for (plans) |plan| {
        var rows: usize = 0;
        var item_alone: [96]c_int = undefined;
        var item_all: [12]c_int = undefined;
        var most: c_int = 0;
        for (plan.counts, 0..) |n, e| {
            item_all[3 * e] = @intCast(e);
            item_all[3 * e + 1] = @intCast(rows);
            item_all[3 * e + 2] = n;
            most = @max(most, n);
            for (0..@as(usize, @intCast(n))) |_| {
                item_alone[3 * rows] = @intCast(e);
                item_alone[3 * rows + 1] = @intCast(rows);
                item_alone[3 * rows + 2] = 1;
                rows += 1;
            }
        }
        var plan_alone = try Buffer.fromHost(t.base, @as([]const c_int, item_alone[0 .. 3 * rows]));
        defer plan_alone.free();
        var plan_all = try Buffer.fromHost(t.base, @as([]const c_int, &item_all));
        defer plan_all.free();
        var arg: affine.Arg = .{
            .x = x.ptr,
            .words = words.ptr,
            .scale = .{ .p = scale.ptr, .kind = 1 },
            .bias = .{ .p = bias.ptr, .kind = 1 },
            .out = if (pair) 0 else out.ptr,
            .out16 = if (pair) out.ptr else 0,
            .m = 1,
            .n = @intCast(c.n),
            .k = @intCast(c.k),
            .bits = c.bits,
            .group = c.group,
            .fp16 = @intFromBool(t.fp16),
            .route = .{ .items = plan_alone.ptr, .members = mem.ptr, .x_div = 1 },
        };
        try out.fill8(0xA5, null);
        try launchRouted(t, arg, @intCast(rows), pair);
        try out.download(0, alone[0 .. rows * cols * size]);
        arg.m = most;
        arg.route.items = plan_all.ptr;
        try out.fill8(0xA5, null);
        try launchRouted(t, arg, 4, pair);
        try out.download(0, got[0 .. rows * cols * size]);
        for (0..rows) |i| {
            const row = i * cols * size;
            if (!std.mem.eql(u8, got[row..][0 .. cols * size], alone[row..][0 .. cols * size])) {
                if (bad < 6) std.debug.print("  routed {s} {s}: pair {d} of {d} (most {d} rows) differs from the pair alone\n", .{ c.name, if (pair) "act" else "fp32", i, rows, most });
                bad += 1;
            }
        }
    }
    return bad;
}

/// Row counts the block shapes are compared at: every short one's edges, ragged ones and a few blocks.
const tier_rows = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 33, 63, 64, 65, 100, 128, 200 };

/// One product (dense, or routed with ragged and empty items) through every block shape: words unlike the 128-row's.
fn tierProduct(t: *Rig, m: usize, n: usize, k: usize, bits: c_int, group: c_int, routed: bool) !usize {
    const experts: usize = if (routed) 5 else 1;
    const words_row = k * @as(usize, @intCast(bits)) / 32;
    const groups = k / @as(usize, @intCast(group));
    var items: [3 * 256]i32 = undefined;
    var item_count: usize = 1;
    var max_rows = m;
    if (routed) {
        item_count = 0;
        max_rows = 1;
        var first: usize = 0;
        while (first < m) {
            const take = @min(m - first, 1 + t.rng.next() % 40);
            items[item_count * 3 ..][0..3].* = .{ @intCast(item_count % experts), @intCast(first), @intCast(take) };
            item_count += 1;
            first += take;
            max_rows = @max(max_rows, take);
        }
        items[item_count * 3 ..][0..3].* = .{ 0, @intCast(m), 0 };
        item_count += 1;
    }
    const pairs = try gpa.alloc(i32, m);
    defer gpa.free(pairs);
    for (pairs, 0..) |*p, i| p.* = @intCast(i);
    const x_div: usize = if (routed) 2 else 1;
    const hx = try data.fill(gpa, u16, (m / x_div + 1) * k, &t.rng, if (t.fp16) data.makeX16 else data.makeXB);
    defer gpa.free(hx);
    const hw = try data.fill(gpa, u32, experts * n * words_row, &t.rng, data.makeWord);
    defer gpa.free(hw);
    const hs = try data.fill(gpa, u16, experts * n * groups, &t.rng, data.makeScale);
    defer gpa.free(hs);
    const hb = try data.fill(gpa, u16, experts * n * groups, &t.rng, data.makeBias);
    defer gpa.free(hb);
    var x = try Buffer.fromHost(t.base, hx);
    defer x.free();
    var words = try Buffer.fromHost(t.base, hw);
    defer words.free();
    var scale = try Buffer.fromHost(t.base, hs);
    defer scale.free();
    var bias = try Buffer.fromHost(t.base, hb);
    defer bias.free();
    var plan = try Buffer.fromHost(t.base, items[0 .. item_count * 3]);
    defer plan.free();
    var members = try Buffer.fromHost(t.base, pairs);
    defer members.free();
    const bytes = m * n * 4;
    var arg: affine.Arg = .{
        .x = x.ptr,
        .words = words.ptr,
        .scale = .{ .p = scale.ptr, .kind = 1 },
        .bias = .{ .p = bias.ptr, .kind = 1 },
        .out = 0,
        .m = @intCast(max_rows),
        .n = @intCast(n),
        .k = @intCast(k),
        .bits = bits,
        .group = group,
        .fp16 = @intFromBool(t.fp16),
        .route = if (routed) .{ .items = plan.ptr, .members = members.ptr, .x_div = @intCast(x_div) } else .{},
    };
    // every tile the registry picks between by rows at this shape, twice, and the engine's own pick
    const k_ptr = t.kernels;
    var set: std.ArrayList(*const Entry) = .empty;
    defer set.deinit(gpa);
    try prod.pickable(k_ptr, arg, item_count, .prefill, k_ptr.env(), &set, gpa);
    var ids: [registry.max_entries]*const Entry = undefined;
    var n_ids: usize = 0;
    for (set.items) |e| if (prod.takesArg(k_ptr, e, arg, item_count, .prefill)) {
        ids[n_ids] = e;
        n_ids += 1;
    };
    if (n_ids == 0) return error.NoTile;
    const big = n_ids - 1;
    var outs: [2 * registry.max_entries + 1]Buffer = undefined;
    const n_outs = 2 * n_ids + 1;
    for (outs[0..n_outs], 0..) |*o, i| {
        o.* = try Buffer.alloc(t.base, bytes);
        errdefer for (outs[0..i]) |*f| f.free();
        try o.fill8(0xA5, null);
    }
    defer for (outs[0..n_outs]) |*o| o.free();
    for (0..2 * n_ids) |i| {
        arg.out = outs[i].ptr;
        try ids[i % n_ids].launch(k_ptr, .{ .r = &t.base.r, .s = t.base.stream.handle, .arg = arg, .items = @intCast(item_count) });
    }
    arg.out = outs[2 * n_ids].ptr;
    try k_ptr.prefillLaunch(&t.base.r, arg, t.base.stream.handle, @intCast(item_count));
    try t.base.stream.synchronize();
    const written = try gpa.alloc(u8, bytes);
    defer gpa.free(written);
    try outs[big].download(0, written);
    t.digest = std.hash.Wyhash.hash(t.digest, written);
    var diff: usize = 0;
    for (0..n_outs) |i| {
        if (i == big) continue;
        const d = try data.sameOnDevice(outs[big], outs[i], bytes);
        if (d != 0) {
            const what: []const u8 = if (i < 2 * n_ids) ids[i % n_ids].id else "the engine's pick";
            std.debug.print("{s} differs from {s} in {d} words\n", .{ what, ids[big].id, d });
            const ha = try gpa.alloc(f32, m * n);
            defer gpa.free(ha);
            const hb2 = try gpa.alloc(f32, m * n);
            defer gpa.free(hb2);
            try outs[big].download(0, std.mem.sliceAsBytes(ha));
            try outs[i].download(0, std.mem.sliceAsBytes(hb2));
            var shown: usize = 0;
            for (ha, hb2, 0..) |u, v, at| {
                if (@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)) and shown < 12) {
                    std.debug.print("  row {d} col {d}: {e} vs {e}\n", .{ at / n, at % n, u, v });
                    shown += 1;
                }
            }
        }
        diff += d;
    }
    return diff;
}

/// The router's logits through the wave-an-expert kernel and the 64 x 64 tiles at `r` rows: the words that differ.
fn routerProduct(t: *Rig, r: usize, d: usize, e: usize) !usize {
    const hx = try data.fill(gpa, u16, r * d, &t.rng, if (t.fp16) data.makeX16 else data.makeXB);
    defer gpa.free(hx);
    const hw = try gpa.alloc(f32, e * d);
    defer gpa.free(hw);
    for (hw) |*v| v.* = data.fine(&t.rng);
    var x = try Buffer.fromHost(t.base, hx);
    defer x.free();
    var w = try Buffer.fromHost(t.base, hw);
    defer w.free();
    var outs: [2]Buffer = undefined;
    for (&outs, 0..) |*o, i| {
        o.* = try Buffer.alloc(t.base, r * e * 4);
        errdefer for (outs[0..i]) |*f| f.free();
        try o.fill8(0xA5, null);
    }
    defer for (&outs) |*o| o.free();
    for (outs, [_]bool{ true, false }) |o, small| {
        try t.base.on.routerWith(@ptrFromInt(x.ptr), if (t.fp16) 1 else 2, @ptrFromInt(w.ptr), @ptrFromInt(o.ptr), @intCast(r), @intCast(d), @intCast(e), t.base.stream.handle, small);
    }
    try t.base.stream.synchronize();
    const written = try gpa.alloc(u8, r * e * 4);
    defer gpa.free(written);
    try outs[0].download(0, written);
    t.digest = std.hash.Wyhash.hash(t.digest, written);
    return data.sameOnDevice(outs[0], outs[1], r * e * 4);
}

/// Prefill's short blocks against the 128-row one at every `tier_rows`, dense and routed: any differing word fails.
fn tiers(t: *Rig) !usize {
    var ran: usize = 0;
    for (shapes.widths) |bits_u| for (shapes.groups) |group_u| for (tier_rows) |m| for ([_]bool{ false, true }) |routed| {
        // n off every block width, k a few stages or a long odd run
        const bits: c_int = bits_u;
        const group: c_int = group_u;
        const ks = [_]usize{ 3 * @as(usize, @intCast(group)), if (group == 32) 2080 else 1024 };
        for (ks, 0..) |k, j| {
            const n: usize = if (j == 0) 97 else 288;
            const diff = try tierProduct(t, m, n, k, bits, group, routed);
            try base.expect(diff == 0, "gemm tiers m{d} n{d} k{d} b{d} g{d} routed {}: {d} words differ from the 128-row block", .{ m, n, k, bits, group, routed, diff });
            ran += 1;
        }
    };
    for (tier_rows) |r| {
        for ([_]usize{ 96, 2048 }) |d| {
            const diff = try routerProduct(t, r, d, 259);
            try base.expect(diff == 0, "router logits r{d} d{d}: {d} words differ between the wave-an-expert kernel and the tiles", .{ r, d, diff });
            ran += 1;
        }
    }
    return ran;
}

/// Decode: the rows of a lane round's launch against each row alone, plain products and routed plans.
pub fn decode(t: *Rig) !void {
    var rng: data.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    var bad: usize = 0;
    for (cases) |c| for ([_]bool{ false, true }) |half| {
        bad += try product(t, &rng, c, half);
    };
    for (routed_cases) |c| for ([_]bool{ false, true }) |pair| {
        bad += try routedProduct(t, &rng, c, pair);
    };
    try base.expect(bad == 0, "rows decode: {d} rows differ from the same row alone", .{bad});
}

/// Prefill: every tile the registry picks between by rows, at every row count, width and group.
pub fn prefill(t: *Rig) !void {
    try base.expect(try tiers(t) > 0, "rows prefill: no product ran", .{});
}

test "a decode row writes the same bytes alone and in a round of 2 to 32, dense and routed, fp32 and rounded" {
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    var rig = Rig.open(b);
    try decode(&rig);
}

test "a prompt row writes the same bytes in every tile, dense and routed, every width and group, and the router's logits" {
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    var rig = Rig.open(b);
    try prefill(&rig);
}
