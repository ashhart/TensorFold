//! The decode walk's softmax stats and partial sums past the one-block walk: the max, and each lane's tiles in order.

const std = @import("std");
const rig = @import("rig.zig");
const util = @import("../launches/util.zig");
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

fn ptr(b: DeviceBuffer) ?*anyopaque {
    return @ptrFromInt(at(b));
}

/// bf16 rounded to nearest even, as the kernels' float_to_bf16 rounds a finite value.
fn bf16Even(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + ((u >> 16) & 1) + 0x7fff) >> 16);
}

test "partial sums add each lane's tiles in tile order, into fp32 and the activation kinds" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x5A5 };
    const heads = 4;
    var cases: usize = 0;
    for ([_]usize{ 1, 33, 512, 1024 }) |tiles| for ([_]usize{ 64, 128, 256 }) |d| for ([_]usize{ 1, 3 }) |rows| {
        const host = try gpa.alloc(f32, rows * heads * tiles * d);
        defer gpa.free(host);
        for (host) |*v| v.* = @floatCast(rng.normal());
        var partials = try t.upload(host);
        defer partials.free();
        var out = try t.alloc(rows * heads * d * 4);
        defer out.free();
        var half = try t.alloc(rows * heads * d * 2);
        defer half.free();
        const want = try gpa.alloc(f32, rows * heads * d);
        defer gpa.free(want);
        // a lane adds its column's tiles one after another from tile 0
        for (0..rows * heads) |row| for (0..d) |lane| {
            var acc: f32 = 0;
            for (0..tiles) |tile| acc += host[(row * tiles + tile) * d + lane];
            want[row * d + lane] = acc;
        };
        const s = t.stream.handle;
        var e: util.Args = .{};
        try e.add(ptr(partials));
        try e.add(ptr(out));
        try e.add(@as(c_int, @intCast(tiles)));
        try e.add(@as(c_int, @intCast(d)));
        try t.on.go(t.on.sum_partials, util.dim(heads, rows, 1), util.dim(256, 1, 1), 0, s, &e);
        try t.stream.synchronize();
        const got = try rig.download(f32, out);
        defer gpa.free(got);
        for (got, want, 0..) |g, w, i| try rig.expect(
            @as(u32, @bitCast(g)) == @as(u32, @bitCast(w)),
            "sum of {d} tiles, d {d}, {d} rows: element {d} is {e}, the sum in tile order {e}",
            .{ tiles, d, rows, i, g, w },
        );
        for ([_]c_int{ 1, 2 }) |kind| {
            var a: util.Args = .{};
            try a.add(ptr(partials));
            try a.add(ptr(half));
            try a.add(@as(c_int, @intCast(tiles)));
            try a.add(@as(c_int, @intCast(d)));
            try a.add(kind);
            try t.on.go(t.on.plan.sum_act, util.dim(heads, rows, 1), util.dim(256, 1, 1), 0, s, &a);
            try t.stream.synchronize();
            const bits = try rig.download(u16, half);
            defer gpa.free(bits);
            for (bits, want, 0..) |b, w, i| {
                const r: u16 = if (kind == 1) @bitCast(@as(f16, @floatCast(w))) else bf16Even(w);
                try rig.expect(b == r, "sum into kind {d} of {d} tiles, d {d}: element {d} is {x}, the rounded sum {x}", .{ kind, tiles, d, i, b, r });
            }
        }
        cases += 1;
    };
    std.debug.print("  partial sums: {d} cases, each lane's tiles in tile order\n", .{cases});
}

test "softmax stats take the row's max exactly and a sum that bounds the row, past the one-block walk" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x57A7 };
    const heads = 4;
    var cases: usize = 0;
    for ([_][2]usize{ .{ 4352, 4097 }, .{ 16384, 16384 }, .{ 16384, 9001 }, .{ 65536, 65536 }, .{ 65536, 65535 } }) |sv| for ([_]usize{ 1, 2 }) |rows| {
        const span = sv[0];
        const visible = sv[1];
        const host = try gpa.alloc(f32, rows * heads * span);
        defer gpa.free(host);
        // keys past the visible ones hold -inf, as the score kernel writes them
        for (host, 0..) |*v, i| v.* = if (i % span < visible) @floatCast(8 * rng.normal()) else -std.math.inf(f32);
        // each row's max, one above the rest, sits at a key of its own: last visible, first, past 4096, mid-row
        const tops = [_]usize{ visible - 1, 0, 4096, visible / 2 + 255 };
        for (0..rows * heads) |row| {
            const keys = host[row * span ..][0..visible];
            var top: f32 = -std.math.inf(f32);
            for (keys) |k| top = @max(top, k);
            keys[tops[row % tops.len]] = top + 1;
        }
        var scores = try t.upload(host);
        defer scores.free();
        var stats = try t.alloc(rows * heads * 2 * 4);
        defer stats.free();
        var b: util.Args = .{};
        try b.add(ptr(scores));
        try b.add(ptr(stats));
        try b.add(@as(c_int, @intCast(span)));
        try b.add(@as(c_int, @intCast(visible)));
        try b.add(@as(?*const c_int, null));
        try t.on.go(t.on.softmax_stats, util.dim(heads, rows, 1), util.dim(256, 1, 1), 0, t.stream.handle, &b);
        try t.stream.synchronize();
        const got = try rig.download(f32, stats);
        defer gpa.free(got);
        for (0..rows * heads) |row| {
            const keys = host[row * span ..][0..visible];
            var top: f32 = -std.math.inf(f32);
            for (keys) |k| top = @max(top, k);
            var sum: f64 = 0;
            for (keys) |k| sum += @exp(@as(f64, k) - top);
            const g = got[row * 2 ..][0..2];
            try rig.expect(@as(u32, @bitCast(g[0])) == @as(u32, @bitCast(top)), "stats over {d} of {d} keys: max {e}, the row's {e}", .{ visible, span, g[0], top });
            try rig.expect(@abs(g[1] - sum) <= 1e-5 * sum, "stats over {d} of {d} keys: sum {e}, float64 {e}", .{ visible, span, g[1], sum });
        }
        cases += 1;
    };
    std.debug.print("  softmax stats: {d} cases, the max exactly and the sum within 1e-5\n", .{cases});
}
