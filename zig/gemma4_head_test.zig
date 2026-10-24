//! Production prompt/decode bindings, evaluated by a CPU oracle; this does not emulate Metal reduction precision.
const std = @import("std");
const mtl = @import("metal");
const weights = @import("src/families/gemma4/weights.zig");
const prompt = @import("src/families/gemma4/prompt.zig");
const cfg = @import("src/families/gemma4/config.zig");
const row = @import("src/core/row_projection.zig");
const lane = @import("src/core/lane_projection.zig");
const D = 64;
const Row = [D]u16;

fn bf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% (0x7fff + ((u >> 16) & 1))) >> 16);
}
fn value(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}
fn norm(x: Row) Row {
    var sum: f32 = 0;
    for (x) |v| sum += value(v) * value(v);
    const inv = 1 / @sqrt(sum / D + cfg.eps);
    var y: Row = undefined;
    for (x, &y) |v, *o| o.* = bf(value(v) * inv);
    return y;
}
fn add(a: Row, b: Row) Row {
    var y: Row = undefined;
    for (a, b, &y) |x, v, *o| o.* = bf(value(x) + value(v));
    return y;
}

// Two small causal-attention/residual/GeGLU layers with BF16 boundaries and a final RMS, no checkpoint payload.
fn stack(rows: usize, out: []Row) void {
    for (out, 0..) |*x, r| for (x, 0..) |*v, j| {
        const t = @as(f32, @floatFromInt(r)) / @as(f32, @floatFromInt(@max(rows - 1, 1)));
        v.* = bf(if (j == 0) 3 - 2.75 * t else if (j == 1) 0.25 + 2.75 * t else 0.01 * @as(f32, @floatFromInt(j % 7)));
    };
    for (0..2) |_| {
        var prefix: [D]f32 = @splat(0);
        for (out, 0..) |*x, r| {
            const nx = norm(x.*);
            var attn: Row = undefined;
            for (nx, &prefix, &attn) |v, *p, *o| {
                p.* += value(v);
                o.* = bf(0.03125 * p.* / @as(f32, @floatFromInt(r + 1)));
            }
            const h = add(x.*, norm(attn));
            const nm = norm(h);
            var down: Row = undefined;
            for (nm, &down) |v, *o| {
                const gate = value(v) * 0.125;
                const gelu = 0.5 * gate * (1 + std.math.tanh(0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)));
                o.* = bf(value(bf(gelu * value(v) * 0.125)) * 0.03125);
            }
            x.* = add(h, norm(down));
        }
    }
    for (out) |*x| x.* = norm(x.*);
}
fn code(j: usize, k: usize) u4 {
    return if (j < 2 and j == k) 15 else if (j >= 2 and k == (j + 2) % D) 1 else 0;
}
fn logits(x: Row) Row {
    var y: Row = undefined;
    for (&y, 0..) |*v, j| {
        var sum: f32 = 0;
        for (x, 0..) |a, k| sum += value(a) * @as(f32, @floatFromInt(code(j, k))) * 0.0625;
        v.* = bf(cfg.softcap * std.math.tanh(value(bf(sum)) / cfg.softcap));
    }
    return y;
}
fn argmax(y: Row) usize {
    var best: usize = 0;
    for (y, 0..) |v, j| if (value(v) > value(y[best])) {
        best = j;
    };
    return best;
}
fn buf(id: usize) mtl.Buffer {
    return .{ .id = id };
}
fn linear(raw: bool) weights.Linear {
    return .{ .w = buf(2), .sb = buf(3), .n = D, .k = D, .format = .{ .bits = 4, .group = 64 }, .raw = if (raw) row.Weights{ .w = buf(2), .scales = buf(3), .biases = buf(4), .n = D, .k = D } else null };
}
fn simdPipes() row.simd_bits.Pipelines {
    var p = row.simd_bits.Pipelines{ .device = .{}, .lib = .{} };
    for (&p.built) |*a| for (a) |*b| for (b) |*c| for (c) |*v| {
        v.* = .{ .id = 101 };
    };
    return p;
}
fn projection() lane.Projection {
    return .{ .pipe = .{ .id = 201 }, .n = D, .k = D, .group_count = 1, .weight_bytes = D * D / 2, .metadata_bytes = D * 4, .output_bytes = 2, .y_stride = D, .tile_count = 2, .sk = 1, .reg = true, .sums_pipe = .{ .id = 200 }, .sums_buffer = buf(7) };
}
fn args(p: mtl.Pass) row.simd_bits.Args {
    return std.mem.bytesToValue(row.simd_bits.Args, p.values[4][0..@sizeOf(row.simd_bits.Args)]);
}

test "gemma prompt head: both paths equal all-row matrix logits on the last synthetic layer row" {
    const a = std.testing.allocator;
    for ([_]usize{ 1, 2, 7, 16, 17, 511, 512 }) |rows| {
        const hidden = try a.alloc(Row, rows);
        defer a.free(hidden);
        stack(rows, hidden);
        // Independent matrix oracle evaluates every row; it never uses the launcher's selected address.
        const all = try a.alloc(Row, rows);
        defer a.free(all);
        for (hidden, all) |x, *y| y.* = logits(x);
        if (rows > 1) {
            try std.testing.expectEqual(@as(usize, 0), argmax(all[0]));
            try std.testing.expectEqual(@as(usize, 1), argmax(all[rows - 1]));
        }
        var both: [2]Row = undefined;
        for ([_]bool{ true, false }, 0..) |raw, pi| {
            const l = linear(raw);
            var ps = weights.Projections{ .geo = cfg.unified12b, .row_simd = simdPipes() };
            try ps.list.append(a, .{ .x = l, .p = projection() });
            defer ps.list.deinit(a);
            var record: mtl.Record = .{};
            try prompt.projectHead(a, .{}, &ps, .{ .r = &record }, l, buf(1), buf(5), rows);
            try std.testing.expectEqual(@as(usize, if (raw) 1 else 2), record.n);
            const p = record.passes[record.n - 1];
            if (!raw) try std.testing.expectEqual(p.buffers[0], record.passes[0].buffers[0]);
            const input = p.buffers[0];
            try std.testing.expectEqual(@as(usize, 1), input.id);
            try std.testing.expect(input.off % @sizeOf(Row) == 0);
            const selected = input.off / @sizeOf(Row);
            try std.testing.expect(selected < rows);
            both[pi] = logits(hidden[selected]);
            try std.testing.expectEqualSlices(u16, &all[rows - 1], &both[pi]);
            try std.testing.expectEqual((rows - 1) * D * 2, input.off);
            if (raw) {
                try std.testing.expectEqual(@as(i32, 1), args(p).r);
                try std.testing.expectEqual(@as(i32, D), args(p).ld);
                try std.testing.expectEqual(mtl.Bound{ .id = 5 }, p.buffers[5]);
            } else {
                const shape = std.mem.bytesToValue([2]i32, p.values[3][0..8]);
                try std.testing.expectEqual(@as(i32, 1), shape[0]);
                try std.testing.expectEqual(mtl.Bound{ .id = 5 }, p.buffers[4]);
            }
        }
        try std.testing.expectEqualSlices(u16, &both[0], &both[1]);
    }
}

test "gemma prompt head: a selected input ref survives the real row encoder at each code width" {
    for (row.simd_bits.bit_widths) |bits| for ([_]usize{ 1, 2, 16, 17 }) |rows| {
        var ps = weights.Projections{ .geo = cfg.unified12b, .row_simd = simdPipes() };
        var l = linear(true);
        l.format.bits = bits;
        l.raw.?.bits = bits;
        l.col = 32;
        l.y_stride = 128;
        var record: mtl.Record = .{};
        try weights.projectRow(&ps, .{}, .{ .r = &record }, l.raw.?, l, .{ .buf = buf(1), .off = 256 }, buf(5), rows);
        const p = record.passes[0];
        try std.testing.expectEqual(mtl.Bound{ .id = 1, .off = 256 }, p.buffers[0]);
        try std.testing.expectEqual(mtl.Bound{ .id = 5, .off = 64 }, p.buffers[5]);
        try std.testing.expectEqual(@as(i32, @intCast(rows)), args(p).r);
        try std.testing.expectEqual(@as(i32, 128), args(p).ld);
    };
}

test "gemma prompt head: zero rows refused and first decode starts from its own row" {
    var ps = weights.Projections{ .geo = cfg.unified12b, .row_simd = simdPipes() };
    var record: mtl.Record = .{};
    const e = mtl.ComputeEncoder{ .r = &record };
    const l = linear(true);
    try std.testing.expectError(error.BadChunk, prompt.projectHead(std.testing.allocator, .{}, &ps, e, l, buf(1), buf(5), 0));
    try std.testing.expectEqual(@as(usize, 0), record.n);
    // The production decode caller supplies this zero-offset Ref for its first (or window's first) row.
    try weights.projectRow(&ps, .{}, e, l.raw.?, l, .{ .buf = buf(6) }, buf(5), 1);
    try std.testing.expectEqual(mtl.Bound{ .id = 6 }, record.passes[0].buffers[0]);
    try std.testing.expectEqual(@as(i32, 1), args(record.passes[0]).r);
}

test "gemma prompt head: actual 12B and 31B hidden widths select the final chunk row" {
    for ([_]cfg.Geometry{ cfg.unified12b, cfg.dense31b }) |geo| for ([_]usize{ 2, 511, 512 }) |rows| {
        var ps = weights.Projections{ .geo = geo, .row_simd = simdPipes() };
        var l = linear(true);
        l.n = cfg.vocab;
        l.k = geo.hidden;
        l.raw.?.n = l.n;
        l.raw.?.k = l.k;
        var record: mtl.Record = .{};
        try prompt.projectHead(std.testing.allocator, .{}, &ps, .{ .r = &record }, l, buf(1), buf(5), rows);
        try std.testing.expectEqual(mtl.Bound{ .id = 1, .off = (rows - 1) * geo.hidden * 2 }, record.passes[0].buffers[0]);
        try std.testing.expectEqual(@as(i32, 1), args(record.passes[0]).r);
        try std.testing.expectEqual(@as(i32, cfg.vocab), args(record.passes[0]).n);
    };
}
