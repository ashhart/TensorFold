const std = @import("std");
const gpu = @import("tensor_metal");
const st = @import("safetensors");

pub fn checkUniform(g: *gpu.Gpu) !void {
    for ([_][3]usize{ .{ 5184, 1, 1 }, .{ 5185, 1, 1 }, .{ 65, 2, 2 } }) |shape| {
        const rows, const heads, const batch = shape;
        const q = try g.alloc(rows * heads * batch * 256, .f32);
        defer q.deinit();
        const k = try g.alloc(q.count, .f32);
        defer k.deinit();
        const v = try g.alloc(q.count, .f32);
        defer v.deinit();
        @memset(q.slice(f32, q.count), 0);
        @memset(k.slice(f32, k.count), 0);
        for (v.slice(f32, v.count), 0..) |*value, i| {
            const key = i / 256 / heads % rows;
            const h = i / 256 % heads;
            const b = i / 256 / heads / rows;
            const coefficient: i32 = @as(i32, @intCast(key % 31)) - 15 + @as(i32, @intCast(h + 3 * b));
            value.* = @as(f32, @floatFromInt(coefficient * @as(i32, @intCast(i % 256 % 17 + 1)))) / 512;
        }
        for ([_]bool{ false, true }) |causal| {
            if (rows > 65 and causal) continue;
            const out = try gpu.ops.attentionBatched(g, q, k, v, rows, heads, heads, causal, 0.0625, batch);
            defer out.deinit();
            try out.checkFinite("SAM3 uniform D256 attention");
            for (0..batch) |b| for (0..heads) |h| {
                var prefix: i64 = 0;
                var total: i64 = 0;
                for (0..rows) |key| total += @as(i64, @intCast(key % 31)) - 15 + @as(i64, @intCast(h + 3 * b));
                for (0..rows) |row| {
                    prefix += @as(i64, @intCast(row % 31)) - 15 + @as(i64, @intCast(h + 3 * b));
                    for (0..256) |d| {
                        const numerator = (if (causal) prefix else total) * @as(i64, @intCast(d % 17 + 1));
                        const denominator = (if (causal) row + 1 else rows) * 512;
                        const want = @as(f64, @floatFromInt(numerator)) / @as(f64, @floatFromInt(denominator));
                        const got = out.get(((b * rows + row) * heads + h) * 256 + d);
                        const bound = 3 * std.math.floatEps(f32) * @abs(want);
                        if (@abs(want - got) > bound) return error.SamUniformAttentionParity;
                    }
                }
            };
            std.debug.print("SAM D256 uniform attention rows={d} heads={d} batch={d} causal={}: every output matches FP64\n", .{ rows, heads, batch, causal });
        }
    }
}

fn load(init: std.process.Init, g: *gpu.Gpu, path: []const u8) !gpu.Tensor {
    var file = try st.File.open(init.gpa, init.io, path);
    defer file.close(init.io);
    if (file.names.count() != 1) return error.ExpectedOneReplayTensor;
    const source = file.get(file.names.keys()[0]).?;
    if (source.dtype != .f32 or source.rank != 2 or source.dim(1) != 256) return error.ExpectedSam3AttentionMatrix;
    const tensor = try g.alloc(source.numel(), .f32);
    errdefer tensor.deinit();
    @memcpy(tensor.slice(u8, source.bytes.len), source.bytes);
    try tensor.checkFinite(path);
    return tensor;
}

fn firstNonFinite(tensor: gpu.Tensor) ?usize {
    for (tensor.slice(f32, tensor.count), 0..) |value, i| if (!std.math.isFinite(value)) return i;
    return null;
}

pub fn run(init: std.process.Init, g: *gpu.Gpu, args: []const [:0]const u8) !void {
    if (args.len != 3 and args.len != 4) return error.ExpectedQueryKeyValueAndOptionalRow;
    const q = try load(init, g, args[0]);
    defer q.deinit();
    const k = try load(init, g, args[1]);
    defer k.deinit();
    const v = try load(init, g, args[2]);
    defer v.deinit();
    if (k.count != v.count) return error.AttentionReplayShape;
    const rows = q.count / 256;
    const keys = k.count / 256;
    const full = try gpu.ops.attentionCross(g, q, k, v, rows, keys, 1, 0.0625);
    defer full.deinit();
    for (1..5) |_| {
        const repeated = try gpu.ops.attentionCross(g, q, k, v, rows, keys, 1, 0.0625);
        defer repeated.deinit();
        if (!std.mem.eql(u8, full.slice(u8, full.count * 4), repeated.slice(u8, repeated.count * 4)))
            return error.AttentionReplayNotRepeatable;
    }
    const bad = firstNonFinite(full);
    const row = if (args.len == 4) try std.fmt.parseInt(usize, args[3], 10) else if (bad) |i| i / 256 else 0;
    if (row >= rows) return error.AttentionReplayRow;
    var selected = q;
    selected.offset += row * 256 * 4;
    selected.count = 256;
    selected.owned = false;
    const single = try gpu.ops.attentionCross(g, selected, k, v, 1, keys, 1, 0.0625);
    defer single.deinit();
    const scores = try init.gpa.alloc(f64, keys);
    defer init.gpa.free(scores);
    var top = -std.math.inf(f64);
    var bottom = std.math.inf(f64);
    for (scores, 0..) |*score, key| {
        score.* = 0;
        for (0..256) |d| score.* += @as(f64, q.get(row * 256 + d)) * k.get(key * 256 + d);
        score.* *= 0.0625;
        top = @max(top, score.*);
        bottom = @min(bottom, score.*);
    }
    var total: f64 = 0;
    for (scores) |*score| {
        score.* = @exp(score.* - top);
        total += score.*;
    }
    var full_error: f64 = 0;
    var single_error: f64 = 0;
    for (0..256) |d| {
        var value: f64 = 0;
        for (scores, 0..) |score, key| value += score * v.get(key * 256 + d);
        value /= total;
        const x = full.get(row * 256 + d);
        const y = single.get(d);
        full_error = @max(full_error, if (std.math.isFinite(x)) @abs(value - x) else std.math.inf(f64));
        single_error = @max(single_error, if (std.math.isFinite(y)) @abs(value - y) else std.math.inf(f64));
    }
    const equal = std.mem.eql(u8, full.slice(u8, full.count * 4)[row * 256 * 4 ..][0 .. 256 * 4], single.slice(u8, 256 * 4));
    std.debug.print("SAM3 attention replay {d}x{d} D256: first nonfinite {?d}, row {d}, single nonfinite {?d}, byte-equal {}; FP64 maximum errors full {e}, single {e}\n", .{ rows, keys, bad, row, firstNonFinite(single), equal, full_error, single_error });
    std.debug.print("FP64 scores [{e}, {e}], probability span {e}\n", .{ bottom, top, @exp(bottom - top) });
    if (bad != null) {
        const zero = try g.alloc(q.count, .f32);
        defer zero.deinit();
        @memset(zero.slice(f32, zero.count), 0);
        const uniform = try gpu.ops.attentionCross(g, zero, k, v, rows, keys, 1, 0.0625);
        defer uniform.deinit();
        std.debug.print("Zero-query replay first nonfinite {?d}\n", .{firstNonFinite(uniform)});
    }
    if (bad != null or firstNonFinite(single) != null) return error.NonFiniteAttentionReplay;
    if (!equal) return error.AttentionReplayRowsDiffer;
    std.debug.print("Five full-grid runs are byte-identical\n", .{});
}
