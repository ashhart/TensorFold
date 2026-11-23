//! Independent binary64 selected-attention loss checks the native device body, including CUDA online operand cuts.
const std = @import("std");
const core = @import("core");
const spec = core.living_qsa;
const C = @import("learning_fixture.zig").Case(spec);
const a = std.testing.allocator;
const rows = 67;
const heads = 4;
const kv = 2;
const dim = 4;
const idw = 67;
fn promote(b: u16) f64 {
    return @as(f32, @bitCast(@as(u32, b) << 16));
}
fn selected(row: usize, j: usize) usize {
    return if (row > 34) (if (j < 3) j * 7 else row - 2 + j - 3) else j;
}
fn count(row: usize) usize {
    return if (row > 34) 6 else row + 1;
}
fn score(q: []const f64, k: []const f64, row: usize, head: usize, key: usize) f64 {
    var v: f64 = 0;
    for (0..dim) |i| v += q[(row * heads + head) * dim + i] * k[(key * kv + head / (heads / kv)) * dim + i];
    return v * 0.5;
}
fn loss(q: []const f64, k: []const f64, v: []const f64, upstream: []const f32, offsets: []const f64, score_offsets: []const f64, cut: bool) f64 {
    var out: f64 = 0;
    for (0..rows) |row| for (0..heads) |head| {
        const n = count(row);
        var s: [rows]f64 = undefined;
        var maximum: f64 = -std.math.inf(f64);
        for (0..n) |j| {
            const key = selected(row, j);
            s[j] = score(q, k, row, head, key) + score_offsets[(row * heads + head) * rows + key];
            maximum = @max(maximum, s[j]);
        }
        var total: f64 = 0;
        for (s[0..n]) |x| total += @exp(x - maximum);
        for (0..n) |j| {
            const key = selected(row, j);
            const end = @min(n, (j / 32 + 1) * 32);
            var tilemax: f64 = -std.math.inf(f64);
            for (s[0..end]) |x| tilemax = @max(tilemax, x);
            const weight = (@exp(s[j] - maximum) + (if (cut) offsets[(row * heads + head) * rows + key] * @exp(tilemax - maximum) else 0)) / total;
            for (0..dim) |d| out += weight * v[(key * kv + head / (heads / kv)) * dim + d] * upstream[(row * heads + head) * dim + d];
        }
    };
    return out;
}
fn check(cut: bool) !void {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var p: spec.Params = .{};
    p.u[0] = rows;
    p.u[1] = heads;
    p.u[2] = kv;
    p.u[3] = dim;
    p.u[4] = idw;
    p.u[5] = @intFromBool(cut);
    p.f[0] = 0.5;
    var scores = try C.init(.scores, p);
    defer scores.deinit();
    var soft = try C.init(.softmax, p);
    defer soft.deinit();
    var query = try C.init(.query, p);
    defer query.deinit();
    var keys = try C.init(.key_value, p);
    defer keys.deinit();
    var input: [3][]f64 = undefined;
    for (&input, [_]usize{ rows * heads * dim, rows * kv * dim, rows * kv * dim }, 0..) |*vec, size, which| {
        vec.* = try mem.alloc(f64, size);
        const bf = std.mem.bytesAsSlice(u16, scores.data[which]);
        for (vec.*, 0..) |*v, i| {
            var seed: u32 = @truncate((i + 1) *% 0x9e3779b9 +% which *% 0x85ebca6b);
            seed ^= seed >> 16;
            seed *%= 0x7feb352d;
            seed ^= seed >> 15;
            const x = @as(f32, @floatFromInt(@as(i32, @intCast(seed % 1021)) - 510)) * 0.00390625;
            bf[i] = core.quant.convert.bf16(x);
            v.* = promote(bf[i]);
        }
    }
    const up = scores.floats(3)[0 .. rows * heads * dim];
    for (up, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.03125;
    const ids = std.mem.bytesAsSlice(i32, scores.data[4]);
    const counts = std.mem.bytesAsSlice(i32, scores.data[5]);
    const sparse = std.mem.bytesAsSlice(i32, scores.data[6]);
    @memset(ids, -1000);
    for (0..rows) |row| {
        counts[row] = @intCast(count(row));
        sparse[row] = @intFromBool(row > 34);
        if (sparse[row] != 0) for (0..count(row)) |j| {
            ids[row * idw + j] = @intCast(selected(row, j));
        };
    }
    try scores.run();
    try soft.put(0, scores.data[8]);
    try soft.put(1, scores.data[9]);
    for (2..5) |i| try soft.put(i, scores.data[i + 2]);
    try soft.run();
    try query.put(0, soft.data[9]);
    try query.put(1, scores.data[1]);
    for (2..5) |i| try query.put(i, scores.data[i + 2]);
    try query.run();
    try keys.put(0, soft.data[9]);
    try keys.put(1, soft.data[8]);
    try keys.put(2, scores.data[0]);
    try keys.put(3, scores.data[3]);
    for (4..7) |i| try keys.put(i, scores.data[i]);
    try keys.run();
    const offset = try mem.alloc(f64, rows * heads * rows);
    const so = try mem.alloc(f64, offset.len);
    @memset(offset, 0);
    @memset(so, 0);
    for (0..rows) |row| for (0..heads) |head| {
        const n = count(row);
        for (0..n) |j| {
            const key = selected(row, j);
            const at = (row * heads + head) * rows + key;
            so[at] = @as(f64, scores.floats(8)[at]) - score(input[0], input[1], row, head, key);
            var mx: f32 = -std.math.inf(f32);
            for (0..@min(n, (j / 32 + 1) * 32)) |i| mx = @max(mx, scores.floats(8)[(row * heads + head) * rows + selected(row, i)]);
            const exp = @exp(scores.floats(8)[at] - mx);
            offset[at] = promote(core.quant.convert.bf16(exp)) - @as(f64, exp);
        }
    };
    const grads = [_][]const f32{ query.floats(8), keys.floats(8), keys.floats(9) };
    var checked: usize = 0;
    for (&input, 0..) |*vec, which| {
        for ([_]usize{ 0, 3, 14, 127, 135, vec.len / 2, vec.len - 3, vec.len - 1 }) |coord| {
            const orig = vec.*[coord];
            for ([_]f64{ 0.0005, 0.00025 }) |step| {
                vec.*[coord] = orig + step;
                const hi = loss(input[0], input[1], input[2], up, offset, so, cut);
                vec.*[coord] = orig - step;
                const lo = loss(input[0], input[1], input[2], up, offset, so, cut);
                vec.*[coord] = orig;
                const slope = (hi - lo) / (2 * step);
                @import("learning_fixture.zig").near(grads[which][coord], slope) catch |err| {
                    std.debug.print("QSA_NATIVE_FD cut={} which={d} coord={d} grad={d} slope={d}\n", .{ cut, which, coord, grads[which][coord], slope });
                    return err;
                };
                checked += 1;
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 48), checked);
}
test "Qwen4 learning: native Metal and CUDA QSA reverse matches independent causal GQA slopes and online cuts" {
    try check(false);
    try check(true);
}
