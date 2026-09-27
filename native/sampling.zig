const std = @import("std");
const mx = @import("mlx.zig");
pub const Candidate = struct { id: i32, value: f64 };
/// Same prompt-derived default as engine/exact_sampling.py, salt zero.
pub fn seedFor(tokens: []const i32) u64 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var buffer: [16]u8 = undefined;
    for (tokens, 0..) |token, i| {
        if (i != 0) hash.update(",");
        hash.update(std.fmt.bufPrint(&buffer, "{d}", .{token}) catch unreachable);
    }
    hash.update("|0");
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return std.mem.readInt(u64, digest[0..8], .little) & 0x7fffffffffffffff;
}
pub const Sampling = struct {
    metal: bool = false,
    seed: u64 = 0,
    temperature: f64 = 1,
    top_k: usize = 20,
    top_p: f64 = 0.95,
    pub fn validate(s: Sampling) !void {
        if (!std.math.isFinite(s.temperature) or s.temperature < 0 or !std.math.isFinite(s.top_p) or s.top_p <= 0 or s.top_p > 1) return error.InvalidSampling;
    }
    pub fn choose(s: Sampling, sorted: []const Candidate, position: u64) i32 {
        if (s.temperature == 0) return sorted[0].id;
        const n = if (s.top_k == 0) sorted.len else @min(s.top_k, sorted.len);
        const temp = @max(s.temperature, 1e-6);
        const max = sorted[0].value / temp;
        var total: f64 = 0;
        for (sorted[0..n]) |v| total += @exp(v.value / temp - max);
        var cumulative: f64 = 0;
        var best: f64 = -std.math.inf(f64);
        var chosen = sorted[0].id;
        for (sorted[0..n]) |v| {
            const score = v.value / temp + noise(s.seed, position, @intCast(v.id));
            if (score > best) {
                best = score;
                chosen = v.id;
            }
            cumulative += @exp(v.value / temp - max) / total;
            if (cumulative >= s.top_p) break;
        }
        return chosen;
    }
};
fn mix(v: u64) u64 {
    var x = v;
    x ^= x >> 30;
    x *%= 0xbf58476d1ce4e5b9;
    x ^= x >> 27;
    x *%= 0x94d049bb133111eb;
    return x ^ (x >> 31);
}
pub fn uniform(seed: u64, position: u64, id: u64) f64 {
    var x = mix(seed +% 0x9e3779b97f4a7c15);
    x = mix(x ^ (position *% 0xd1b54a32d192ed03));
    x = mix(x ^ id);
    return @as(f64, @floatFromInt(x >> 11)) * 0x1p-53 + 0x1p-54;
}
pub fn noise(seed: u64, position: u64, id: u64) f64 {
    return -@log(-@log(uniform(seed, position, id)));
}
pub fn less(_: void, a: Candidate, b: Candidate) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}
pub fn top(allocator: std.mem.Allocator, values: []const f32, n: usize) ![]Candidate {
    // Keep a sorted bounded set. k=20 is the model default; k=0 uses a full sort.
    const count = if (n == 0) values.len else @min(n, values.len);
    const result = try allocator.alloc(Candidate, count);
    if (n == 0) {
        for (values, 0..) |v, i| result[i] = .{ .id = @intCast(i), .value = v };
        std.mem.sort(Candidate, result, {}, less);
        return result;
    }
    var used: usize = 0;
    for (values, 0..) |v, i| {
        const candidate = Candidate{ .id = @intCast(i), .value = v };
        if (used == count and !less({}, candidate, result[count - 1])) continue;
        var at = @min(used, count - 1);
        while (at > 0 and less({}, candidate, result[at - 1])) : (at -= 1) {
            result[at] = result[at - 1];
        }
        result[at] = candidate;
        used = @min(used + 1, count);
    }
    return result;
}
pub fn rows(k: *mx.Kernels, s: *mx.Scope, logits: mx.Array, positions: []const i32, settings: Sampling) ![]i32 {
    return rowsMapped(k, s, logits, positions, settings, null);
}
/// Mapping must be sorted ascending, preserving token-ID tie ordering.
pub fn rowsMapped(k: *mx.Kernels, s: *mx.Scope, logits: mx.Array, positions: []const i32, settings: Sampling, mapping: ?mx.Array) ![]i32 {
    if (mapping) |ids| {
        if (mx.dtype(ids) != mx.c.MLX_UINT32 or mx.c.mlx_array_size(ids) != @as(usize, @intCast(mx.dim(logits, -1)))) return error.InvalidSamplingMapping;
    }
    const out = try mx.allocator.alloc(i32, positions.len);
    errdefer mx.allocator.free(out);
    if (settings.metal) {
        const ids = try @import("gpu_sampling.zig").sample(k, s, logits, positions, settings, mapping);
        try mx.eval(ids);
        for (out, 0..) |*v, i| v.* = @intCast(mx.c.mlx_array_data_uint32(ids)[i]);
        return out;
    }
    if (settings.temperature == 0) {
        const picked = try s.argmax(logits);
        const ids = if (mapping) |ids| try s.take(ids, picked, 0) else picked;
        try mx.eval(ids);
        for (out, 0..) |*v, i| v.* = @intCast(mx.c.mlx_array_data_uint32(ids)[i]);
        return out;
    }
    const f = try s.cast(logits, mx.f32t);
    try mx.eval(f);
    const width: usize = @intCast(mx.dim(f, -1));
    const ptr = mx.c.mlx_array_data_float32(f);
    const id_map = if (mapping) |ids| blk: {
        try mx.eval(ids);
        break :blk mx.c.mlx_array_data_uint32(ids)[0..width];
    } else null;
    for (positions, 0..) |pos, i| {
        const candidates = try top(mx.allocator, ptr[i * width ..][0..width], settings.top_k);
        defer mx.allocator.free(candidates);
        if (id_map) |ids| for (candidates) |*candidate| {
            candidate.id = @intCast(ids[@intCast(candidate.id)]);
        };
        out[i] = settings.choose(candidates, @intCast(pos));
    }
    return out;
}
test "sampling is position keyed, greedy ties use token id" {
    const values = [_]f32{ 1, 3, 3, -1 };
    const candidates = try top(std.testing.allocator, &values, 3);
    defer std.testing.allocator.free(candidates);
    try std.testing.expectEqual(@as(i32, 1), (Sampling{ .temperature = 0 }).choose(candidates, 3));
    const s = Sampling{ .seed = 1234 };
    try std.testing.expectEqual(s.choose(candidates, 4), s.choose(candidates, 4));
    try std.testing.expect(uniform(1234, 4, 2) > 0 and uniform(1234, 4, 2) < 1);
}
