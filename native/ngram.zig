//! Flash Next's signed-int64 n-gram hash and exact checkpoint constants.
const std = @import("std");
const mx = @import("mlx.zig");
pub const NGram = struct {
    sizes: [16]i64 = undefined,
    offsets: [16]i64 = undefined,
    multipliers: [3]i64 = undefined,
    fn mix(value: u64) u64 {
        var x = value +% 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) *% 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) *% 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }
    fn prime(v: i64) bool {
        if (v < 2) return false;
        if (@mod(v, 2) == 0) return v == 2;
        var d: i64 = 3;
        while (d * d <= v) : (d += 2) {
            if (@mod(v, d) == 0) return false;
        }
        return true;
    }
    pub fn init() NGram {
        var n = NGram{};
        var next: i64 = 19999999;
        var offset: i64 = 0;
        for (0..16) |i| {
            next += 1;
            while (!prime(next)) next += 1;
            n.sizes[i] = next;
            n.offsets[i] = offset;
            offset += next;
        }
        const half: u64 = @divTrunc(@divTrunc(std.math.maxInt(i64), 248320), 2);
        for (0..3) |i| n.multipliers[i] = @intCast(2 * (mix(1234 +% (@as(u64, @intCast(i + 1)) *% 0x9e3779b97f4a7c15)) % half) + 1);
        return n;
    }
    pub fn ids(n: NGram, history: [2]i32, token: i32) [16]i64 {
        const prev = [_]i32{ token, if (history[1] == 248044) 248044 else history[1], if (history[1] == 248044 or history[0] == 248044) 248044 else history[0] };
        var result: [16]i64 = undefined;
        var h = @as(i64, prev[0]) *% n.multipliers[0];
        for (1..3) |p| {
            h ^= @as(i64, prev[p]) *% n.multipliers[p];
            for (0..8) |j| {
                const head = (p - 1) * 8 + j;
                result[head] = @mod(h, n.sizes[head]) + n.offsets[head];
            }
        }
        return result;
    }
    /// joined contains the two previous token IDs followed by 1..2048 new IDs.
    /// All operations remain lazy GPU arrays; no token or history host read.
    pub fn idsArray(n: NGram, s: *mx.Scope, joined: mx.Array) !mx.Array {
        if (joined.ctx == null or mx.shape(joined).len != 1) return error.InvalidNGramHistory;
        const count = mx.dim(joined, 0);
        if (count < 3 or count > 2050 or (mx.dtype(joined) != mx.i32t and mx.dtype(joined) != mx.c.MLX_UINT32 and mx.dtype(joined) != mx.c.MLX_INT64)) return error.InvalidNGramHistory;
        const rows = count - 2;
        const tokens = try s.cast(joined, mx.c.MLX_INT64);
        const previous = try s.slice(tokens, 0, 1, rows + 1);
        const oldest = try s.slice(tokens, 0, 0, rows);
        const eos = try s.data(&@as(i64, 248044), &.{1}, mx.c.MLX_INT64);
        const reset = try s.binary(mx.c.mlx_logical_or, try s.binary(mx.c.mlx_equal, previous, eos), try s.binary(mx.c.mlx_equal, oldest, eos));
        var selected = mx.c.mlx_array_new();
        const rc = mx.c.mlx_where(&selected, reset, eos, oldest, mx.stream);
        const context = try s.result(rc, selected);
        const m0 = try s.data(&n.multipliers[0], &.{1}, mx.c.MLX_INT64);
        const m1 = try s.data(&n.multipliers[1], &.{1}, mx.c.MLX_INT64);
        const m2 = try s.data(&n.multipliers[2], &.{1}, mx.c.MLX_INT64);
        const h2 = try s.binary(mx.c.mlx_bitwise_xor, try s.binary(mx.c.mlx_multiply, try s.slice(tokens, 0, 2, count), m0), try s.binary(mx.c.mlx_multiply, previous, m1));
        const h3 = try s.binary(mx.c.mlx_bitwise_xor, h2, try s.binary(mx.c.mlx_multiply, context, m2));
        var heads: [2]mx.Array = undefined;
        for ([_]mx.Array{ h2, h3 }, 0..) |hash, i| {
            const sizes = try s.data(n.sizes[i * 8 ..][0..8].ptr, &.{ 1, 8 }, mx.c.MLX_INT64);
            const offsets = try s.data(n.offsets[i * 8 ..][0..8].ptr, &.{ 1, 8 }, mx.c.MLX_INT64);
            heads[i] = try s.binary(mx.c.mlx_add, try s.binary(mx.c.mlx_remainder, try s.reshape(hash, &.{ rows, 1 }), sizes), offsets);
        }
        return s.cast(try s.cat(&heads, 1), mx.c.MLX_UINT32);
    }
};
pub fn checkGpu() !void {
    try mx.init();
    defer mx.shutdown();
    var rng = std.Random.DefaultPrng.init(0x504c4548415348);
    var checked: usize = 0;
    {
        var s = mx.Scope{};
        defer s.deinit();
        const n = NGram.init();
        try std.testing.expectError(error.InvalidNGramHistory, n.idsArray(&s, mx.empty));
        try std.testing.expectError(error.InvalidNGramHistory, n.idsArray(&s, try s.zeros(&.{2}, mx.i32t)));
        try std.testing.expectError(error.InvalidNGramHistory, n.idsArray(&s, try s.zeros(&.{2051}, mx.i32t)));
        try std.testing.expectError(error.InvalidNGramHistory, n.idsArray(&s, try s.zeros(&.{ 1, 3 }, mx.i32t)));
        try std.testing.expectError(error.InvalidNGramHistory, n.idsArray(&s, try s.zeros(&.{3}, mx.f32t)));
    }
    for (0..2) |overflow| {
        var n = NGram.init();
        if (overflow == 1) n.multipliers = .{ std.math.maxInt(i64), std.math.minInt(i64) + 37, -1 };
        for ([_]usize{ 1, 3, 16, 17, 64, 257, 2048 }) |rows| {
            for (0..64) |trial| {
                var joined: [2050]i32 = undefined;
                for (joined[0 .. rows + 2]) |*id| id.* = rng.random().intRangeAtMost(i32, 0, 248319);
                // Every EOS location, adjacent EOS, non-reset EOS, and vocabulary ends.
                joined[trial % (rows + 2)] = switch (trial % 4) {
                    0 => 248044,
                    1 => 248046,
                    2 => 0,
                    else => 248319,
                };
                if (trial < rows + 2) joined[trial] = 248044;
                if (trial % 5 == 0) joined[0..2].* = .{ 248044, 248044 };
                var expected: [2048 * 16]u32 = undefined;
                for (0..rows) |i| {
                    const ids = n.ids(.{ joined[i], joined[i + 1] }, joined[i + 2]);
                    for (ids, 0..) |id, h| expected[i * 16 + h] = @intCast(id);
                }
                var s = mx.Scope{};
                defer s.deinit();
                const input = try s.binary(mx.c.mlx_add, try s.ints(joined[0 .. rows + 2]), try s.ints(&.{0}));
                const ids = try n.idsArray(&s, input);
                try mx.eval(ids);
                // Never cast IDs to float: that would hide low-bit hash errors.
                try std.testing.expectEqualSlices(u32, expected[0 .. rows * 16], mx.c.mlx_array_data_uint32(ids)[0 .. rows * 16]);
                checked += 1;
            }
        }
    }
    try mx.check(mx.c.mlx_synchronize(mx.stream));
    var active: usize = 0;
    try mx.check(mx.c.mlx_get_active_memory(&active));
    try std.testing.expectEqual(@as(usize, 0), active);
    std.debug.print("PASS: {d} GPU n-gram windows match integer CPU IDs exactly, including EOS and signed overflow\n", .{checked});
}
test "EOS resets n-gram context and ids stay inside each head" {
    const n = NGram.init();
    const a = n.ids(.{ 7, 248044 }, 9);
    const b = n.ids(.{ 248044, 248044 }, 9);
    try std.testing.expectEqualSlices(i64, &a, &b);
    for (a, 0..) |id, i| try std.testing.expect(id >= n.offsets[i] and id < n.offsets[i] + n.sizes[i]);
}
