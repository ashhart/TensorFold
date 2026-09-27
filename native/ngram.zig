//! Flash Next's signed-int64 n-gram hash and exact checkpoint constants.
const std = @import("std");
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
};
test "EOS resets n-gram context and ids stay inside each head" {
    const n = NGram.init();
    const a = n.ids(.{ 7, 248044 }, 9);
    const b = n.ids(.{ 248044, 248044 }, 9);
    try std.testing.expectEqualSlices(i64, &a, &b);
    for (a, 0..) |id, i| try std.testing.expect(id >= n.offsets[i] and id < n.offsets[i] + n.sizes[i]);
}
