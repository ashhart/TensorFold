//! The packed reader against independent placement in experts_pack.cu, at both group sizes.
const std = @import("std");
fn shuffle(w: u32) u32 {
    var v: u32 = 0;
    for (0..8) |j| v |= ((w >> @intCast(4 * j)) & 15) << @intCast(4 * ((j % 2) * 4 + j / 2));
    return v;
}
fn word(block: []const u32, h: usize, rr: usize, kk: usize) u32 {
    const t = rr / 8;
    const r = rr % 8;
    const q = kk / h;
    const j = kk % h;
    const f = ((t * h + j) * 8 + r) * 4 + q;
    const p = block[(f / 128) * 128 + (f % 32) * 4 + (f / 32) % 4];
    var out: u32 = 0;
    for (0..4) |i| out |= ((p >> @intCast(4 * i)) & 15) << @intCast(8 * i) | ((p >> @intCast(16 + 4 * i)) & 15) << @intCast(8 * i + 4);
    return out;
}
test "MoE packed: every code word and scale/bias row at group 32 and 64 matches the engine packer" {
    for ([_]usize{ 1, 2 }) |h| {
        var block: [288]u32 = @splat(0);
        var raw: [256]u32 = undefined;
        for (raw[0 .. 128 * h], 0..) |*v, i| {
            v.* = @as(u32, @intCast(i)) *% 0x9e3779b9 +% 0x12345678;
            const rr = i / (4 * h);
            const kk = i % (4 * h);
            const f = ((rr / 8 * h + kk % h) * 8 + rr % 8) * 4 + kk / h;
            block[f / 128 * 128 + f % 32 * 4 + f / 32 % 4] = shuffle(v.*);
        }
        for (0..32) |rr| for (0..4 * h) |kk| {
            try std.testing.expectEqual(raw[rr * 4 * h + kk], word(&block, h, rr, kk));
            if (h == 1) {
                const i = rr * 4 + kk;
                try std.testing.expectEqual(shuffle(raw[i]), block[i % 32 * 4 + i / 32]);
            }
        };
        for (0..32) |i| {
            const p = i / 8;
            const t = i % 4;
            const rr = t * 8 + p * 2;
            block[128 * h + i] = @as(u32, @intCast(rr + 1)) | (@as(u32, @intCast(rr + 2)) << 16);
        }
        for (0..32) |rr| {
            const t = rr / 8;
            const p = rr % 8 / 2;
            for ([_]usize{ 0, 4 }) |bias| {
                const sb = block[128 * h + p * 8 + bias + t];
                const got: u16 = @truncate(sb >> @intCast((rr % 2) * 16));
                try std.testing.expectEqual(@as(u16, @intCast(rr + 1)), got);
            }
        }
    }
}
