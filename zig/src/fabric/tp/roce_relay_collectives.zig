//! Host consumers of a completed relay frame; the production multirank GPU ABI remains gather-only.
const std = @import("std");
const guard = @import("roce_multi_guard.zig");
const bf16 = @import("bf16_round");
pub const Mode = enum { gather, sum_f32, sum_bf16, exchange };
fn input(region: []const u8, l: guard.Layout, me: u32, src: u32, seq: u32, n: usize) []const u8 {
    const at = if (src == me) l.sendAt(seq & 1) else l.recvAt(src, seq & 1);
    return region[at..][0..n];
}
/// The caller has acquired every source's two flags and retains custody until this function returns.
pub fn consume(region: []const u8, l: guard.Layout, me: u32, seq: u32, n: usize, mode: Mode, out: []u8) !void {
    if (l.world != 4 or me >= 4 or l.slot > guard.max_write or region.len < l.len or n > l.slot or n % 4 != 0 or
        out.len != n * @as(usize, if (mode == .gather) 4 else 1)) return error.BadCollective;
    if (mode == .gather) {
        for (0..4) |src| @memcpy(out[src * n ..][0..n], input(region, l, me, @intCast(src), seq, n));
    } else if (mode == .exchange) {
        @memcpy(out, input(region, l, me, (me + 2) % 4, seq, n));
    } else {
        const width: usize = if (mode == .sum_f32) 4 else 2;
        for (0..n / width) |i| {
            var sum: f32 = undefined;
            for (0..4) |src| {
                const bytes = input(region, l, me, @intCast(src), seq, n);
                const value: f32 = if (mode == .sum_f32)
                    @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little))
                else
                    @bitCast(@as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16);
                sum = if (src == 0) value else sum + value;
            }
            if (mode == .sum_f32) std.mem.writeInt(u32, out[i * 4 ..][0..4], @bitCast(sum), .little) else std.mem.writeInt(u16, out[i * 2 ..][0..2], bf16.bits(sum), .little);
        }
    }
}
