// Shared logical KV transactions. Backends supply copies and explicit physical formats.
const std = @import("std");
pub const Encoding = enum { bf16, f32, cuda_fp8_584, cuda_index_fp8_132, metal_fp4_288, metal_index_fp4_68 };
pub const Region = struct {
    live: u64,
    backup: u64,
    row_bytes: usize,
    rows: u32,
    kind: enum { ring, carry },
    encoding: Encoding = .bf16,
    pub fn bytes(r: Region) usize {
        return r.row_bytes * r.rows;
    }
};
pub const Copy = struct {
    ctx: *anyopaque,
    run: *const fn (*anyopaque, u64, u64, usize) anyerror!void,
    pub fn copy(c: Copy, dst: u64, src: u64, bytes: usize) !void {
        try c.run(c.ctx, dst, src, bytes);
    }
};
pub const State = struct {
    position: u64 = 0,
    pending: ?u32 = null,
    // A failed copy leaves the transaction active. The owner must retry or retain its buffers.
    pub fn begin(s: *State, regions: []const Region, c: Copy, n: u32) !void {
        if (s.pending != null) return error.CacheTransactionActive;
        if (n == 0 or n > 64 or s.position > std.math.maxInt(u64) - @as(u64, n)) return error.BadWindow;
        for (regions) |r| {
            _ = std.math.mul(usize, r.row_bytes, r.rows) catch return error.BadRegion;
            if (r.row_bytes == 0 or r.rows == 0 or (r.kind == .ring and n > r.rows)) return error.BadRegion;
        }
        for (regions) |r| try c.copy(r.backup, r.live, r.bytes());
        s.pending = n;
    }
    // Carries are selected from the accepted projection before keep. A full rollback restores them here.
    pub fn keep(s: *State, regions: []const Region, c: Copy, count: u32) !void {
        const n = s.pending orelse return error.NoCacheTransaction;
        if (count > n) return error.BadWindow;
        for (regions) |r| {
            if (count == 0) {
                try c.copy(r.live, r.backup, r.bytes());
            } else if (r.kind == .ring) {
                // Only rejected positions were overwritten. Accepted rows stay in their physical ring slots.
                for (count..n) |i| {
                    const off = ((s.position + i) % r.rows) * r.row_bytes;
                    try c.copy(r.live + off, r.backup + off, r.row_bytes);
                }
            }
        }
        s.position += count;
        s.pending = null;
    }
    pub fn rollback(s: *State, regions: []const Region, c: Copy) !void {
        try s.keep(regions, c, 0);
    }
};

test {
    _ = @import("kv_state_test.zig");
}
