//! The shared pool's device side: one latent and one index-key buffer a KV source, a plan's moves and zeroed blocks.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("dsv41").config.Config;
const kv = @import("kv.zig");
const pool = @import("pool.zig");

/// Bytes a block of positions takes across every KV source's latents and index keys.
pub fn bytesPerBlock(c: *const Config) u64 {
    var n: u64 = 0;
    var l: u32 = 0;
    while (l < c.layers) : (l += 1) if (c.isKvSource(l)) {
        n += rowsPerBlock(c, l) * (@as(u64, c.head_dim) * 2 + kv.index_key_bytes);
    };
    return n;
}

/// Compressed rows a block holds at a KV source.
fn rowsPerBlock(c: *const Config, layer: u32) u64 {
    return pool.block / c.compressRatio(layer);
}

/// The byte spans of `blocks` blocks from block `first` at one KV source: latents, then index keys.
fn spans(c: *const Config, layer: u32, first: u32, blocks: u32) [2][2]usize {
    const r0: usize = @intCast(first * rowsPerBlock(c, layer));
    const n: usize = @intCast(blocks * rowsPerBlock(c, layer));
    const d: usize = @as(usize, c.head_dim) * 2;
    return .{ .{ r0 * d, n * d }, .{ r0 * kv.index_key_bytes, n * kv.index_key_bytes } };
}

pub const Bufs = struct {
    a: std.mem.Allocator,
    pools: []?mtl.Buffer,
    keys: []?mtl.Buffer,
    blocks: u32,

    /// Every KV source's buffers for `blocks` blocks, zeroed; BufferTooLong past the device's longest buffer.
    pub fn init(a: std.mem.Allocator, device: mtl.Device, c: *const Config, blocks: u32) !Bufs {
        const pools = try a.alloc(?mtl.Buffer, c.layers);
        errdefer a.free(pools);
        @memset(pools, null);
        const keys = try a.alloc(?mtl.Buffer, c.layers);
        errdefer a.free(keys);
        @memset(keys, null);
        var b: Bufs = .{ .a = a, .pools = pools, .keys = keys, .blocks = blocks };
        errdefer b.release();
        var l: u32 = 0;
        while (l < c.layers) : (l += 1) if (c.isKvSource(l)) {
            const all = spans(c, l, 0, blocks);
            if (all[0][1] > device.maxBufferLength()) return error.BufferTooLong;
            pools[l] = try device.buffer(all[0][1], mtl.ResourceOptions.shared);
            @memset(pools[l].?.contents()[0..all[0][1]], 0);
            keys[l] = try device.buffer(all[1][1], mtl.ResourceOptions.shared);
            @memset(keys[l].?.contents()[0..all[1][1]], 0);
        };
        return b;
    }

    fn release(b: *Bufs) void {
        for (b.pools) |x| if (x) |y| y.deinit();
        for (b.keys) |x| if (x) |y| y.deinit();
    }

    pub fn deinit(b: *Bufs) void {
        b.release();
        b.a.free(b.keys);
        b.a.free(b.pools);
        b.* = undefined;
    }

    /// The buffers as the slots borrow them.
    pub fn shared(b: *const Bufs) kv.Shared {
        return .{ .pools = b.pools, .keys = b.keys };
    }

    /// A placement on the host, with no GPU work in flight: live runs copied down in order, then the new run zeroed.
    pub fn apply(b: *const Bufs, c: *const Config, plan: pool.Plan) void {
        for (plan.moves) |m| b.each(c, m.from, m.blocks, m.to, move);
        for (plan.zero) |z| b.each(c, z.start, z.blocks, 0, zero);
    }

    fn move(bytes: []u8, from: [2]usize, to: [2]usize) void {
        std.mem.copyForwards(u8, bytes[to[0]..][0..to[1]], bytes[from[0]..][0..from[1]]);
    }

    fn zero(bytes: []u8, at: [2]usize, _: [2]usize) void {
        @memset(bytes[at[0]..][0..at[1]], 0);
    }

    /// `f` over every KV source's latents and keys at blocks [first, first + blocks) (and the same span from `to`).
    fn each(b: *const Bufs, c: *const Config, first: u32, blocks: u32, to: u32, f: fn ([]u8, [2]usize, [2]usize) void) void {
        for (b.pools, b.keys, 0..) |p, k, l| if (p) |pb| {
            const from = spans(c, @intCast(l), first, blocks);
            const dest = spans(c, @intCast(l), to, blocks);
            f(pb.contents()[0..pb.length()], from[0], dest[0]);
            f(k.?.contents()[0..k.?.length()], from[1], dest[1]);
        };
    }
};

test "a block's rows start 256-byte aligned in every pooled buffer at the release's shapes" {
    const c: Config = .{};
    var l: u32 = 0;
    while (l < c.layers) : (l += 1) if (c.isKvSource(l)) {
        for (spans(&c, l, 1, 1)) |s| try std.testing.expectEqual(@as(usize, 0), s[0] % 256);
    };
    // three ratio-2 sources and one ratio-1 source: 3 x 546 + 1092 bytes a position
    try std.testing.expectEqual(@as(u64, 2730 * pool.block), bytesPerBlock(&c));
}
