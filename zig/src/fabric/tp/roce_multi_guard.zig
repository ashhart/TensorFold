//! Every RoCE post's bounds before the verb: each HCA's stripe inside its slot and the registered region.
const std = @import("std");

pub const max_hcas = 2;
pub const flag_stride = 128;
/// The largest world the wire takes (tp_roce's pollers: one lane a peer and HCA in a warp).
pub const max_peers = 8;
/// Send and receive slots by seq parity.
pub const slots = 2;

pub const Error = error{ OutOfRegion, WriteTooLarge };

/// The largest RDMA write a post may make, whatever the slot (a 9.6 MB post faulted a DGX Spark's HCA); never raised.
pub const max_write: u32 = 1 << 20;

/// The `i`-th peer `me` posts to: the next rank first, so the senders do not all hit one receiver at once.
pub fn peerAt(me: u32, world: u32, i: u32) u32 {
    return (me + 1 + i) % world;
}

/// Every rank's region layout, the same on every rank: ctrl [4 KiB], flags and receive slots one entry per source rank.
pub const Layout = struct {
    /// Ranks: the flags and receive slots have one entry a source rank.
    world: u32 = 2,
    send: u64,
    recv: u64,
    flags: u64,
    slot: u64,
    len: u64,

    pub const ctrl: u64 = 0;
    /// The flags start: a page after the control words.
    pub const flags_at: u64 = 4096;

    /// `world` ranks at a `slot`-byte slot: the offsets roce.zig's constants were, for any world.
    pub fn of(world: u32, slot: u64) Layout {
        const flags = flags_at;
        const send = flags + std.mem.alignForward(u64, flagBytes(world), 4096);
        const recv = send + slots * slot;
        return .{ .world = world, .send = send, .recv = recv, .flags = flags, .slot = slot, .len = recv + @as(u64, world) * slots * slot };
    }

    pub fn flagBytes(world: u32) u64 {
        return @as(u64, world) * slots * max_hcas * flag_stride;
    }

    pub fn sendAt(l: Layout, slot: u32) u64 {
        return l.send + @as(u64, slot) * l.slot;
    }

    /// Where rank `src`'s slot `slot` lands in the receiver's region.
    pub fn recvAt(l: Layout, src: u32, slot: u32) u64 {
        return l.recv + (@as(u64, src) * slots + slot) * l.slot;
    }

    /// Where rank `src`'s seq for slot `slot` on its HCA `hca` lands in the receiver's region.
    pub fn flagAt(l: Layout, src: u32, slot: u32, hca: u32) u64 {
        return l.flags + ((@as(u64, src) * slots + slot) * max_hcas + hca) * flag_stride;
    }
};

/// One HCA's share of a post: its data stripe in this rank's send slot and the peer's receive slot, and its flag word.
pub const Stripe = struct { local: u64, remote: u64, len: u32, flag: u64 };

/// Slot `slot`'s first `padded` bytes striped over `n_hca` HCAs, `src` to `dst`; error.OutOfRegion before any verb.
pub fn stripes(l: Layout, src: u32, dst: u32, n_hca: u32, slot: u32, padded: u32, out: *[max_hcas]Stripe) Error!void {
    if (l.world < 3 or l.world > max_peers or l.slot == 0 or l.slot > max_write) return error.OutOfRegion;
    const canonical = Layout.of(l.world, l.slot);
    if (!std.meta.eql(l, canonical) or src == dst) return error.OutOfRegion;
    if (n_hca == 0 or n_hca > max_hcas or slot > 1) return error.OutOfRegion;
    if (src >= l.world or dst >= l.world) return error.OutOfRegion;
    if (padded > max_write) return error.WriteTooLarge;
    if (padded == 0 or padded % 16 != 0 or padded > l.slot) return error.OutOfRegion;
    const packs = padded / 16;
    var offset: u64 = 0;
    for (out[0..n_hca], 0..) |*o, i| {
        var stripe = packs / n_hca;
        if (i < packs % n_hca) stripe += 1;
        o.* = .{ .local = l.sendAt(slot) + offset, .remote = l.recvAt(src, slot) + offset, .len = stripe * 16, .flag = l.flagAt(src, slot, @intCast(i)) };
        if (!within(o.local, o.len, l.sendAt(slot), l.slot) or !within(o.remote, o.len, l.recvAt(src, slot), l.slot)) return error.OutOfRegion;
        if (!within(o.flag, 4, l.flags, l.send - l.flags) or !within(o.local, o.len, 0, l.len) or !within(o.remote, o.len, 0, l.len)) return error.OutOfRegion;
        offset += o.len;
    }
}

fn within(at: u64, len: u64, lo: u64, size: u64) bool {
    return at >= lo and len <= size and at - lo <= size - len;
}

/// roce.zig's layout for a 64 KiB slot: ctrl 4 KiB, flags, send [2][slot], recv [world][2][slot].
fn testLayout() Layout {
    return Layout.of(3, 64 << 10);
}

test "a post striped over two HCAs: the stripes tile the slot's first bytes, each flag on its own word" {
    const l = testLayout();
    var s: [max_hcas]Stripe = undefined;
    try stripes(l, 0, 1, 2, 1, 48, &s);
    try std.testing.expectEqual(Stripe{ .local = l.send + l.slot, .remote = l.recv + l.slot, .len = 32, .flag = l.flags + 2 * flag_stride }, s[0]);
    try std.testing.expectEqual(Stripe{ .local = l.send + l.slot + 32, .remote = l.recv + l.slot + 32, .len = 16, .flag = l.flags + 3 * flag_stride }, s[1]);
    try stripes(l, 0, 1, 2, 1, @intCast(l.slot), &s);
    try std.testing.expectEqual(l.recv + 2 * l.slot, s[1].remote + s[1].len);
}

test "a post past its slot, a third slot, an unpadded or empty post, or no HCA is refused" {
    const l = testLayout();
    var s: [max_hcas]Stripe = undefined;
    try std.testing.expectError(error.OutOfRegion, stripes(l, 0, 1, 2, 0, @intCast(l.slot + 16), &s));
    try std.testing.expectError(error.OutOfRegion, stripes(l, 0, 1, 2, 2, 16, &s));
    try std.testing.expectError(error.OutOfRegion, stripes(l, 0, 1, 2, 0, 24, &s));
    try std.testing.expectError(error.OutOfRegion, stripes(l, 0, 1, 2, 0, 0, &s));
    try std.testing.expectError(error.OutOfRegion, stripes(l, 0, 1, 0, 0, 16, &s));
    try std.testing.expectError(error.OutOfRegion, stripes(l, 0, 1, 3, 0, 16, &s));
    const big = Layout.of(3, 1 << 20);
    try std.testing.expectError(error.WriteTooLarge, stripes(big, 0, 1, 2, 0, max_write + 16, &s));
    try stripes(big, 0, 1, 2, 0, max_write, &s);
    var short = l;
    short.len = l.send + 2 * l.slot - 16; // the send slots' last 16 bytes past the registered region
    try std.testing.expectError(error.OutOfRegion, stripes(short, 0, 1, 2, 1, @intCast(l.slot), &s));
}

test "four ranks: rank 0's slot 1 lands where the receiver keeps source 0, its second HCA's flag one stride on" {
    const l = Layout.of(4, 64 << 10);
    var s: [max_hcas]Stripe = undefined;
    try stripes(l, 0, 3, 2, 1, 32, &s);
    try std.testing.expectEqual(l.recv + 0 * 2 * l.slot + l.slot, s[0].remote);
    try std.testing.expectEqual(l.flags + (0 * 2 + 1) * max_hcas * flag_stride + flag_stride, s[1].flag);
    // rank 3's slot 0 in rank 0's region, past sources 0..2's slots
    try stripes(l, 3, 0, 1, 0, 16, &s);
    try std.testing.expectEqual(l.recv + 3 * 2 * l.slot, s[0].remote);
    // past the world: refused
    try std.testing.expectError(error.OutOfRegion, stripes(l, 4, 0, 1, 0, 16, &s));
    try std.testing.expectError(error.OutOfRegion, stripes(l, 0, 4, 1, 0, 16, &s));
    try std.testing.expectEqual(@as(u32, 2), peerAt(1, 4, 0));
    try std.testing.expectEqual(@as(u32, 0), peerAt(3, 4, 0));
}
