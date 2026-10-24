//! Every RoCE post's bounds before the verb: each HCA's stripe inside its slot and the registered region.
const std = @import("std");

pub const max_hcas = 2;
pub const flag_stride = 128;

pub const Error = error{ OutOfRegion, WriteTooLarge };

/// The largest RDMA write a post may make, whatever the slot (a 9.6 MB post faulted a DGX Spark's HCA); never raised.
pub const max_write: u32 = 1 << 20;

/// Both ranks' region layout (roce.zig's): byte offsets and the registered length every post must stay inside.
pub const Layout = struct { send: u64, recv: u64, flags: u64, slot: u64, len: u64 };

/// One HCA's share of a post: its data stripe in this rank's send slot and the peer's receive slot, and its flag word.
pub const Stripe = struct { local: u64, remote: u64, len: u32, flag: u64 };

/// Slot `slot`'s first `padded` bytes striped over `n_hca` HCAs; error.OutOfRegion before any verb past a slot or MR.
pub fn stripes(l: Layout, n_hca: u32, slot: u32, padded: u32, out: *[max_hcas]Stripe) Error!void {
    if (n_hca == 0 or n_hca > max_hcas or slot > 1) return error.OutOfRegion;
    if (padded > max_write) return error.WriteTooLarge;
    if (padded == 0 or padded % 16 != 0 or padded > l.slot) return error.OutOfRegion;
    const packs = padded / 16;
    var offset: u64 = 0;
    for (out[0..n_hca], 0..) |*o, i| {
        var stripe = packs / n_hca;
        if (i < packs % n_hca) stripe += 1;
        o.* = .{ .local = l.send + slot * l.slot + offset, .remote = l.recv + slot * l.slot + offset, .len = stripe * 16, .flag = l.flags + (@as(u64, slot) * max_hcas + i) * flag_stride };
        if (!within(o.local, o.len, l.send + slot * l.slot, l.slot) or !within(o.remote, o.len, l.recv + slot * l.slot, l.slot)) return error.OutOfRegion;
        if (!within(o.flag, 4, l.flags, l.send - l.flags) or !within(o.local, o.len, 0, l.len) or !within(o.remote, o.len, 0, l.len)) return error.OutOfRegion;
        offset += o.len;
    }
}

fn within(at: u64, len: u64, lo: u64, size: u64) bool {
    return at >= lo and at + len <= lo + size;
}

/// roce.zig's layout for a 64 KiB slot: ctrl 4 KiB, flags 4 KiB, send [2][slot], recv [2][slot].
fn testLayout() Layout {
    const slot: u64 = 64 << 10;
    return .{ .send = 8192, .recv = 8192 + 2 * slot, .flags = 4096, .slot = slot, .len = 8192 + 4 * slot };
}

test "a post striped over two HCAs: the stripes tile the slot's first bytes, each flag on its own word" {
    const l = testLayout();
    var s: [max_hcas]Stripe = undefined;
    try stripes(l, 2, 1, 48, &s);
    try std.testing.expectEqual(Stripe{ .local = l.send + l.slot, .remote = l.recv + l.slot, .len = 32, .flag = l.flags + 2 * flag_stride }, s[0]);
    try std.testing.expectEqual(Stripe{ .local = l.send + l.slot + 32, .remote = l.recv + l.slot + 32, .len = 16, .flag = l.flags + 3 * flag_stride }, s[1]);
    try stripes(l, 2, 1, @intCast(l.slot), &s);
    try std.testing.expectEqual(l.recv + 2 * l.slot, s[1].remote + s[1].len);
}

test "a post past its slot, a third slot, an unpadded or empty post, or no HCA is refused" {
    const l = testLayout();
    var s: [max_hcas]Stripe = undefined;
    try std.testing.expectError(error.OutOfRegion, stripes(l, 2, 0, @intCast(l.slot + 16), &s));
    try std.testing.expectError(error.OutOfRegion, stripes(l, 2, 2, 16, &s));
    try std.testing.expectError(error.OutOfRegion, stripes(l, 2, 0, 24, &s));
    try std.testing.expectError(error.OutOfRegion, stripes(l, 2, 0, 0, &s));
    try std.testing.expectError(error.OutOfRegion, stripes(l, 0, 0, 16, &s));
    try std.testing.expectError(error.OutOfRegion, stripes(l, 3, 0, 16, &s));
    var big = l;
    big.slot = 10 << 20;
    big.recv = big.send + 2 * big.slot;
    big.len = big.send + 4 * big.slot;
    try std.testing.expectError(error.WriteTooLarge, stripes(big, 2, 0, max_write + 16, &s));
    try stripes(big, 2, 0, max_write, &s);
    var short = l;
    short.len -= 16;
    try std.testing.expectError(error.OutOfRegion, stripes(short, 2, 1, @intCast(l.slot), &s));
}
