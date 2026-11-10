//! The RoCE all-gather's shared geometry: peer numbering, the fabric contract, the region layout and its ctrl words.
const std = @import("std");
const verbs = @import("roce_verbs.zig");

pub const max_world = 8; // ranks a group may have on this path (GB10 kits are 2 or 4)
pub const slots = 2;
pub const flag_stride = 128;

/// A rank's peers in rank order, without itself: rank `peer` is this rank's peer number `peerIndex`.
pub fn peerIndex(rank: usize, peer: usize) usize {
    return if (peer < rank) peer else peer - 1;
}

/// The rank behind peer number `p` of `rank` (the inverse of peerIndex).
pub fn senderOf(rank: usize, p: usize) usize {
    return if (p < rank) p else p + 1;
}

/// The first rank `rank` cannot reach on a fabric described by `reaches`, or null; `Gather.probe` proves it by RDMA.
pub fn firstUnreachable(rank: usize, world: usize, ctx: anytype, reaches: *const fn (@TypeOf(ctx), usize, usize) bool) ?usize {
    for (0..world) |peer| if (peer != rank and !reaches(ctx, rank, peer)) return peer;
    return null;
}

/// The region, the same on every rank; at world 2 (one peer) every offset is the two-rank layout's, byte for byte.
pub const Layout = struct {
    slot_bytes: usize,
    peers: usize,
    recv: usize, // recv[slot][peer][slot_bytes]: each peer's shard, written by that peer's NIC
    flags: usize, // flags[slot][peer][hca]: the sequence a peer's NIC writes after its stripe on that HCA
    send: usize, // send[slot][slot_bytes]: this rank's shard, staged for its NICs
    ctrl: usize, // the doorbell, sizes, error words and the connect probe's tokens
    total: usize,

    pub fn of(max_bytes: usize, world: usize) Layout {
        const peers = world - 1;
        const s = std.mem.alignForward(usize, max_bytes, 4096);
        const flags = slots * peers * s;
        const send = flags + slots * peers * verbs.max_hcas * flag_stride;
        const ctrl = send + slots * s;
        return .{ .slot_bytes = s, .peers = peers, .recv = 0, .flags = flags, .send = send, .ctrl = ctrl, .total = ctrl + flag_stride };
    }

    /// Where peer number `p`'s shard of slot `slot` lands.
    pub fn recvAt(l: Layout, slot: usize, p: usize) usize {
        return l.recv + (slot * l.peers + p) * l.slot_bytes;
    }

    /// Where peer number `p`'s flag for HCA `h` of slot `slot` lands (`n_hca` HCAs in use: the kernel's stride).
    pub fn flagAt(l: Layout, slot: usize, p: usize, h: usize, n_hca: usize) usize {
        return l.flags + ((slot * l.peers + p) * n_hca + h) * flag_stride;
    }
};
// ctrl words (32 of them in flag_stride bytes)
pub const c_seq = 0;
pub const c_nbytes = 1; // + slot
pub const c_error = 3;
pub const c_hca = 4;
pub const c_peer = 5; // the rank whose stripe a timed-out wait missed
pub const c_probe = 8; // + peer * verbs.max_hcas + hca: the connect probe's token (rank + 1) from that peer on that HCA

comptime {
    std.debug.assert(c_probe + (max_world - 1) * verbs.max_hcas <= flag_stride / 4);
}
