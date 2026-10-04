//! What each fabric primitive costs per link kind: a write's latency, bandwidth, and the GPU handoff around a step.
const std = @import("std");
const LinkKind = @import("rdma.zig").LinkKind;

/// Where a constant comes from: MCDMA's published docs, our own link measurements or probes, or an assumption.
pub const Source = enum { documented, measured, probed, assumed };

pub const Link = struct {
    kind: LinkKind,
    /// One small message, one way: posted until the peer can use it.
    latency_ns: f64,
    /// Sustained payload bytes a second, one way, on one link.
    bandwidth: f64,
    /// A node's ceiling over all its links of this kind when they share one bottleneck; null when each link is its own.
    node_bandwidth: ?f64 = null,
    source: Source,
};

/// MCDMA's published CX5 tables: 4 KiB queue-depth-one write latency and sustained 4 MiB host-buffer bandwidth.
pub const cx5_mac_to_cuda: Link = .{ .kind = .cx5, .latency_ns = 7_625, .bandwidth = 29.4e9 / 8.0, .node_bandwidth = 30.8e9 / 8.0, .source = .documented };
pub const cx5_cuda_to_mac: Link = .{ .kind = .cx5, .latency_ns = 3_680, .bandwidth = 51.0e9 / 8.0, .node_bandwidth = 51.2e9 / 8.0, .source = .documented };
/// Thunderbolt 5 between two M3 Ultras: half the 8 B-4 KiB ping-pong round trip, and streaming SENDs of 16 KiB-4 MiB.
pub const tb5: Link = .{ .kind = .tb5, .latency_ns = 3_500, .bandwidth = 9.38e9, .source = .measured };
/// A same-node copy through unified memory.
pub const local: Link = .{ .kind = .local, .latency_ns = 1_000, .bandwidth = 400e9, .source = .assumed };

/// The handoff between the GPU and the posting CPU around one step, both ways.
pub const Handoff = struct {
    gpu_to_host_ns: f64,
    host_to_gpu_ns: f64,
    source: Source,
};

/// A command buffer signals a shared event the host waits on, and the host's signal wakes the next one (our M5 wake time).
pub const event_handoff: Handoff = .{ .gpu_to_host_ns = 10_000, .host_to_gpu_ns = 100_000, .source = .probed };
/// A resident kernel polls flag words the NIC writes into a shared buffer; unverified on Apple GPUs.
pub const polled_handoff: Handoff = .{ .gpu_to_host_ns = 2_000, .host_to_gpu_ns = 3_000, .source = .assumed };
/// The network alone, for the floor.
pub const no_handoff: Handoff = .{ .gpu_to_host_ns = 0, .host_to_gpu_ns = 0, .source = .assumed };

pub const Primitive = enum { send, barrier, broadcast, all_reduce, all_to_all, gather, bulk };

/// The shape of one call: ranks taking part, payload bytes each rank sends to each peer, and whether peers have their own links.
pub const Shape = struct {
    ranks: u32,
    bytes: u64,
    mesh: bool = true,
    /// Bandwidth the GPU reads partials at for the local sum (an all-reduce's last stage).
    local_bandwidth: f64 = 400e9,
};

/// Estimated nanoseconds for one call of `p`; one-shot over a full mesh unless `shape.mesh` is false.
pub fn estimateNs(p: Primitive, link: Link, h: Handoff, s: Shape) f64 {
    const peers: f64 = @floatFromInt(@max(s.ranks, 1) - 1);
    const b: f64 = @floatFromInt(s.bytes);
    const per_link = if (s.mesh) b else peers * b;
    var wire = per_link / link.bandwidth;
    if (link.node_bandwidth) |node| wire = @max(wire, peers * b / node);
    const sync = h.gpu_to_host_ns + h.host_to_gpu_ns;
    return switch (p) {
        .bulk => link.latency_ns + b / link.bandwidth * 1e9,
        .barrier => sync + link.latency_ns,
        .send => sync + link.latency_ns + b / link.bandwidth * 1e9,
        .broadcast, .all_to_all, .gather => sync + link.latency_ns + wire * 1e9,
        .all_reduce => sync + link.latency_ns + wire * 1e9 + (peers + 1) * b / s.local_bandwidth * 1e9,
    };
}

/// One lane round's communication: `steps` collectives of `p`, each moving `rows` rows of `row_bytes`.
pub fn roundNs(p: Primitive, steps: u32, rows: u32, row_bytes: u64, link: Link, h: Handoff, ranks: u32) f64 {
    const one = estimateNs(p, link, h, .{ .ranks = ranks, .bytes = @as(u64, rows) * row_bytes });
    return @as(f64, @floatFromInt(steps)) * one;
}

test "costs grow with bytes and handoff, and a shared bottleneck caps fan-out" {
    const small = estimateNs(.all_reduce, tb5, no_handoff, .{ .ranks = 4, .bytes = 14_336 });
    const big = estimateNs(.all_reduce, tb5, no_handoff, .{ .ranks = 4, .bytes = 64 * 14_336 });
    try std.testing.expect(big > small and small > tb5.latency_ns);
    const evented = estimateNs(.all_reduce, tb5, event_handoff, .{ .ranks = 4, .bytes = 14_336 });
    try std.testing.expectApproxEqAbs(small + 110_000, evented, 1);
    const fan = estimateNs(.broadcast, cx5_cuda_to_mac, no_handoff, .{ .ranks = 3, .bytes = 1 << 20 });
    const lone = estimateNs(.broadcast, cx5_cuda_to_mac, no_handoff, .{ .ranks = 2, .bytes = 1 << 20 });
    try std.testing.expect(fan > lone);
    try std.testing.expectApproxEqRel(@as(f64, 186) * small, roundNs(.all_reduce, 186, 1, 14_336, tb5, no_handoff, 4), 1e-9);
}
