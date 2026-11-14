//! TF_GLM_KV_SPLIT's carry: a kept state restored off its home gets its latent rows from the home's copy, in pieces.
const std = @import("std");
const cuda = @import("cuda");
const Plan = @import("cuda_snapshot_layout.zig").Plan;
const page = @import("cuda_snapshot_pages.zig").page;

/// This rank's part in a carry: the home sends its copy's latent rows, the stream's owner receives them.
pub const Role = enum { none, send, recv };

/// The role of `rank` restoring a state kept at `home` onto a stream `owner` holds (at home: no carry).
pub fn role(home: u32, owner: u32, rank: usize) Role {
    if (home == owner) return .none;
    return if (rank == home) .send else if (rank == owner) .recv else .none;
}

/// The latent rows' bytes, as many on every rank: the sum of the plan's latent parts.
pub fn bytes(plan: *const Plan) usize {
    var n: usize = 0;
    for (plan.parts[0..plan.count]) |part| n += if (part.alt) part.len else 0;
    return n;
}

/// The bytes a kept state's copy holds of `plan`: every part, its latent ones only where `lat` (the home's copy).
pub fn kept(plan: *const Plan, lat: bool) usize {
    var n: usize = 0;
    for (plan.parts[0..plan.count]) |part| n += if (!part.alt or lat) part.len else 0;
    return n;
}

/// A walk over the latent rows' bytes in plan order: in the home's lease (its copy holds every part), or the pages.
const Walk = struct {
    plan: *const Plan,
    lease: []const u32, // .send: the home's lease, `base` its pages' first byte
    base: u64,
    side: Role,
    i: usize = 0, // the part
    done: usize = 0, // its bytes walked
    off: usize = 0, // .send: the part's first byte in the lease

    /// The next run of at most `max` bytes, where it sits.
    fn next(w: *Walk, max: usize) !struct { at: u64, len: usize } {
        while (w.i < w.plan.count and (!w.plan.parts[w.i].alt or w.done == w.plan.parts[w.i].len)) {
            w.off += w.plan.parts[w.i].len; // the home's copy holds every part, latent or not
            w.i += 1;
            w.done = 0;
        }
        if (w.i == w.plan.count) return error.KvSplitCarry;
        const part = w.plan.parts[w.i];
        if (w.side == .recv) {
            const x = w.plan.piece(part, w.done);
            const n = @min(x.len, max);
            w.done += n;
            return .{ .at = x.live, .len = n };
        }
        const at = w.off + w.done;
        if (at / page >= w.lease.len) return error.SnapshotSize;
        const n = @min(@min(part.len - w.done, page - at % page), max);
        w.done += n;
        return .{ .at = w.base + @as(u64, w.lease[at / page]) * page + at % page, .len = n };
    }
};

/// This rank's side through `room_len` bytes at `room` (`d`: copy, send, recv); `lease` at `base`: the home's copy.
pub fn run(d: anytype, plan: *const Plan, side: Role, peer: u32, lease: []const u32, base: u64, room: u64, room_len: usize) !usize {
    if (side == .none) return 0;
    if (side == .recv and plan.lat_ids.len == 0) return error.KvSplitCarry; // the owner holds the stream's rows
    if (room_len == 0) return error.KvSplitRoom;
    const total = bytes(plan);
    var w: Walk = .{ .plan = plan, .lease = lease, .base = base, .side = side };
    var start: usize = 0;
    while (start < total) {
        const m = @min(room_len, total - start);
        if (side == .recv) try d.recv(room, m, peer);
        var k: usize = 0;
        while (k < m) {
            const x = try w.next(m - k);
            if (side == .send) try d.copy(room + k, x.at, x.len) else try d.copy(x.at, room + k, x.len);
            k += x.len;
        }
        if (side == .send) try d.send(room, m, peer);
        start += m;
    }
    return total;
}

/// The device side: copies on `stream`, and the owner-attention room's NCCL sends and receives.
pub const Gpu = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,

    pub fn copy(g: Gpu, dst: u64, src: u64, n: usize) !void {
        try g.d.check(g.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, g.stream.handle), "cuMemcpyDtoDAsync");
    }
    pub fn send(g: Gpu, ptr: u64, n: usize, peer: u32) !void {
        try g.lib.check(g.lib.api.ncclSend(ptr, n, .u8, @intCast(peer), g.comm, g.stream.handle), "ncclSend");
    }
    pub fn recv(g: Gpu, ptr: u64, n: usize, peer: u32) !void {
        try g.lib.check(g.lib.api.ncclRecv(ptr, n, .u8, @intCast(peer), g.comm, g.stream.handle), "ncclRecv");
    }
};
