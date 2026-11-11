//! Shared-MTP oracle: owned verify rows and the actual unpaged BF16 MTP caches on every rank.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const Engine = glm.engine.Engine;
const State = glm.forward.State;

pub const Counts = struct {
    rounds: usize = 0,
    streams: usize = 0,
    levels: usize = 0,
    drafts: usize = 0,
    chances: usize = 0,
    lengths: usize = 0,
    caches: usize = 0,
    upstream: usize = 0,
};

pub fn saveRows(e: *Engine, owned: cuda.DeviceBuffer, rows: usize) !void {
    try e.copy(owned.ptr, e.cur.rows_at, rows * e.c.hidden * 2);
    e.cur.rows_at = owned.ptr;
}

fn regions(e: *const Engine, st: *const State) [5]struct { ptr: u64, bytes: usize } {
    const c = st.mtp_caches;
    const j = 2 * st.caches.len;
    return .{
        .{ .ptr = c.lc, .bytes = st.capacity * e.c.kv_lora * 2 },
        .{ .ptr = st.rings[j], .bytes = st.ring_rows * st.ring_row },
        .{ .ptr = st.rings[j + 1], .bytes = st.ring_rows * st.ring_row },
        .{ .ptr = c.pk, .bytes = c.pk_rows * e.c.i_dim * 2 },
        .{ .ptr = st.mtp_pos_dev, .bytes = 4 },
    };
}

/// Initialize inactive tails alike so a full physical-cache comparison also detects stray writes.
pub fn zeroMtp(e: *Engine) !void {
    const st = &e.cur.st;
    if (st.table != 0 or st.mtp_caches.pt != 0 or st.mtp_caches.ls != 0) return error.CacheModeNotCovered;
    for (regions(e, st)) |r| try e.d.check(e.d.api.cuMemsetD8Async(r.ptr, 0, r.bytes, e.stream.handle), "cuMemsetD8Async");
}

fn sameDevice(e: *const Engine, a: u64, b: u64, bytes: usize, what: []const u8, count: *usize) !void {
    const x = try e.gpa.alloc(u8, bytes);
    defer e.gpa.free(x);
    const y = try e.gpa.alloc(u8, bytes);
    defer e.gpa.free(y);
    try cuda.memory.download(e.d, a, x);
    try cuda.memory.download(e.d, b, y);
    if (std.mem.indexOfDiff(u8, x, y)) |at| {
        count.* += 1;
        if (count.* <= 8) std.debug.print("FAIL rank={d} {s} first_byte={d} got={x} want={x}\n", .{ e.share.rank, what, at, x[at], y[at] });
    }
}

pub fn target(e: *const Engine, b: cuda.DeviceBuffer, s: cuda.DeviceBuffer, rows: usize, got: []const u32, want: []const u32, n: *Counts) !void {
    try e.stream.synchronize();
    if (!std.mem.eql(u32, got, want)) n.upstream += 1;
    try sameDevice(e, b.ptr, s.ptr, rows * e.c.hidden * 2, "target hidden before MTP", &n.upstream);
}

/// Compare every round on every rank, including speculative rows and device/host cursor agreement.
pub fn stream(e: *Engine, b: usize, s: usize, got: []const u32, want: []const u32, gc: ?[]const f64, wc: ?[]const f64, n: *Counts) !void {
    n.streams += 1;
    n.levels += got.len;
    for (got, want) |x, y| if (x != y) {
        n.drafts += 1;
    };
    if (gc) |xs| for (xs, wc.?) |x, y| if (@as(u64, @bitCast(x)) != @as(u64, @bitCast(y))) {
        n.chances += 1;
    };
    const qb = e.seqs.items[b].?;
    const qs = e.seqs.items[s].?;
    const a = &qb.st;
    const z = &qs.st;
    if (a.pos != z.pos or a.mtp_len != z.mtp_len or a.mtp_base != z.mtp_base or qb.in_head != qs.in_head) n.lengths += 1;
    try e.stream.synchronize();
    for (regions(e, a), regions(e, z), 0..) |ra, rz, j| {
        if (ra.bytes != rz.bytes) return error.CacheShapeMismatch;
        try sameDevice(e, ra.ptr, rz.ptr, ra.bytes, switch (j) {
            0 => "MTP latent",
            1 => "MTP index keys",
            2 => "MTP index gates",
            3 => "MTP pooled keys",
            else => "MTP device cursor",
        }, &n.caches);
    }
}
