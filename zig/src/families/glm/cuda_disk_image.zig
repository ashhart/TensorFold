//! A GLM kept state's disk image on one rank: its parts at a position, and where each image byte sits in its lease.
const std = @import("std");
const cuda = @import("cuda");
const st_mod = @import("cuda_state.zig");
const sparse = @import("cuda_sparse.zig");
const Config = @import("config.zig").Config;
const pages = @import("cuda_snapshot_pages.zig");
const State = st_mod.State;
const window_rows = st_mod.window_rows;

/// The image's first bytes: the format, the position, the head's rows, both ring bases and the KDA copy bits.
pub const Words = struct {
    at: u64,
    mtp_len: u64,
    base: u64,
    mtp_base: u64,
    cur: u64,

    pub const bytes: u64 = 48;
    const format: u64 = 1;

    pub fn encode(w: Words) [6]u64 {
        return .{ format, w.at, w.mtp_len, w.base, w.mtp_base, w.cur };
    }

    /// Words read back: refused unless the format is this one and the state stands at `at`.
    pub fn decode(x: [6]u64, at: u32) !Words {
        if (x[0] != format or x[1] != at) return error.DiskWords;
        return .{ .at = x[1], .mtp_len = x[2], .base = x[3], .mtp_base = x[4], .cur = x[5] };
    }
};

/// A part as the rows store takes it: `row` bytes for each `per` tokens from position 0, or the state's own (row 0).
pub const Part = struct { bytes: u64, row: u32 = 0, per: u32 = 1 };

/// `len` image bytes from `img`: the words, `lease` bytes into the state's lease, or zeros the lease does not hold.
pub const Seg = struct { img: u64, len: u64, from: union(enum) { words, lease: u64, zeros } };

/// What a rank's kept state holds, read from a sequence of the engine (its addresses are not used past here).
pub const Shape = struct {
    prefix: u64, // the sequence's own fields before its caches: KDA conv and states, a window's saved rows
    tail: u64, // its fields after them: the head's position and first-draft row
    layers: u32, // DSA layers, whose rows are shared by position
    latent: u32, // a latent row's bytes
    scales: bool, // fp8 latent rows: a 4-byte scale each
    ring_rows: u32,
    ring_row: u32,
    pk_rows: u32,
    pooled_row: u32,
    capacity: u32,
    head: bool, // the MTP head's caches are kept (drafting)

    pub fn of(st: *const State, c: *const Config, capacity: usize, head: bool) !Shape {
        if (st.rings.len == 0) return error.SnapshotShape;
        const base = st.arena.buf.ptr;
        const first = if (st.table != 0) st.rings[0] else if (st.caches.len > 0) st.caches[0].lc else st.mtp_caches.lc;
        const end = if (st.table != 0) st.table else base + st.arena.buf.len;
        if (first < base or st.mtp_pos_dev < first or end < st.mtp_pos_dev) return error.SnapshotShape;
        const fp8 = st.mtp_caches.ls != 0;
        return .{
            .prefix = first - base,
            .tail = end - st.mtp_pos_dev,
            .layers = @intCast(st.caches.len),
            .latent = c.kv_lora * @as(u32, if (fp8) 1 else 2),
            .scales = fp8,
            .ring_rows = @intCast(st.ring_rows),
            .ring_row = @intCast(st.ring_row),
            .pk_rows = @intCast(capacity / sparse.pool + 2),
            .pooled_row = c.i_dim * 2,
            .capacity = @intCast(capacity),
            .head = head,
        };
    }

    /// Rows the image keeps of the head's caches at `at`: a verify window past it, as the cache charges a state.
    pub fn headRows(sh: Shape, at: u32) u32 {
        return @min(sh.capacity, at + window_rows);
    }
};

/// One cache's parts in the lease's order, for `n` rows: latent, scales, the two index rings, the pooled keys.
fn cache(sh: Shape, n: u64) [5]u64 {
    return .{
        n * sh.latent,
        if (sh.scales) n * 4 else 0,
        @min(n, sh.ring_rows) * sh.ring_row,
        @min(n, sh.ring_rows) * sh.ring_row,
        @min(n / sparse.pool + 2, sh.pk_rows) * sh.pooled_row,
    };
}

/// The image at `at` (a chunk start): the words, the prefix, each DSA layer's rows, the head's caches, the tail.
pub fn parts(sh: Shape, at: u32, out: *std.ArrayList(Part), a: std.mem.Allocator) !void {
    if (at == 0 or at % sparse.pool != 0 or at > sh.capacity) return error.SnapshotShape;
    try out.appendSlice(a, &.{ .{ .bytes = Words.bytes }, .{ .bytes = sh.prefix } });
    const pooled: u64 = at / sparse.pool;
    for (0..sh.layers) |_| {
        const c = cache(sh, at);
        try out.append(a, .{ .bytes = c[0], .row = sh.latent });
        if (sh.scales) try out.append(a, .{ .bytes = c[1], .row = 4 });
        try out.appendSlice(a, &.{ .{ .bytes = c[2] }, .{ .bytes = c[3] } });
        // whole pools are rows by position; the pool past them and the spare row stay the state's own
        try out.appendSlice(a, &.{ .{ .bytes = pooled * sh.pooled_row, .row = sh.pooled_row, .per = sparse.pool }, .{ .bytes = c[4] - pooled * sh.pooled_row } });
    }
    if (sh.head) for (cache(sh, sh.headRows(at))) |n| if (n > 0) try out.append(a, .{ .bytes = n });
    try out.append(a, .{ .bytes = sh.tail });
}

/// The image's segments in order for a state at `at` whose head holds `mtp_len` rows (0 without a head).
pub fn segments(sh: Shape, at: u32, mtp_len: u64, out: *std.ArrayList(Seg), a: std.mem.Allocator) !void {
    if (sh.head and mtp_len > sh.headRows(at)) return error.SnapshotShape;
    var img: u64 = Words.bytes;
    var lease: u64 = 0;
    try out.append(a, .{ .img = 0, .len = Words.bytes, .from = .words });
    var front: u64 = sh.prefix; // the prefix and every DSA layer's parts sit in the lease as in the image
    for (0..sh.layers) |_| for (cache(sh, at)) |n| {
        front += n;
    };
    try out.append(a, .{ .img = img, .len = front, .from = .{ .lease = 0 } });
    img += front;
    lease += front;
    if (sh.head) for (cache(sh, sh.headRows(at)), cache(sh, mtp_len)) |full, held| {
        if (held > 0) try out.append(a, .{ .img = img, .len = held, .from = .{ .lease = lease } });
        if (full > held) try out.append(a, .{ .img = img + held, .len = full - held, .from = .zeros });
        img += full;
        lease += held;
    };
    if (sh.tail > 0) try out.append(a, .{ .img = img, .len = sh.tail, .from = .{ .lease = lease } });
}

/// The lease bytes of a state at `at` whose head holds `mtp_len` rows (layout.Plan's length for it).
pub fn leaseBytes(sh: Shape, at: u32, mtp_len: u64) u64 {
    var n: u64 = sh.prefix + sh.tail;
    for (0..sh.layers) |_| for (cache(sh, at)) |x| {
        n += x;
    };
    if (sh.head) for (cache(sh, mtp_len)) |x| {
        n += x;
    };
    return n;
}

/// Copies between `host` and image bytes [off, off + host.len) of a lease: out (lease to host) or in.
pub const Copy = struct {
    d: *const cuda.Driver,
    pool: *const pages.Pool,
    lease: *const pages.Lease,
    segs: []const Seg,
    words: *[6]u64,

    pub fn run(c: Copy, off: u64, host: []u8, out: bool) !void {
        const end = off + host.len;
        for (c.segs) |s| {
            const lo = @max(off, s.img);
            const hi = @min(end, s.img + s.len);
            if (lo >= hi) continue;
            const h = host[@intCast(lo - off)..][0..@intCast(hi - lo)];
            switch (s.from) {
                .words => {
                    const w = std.mem.asBytes(c.words)[@intCast(lo - s.img)..][0..h.len];
                    if (out) @memcpy(h, w) else @memcpy(w, h);
                },
                .zeros => if (out) @memset(h, 0),
                .lease => |at| try c.leased(at + (lo - s.img), h, out),
            }
        }
    }

    /// Lease bytes from `at`, a page at a time (each page is a run of the snapshot arena).
    fn leased(c: Copy, at: u64, host: []u8, out: bool) !void {
        const arena = c.pool.arena orelse return error.NoSnapshotRoom;
        if (at + host.len > c.lease.ids.items.len * pages.page) return error.SnapshotSize;
        var done: usize = 0;
        while (done < host.len) {
            const x = at + done;
            const n = @min(host.len - done, pages.page - x % pages.page);
            const dev = arena.ptr + @as(u64, c.lease.ids.items[@intCast(x / pages.page)]) * pages.page + x % pages.page;
            const p = host[done..][0..n];
            if (out) {
                try c.d.check(c.d.api.cuMemcpyDtoH_v2(p.ptr, dev, n), "cuMemcpyDtoH");
            } else try c.d.check(c.d.api.cuMemcpyHtoD_v2(dev, p.ptr, n), "cuMemcpyHtoD");
            done += n;
        }
    }
};

test {
    _ = @import("cuda_disk_image_test.zig");
}
