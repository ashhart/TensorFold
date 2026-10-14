//! GLM-5.3-Flash on CUDA: a stream's committed caches (forward.py's State), sized to its request. KDA conv windows and
//! recurrent states (two parities), the main and MTP latent caches, the indexer's keys and gates (rings: only a
//! window's rows and the 3 before it are read) and pools, and the kept prompt state a later prompt can resume.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const kern = @import("kern.zig");

const D = kern.D;
const LL = kern.LL;

pub const MAX_ROWS: i64 = 16; // a stream's verify window: the pending token and up to 15 drafts (long copies)
pub const PROJ_W: i64 = 3 * LL * 128 + 128 + 128 + LL; // q | k | v | f_a | g_a | b
pub const CONV_C: i64 = 3 * LL * 128; // the conv's channels: q | k | v
pub const REC: i64 = LL * 128 * 128; // a layer's recurrent state (fp32 values)

/// Image features for a prompt: ``spans`` [begin, end) prompt positions of each image's placeholder run, in order,
/// and ``feats`` their rows [sum of runs, 4096] bf16 on the device (the caller's, live through the prefill).
pub const Media = struct { spans: []const [2]i64, feats: u64 };

/// Rows of the index rings (TF_GLM_INDEX_RING, default 16384, at least 4096; the capacity when that is smaller).
pub fn ringRows(capacity: i64) i64 {
    var rows: i64 = 16384;
    if (std.c.getenv("TF_GLM_INDEX_RING")) |v| rows = std.fmt.parseInt(i64, std.mem.span(v), 10) catch rows;
    return @min(capacity, @max(4096, rows));
}

/// Rows kept before a rebased window's first pool: a rejected draft's window starts up to MAX_ROWS rows back.
const RING_SLACK: i64 = 2 * MAX_ROWS;

pub const Seq = struct {
    uid: u64 = 0,
    capacity: i64,
    nkda: usize,
    ndsa: usize, // the DSA layers' latent caches
    caches: usize, // all latent caches: the DSA layers', then the MTP layer's (when the model has one)
    pos: i64 = 0,
    mtp_len: i64 = 0,
    mtp_drafted: i64 = 0,
    cur: u1 = 0,
    mem: cuda.DeviceBuffer = undefined,
    conv: u64 = 0, // [nkda, 3, CONV_C] bf16
    rec: u64 = 0, // [2, nkda, LL, 128, 128] fp32
    proj: u64 = 0, // [nkda, MAX_ROWS, PROJ_W] bf16: a window's conv channels (the commit's conv shift reads them)
    ks: kern.KdaScratch = undefined, // [nkda, MAX_ROWS, ...] the window's replay inputs
    kc: [12]u64 = @splat(0), // latent caches [capacity, 512] bf16
    ik: [12]u64 = @splat(0), // index keys [ring, 128] bf16: rows ring_base.. of the positions
    ig: [12]u64 = @splat(0), // index gates [ring, 128] bf16
    pk: [12]u64 = @splat(0), // pooled keys [capacity / 4 + 2, 128] bf16
    /// The index rows' first position (a multiple of 4, less the slack): the DSA layers' (0) and the MTP layer's (1).
    ring_base: [2]i64 = .{ 0, 0 },
    ring: i64 = 0,
    /// The prompt's images (attach): their placeholder runs' positions and features, read by the prompt's chunks.
    media: ?Media = null,
    /// The last prompt chunk's final-normed last row, the first draft's input.
    last_hidden: u64 = 0,
    /// The row a window's final-normed rows start at in the step that ran it (the next draft's inputs).
    hidden: u64 = 0,
    /// The reply's draw rule (null or temperature 0: greedy); drafts stay greedy (they only propose).
    sampling: ?lanes.Sampling = null,
    /// The kept prompt state: KDA states and conv windows at ``snap_at``, the MTP head's pending row and its cache
    /// length, each cache's index rows of the point's incomplete pool. The attention rows below the point stay valid
    /// in the live caches while the next prompt extends the kept prefix. 0: none.
    snap_at: usize = 0,
    snap_mtp_len: i64 = 0,
    snap_rec: u64 = 0, // [nkda, LL, 128, 128] fp32
    snap_conv: u64 = 0, // [nkda, 3, CONV_C] bf16
    snap_pending: u64 = 0, // [D] bf16
    snap_ik: u64 = 0, // [12, 3, 128] bf16
    snap_ig: u64 = 0,
    // DFlash2 (dflash.zig): the context ring, its first valid position, and the last verify's rows (the step it
    // ran in, the rows' first row in it, their first position, how many)
    draft: u64 = 0,
    d_from: i64 = 0,
    v_step: u64 = 0,
    v_row0: i64 = 0,
    v_pos0: i64 = 0,
    v_rows: usize = 0,

    pub fn sampled(s: *const Seq) ?lanes.Sampling {
        const r = s.sampling orelse return null;
        return if (r.temperature > 0) r else null;
    }

    pub fn pools(s: *const Seq) i64 {
        return @divFloor(s.capacity, 4);
    }

    pub fn recAt(s: *const Seq, parity: u1, li: usize) u64 {
        return s.rec + (@as(u64, parity) * s.nkda + li) * @as(u64, @intCast(REC * 4));
    }

    pub fn convAt(s: *const Seq, li: usize) u64 {
        return s.conv + li * @as(u64, @intCast(3 * CONV_C * 2));
    }

    /// Cache ``ci``'s ring as the kernels address it (position p at row p - ring_base).
    pub fn ringPtr(s: *const Seq, ring: u64, ci: usize) u64 {
        const which: usize = if (ci < s.ndsa) 0 else 1;
        return ring -% @as(u64, @intCast(s.ring_base[which] * kern.IDX_DIM * 2));
    }

    pub fn kdaScratch(s: *const Seq, li: usize) kern.KdaScratch {
        const l: u64 = @intCast(li);
        const rows: u64 = @intCast(MAX_ROWS);
        const hw: u64 = @intCast(LL * 128);
        return .{ .out = s.ks.out + l * rows * hw * 2, .k = s.ks.k + l * rows * hw * 4, .v = s.ks.v + l * rows * hw * 2,
            .g = s.ks.g + l * rows * hw * 4, .b = s.ks.b + l * rows * @as(u64, @intCast(LL)) * 4 };
    }
};

/// Device bytes a stream of ``capacity`` cache slots takes.
pub fn bytes(nkda: usize, caches: usize, capacity: i64, extra: u64) u64 {
    const n: i64 = @intCast(nkda);
    const c: i64 = @intCast(caches);
    const per = n * 3 * CONV_C * 2 + 2 * n * REC * 4 + n * MAX_ROWS * PROJ_W * 2 + n * MAX_ROWS * LL * 128 * (2 + 4 + 2 + 4) + n * MAX_ROWS * LL * 4 +
        c * (capacity * kern.KV_LORA * 2 + 2 * ringRows(capacity) * kern.IDX_DIM * 2 + (@divFloor(capacity, 4) + 2) * kern.IDX_DIM * 2) + D * 2 +
        n * REC * 4 + n * 3 * CONV_C * 2 + D * 2 + 2 * 12 * 3 * kern.IDX_DIM * 2; // the kept state
    return @as(u64, @intCast(per + 64 * 512)) + extra;
}

const Carve = struct {
    at: u64 = 0,
    fn take(self: *Carve, n: i64) u64 {
        const p = self.at;
        self.at += std.mem.alignForward(u64, @intCast(@max(n, 1)), 512);
        return p;
    }
};

/// ``extra``: bytes of a drafter's ring (0: none).
pub fn alloc(gpa: std.mem.Allocator, d: *const cuda.Driver, nkda: usize, ndsa: usize, caches: usize, capacity: i64, extra: u64) !*Seq {
    const s = try gpa.create(Seq);
    errdefer gpa.destroy(s);
    s.* = .{ .capacity = capacity, .nkda = nkda, .ndsa = ndsa, .caches = caches, .ring = ringRows(capacity) };
    var c: Carve = .{};
    const n: i64 = @intCast(nkda);
    s.conv = c.take(n * 3 * CONV_C * 2);
    s.rec = c.take(2 * n * REC * 4);
    s.proj = c.take(n * MAX_ROWS * PROJ_W * 2);
    s.ks = .{ .out = c.take(n * MAX_ROWS * LL * 128 * 2), .k = c.take(n * MAX_ROWS * LL * 128 * 4), .v = c.take(n * MAX_ROWS * LL * 128 * 2),
        .g = c.take(n * MAX_ROWS * LL * 128 * 4), .b = c.take(n * MAX_ROWS * LL * 4) };
    for (0..caches) |i| {
        s.kc[i] = c.take(capacity * kern.KV_LORA * 2);
        s.ik[i] = c.take(s.ring * kern.IDX_DIM * 2);
        s.ig[i] = c.take(s.ring * kern.IDX_DIM * 2);
        s.pk[i] = c.take((@divFloor(capacity, 4) + 2) * kern.IDX_DIM * 2);
    }
    s.last_hidden = c.take(D * 2);
    s.snap_rec = c.take(n * REC * 4);
    s.snap_conv = c.take(n * 3 * CONV_C * 2);
    s.snap_pending = c.take(D * 2);
    s.snap_ik = c.take(12 * 3 * kern.IDX_DIM * 2);
    s.snap_ig = c.take(12 * 3 * kern.IDX_DIM * 2);
    if (extra > 0) s.draft = c.take(@intCast(extra));
    s.mem = try cuda.DeviceBuffer.alloc(d, c.at);
    const base = s.mem.ptr;
    inline for (.{ "conv", "rec", "proj", "last_hidden", "snap_rec", "snap_conv", "snap_pending", "snap_ik", "snap_ig" }) |f| @field(s, f) += base;
    if (extra > 0) s.draft += base;
    inline for (.{ "out", "k", "v", "g", "b" }) |f| @field(s.ks, f) += base;
    for (0..caches) |i| {
        s.kc[i] += base;
        s.ik[i] += base;
        s.ig[i] += base;
        s.pk[i] += base;
    }
    try d.check(d.api.cuMemsetD8_v2(s.mem.ptr, 0, c.at), "zero sequence");
    return s;
}

pub fn free(gpa: std.mem.Allocator, s: *Seq) void {
    s.mem.free();
    gpa.destroy(s);
}

/// State.reset: the conv windows and recurrent states zeroed, positions 0 (the caches' stale rows stay: as Python).
pub fn reset(k: *kern.K, s: *Seq) !void {
    const d = k.d;
    const n: i64 = @intCast(s.nkda);
    try d.check(d.api.cuMemsetD8Async(s.conv, 0, @intCast(n * 3 * CONV_C * 2), k.stream.handle), "reset conv");
    try d.check(d.api.cuMemsetD8Async(s.rec, 0, @intCast(2 * n * REC * 4), k.stream.handle), "reset rec");
    s.cur = 0;
    s.pos = 0;
    s.mtp_len = 0;
    s.mtp_drafted = 0;
    s.ring_base = .{ 0, 0 };
}

fn ringCaches(s: *const Seq, which: u1) []const usize {
    const all = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    return if (which == 0) all[0..s.ndsa] else all[s.ndsa..s.caches];
}

/// The index ring ``which`` (0: the DSA layers', 1: the MTP layer's) holding rows host_pos - host_pos % 4 ..
/// host_pos + R: moved down (the rows before the window kept) when the window runs past its end. The kernels read
/// the rings through the step's tables, so nothing captured holds the old base.
pub fn ensureRing(k: *kern.K, s: *Seq, which: u1, host_pos: i64, R: i64) !void {
    const lo = host_pos - @mod(host_pos, 4);
    const base = s.ring_base[which];
    if (lo >= base and host_pos + R - base <= s.ring) return;
    const nb = @max(0, lo - RING_SLACK);
    if (nb < base or host_pos - base > s.ring or host_pos + R - nb > s.ring) return error.IndexRowsLost;
    const kept: u64 = @intCast(host_pos - nb);
    const row: u64 = @intCast(kern.IDX_DIM * 2);
    if (kept > 0) for (ringCaches(s, which)) |ci| {
        const from: u64 = @intCast(nb - base);
        try k.copy(s.ik[ci], s.ik[ci] + from * row, kept * row);
        try k.copy(s.ig[ci], s.ig[ci] + from * row, kept * row);
    };
    s.ring_base[which] = nb;
}

/// The kept point's incomplete pool rows (positions at - at % 4 .. at) of ring ``which`` -> the snapshot (or back).
pub fn ringSnap(k: *kern.K, s: *Seq, which: u1, at: i64, restore: bool) !void {
    const lo = at - @mod(at, 4);
    const n: u64 = @intCast(at - lo);
    const row: u64 = @intCast(kern.IDX_DIM * 2);
    if (restore) s.ring_base[which] = lo;
    if (n == 0) return;
    const off: u64 = @intCast(lo - s.ring_base[which]);
    for (ringCaches(s, which)) |ci| {
        const slot = s.snap_ik + ci * 3 * row;
        const gslot = s.snap_ig + ci * 3 * row;
        if (restore) {
            try k.copy(s.ik[ci] + off * row, slot, n * row);
            try k.copy(s.ig[ci] + off * row, gslot, n * row);
        } else {
            try k.copy(slot, s.ik[ci] + off * row, n * row);
            try k.copy(gslot, s.ig[ci] + off * row, n * row);
        }
    }
}
