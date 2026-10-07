//! Device buffers of one Nemotron sequence: its caches and recurrent state, and the scratch a window or chunk uses.

const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const kern = @import("cuda_kernels.zig");
const sampler = @import("cuda_sampler.zig");

pub const max_rows = 16; // a verify window's rows (the row tile of the row-parallel kernels)
pub const prefill_rows = 2048; // rows of a prompt chunk
pub const chunk_keys = 512; // keys an attention chunk holds, at fixed absolute positions
const shared_words = 2 * max_rows + 8 * max_rows; // the shared expert's members, then (items, counts) a row count

/// Sub-allocations of one device allocation, 256-byte aligned like the driver's own.
const Arena = struct {
    buf: cuda.DeviceBuffer,
    used: usize = 0,

    fn take(a: *Arena, bytes: usize) u64 {
        const at = std.mem.alignForward(usize, a.used, 256);
        a.used = at + bytes;
        std.debug.assert(a.used <= a.buf.len);
        return a.buf.ptr + at;
    }
};

/// A sequence's own Buffers fields: caches and state (the first seven, one range), window io and its sampled rule.
pub const seq_fields = [_][]const u8{ "k_cache", "v_cache", "ssm", "conv_base", "raw", "xc", "dt", "meta", "ids", "hidden", "sampled", "rule" };

fn seqSizes(c: Config, max_len: usize) [seq_fields.len]usize {
    const W: usize = max_rows;
    const nm: usize = c.count(.mamba);
    const na: usize = c.count(.attention);
    const kv = na * max_len * c.kv_heads * c.head_dim * 2;
    const cd: usize = c.convDim();
    return .{ kv, kv, nm * c.mamba_heads * c.mamba_head_dim * c.state * 4, nm * 3 * cd * 2, nm * 2 * W * cd * 2, nm * 2 * W * cd * 2, nm * 2 * W * c.mamba_heads * 4, 16, W * 4, W * c.hidden * 2, W * 4, @sizeOf(sampler.Rule) };
}

const scratch_count = 35;

/// Scratch windows and chunks share on one stream: window logits, chunk io, activations, the expert plan.
fn scratchSizes(c: Config, nch: usize) [scratch_count]usize {
    const R: usize = prefill_rows;
    const W: usize = max_rows;
    const D: usize = c.hidden;
    const ns: usize = c.slots();
    const qd: usize = c.heads * c.head_dim;
    const cd: usize = c.convDim();
    const xd: usize = c.inner();
    const pairs = R * ns;
    const items = kern.maxItems(@intCast(pairs), c.experts + 2, 16);
    return .{
        W * c.vocab * 2,  16,                   R * 4,              R * D * 2,       c.vocab * 2,    16,
        R * D * 2,        R * D * 2,            R * D * 2,          R * D * 2,       R * D / 64 * 4, R * D * 2,
        R * c.projDim() * 2, R * cd * 2,         R * xd * 2,         R * xd * 2,      R * xd / 64 * 4, R * c.qkvDim() * 2,
        R * qd * 2,       R * qd * 2,           R * qd / 64 * 4,    W * nch * qd * 4, W * nch * c.heads * 4,
        W * nch * c.heads * 4, 6 * R * c.experts * 4, R * ns * 4,  R * ns * 4,      pairs * 4, items * 12,
        8,                pairs * 4,            (pairs + 1023) / 1024 * (c.experts + 2) * 4, pairs * c.expert_width * 2,
        pairs * D * 4,    shared_words * 4,
    };
}

/// One sequence: its own buffers (seq_fields, then its MTP head's) and where it stands while another is bound.
pub const Seq = struct {
    arena: ?Arena, // null: the engine's own buffers
    ptr: [seq_fields.len]u64,
    head: [3]u64 = @splat(0), // the MTP head's seq_fields (cuda_mtp.zig)
    pos: usize = 0,
    parity: usize = 0,
    prev_keep: usize = 0,
    rows: usize = 0,
    head_pos: usize = 0,
    sampling: ?sampler.Sampling = null,

    /// The device bytes one sequence takes: its buffers for `max_len` cache rows and a head's `head` buffers.
    pub fn bytes(c: Config, max_len: usize, head: []const usize) usize {
        var total: usize = 0;
        for (seqSizes(c, max_len)) |n| total += n + 256;
        for (head) |n| total += n + 256;
        return total;
    }

    /// Cache and head buffers are zeroed on their owning stream.
    pub fn init(d: *const cuda.Driver, c: Config, max_len: usize, head: []const usize, stream: cuda.Stream) !Seq {
        const sizes = seqSizes(c, max_len);
        var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, bytes(c, max_len, head)) };
        errdefer a.buf.free();
        try a.buf.fill8(0, stream.handle);
        var s: Seq = .{ .arena = null, .ptr = undefined };
        for (&s.ptr, sizes) |*p, n| p.* = a.take(n);
        for (s.head[0..head.len], head) |*p, n| p.* = a.take(n);
        s.arena = a;
        return s;
    }

    /// The engine's own buffers as a sequence.
    pub fn view(b: *const Buffers) Seq {
        var s: Seq = .{ .arena = null, .ptr = undefined };
        inline for (seq_fields, &s.ptr) |name, *p| p.* = @field(b, name);
        return s;
    }

    pub fn deinit(s: *Seq) void {
        if (s.arena) |*a| a.buf.free();
        s.* = undefined;
    }
};

pub const Buffers = struct {
    arena: Arena,
    // sequence state (Engine.STATE in the Python engine)
    k_cache: u64,
    v_cache: u64,
    ssm: u64,
    conv_base: u64,
    raw: u64,
    xc: u64,
    dt: u64,
    state_bytes: [7]usize,
    // a window's inputs and outputs
    meta: u64,
    ids: u64,
    hidden: u64,
    logits: u64,
    sampled: u64,
    // a prompt chunk's own inputs and outputs
    p_meta: u64,
    p_ids: u64,
    p_hidden: u64,
    p_logits: u64,
    p_sampled: u64,
    // scratch shared by windows and chunks (they never overlap on the stream)
    emb: u64,
    h: [2]u64,
    y: u64,
    xs: u64,
    delta: u64,
    proj: u64,
    p_xc: u64,
    sy: u64,
    g: u64,
    gxs: u64,
    qkv: u64,
    q: u64,
    att: u64,
    axs: u64,
    po: u64,
    pm: u64,
    pl: u64,
    part: u64,
    pick: u64,
    wts: u64,
    plan: kern.Plan,
    act: u64,
    ymoe: u64,
    rule: u64, // the sequence's sampled rule, as sample.cu reads it
    shared: u64, // the shared expert's fixed plans: 32 members, then (items, counts) for 1..16 rows

    /// Sizes every buffer for `max_len` cache rows and `nch` attention chunk partials a row; the own sequence's first.
    pub fn init(d: *const cuda.Driver, c: Config, max_len: usize, nch: usize) !Buffers {
        const own = seqSizes(c, max_len);
        const scratch = scratchSizes(c, nch);
        var total: usize = 0;
        for (own) |n| total += n + 256;
        for (scratch) |n| total += n + 256;
        var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, total) };
        errdefer a.buf.free();
        var b: Buffers = undefined;
        b.state_bytes = own[0..7].*;
        inline for (seq_fields, own) |name, n| @field(b, name) = a.take(n);
        for (b.scratchPtrs(), scratch) |f, n| f.* = a.take(n);
        b.arena = a;
        return b;
    }

    /// Its own scratch for another prompt segment, sharing `b`'s caches and state (follow); deinit frees the scratch.
    pub fn sibling(b: *const Buffers, d: *const cuda.Driver, c: Config, nch: usize) !Buffers {
        const scratch = scratchSizes(c, nch);
        var total: usize = 0;
        for (scratch) |n| total += n + 256;
        var s: Buffers = b.*;
        s.arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, total) };
        for (s.scratchPtrs(), scratch) |f, n| f.* = s.arena.take(n);
        return s;
    }

    /// A sibling takes `b`'s sequence buffers (they move when the engine binds another sequence).
    pub fn follow(s: *Buffers, b: *const Buffers) void {
        inline for (seq_fields) |name| @field(s, name) = @field(b, name);
        s.state_bytes = b.state_bytes;
    }

    /// Every scratch field, in scratchSizes' order.
    fn scratchPtrs(b: *Buffers) [scratch_count]*u64 {
        return .{
            &b.logits, &b.p_meta, &b.p_ids, &b.p_hidden, &b.p_logits, &b.p_sampled, &b.emb,   &b.h[0],  &b.h[1],  &b.y,
            &b.xs,     &b.delta,  &b.proj,  &b.p_xc,     &b.sy,       &b.g,         &b.gxs,   &b.qkv,   &b.q,     &b.att,
            &b.axs,    &b.po,     &b.pm,    &b.pl,       &b.part,     &b.pick,      &b.wts,   &b.plan.members, &b.plan.items,
            &b.plan.counts, &b.plan.rank, &b.plan.hist, &b.act, &b.ymoe, &b.shared,
        };
    }

    /// The shared expert's plan at `rows` rows: halves `experts` and `experts + 1` over slots top_k and top_k + 1.
    pub fn sharedPlan(b: *const Buffers, rows: usize) kern.Plan {
        const items = b.shared + (2 * max_rows + 8 * @as(u64, rows - 1)) * 4;
        return .{ .members = b.shared, .items = items, .counts = items + 6 * 4, .rank = 0, .hist = 0 };
    }

    /// Writes every row count's shared plan (pairs in row order, one item a half since rows <= 16).
    pub fn initShared(b: *const Buffers, ops: kern.Ops, c: Config) !void {
        var w: [shared_words]u32 = undefined;
        for (0..max_rows) |r| {
            w[r] = @intCast(r * c.slots() + c.top_k);
            w[max_rows + r] = @intCast(r * c.slots() + c.top_k + 1);
        }
        for (1..max_rows + 1) |rows| {
            const at = 2 * max_rows + 8 * (rows - 1);
            const n: u32 = @intCast(rows);
            w[at..][0..8].* = .{ @intCast(c.experts), 0, n, @intCast(c.experts + 1), max_rows, n, 2, 2 };
        }
        try ops.upload(b.shared, std.mem.sliceAsBytes(&w));
        try ops.s.synchronize();
    }

    pub fn deinit(b: *Buffers) void {
        b.arena.buf.free();
        b.* = undefined;
    }

    /// The caches and recurrent state as one device range (they are the arena's first buffers).
    pub fn stateBytes(b: *const Buffers) usize {
        return b.dt + b.state_bytes[6] - b.k_cache;
    }

    /// Engine.snapshot: a device copy of every cache and state buffer.
    pub fn snapshot(b: *const Buffers, ops: kern.Ops) !cuda.DeviceBuffer {
        const copy = try cuda.DeviceBuffer.alloc(ops.k.d, b.stateBytes());
        try ops.copy(copy.ptr, b.k_cache, b.stateBytes());
        return copy;
    }

    pub fn restore(b: *const Buffers, ops: kern.Ops, saved: cuda.DeviceBuffer) !void {
        try ops.copy(b.k_cache, saved.ptr, b.stateBytes());
    }

    /// Engine.reset: every cache and state buffer zeroed, as a fresh request starts.
    pub fn reset(b: *const Buffers, ops: kern.Ops) !void {
        const ptrs = [_]u64{ b.k_cache, b.v_cache, b.ssm, b.conv_base, b.raw, b.xc, b.dt };
        for (ptrs, b.state_bytes) |p, n| try ops.fill32(p, 0, n / 4);
    }
};
