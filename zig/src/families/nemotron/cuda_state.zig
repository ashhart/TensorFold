//! Device buffers of one Nemotron sequence: its caches and recurrent state, and the scratch a window or chunk uses.

const std = @import("std");
const lesson_route = @import("living").lesson_route;
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const kern = @import("cuda_kernels.zig");
const glue = @import("cuda_glue.zig");
const sampler = @import("cuda_sampler.zig");
const seq_graphs = @import("cuda_seq_graphs.zig");
const latent = @import("cuda_latent.zig");
const worker = @import("cuda_worker.zig");

pub const max_rows = 32; // a verify window's rows at most (the widest row tile of the row-parallel kernels)
pub const prefill_rows = 2048; // rows of a staggered segment's chunk, and of a pass under dumps or lessons
pub const pass_rows = 2 * prefill_rows; // rows a serial prompt pass runs at most (a row's bits are the same at any cut)
pub const yield_rows = 256; // rows of a background pass's chunk: a request that arrives waits for one at most
pub const chunk_keys = 512; // keys an attention chunk holds, at fixed absolute positions
const shared_rows = cuda.qlinear.Affine4Kernels.max_rows; // rows a shared plan covers (windowRows without a latent MoE)
const shared_words = 2 * shared_rows + 8 * shared_rows; // the shared expert's members, then (items, counts) a row count

/// Super verifies 32 rows; Bolt copies and non-latent affine models retain their qualified 16-row window.
pub fn windowRows(c: Config) usize {
    return if (c.latent > 0 and !c.bolt) max_rows else cuda.qlinear.Affine4Kernels.max_rows;
}

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

/// A sequence's KV planes in the display carveout, given back when the sequence goes.
pub const Carved = struct {
    carve: *cuda.Carveout,
    kv: [2]cuda.DeviceBuffer,

    /// Both KV planes of `n` bytes each from `carve`, or null when it is off or cannot hold both.
    fn take(carve: ?*cuda.Carveout, n: usize) !?Carved {
        const cv = carve orelse return null;
        const k = (try cv.take(n)) orelse return null;
        errdefer cv.give(k);
        const v = (try cv.take(n)) orelse {
            cv.give(k);
            return null;
        };
        return .{ .carve = cv, .kv = .{ k, v } };
    }

    fn give(x: Carved) void {
        x.carve.give(x.kv[1]);
        x.carve.give(x.kv[0]);
    }

    /// Arena bytes for `sizes` with the carved planes left out.
    fn arenaBytes(x: ?Carved, sizes: []const usize) usize {
        var total: usize = 0;
        for (sizes, 0..) |n, i| total += if (x != null and i < 2) 0 else n + 256;
        return total;
    }
};

/// A sequence's own Buffers fields: cache and state planes (the first seven), window io and its sampled rule.
pub const seq_fields = [_][]const u8{ "k_cache", "v_cache", "ssm", "conv_base", "raw", "xc", "dt", "meta", "ids", "hidden", "sampled", "rule" };

/// Bytes of one token's key (or value) row in one attention layer: every KV head's dims in bf16.
pub fn kvRowBytes(c: Config) usize {
    return c.kv_heads * c.head_dim * 2;
}

/// Bytes of every Mamba layer's SSM and conv states (seq_fields' ssm, conv_base): a prompt mark keeps both whole.
pub fn mambaPlanes(c: Config) [2]usize {
    const nm: usize = c.count(.mamba);
    return .{ nm * c.mamba_heads * c.mamba_head_dim * c.state * 4, nm * 3 * c.convDim() * 2 };
}

fn seqSizes(c: Config, max_len: usize) [seq_fields.len]usize {
    const W: usize = windowRows(c);
    const nm: usize = c.count(.mamba);
    const kv = c.count(.attention) * max_len * kvRowBytes(c);
    const cd: usize = c.convDim();
    const m = mambaPlanes(c);
    return .{ kv, kv, m[0], m[1], nm * 2 * W * cd * 2, nm * 2 * W * cd * 2, nm * 2 * W * c.mamba_heads * 4, 16, W * 4, W * c.hidden * 2, W * 4, @sizeOf(sampler.Rule) };
}

/// The most fp32 slice partials the lane matmul writes when it reduces them: every dense shape, rows short of fusing.
fn lanePartBytes(c: Config) usize {
    const shapes = [_][2]usize{ .{ c.projDim(), c.hidden }, .{ c.hidden, c.inner() }, .{ c.qkvDim(), c.hidden }, .{ c.hidden, c.heads * c.head_dim }, .{ c.vocab, c.hidden }, .{ c.hidden, 2 * c.hidden }, .{ c.latent, c.hidden }, .{ c.hidden, c.latent } };
    const rows = cuda.qmmf.fused_rows - 1;
    var most: usize = 16;
    for (shapes) |s| {
        const p = cuda.qmmf.plan(rows, s[0], s[1], false);
        if (p.reduce) most = @max(most, @as(usize, p.sk) * rows * s[0] * 4);
    }
    return most;
}
const scratch_count = 44;

/// Bytes of the shared expert's fixed plans (Buffers.shared): shared_words, or a latent MoE's plan for every row count.
fn sharedBytes(c: Config) usize {
    return (if (c.latent > 0 and !c.bolt) windowRows(c) * Buffers.latentPlanWords(c) else shared_words) * 4;
}

/// Scratch windows and chunks share on one stream: window logits, chunk io, activations, the expert plan.
fn scratchSizes(c: Config, nch: usize) [scratch_count]usize {
    const R: usize = pass_rows;
    const W: usize = windowRows(c);
    const D: usize = c.hidden;
    const ns: usize = c.slots();
    const qd: usize = c.heads * c.head_dim;
    const cd: usize = c.convDim();
    const xd: usize = c.inner();
    const pairs = R * ns;
    // a latent MoE plans pairs (row, pick, share held) over its share tables; its experts write fp32 to the exchange
    const lat = c.latent > 0 and !c.bolt;
    const own: usize = if (lat) c.ownShares() else 1;
    const plan_experts = if (lat) c.experts * own else c.experts + 2;
    const items = kern.maxItems(@intCast(pairs * own), plan_experts, 16);
    const ymoe = if (lat) c.world * latent.partFloats(c, R) * 4 else pairs * D * 4; // a region a rank
    const job = if (lat and c.world > 1) worker.jobBytes(c) else 0;
    const sw = if (lat) c.shared_width / c.shares else 0;
    // the checkpoint's math: e4m3 rows of the widest FP8 projection input, and a latent MoE's NVFP4 rows
    const router_slices: usize = if (lat) latent.routerSlices(D) else if (c.bolt) glue.max_router_slices else 6;
    const ck8 = if (c.ck) R * @max(D, @max(xd, qd)) else 0;
    const ck4 = if (c.ck and lat) @import("cuda_latent_ck.zig").rowsBytes(c, R) else 0;
    // its passes of prompt_tokens rows or more write up's rows to ck4: bf16 activations only for the calls under them
    const act_rows: usize = if (c.ck and lat) cuda.ck.prompt_tokens - 1 else R;
    return .{
        W * c.vocab * 2,                   16,                                             R * 4,
        R * D * 2,                         c.vocab * 2,                                    16,
        R * D * 2,                         R * D * 2,                                      R * D * 2,
        R * D * 2,                         R * D / 64 * 4,                                 R * D * 2,
        R * c.projDim() * 2,               R * cd * 2,                                     R * xd * 2,
        R * xd * 2,                        R * xd / 64 * 4,                                R * c.qkvDim() * 2,
        R * qd * 2,                        R * qd * 2,                                     R * qd / 64 * 4,
        W * nch * qd * 4,                  W * nch * c.heads * 4,                          W * nch * c.heads * 4,
        router_slices * R * c.experts * 4, pairs * own * 4,                                R * ns * 4,
        pairs * own * 4,                   items * 12,                                     8,
        pairs * own * 4,                   (pairs * own + 1023) / 1024 * plan_experts * 4, act_rows * ns * c.expert_width * 2,
        ymoe,                              sharedBytes(c),                                 R * kern.pocket_width * 4,
        R * c.latent * 2,                  R * c.latent * 2,                               act_rows * own * sw * 2,
        R * own * 4,                       job,                                            ck8,
        ck4,                               lanePartBytes(c),
    };
}

/// One sequence: its own buffers (seq_fields, then its MTP head's) and where it stands while another is bound.
pub const Seq = struct {
    arena: ?Arena, // null: the engine's own buffers
    carved: ?Carved = null,
    ptr: [seq_fields.len]u64,
    table: u64 = 0, // its page table on the device (0: contiguous KV planes of max_len rows)
    pages: []const u32 = &.{}, // the pages the table names, as the host holds them (cuda_pages.zig)
    cap: usize = 0, // rows its caches hold (0: the engine's max_len)
    head: [3]u64 = @splat(0), // the MTP head's seq_fields (cuda_mtp.zig)
    graphs: seq_graphs.Graphs = .{}, // what this sequence captured for itself (the own one: masked windows alone)
    epoch: u32 = 0, // the engine's graph epoch its graphs were captured in
    pos: usize = 0,
    parity: usize = 0,
    prev_keep: usize = 0,
    rows: usize = 0,
    head_pos: usize = 0,
    sampling: ?sampler.Sampling = null,
    slot: u32 = 0, // its lesson latch slot (lesson_route)
    decide: i64 = lesson_route.none, // the prompt row where its lessons decide whether they act on the reply
    untaught: bool = false, // reads the model as shipped: no learned neuron acts on its rows (a lesson writer's reads)

    /// The device bytes one sequence takes: its buffers for `max_len` cache rows and a head's `head` buffers.
    pub fn bytes(c: Config, max_len: usize, head: []const usize) usize {
        var total: usize = 0;
        for (seqSizes(c, max_len)) |n| total += n + 256;
        for (head) |n| total += n + 256;
        return total;
    }

    /// Cache and head buffers are zeroed on their owning stream; the KV pair goes to `carve` while it holds both.
    pub fn init(d: *const cuda.Driver, c: Config, max_len: usize, head: []const usize, stream: cuda.Stream, carve: ?*cuda.Carveout) !Seq {
        const sizes = seqSizes(c, max_len);
        const carved = try Carved.take(carve, sizes[0]);
        errdefer if (carved) |x| x.give();
        var total = Carved.arenaBytes(carved, &sizes);
        for (head) |n| total += n + 256;
        var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, total) };
        errdefer a.buf.free();
        try a.buf.fill8(0, stream.handle);
        var s: Seq = .{ .arena = null, .ptr = undefined, .carved = carved };
        for (&s.ptr, sizes, 0..) |*p, n, i| p.* = if (carved != null and i < 2) carved.?.kv[i].ptr else a.take(n);
        for (s.head[0..head.len], head) |*p, n| p.* = a.take(n);
        s.arena = a;
        return s;
    }

    /// Fixed bytes of a paged sequence: every plane but the KV planes, its head's `head` buffers, a table of `pages`.
    pub fn pagedBytes(c: Config, head: []const usize, pages: usize) usize {
        return bytes(c, 0, head) + pages * 4 + 256;
    }

    /// A sequence whose KV planes are pages at `kv` (layer l's K, then V, `chunk_keys` rows each) through its table.
    pub fn initPaged(d: *const cuda.Driver, c: Config, head: []const usize, stream: cuda.Stream, kv: u64, pages: usize) !Seq {
        const sizes = seqSizes(c, 0);
        var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, pagedBytes(c, head, pages)) };
        errdefer {
            cuda.completion.drain(d, stream, "a paged sequence clear did not drain; its arena stays alive");
            a.buf.free();
        }
        try a.buf.fill8(0, stream.handle);
        var s: Seq = .{ .arena = null, .ptr = undefined };
        for (&s.ptr, sizes) |*p, n| p.* = a.take(n);
        for (s.head[0..head.len], head) |*p, n| p.* = a.take(n);
        const plane: u64 = chunk_keys * kvRowBytes(c);
        s.ptr[0] = kv;
        s.ptr[1] = kv + plane;
        if (head.len > 0) s.head[0..2].* = .{ kv + c.count(.attention) * 2 * plane, kv + (c.count(.attention) * 2 + 1) * plane };
        s.table = a.take(pages * 4);
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
        if (s.carved) |x| x.give();
        s.* = undefined;
    }
};

pub const Buffers = struct {
    arena: Arena,
    carved: ?Carved, // the own sequence's KV planes, when the display carveout holds them
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
    table: u64, // the bound sequence's page table (0: contiguous KV planes)
    shared: u64, // the shared expert's fixed plans: one initialized plan per row count
    pocket: u64, // the pocket's activations, fp32 [rows, pocket_width]
    lane_part: u64, // the lane matmul's fp32 slice partials on a card without clusters (rows short of fusing)
    lat: u64, // a latent MoE's rows in the latent width (bf16)
    comb: u64, // its routed pairs' latent sum a row (bf16)
    sact: u64, // its shared expert's activations (bf16)
    sids: u64, // its shared expert's pair ids, share i % own (cuda_latent.initShareIds)
    job: u64, // rank 0's job to the other ranks: a header and the rows their shares read (cuda_worker.zig)
    ck8: u64, // the checkpoint's math: e4m3 rows an FP8 projection reads
    ck4: u64, // and a latent MoE's NVFP4 rows (cuda_latent_ck.zig)

    /// Sizes every buffer for `max_len` cache rows and `nch` attention chunk partials a row; the own sequence's first.
    pub fn init(d: *const cuda.Driver, c: Config, max_len: usize, nch: usize, carve: ?*cuda.Carveout) !Buffers {
        const own = seqSizes(c, max_len);
        const scratch = scratchSizes(c, nch);
        const carved = try Carved.take(carve, own[0]);
        errdefer if (carved) |x| x.give();
        var total = Carved.arenaBytes(carved, &own);
        for (scratch) |n| total += n + 256;
        var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, total) };
        errdefer a.buf.free();
        var b: Buffers = undefined;
        b.carved = carved;
        b.table = 0;
        b.state_bytes = own[0..7].*;
        inline for (seq_fields, own, 0..) |name, n, i| @field(b, name) = if (carved != null and i < 2) carved.?.kv[i].ptr else a.take(n);
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
        s.carved = null; // the owner gives its planes back
        s.arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, total) };
        for (s.scratchPtrs(), scratch) |f, n| f.* = s.arena.take(n);
        // the shared expert's plans and pair ids are written once, by the owner's load (initShared)
        s.shared = b.shared;
        s.sids = b.sids;
        return s;
    }

    /// A sibling takes `b`'s sequence buffers (they move when the engine binds another sequence).
    pub fn follow(s: *Buffers, b: *const Buffers) void {
        inline for (seq_fields) |name| @field(s, name) = @field(b, name);
        s.table = b.table;
        s.state_bytes = b.state_bytes;
    }

    /// Every scratch field, in scratchSizes' order.
    fn scratchPtrs(b: *Buffers) [scratch_count]*u64 {
        return .{
            &b.logits,    &b.p_meta,    &b.p_ids, &b.p_hidden,  &b.p_logits, &b.p_sampled, &b.emb, &b.h[0],         &b.h[1],       &b.y,
            &b.xs,        &b.delta,     &b.proj,  &b.p_xc,      &b.sy,       &b.g,         &b.gxs, &b.qkv,          &b.q,          &b.att,
            &b.axs,       &b.po,        &b.pm,    &b.pl,        &b.part,     &b.pick,      &b.wts, &b.plan.members, &b.plan.items, &b.plan.counts,
            &b.plan.rank, &b.plan.hist, &b.act,   &b.ymoe,      &b.shared,   &b.pocket,    &b.lat, &b.comb,         &b.sact,       &b.sids,
            &b.job,       &b.ck8,       &b.ck4,   &b.lane_part,
        };
    }

    /// The shared expert's plan at `rows` rows: halves `experts` and `experts + 1` over slots top_k and top_k + 1.
    pub fn sharedPlan(b: *const Buffers, rows: usize) kern.Plan {
        const items = b.shared + (2 * shared_rows + 8 * @as(u64, rows - 1)) * 4;
        return .{ .members = b.shared, .items = items, .counts = items + 6 * 4, .rank = 0, .hist = 0 };
    }

    /// Words one row count's latent shared plan takes: the pairs of a window's rows, then their items and counts.
    fn latentPlanWords(c: Config) usize {
        return latentPlanAt(c).counts + 2;
    }

    /// Where a latent shared plan's items and counts start, in words from its members.
    fn latentPlanAt(c: Config) struct { items: usize, counts: usize } {
        const pairs = windowRows(c) * c.ownShares();
        return .{ .items = pairs, .counts = pairs + 3 * kern.maxItems(pairs, c.ownShares(), 16) };
    }

    /// A latent MoE's shared plan at `rows` rows (1 to max_rows) as initShared wrote it; rank and hist are b.plan's.
    pub fn latentPlan(b: *const Buffers, c: Config, rows: usize) kern.Plan {
        const at = latentPlanAt(c);
        const members = b.shared + (rows - 1) * latentPlanWords(c) * 4;
        return .{ .members = members, .items = members + at.items * 4, .counts = members + at.counts * 4, .rank = b.plan.rank, .hist = b.plan.hist };
    }

    /// Writes every row count's shared plan: a latent MoE's (row, share) pairs by Ops.plan, else pairs in row order.
    pub fn initShared(b: *const Buffers, ops: kern.Ops, c: Config) !void {
        if (c.latent > 0 and !c.bolt) {
            try latent.initShareIds(ops.k.latent.?, ops.s, b.sids, pass_rows, c.ownShares());
            for (1..windowRows(c) + 1) |rows| try ops.plan(b.sids, rows * c.ownShares(), c.ownShares(), 16, b.latentPlan(c, rows));
            return ops.s.synchronize();
        }
        var w: [shared_words]u32 = undefined;
        for (0..shared_rows) |r| {
            w[r] = @intCast(r * c.slots() + c.top_k);
            w[shared_rows + r] = @intCast(r * c.slots() + c.top_k + 1);
        }
        for (1..shared_rows + 1) |rows| {
            const at = 2 * shared_rows + 8 * (rows - 1);
            const n: u32 = @intCast(rows);
            w[at..][0..8].* = .{ @intCast(c.experts), 0, n, @intCast(c.experts + 1), shared_rows, n, 2, 2 };
        }
        try ops.upload(b.shared, std.mem.sliceAsBytes(&w));
        if (c.latent > 0) {
            const z: [pass_rows]u32 = @splat(0);
            try ops.upload(b.sids, std.mem.sliceAsBytes(&z));
        }
        try ops.s.synchronize();
    }

    pub fn deinit(b: *Buffers) void {
        b.arena.buf.free();
        if (b.carved) |x| x.give();
        b.* = undefined;
    }

    /// The cache and state planes' addresses, in seq_fields' order (the KV pair may live in the carveout).
    pub fn planes(b: *const Buffers) [7]u64 {
        return .{ b.k_cache, b.v_cache, b.ssm, b.conv_base, b.raw, b.xc, b.dt };
    }

    /// Bytes of every cache and state plane together.
    pub fn stateBytes(b: *const Buffers) usize {
        var total: usize = 0;
        for (b.state_bytes) |n| total += n;
        return total;
    }

    /// Engine.snapshot: a device copy of every cache and state plane, packed.
    pub fn snapshot(b: *const Buffers, ops: kern.Ops) !cuda.DeviceBuffer {
        var copy = try cuda.DeviceBuffer.alloc(ops.k.d, b.stateBytes());
        errdefer copy.free();
        var at: usize = 0;
        for (b.planes(), b.state_bytes) |p, n| {
            try ops.copy(copy.ptr + at, p, n);
            at += n;
        }
        return copy;
    }

    pub fn restore(b: *const Buffers, ops: kern.Ops, saved: cuda.DeviceBuffer) !void {
        var at: usize = 0;
        for (b.planes(), b.state_bytes) |p, n| {
            try ops.copy(p, saved.ptr + at, n);
            at += n;
        }
    }

    /// Engine.reset: every cache and state buffer zeroed, as a fresh request starts (pages: written before any read).
    pub fn reset(b: *const Buffers, ops: kern.Ops) !void {
        for (b.planes(), b.state_bytes, 0..) |p, n, i| if (i >= 2 or b.table == 0) try ops.fill32(p, 0, n / 4);
    }

    /// Bytes of the own sequence's planes in the display carveout.
    pub fn carvedBytes(b: *const Buffers) usize {
        return if (b.carved) |x| x.kv[0].len + x.kv[1].len else 0;
    }
};

test "each row count's latent shared plan holds its pairs and items, apart from the next, inside the shared slot" {
    var c: Config = undefined;
    c.latent = 1024;
    c.bolt = false;
    c.shares = 2;
    var b: Buffers = undefined;
    b.shared = 0;
    b.plan.rank = 0;
    b.plan.hist = 0;
    for (1..c.shares + 1) |world| {
        c.world = world;
        const own = c.ownShares();
        for (1..max_rows + 1) |rows| {
            const p = b.latentPlan(c, rows);
            try std.testing.expect(p.items - p.members >= rows * own * 4);
            try std.testing.expect(p.counts - p.items >= 3 * kern.maxItems(rows * own, own, 16) * 4);
            const end = p.counts + 8;
            try std.testing.expect(if (rows < max_rows) end <= b.latentPlan(c, rows + 1).members else end <= sharedBytes(c));
        }
    }
}
