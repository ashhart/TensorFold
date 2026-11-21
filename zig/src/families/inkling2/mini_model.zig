//! Mini's forward on one GPU: 48 visit slots a stream (40 rings, 8 whole caches, conv states), prompts and decode rows.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const wts = @import("mini_weights.zig");
const attn = @import("kernels.zig");
const ops = @import("ops.zig");
const loop = @import("loop.zig");
const visit = @import("mini_visit.zig");

const DeviceBuffer = cuda.DeviceBuffer;

/// Prompt rows a forward takes at most: one stream's chunk.
pub const prompt_rows = 512;
/// Decode rows a forward takes over every stream.
pub const round_rows = 64;
/// The widest window one stream verifies in a decode forward.
pub const window_rows = 16;
/// Conv sites a visit keeps state for: the key and value convs, after attention, after the MLP.
pub const Site = enum { k, v, attn, mlp };

/// A stream's rows in one forward: its cache slot, first position and tokens.
pub const Chain = struct { slot: usize, p0: usize, tokens: []const u32 };

pub const Kernels = struct {
    attn: attn.Kernels,
    ops: ops.Kernels,
    loop: loop.Kernels,
    lane: cuda.qmmf.Lane,
    dense: cuda.dense_rows.DenseRows,
    nv: cuda.experts.Nvfp4Experts,
    mx: cuda.experts_mx.Mxfp8Experts,
    glue: cuda.moe_glue.MoeGlue,
    plan_mod: cuda.Module,
    router: cuda.grouped.Router,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        const d = ctx.d;
        var k: Kernels = undefined;
        k.attn = try attn.Kernels.load(d);
        errdefer k.attn.unload();
        k.ops = try ops.Kernels.load(d);
        errdefer k.ops.unload();
        k.loop = try loop.Kernels.load(d);
        errdefer k.loop.unload();
        k.lane = try cuda.qmmf.Lane.load(d, (try ctx.capability()) / 10);
        errdefer k.lane.unload();
        k.dense = try cuda.dense_rows.DenseRows.load(d);
        errdefer k.dense.unload();
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        k.nv = try cuda.experts.Nvfp4Experts.load(d, sms);
        errdefer k.nv.unload();
        k.mx = try cuda.experts_mx.Mxfp8Experts.load(d, sms);
        errdefer k.mx.unload();
        k.glue = try cuda.moe_glue.MoeGlue.load(d);
        errdefer k.glue.unload();
        k.plan_mod = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer k.plan_mod.unload();
        k.router = try cuda.grouped.Router.resolve(k.plan_mod);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.plan_mod.unload();
        k.glue.unload();
        k.mx.unload();
        k.nv.unload();
        k.dense.unload();
        k.lane.unload();
        k.loop.unload();
        k.ops.unload();
        k.attn.unload();
    }

    pub fn experts(k: *const Kernels) cuda.experts.Grouped {
        return .{ .nvfp4 = &k.nv, .mxfp8 = &k.mx };
    }
};

/// The forward's device scratch, sized for `prompt_rows` rows (decode rounds use the first `round_rows`).
pub const Scratch = struct {
    ids: u64, // int32 [R]
    pos: u64, // int32 [R]
    seg_of: u64, // int32 [R]: the row's stream in the call
    segs: [2]u64, // attention Seg [streams]: sliding, then full caches (offsets differ by cache length)
    conv_segs: u64, // int4 [streams]: row0, rows, slot
    tiles: u64, // attention Tile [max tiles]
    h: [2]u64, // bf16 [R, H]: the running state in, the next visit's out
    blocks: u64, // bf16 [max blocks][R, H]
    bank: u64, // bf16 [bank blocks][R, H]: the previous pass's blocks past its entry
    bank_h: u64, // bf16 [R, H]: the previous pass's output
    msg: u64, // fp32 [4][R, H]
    mixf: u64, // fp32 [3][R, H]
    gdown: u64, // bf16 [3][R][rank]
    xin: u64, // bf16 [3][R, H]
    q: u64,
    kraw: u64,
    vraw: u64,
    rel: u64,
    gate: u64,
    sink: u64,
    dyn: [2]u64, // fp32 [R, kv * 4]
    kc: u64,
    vc: u64,
    qp: u64,
    relp: u64,
    tab: u64,
    part_o: u64,
    part_ml: u64,
    att: u64,
    o: u64,
    h32: u64, // fp32 [R, H]
    xm: u64,
    scores: u64,
    pick: u64,
    rw: u64,
    plan: cuda.grouped.Plan,
    act: u64,
    y: u64,
    moe: u64,
    xf: u64,
    logits: u64, // fp32 [round_rows, unpadded vocab]
    lane_part: u64,
    entries: usize, // (row, head, partial) slots part_o and part_ml hold
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *const wts.Weights,
    c: *const config.Text,
    context: usize, // positions a slot holds
    slots: usize,
    sliding: attn.Layer,
    full: attn.Layer,
    kv: [2][config.max_visits]u64 = undefined, // a visit's K and V: [slots][kv][len][128] bf16
    conv: [config.max_visits][4]u64 = undefined, // a visit's conv states: fp32 [slots][3][C] by Site
    s: Scratch = undefined,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    planned: ?usize = null, // a plan: the bytes counted, nothing allocated

    fn alloc(m: *Model, bytes: usize) !u64 {
        if (m.planned) |*p| {
            p.* += std.mem.alignForward(usize, @max(bytes, 256), 256);
            return 0;
        }
        var b = try DeviceBuffer.alloc(m.d, @max(bytes, 256));
        errdefer b.free();
        try m.buffers.append(m.gpa, b);
        return b.ptr;
    }

    fn zeros(m: *Model, s: cuda.Stream, bytes: usize) !u64 {
        const p = try m.alloc(bytes);
        if (m.planned == null) try m.d.check(m.d.api.cuMemsetD8Async(p, 0, bytes, s.handle), "cuMemsetD8Async");
        return p;
    }

    fn blank(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const wts.Weights, context: usize, slots: usize) Model {
        return .{
            .gpa = gpa,
            .d = d,
            .w = w,
            .c = &w.config.text,
            .context = context,
            .slots = slots,
            .sliding = attn.Layer.sliding(w.config.text.window, prompt_rows),
            .full = attn.Layer.global(@intCast(context)),
        };
    }

    /// The caches and scratch of `slots` streams at `context` positions.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, k: *const Kernels, context: usize, slots: usize) !Model {
        var m = blank(gpa, d, w, context, slots);
        errdefer m.deinit();
        try m.build(s, k);
        try s.synchronize();
        return m;
    }

    /// The device bytes init would allocate.
    pub fn openBytes(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, k: *const Kernels, context: usize, slots: usize) !usize {
        var m = blank(gpa, d, w, context, slots);
        m.planned = 0;
        try m.build(s, k);
        return m.planned.?;
    }

    /// One stream's caches: 40 sliding rings, 8 whole caches and 48 conv states (Mini).
    pub fn streamBytes(m: *const Model) usize {
        const t = m.c;
        const kvw: usize = @as(usize, t.local.kv) * t.local.dim;
        const v = &m.w.plan.visits;
        const caches = v.count(.sliding) * m.sliding.ring + v.count(.full) * m.full.ring;
        return caches * kvw * 2 * 2 + v.count(.conv) * 3 * (2 * kvw + 2 * @as(usize, t.hidden)) * 4;
    }

    pub fn deinit(m: *Model) void {
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
    }

    fn build(m: *Model, s: cuda.Stream, k: *const Kernels) !void {
        const t = m.c;
        const kvw: usize = @as(usize, t.local.kv) * t.local.dim;
        for (m.w.plan.visits.slice(), 0..) |step, i| {
            const len: usize = if (step.of(.full) != null) m.full.ring else m.sliding.ring;
            for (0..2) |side| m.kv[side][i] = try m.alloc(m.slots * len * kvw * 2);
            for (std.enums.values(Site)) |site| {
                const width: usize = if (site == .k or site == .v) kvw else t.hidden;
                m.conv[i][@backingInt(site)] = try m.zeros(s, m.slots * 3 * width * 4);
            }
        }
        try m.scratch(k);
    }

    fn scratch(m: *Model, k: *const Kernels) !void {
        const t = m.c;
        const r: usize = prompt_rows;
        const h: usize = t.hidden;
        const q: usize = @as(usize, t.local.q) * t.local.dim;
        const kvw: usize = @as(usize, t.local.kv) * t.local.dim;
        const heads: usize = t.local.q;
        const rank: usize = t.loop.?.rank;
        const pairs = r * t.topk;
        const blocks = (t.layers + t.block - 1) / t.block;
        // partials: every prompt row's sliding or full chunks, or a split decode call's sub-chunks
        const stride_full = attn.slotsFor(m.full, m.context);
        const entries = @max(r * heads * @max(attn.slotsFor(m.sliding, m.context), stride_full), attn.split_rows * heads * stride_full * attn.subs);
        var part: usize = 0;
        for (m.w.layers) |l| for ([_]cuda.qmmf.Weight{ l.attn.wq, l.attn.wk, l.attn.wv, l.attn.wr, l.attn.gate, l.attn.sink, l.attn.wo }) |w| {
            for (1..r + 1) |rows| part = @max(part, k.lane.partBytes(rows, w));
        };
        for (1..round_rows + 1) |rows| part = @max(part, k.lane.partBytes(rows, m.w.unembed));
        const items = cuda.grouped.maxItems(pairs, t.experts, 64);
        m.s = .{
            .ids = try m.alloc(r * 4),
            .pos = try m.alloc(r * 4),
            .seg_of = try m.alloc(r * 4),
            .segs = .{ try m.alloc(round_rows * @sizeOf(attn.Seg)), try m.alloc(round_rows * @sizeOf(attn.Seg)) },
            .conv_segs = try m.alloc(round_rows * 16),
            .tiles = try m.alloc((attn.tileCount(r, 3) + round_rows) * @sizeOf(attn.Tile)),
            .h = .{ try m.alloc(r * h * 2), try m.alloc(r * h * 2) },
            .blocks = try m.alloc(blocks * r * h * 2),
            .bank = try m.alloc(@max(1, m.w.plan.bank_blocks) * r * h * 2),
            .bank_h = try m.alloc(r * h * 2),
            .msg = try m.alloc(4 * r * h * 4),
            .mixf = try m.alloc(3 * r * h * 4),
            .gdown = try m.alloc(3 * r * rank * 2),
            .xin = try m.alloc(3 * r * h * 2),
            .q = try m.alloc(r * q * 2),
            .kraw = try m.alloc(r * kvw * 2),
            .vraw = try m.alloc(r * kvw * 2),
            .rel = try m.alloc(r * heads * t.d_rel * 2),
            .gate = try m.alloc(r * q * 2),
            .sink = try m.alloc(r * heads * 2),
            .dyn = .{ try m.alloc(r * t.global.kv * config.conv_taps * 4), try m.alloc(r * t.global.kv * config.conv_taps * 4) },
            .kc = try m.alloc(r * kvw * 2),
            .vc = try m.alloc(r * kvw * 2),
            .qp = try m.alloc(r * q * 2),
            .relp = try m.alloc(r * heads * t.d_rel * 2),
            .tab = try m.alloc(attn.tableBytes(r, @intCast(heads))),
            .part_o = try m.alloc(entries * attn.head_dim * 4),
            .part_ml = try m.alloc(entries * 8),
            .att = try m.alloc(r * q * 2),
            .o = try m.alloc(r * h * 2),
            .h32 = try m.alloc(r * h * 4),
            .xm = try m.alloc(r * h * 2),
            .scores = try m.alloc(r * t.experts * 4),
            .pick = try m.alloc(pairs * 4),
            .rw = try m.alloc(pairs * 4),
            .plan = .{
                .members = try m.alloc(pairs * 4),
                .items = try m.alloc(items * 12),
                .counts = try m.alloc(8),
                .rank = try m.alloc(pairs * 4),
                .hist = try m.alloc(((pairs + 1023) / 1024) * t.experts * 4),
            },
            .act = try m.alloc(pairs * t.expert_inter * 2),
            .y = try m.alloc(pairs * h * 4),
            .moe = try m.alloc(r * h * 2),
            .xf = try m.alloc(r * h * 2),
            .logits = try m.alloc(round_rows * t.unpadded_vocab * 4),
            .lane_part = try m.alloc(part),
            .entries = entries,
        };
    }

    pub fn upload(m: *const Model, s: cuda.Stream, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(m.d, dst, bytes, s);
    }

    pub fn copy(m: *const Model, s: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, s.handle), "cuMemcpyDtoDAsync");
    }

    /// A slot's conv states to zero: a stream starting at position 0 has no inputs before it.
    pub fn clear(m: *Model, s: cuda.Stream, slot: usize) !void {
        const t = m.c;
        const kvw: usize = @as(usize, t.local.kv) * t.local.dim;
        for (0..m.w.plan.visits.len) |v| for (std.enums.values(Site)) |site| {
            const width: usize = if (site == .k or site == .v) kvw else t.hidden;
            const at = m.conv[v][@backingInt(site)] + slot * 3 * width * 4;
            try m.d.check(m.d.api.cuMemsetD8Async(at, 0, 3 * width * 4, s.handle), "cuMemsetD8Async");
        };
    }

    /// Every chain's rows (one prompt chunk, or decode windows); rows [head0, head0 + heads)'s fp32 logits at s.logits.
    pub fn forward(m: *Model, k: *const Kernels, s: cuda.Stream, chains: []const Chain, prompt: bool, head0: usize, heads: usize) !void {
        var rows: usize = 0;
        for (chains, 0..) |ch, i| {
            if (ch.tokens.len == 0 or ch.slot >= m.slots or ch.p0 + ch.tokens.len > m.context) return error.BadChain;
            if (!prompt and ch.tokens.len > window_rows) return error.WindowTooWide;
            for (chains[0..i]) |o| if (o.slot == ch.slot) return error.BadChain;
            for (ch.tokens) |id| if (id >= m.c.vocab) return error.InvalidToken;
            rows += ch.tokens.len;
        }
        if (rows == 0 or rows > @as(usize, if (prompt) prompt_rows else round_rows) or (prompt and chains.len != 1)) return error.TooManyRows;
        if (chains.len > round_rows or heads > round_rows or head0 + heads > rows) return error.TooManyRows;
        for (chains) |ch| if (ch.p0 == 0) try m.clear(s, ch.slot);
        var r = try visit.Rows.stage(m, s, chains, rows, prompt);
        try visit.run(m, k, s, &r, head0, heads);
    }
};

test "the forward's launches type-check against the shared operators" {
    std.testing.refAllDecls(Model);
    std.testing.refAllDecls(Kernels);
}
