//! A stream's committed state (DeltaNet recurrences, conv windows, attention caches) and the forward's scratch.

const std = @import("std");
const cuda = @import("cuda");
const c = @import("shape.zig");
const kern = @import("kernels.zig");
const tri = @import("triton.zig");
const torch_ops = kern.torch_ops;

/// The most rows a stream's verify window takes (the multi-stream GDN tree kernel's limit).
pub const max_rows = 16;
/// The most streams a shared round takes.
pub const max_streams = 16;
/// The most rows a verify round takes: every stream's whole window.
pub const round_rows = max_rows * max_streams;
/// The most items a batched copy takes: every stream's layers, three rows or a key and a value each.
pub const copy_items = max_streams * 64 * 3 + round_rows * 64 * 2;
/// The most candidates a sampled row reads from the GPU: top_k up to 1,024, and exact_sampling.MARGIN past it.
pub const max_candidates = 1024 + 8;

/// A bump allocator over one device buffer, every slice 256-byte aligned.
pub const Arena = struct {
    buf: ?cuda.DeviceBuffer, // null: a measuring pass, every take counted and none backed
    used: usize = 0,

    pub fn init(d: *const cuda.Driver, len: usize) !Arena {
        return .{ .buf = try cuda.DeviceBuffer.alloc(d, len) };
    }

    pub fn deinit(a: *Arena) void {
        if (a.buf) |*b| b.free();
    }

    pub fn take(a: *Arena, len: usize) !u64 {
        const at = std.mem.alignForward(usize, a.used, 256);
        a.used = at + len;
        const b = a.buf orelse return 0;
        if (a.used > b.len) return error.ArenaFull;
        return b.ptr + at;
    }
};

/// One stream's committed state: DeltaNet recurrences and conv windows, attention keys and values for `capacity`.
pub const Seq = struct {
    arena: Arena,
    rec: []u64, // by layer: (hv, dv, dk) fp32, 0 for attention layers
    conv: []u64, // by layer: (3, conv dim) bf16
    keys: []u64, // by layer: (capacity, kv heads, head dim) bf16
    values: []u64,
    head_in: u64, // the last prompt row normed, (hidden) bf16, and its group sums: the first draw's input
    head_xs: u64,
    capacity: usize,
    pos: usize = 0,
    gpa: std.mem.Allocator,

    /// Device bytes a stream of `capacity` positions takes.
    pub fn bytes(g: c.Geometry, capacity: usize) usize {
        var n: usize = std.mem.alignForward(usize, g.hidden * 2, 256) + std.mem.alignForward(usize, g.hidden / 64 * 4, 256);
        const a = std.mem.alignForward;
        for (0..g.layers) |i| n += if (c.linear(i)) a(usize, g.deltaBytes(), 256) + a(usize, g.convBytes(), 256) else 2 * a(usize, capacity * g.kvInner() * 2, 256);
        return n;
    }

    /// A fresh stream: zero recurrences and conv windows, caches for `capacity` positions.
    pub fn init(gpa: std.mem.Allocator, ops: kern.Ops, g: c.Geometry, capacity: usize) !Seq {
        var s: Seq = .{ .gpa = gpa, .arena = try Arena.init(ops.k.d, bytes(g, capacity)), .rec = &.{}, .conv = &.{}, .keys = &.{}, .values = &.{}, .head_in = 0, .head_xs = 0, .capacity = capacity };
        errdefer s.arena.deinit();
        s.rec = try gpa.alloc(u64, g.layers);
        errdefer gpa.free(s.rec);
        s.conv = try gpa.alloc(u64, g.layers);
        errdefer gpa.free(s.conv);
        s.keys = try gpa.alloc(u64, g.layers);
        errdefer gpa.free(s.keys);
        s.values = try gpa.alloc(u64, g.layers);
        errdefer gpa.free(s.values);
        const bf = 2;
        s.head_in = try s.arena.take(g.hidden * 2);
        s.head_xs = try s.arena.take(g.hidden / 64 * 4);
        for (0..g.layers) |i| {
            s.rec[i] = 0;
            s.conv[i] = 0;
            s.keys[i] = 0;
            s.values[i] = 0;
            const A = &s.arena;
            if (c.linear(i)) {
                s.rec[i] = try A.take(g.deltaBytes());
                s.conv[i] = try A.take(g.convBytes());
            } else {
                s.keys[i] = try A.take(capacity * g.kvInner() * bf);
                s.values[i] = try A.take(capacity * g.kvInner() * bf);
            }
        }
        try s.reset(ops, g);
        return s;
    }

    pub fn deinit(s: *Seq) void {
        s.arena.deinit();
        s.gpa.free(s.rec);
        s.gpa.free(s.conv);
        s.gpa.free(s.keys);
        s.gpa.free(s.values);
        s.* = undefined;
    }

    /// Back to an empty prompt: zero recurrences and conv windows (State.__init__'s zeros).
    pub fn reset(s: *Seq, ops: kern.Ops, g: c.Geometry) !void {
        for (0..g.layers) |i| if (c.linear(i)) {
            try ops.fill32(s.rec[i], 0, g.deltaBytes() / 4);
            try ops.fill32(s.conv[i], 0, g.convBytes() / 4);
        };
        s.pos = 0;
    }
};

/// One DeltaNet layer's window record, kept until commit: what the replay steps and the conv window takes.
pub const GdnRecord = struct { q: u64, k: u64, v: u64, g: u64, beta: u64, qkv: u64 };
/// One attention layer's window record: the rotated keys and the values the window's rows would add.
pub const AttnRecord = struct { k: u64, v: u64 };

/// The forward's device buffers, sized for `rows` rows a call (a decode window, or a prompt chunk).
pub const Scratch = struct {
    arena: Arena,
    rows: usize,
    // the residual stream and the normed rows with their group sums
    x: u64,
    h: u64,
    y: u64,
    xs: u64,
    r: u64,
    pending: u64,
    // DeltaNet
    qkv: u64,
    z: u64,
    b: u64,
    a: u64,
    yr: u64,
    gq: u64, // a prompt chunk's DeltaNet inputs (a window's live in its records)
    gk: u64,
    gv: u64,
    gg: u64,
    gbeta: u64,
    gated: u64,
    gated_xs: u64,
    // attention
    qg: u64,
    key: u64,
    value: u64,
    q_rot: u64,
    attn: u64,
    // MLP
    gate: u64,
    up: u64,
    act: u64,
    act_xs: u64,
    // the head's input, logits and picks
    last_h: u64,
    last_xs: u64,
    logits: u64,
    picks: u64,
    logits32: u64, // the round's logits as fp32, the top-k input
    cand_values: u64, // (rows, max_candidates) fp32 and int64: each row's top candidates
    cand_ids: u64,
    topk_scratch: u64,
    // inputs a call uploads: ids, positions, conv windows, the GDN plan, the attention plan
    ids: u64,
    pos: u64,
    sids: u64, // a batched prompt's rows' streams
    prompt_keys: u64, // a batched prompt's rotated keys and values, before each stream's cache takes its rows
    prompt_values: u64,
    windows: u64,
    plan: u64,
    attn_plan: u64,
    paths: u64,
    depths: u64,
    offs: u64,
    part_o: u64,
    part_m: u64,
    part_l: u64,
    part_chunks: usize,
    origin: u64, // the bf16 origin cache offsets count from
    conv_tmp: u64,
    conv_old: u64, // a round's streams' conv rows that survive its commit, by stream and DeltaNet layer
    copy_items: u64, // two batched-copy tables (kern.Copy), one for each phase of a commit
    taps: u64, // a call's rows of the drafter's taps: (rows, 5 x hidden) bf16
    replay_table: u64,
    replay_rows: u64,
    round: u64, // a shared round's packed int32 inputs (multi_tree_forward's one host copy)
    tables: u64, // (DeltaNet layers, streams) int64: each stream's state of each layer
    conv_cat: u64, // by DeltaNet layer: every stream's conv rows, stacked
    records: []GdnRecord, // by layer: the round's DeltaNet inputs, kept until commit
    attn_records: []AttnRecord, // by layer: the round's rotated keys and values
    gpa: std.mem.Allocator,

    /// Scratch for prompt chunks of `prompt_rows` and windows of `max_rows`, over caches of `capacity` positions.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, g: c.Geometry, prompt_rows: usize, capacity: usize) !Scratch {
        var s: Scratch = undefined;
        s.gpa = gpa;
        s.records = try gpa.alloc(GdnRecord, g.layers);
        errdefer gpa.free(s.records);
        s.attn_records = try gpa.alloc(AttnRecord, g.layers);
        errdefer gpa.free(s.attn_records);
        s.arena = .{ .buf = null };
        try s.layout(g, prompt_rows, capacity);
        s.arena = try Arena.init(d, s.arena.used);
        errdefer s.arena.deinit();
        try s.layout(g, prompt_rows, capacity);
        return s;
    }

    fn layout(s: *Scratch, g: c.Geometry, prompt_rows: usize, capacity: usize) !void {
        const R = @max(prompt_rows, round_rows);
        const W = round_rows;
        const bf = 2;
        const chunks = (capacity + W + tri.chunk - 1) / tri.chunk;
        // attention.plan_host: a query tile's items per committed chunk and folded group
        const codes = capacity / tri.chunk + capacity / (tri.chunk * tri.group_chunks);
        const items = codes * ((W * (g.query_heads / g.kv_heads) + 15) / 16);
        s.rows = R;
        s.part_chunks = chunks;
        s.arena.used = 0;
        const A = &s.arena;
        s.origin = try A.take(128);
        s.x = try A.take(R * g.hidden * bf);
        s.h = try A.take(R * g.hidden * bf);
        s.y = try A.take(R * g.hidden * bf);
        s.r = try A.take(R * g.hidden * bf);
        s.pending = try A.take(R * g.hidden * bf);
        s.xs = try A.take(R * g.hidden / 64 * 4);
        s.qkv = try A.take(R * g.convDim() * bf);
        s.z = try A.take(R * g.vInner() * bf);
        s.b = try A.take(R * g.linear_v_heads * bf);
        s.a = try A.take(R * g.linear_v_heads * bf);
        s.yr = try A.take(R * g.vInner() * bf);
        s.gq = try A.take(R * g.kInner() * bf);
        s.gk = try A.take(R * g.kInner() * bf);
        s.gv = try A.take(R * g.vInner() * bf);
        s.gg = try A.take(R * g.linear_v_heads * 4);
        s.gbeta = try A.take(R * g.linear_v_heads * 4);
        s.gated = try A.take(R * g.inner() * bf);
        s.gated_xs = try A.take(R * g.inner() / 64 * 4);
        s.qg = try A.take(R * 2 * g.qInner() * bf);
        s.key = try A.take(R * g.kvInner() * bf);
        s.value = try A.take(R * g.kvInner() * bf);
        s.q_rot = try A.take(R * g.qInner() * bf);
        s.attn = try A.take(R * g.qInner() * bf);
        s.gate = try A.take(R * g.intermediate * bf);
        s.up = try A.take(R * g.intermediate * bf);
        s.act = try A.take(R * g.intermediate * bf);
        s.act_xs = try A.take(R * g.intermediate / 64 * 4);
        s.last_h = try A.take(W * g.hidden * bf);
        s.last_xs = try A.take(W * g.hidden / 64 * 4);
        s.logits = try A.take(W * c.vocab * bf);
        s.picks = try A.take(W * 4);
        s.logits32 = try A.take(W * c.vocab * 4);
        s.cand_values = try A.take(W * max_candidates * 4);
        s.cand_ids = try A.take(W * max_candidates * 8);
        s.topk_scratch = try A.take(torch_ops.topkScratchBytes(W, c.vocab));
        s.ids = try A.take(R * 4);
        s.pos = try A.take(R * 4);
        s.sids = try A.take(R * 4);
        s.prompt_keys = try A.take(R * g.kvInner() * bf);
        s.prompt_values = try A.take(R * g.kvInner() * bf);
        s.windows = try A.take(R * c.conv_taps * 4);
        s.plan = try A.take(R * 3 * 4);
        s.attn_plan = try A.take((W + 4 + 3 * items + W) * 4);
        s.paths = try A.take(W * tri.max_nodes * 4);
        s.depths = try A.take(W * 4);
        s.offs = try A.take(g.layers * max_streams * 2 * 8);
        s.part_o = try A.take(chunks * W * g.query_heads * c.head_dim * 4);
        s.part_m = try A.take(chunks * W * g.query_heads * 4);
        s.part_l = try A.take(chunks * W * g.query_heads * 4);
        s.conv_tmp = try A.take((c.conv_taps - 1 + R) * g.convDim() * bf);
        s.conv_old = try A.take(max_streams * g.layers * (c.conv_taps - 1) * g.convDim() * bf);
        s.copy_items = try A.take(2 * copy_items * @sizeOf(kern.Copy));
        s.taps = try A.take(R * 5 * g.hidden * bf);
        s.replay_table = try A.take(g.layers * (4 + max_streams) * 8);
        s.replay_rows = try A.take(max_streams * (W + 1) * 4);
        s.round = try A.take((6 * W + max_streams + 1 + W * c.conv_taps + W + 4 * max_streams + 3 * items * max_streams + W) * 4);
        s.tables = try A.take(g.layers * max_streams * 8);
        s.conv_cat = try A.take(g.layers * max_streams * (c.conv_taps - 1) * g.convDim() * bf);
        for (0..g.layers) |i| {
            if (c.linear(i)) {
                s.records[i] = .{
                    .q = try A.take(W * g.kInner() * bf),
                    .k = try A.take(W * g.kInner() * bf),
                    .v = try A.take(W * g.vInner() * bf),
                    .g = try A.take(W * g.linear_v_heads * 4),
                    .beta = try A.take(W * g.linear_v_heads * 4),
                    .qkv = try A.take(W * g.convDim() * bf),
                };
            } else {
                s.attn_records[i] = .{ .k = try A.take(W * g.kvInner() * bf), .v = try A.take(W * g.kvInner() * bf) };
            }
        }
    }

    pub fn deinit(s: *Scratch) void {
        s.arena.deinit();
        s.gpa.free(s.records);
        s.gpa.free(s.attn_records);
        s.* = undefined;
    }
};
