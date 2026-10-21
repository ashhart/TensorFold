//! Flash Next replay internals split along ownership and dispatch boundaries.
const fz = @import("replay.zig");
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const mark_slots = @import("marks.zig");
const lane_w = fz.lane_w;
const ple_scale = fz.ple_scale;
const rows_load = fz.rows_load;
const variants = fz.variants;
const LAYERS = fz.LAYERS;
const VOCAB = fz.VOCAB;
const TRAINED = fz.TRAINED;
const GROUPS = fz.GROUPS;
const MAXR = fz.MAXR;
const Buf = fz.Buf;
const Variant = fz.Variant;
const Run = fz.Run;
const Model = fz.Model;
const SC_WORDS = fz.SC_WORDS;

pub const RING_WORDS = 20;
pub const AR_HLEN = 800;
/// The arena's words, and where its metadata blocks start.
pub const AR_WORDS = 1024;
pub const AR_POS = 4;
pub const AR_NK = 20;
pub const AR_KV = 36;
pub const AR_ABS_POS = 40;
pub const AR_ABS_NK = 56;
pub const AR_ABS_KV = 72;
pub const AR_CHAIN = 76;
pub const AR_CHAIN_STRIDE = 36;

pub const TOP = 512;
pub const KW = 4 * TOP + 3;
/// catchUp's first-block slots: the target's attention layers, then the head.
pub const CATCH = LAYERS / 4 + 1;

/// Slot k's first block into `words` (CATCH slots of 64 words, 256 bytes apart); its byte offset.
pub fn catchSlot(words: []i32, k: usize, first: usize) usize {
    words[k * 64] = @intCast(first);
    return k * 256;
}

test "catch-up slots: every layer's dispatch reads its own first block after the later layers' writes" {
    var words: [CATCH * 64]i32 = undefined;
    var at: [CATCH]usize = undefined;
    for (0..CATCH) |k| at[k] = catchSlot(&words, k, if (k + 1 < CATCH) 648 + k else 0); // pair rank 0: the head pooled none
    for (0..CATCH) |k| try std.testing.expectEqual(@as(i32, if (k + 1 < CATCH) @intCast(648 + k) else 0), words[at[k] / 4]);
}

pub const Select = struct {
    pool: *Variant,
    scores: *Variant,
    select: *Variant,
    start: Buf, // encode's first block, read at run time: one window a command buffer
    starts: Buf, // catchUp's, one slot a layer (CATCH): one command buffer catches up every layer
    sc: Buf,
    keys: Buf,
    complete: Buf,
    ends: Buf,
    counts: Buf,
    sparse: Buf,
    pooled_shape: mtl.Buffer,
    q_shape: mtl.Buffer,
    sc_shape: mtl.Buffer,
    ids_shape: mtl.Buffer,
    nax_scores: ?mtl.Pipeline = null, // prompt rows: the block scores on the tensor units

    pub fn init(r: *Run, rows_max: usize) !?Select {
        var found: [3]?*Variant = .{ null, null, null };
        var it = r.variants.iterator();
        while (it.next()) |kv| {
            const n = kv.key_ptr.*;
            if (std.mem.indexOf(u8, n, "q4_idx_pool_") != null and std.mem.indexOf(u8, n, "_rel") == null) found[0] = kv.value_ptr.*;
            if (std.mem.indexOf(u8, n, "q4_idx_scores_") != null) found[1] = kv.value_ptr.*;
            if (std.mem.indexOf(u8, n, "q4_idx_select_") != null) found[2] = kv.value_ptr.*;
        }
        if (found[0] == null or found[1] == null or found[2] == null) return null;
        const B = struct {
            fn of(rr: *Run, n: usize) !Buf {
                return .{ .b = try rr.buffer(n) };
            }
        };
        return .{
            .pool = found[0].?,
            .scores = found[1].?,
            .select = found[2].?,
            .start = try B.of(r, 16),
            .starts = try B.of(r, CATCH * 256),
            .sc = try B.of(r, @min(rows_max * (r.window.capacity / 4), @max(SC_WORDS, MAXR * (r.window.capacity / 4))) * 4),
            .keys = try B.of(r, rows_max * KW * 4),
            .complete = try B.of(r, rows_max * 4),
            .ends = try B.of(r, rows_max * 4),
            .counts = try B.of(r, rows_max * 4),
            .sparse = try B.of(r, rows_max * 4),
            .pooled_shape = try r.buffer(16),
            .q_shape = try r.buffer(16),
            .sc_shape = try r.buffer(16),
            .ids_shape = try r.buffer(16),
        };
    }

    /// The window's per-row metadata from its first position; true when some row reads selected blocks.
    pub fn meta(s: *Select, pos: usize, rows: usize) bool {
        const cp = s.complete.b.slice(i32, rows);
        const en = s.ends.b.slice(i32, rows);
        const ct = s.counts.b.slice(i32, rows);
        const sp = s.sparse.b.slice(i32, rows);
        var any = false;
        for (0..rows) |i| {
            const e = pos + i + 1;
            const c = e / 4;
            const sparse = c > TOP;
            cp[i], en[i] = .{ @intCast(c), @intCast(e) };
            ct[i] = @intCast(if (sparse) 4 * TOP + e - 4 * c else e);
            sp[i] = @intFromBool(sparse);
            any = any or sparse;
        }
        return any;
    }

    /// Before GPU-side rounds: pool each attention layer's and the head's blocks below `upto`, each from its own slot.
    pub fn catchUp(s: *Select, r: *Run, m: *Model, upto: usize) !void {
        var k: usize = 0;
        for (&m.layers) |*L| if (!L.linear) {
            try s.catchLayer(r, L, m.t.eps, m.t.log2base, upto, k);
            k += 1;
        };
        try s.catchLayer(r, &m.mtp, m.t.eps, m.t.log2base, upto, k);
    }

    /// One layer's catch-up, its first block in slot k (the GPU reads it when it runs the dispatch).
    pub fn catchLayer(s: *Select, r: *Run, L: anytype, eps: Buf, log2base: Buf, upto: usize, k: usize) !void {
        if (upto <= L.pooled_n) return;
        const at = catchSlot(s.starts.b.slice(i32, CATCH * 64), k, L.pooled_n);
        try r.bindV(s.pool, &.{ L.raw, .{ .b = s.starts.b, .off = at }, L.pool, eps, log2base }, &.{.{ .b = L.pooled.b, .off = L.pooled.off + L.pooled_n * 128 * 2 }});
        r.enc.dispatchThreads(mtl.Size.of(128, upto - L.pooled_n, 1), mtl.Size.of(128, 1, 1));
        if (!r.serial) r.enc.barrier();
        L.pooled_n = upto;
    }

    /// Pool the blocks the window completes, score every complete block for each row, select each row's keys.
    pub fn encode(s: *Select, r: *Run, L: anytype, iq: Buf, eps: Buf, log2base: Buf, pos: usize, rows: usize) !void {
        const last = (pos + rows) / 4;
        if (last > L.pooled_n) {
            s.start.b.slice(i32, 1)[0] = @intCast(L.pooled_n); // read at run time: one window a command buffer
            try r.bindV(s.pool, &.{ L.raw, s.start, L.pool, eps, log2base }, &.{.{ .b = L.pooled.b, .off = L.pooled.off + L.pooled_n * 128 * 2 }});
            r.enc.dispatchThreads(mtl.Size.of(128, last - L.pooled_n, 1), mtl.Size.of(128, 1, 1));
            if (!r.serial) r.enc.barrier();
            L.pooled_n = last;
        }
        const nb = L.pooled_n;
        @memcpy(s.pooled_shape.slice(i32, 2), &[2]i32{ @intCast(nb), 128 });
        @memcpy(s.q_shape.slice(i32, 3), &[3]i32{ @intCast(rows), 4, 128 });
        @memcpy(s.sc_shape.slice(i32, 2), &[2]i32{ @intCast(rows), @intCast(nb) });
        @memcpy(s.ids_shape.slice(i32, 2), &[2]i32{ @intCast(rows), KW });
        try r.shapes.put(r.arena, "POOLED_shape", s.pooled_shape);
        try r.shapes.put(r.arena, "Q_shape", s.q_shape);
        try r.shapes.put(r.arena, "SC_shape", s.sc_shape);
        if (s.nax_scores) |pipe| {
            r.enc.setPipeline(pipe);
            r.enc.setBuffer(iq.b, iq.off, 0);
            r.enc.setBuffer(L.pooled.b, L.pooled.off, 1);
            const prm = [2]i32{ @intCast(rows), @intCast(nb) };
            r.enc.setBytes(std.mem.asBytes(&prm), 2);
            r.enc.setBuffer(s.sc.b, s.sc.off, 3);
            r.enc.dispatchThreads(mtl.Size.of(((nb + 63) / 64) * 128, (rows + 31) / 32, 1), mtl.Size.of(128, 1, 1));
        } else {
            try r.bindV(s.scores, &.{ iq, L.pooled, s.complete }, &.{s.sc});
            r.enc.dispatchThreads(mtl.Size.of(((nb + 7) / 8) * 256, (rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
        }
        if (!r.serial) r.enc.barrier();
        try r.bindV(s.select, &.{ s.sc, s.complete, s.ends }, &.{s.keys});
        r.enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
        if (!r.serial) r.enc.barrier();
    }
};

/// Block selection in GPU-side rounds: fz_sel_meta writes each round's rows, pooling range and pooled count; the...
pub const SEL_POOLED = 4 * MAXR + 2;
/// Blocks one round's pool can owe: a window of up to MAXR rows completes at most this many.
pub const POOL_BLOCKS = (MAXR + 3) / 4;

test "a round's pool covers every block its window completes, a copied window's 16 rows too" {
    for (0..64) |tn| for (1..MAXR + 1) |w| try std.testing.expect((tn + w) / 4 - tn / 4 <= POOL_BLOCKS);
    try std.testing.expectEqual(@as(usize, 4), (5 + MAXR) / 4 - 5 / 4);
}

pub const GSelect = struct {
    sel: Buf, // complete[16] ends[16] counts[16] sparse[16] start count pooled
    sc: Buf,
    keys: Buf,
    pooled_shape: mtl.Buffer,
    q_shape: [MAXR + 1]mtl.Buffer, // by window rows
    sc_shape: mtl.Buffer,
    ids_shape: [MAXR + 1]mtl.Buffer,
    nb_ub: usize = 0,

    pub fn init(r: *Run, pooled: usize) !GSelect {
        var g: GSelect = .{
            .sel = .{ .b = try r.buffer(128 * 4) },
            .sc = .{ .b = try r.buffer(MAXR * (r.window.capacity / 4) * 4) },
            .keys = .{ .b = try r.buffer(MAXR * KW * 4) },
            .pooled_shape = try r.buffer(16),
            .q_shape = undefined,
            .sc_shape = try r.buffer(16),
            .ids_shape = undefined,
        };
        g.sel.b.slice(i32, 128)[SEL_POOLED] = @intCast(pooled);
        for (0..MAXR + 1) |w| {
            g.q_shape[w] = (try i32Buf(r, &.{ @intCast(w), 4, 128 })).b;
            g.ids_shape[w] = (try i32Buf(r, &.{ @intCast(w), KW })).b;
        }
        return g;
    }

    pub fn view(g: *GSelect, at: usize) Buf {
        return .{ .b = g.sel.b, .off = at * 4 };
    }
};

/// A pack hyper-connection; `w`: the down and up widths when not 6/32; `rows`: on the per-row kernels (rows_load.zig).
pub const Hc = struct {
    scale: Buf,
    dw: Buf,
    ds: Buf,
    db: Buf,
    uw: Buf,
    us: Buf,
    ub: Buf,
    w: ?[2][2]usize = null,
    rows: ?*rows_load.rows_w.Shared = null,
};
pub const Lane = lane_w.Lane; // the recorded 6/32 pair, or runs at other widths
pub const Layer = struct {
    ahc: Hc,
    mhc: Hc,
    linear: bool,
    proj: Lane,
    out: Lane,
    conv: Buf = undefined,
    alog: Buf = undefined,
    dt: Buf = undefined,
    norm: Buf = undefined,
    qn: Buf = undefined,
    kn: Buf = undefined,
    iqn: Buf = undefined,
    router: Buf,
    ex: [18]Buf, // gate, up, shared gate, shared up (weight, scales, biases), then down, shared down
    xw: ?variants.Experts = null, // the experts' widths when not 6/32 (engine_widths.zig)
    cs: [2]Buf = undefined,
    so: [2]Buf = undefined,
    keys: Buf = undefined,
    vals: Buf = undefined,
    raw: Buf = undefined,
    pool: Buf = undefined,
    pooled: Buf = undefined, // the indexer's pooled block keys [capacity / 4, 128]
    pooled_n: usize = 0,
};

pub const hcOf = variants.hcOf;
pub const laneOf = lane_w.laneOf;

pub const Ple = struct {
    kv: Lane,
    ks: Buf,
    qs: Buf,
    cs: Buf,
    conv: Buf,
    starts: Buf,
    tables: [3 * GROUPS]Buf,
    cin: Buf,
    hist: [2]i64,
    eos: i64,
    mult: [3]i64,
    sizes: [16]i64,
    offsets: [16]i64,
    w: ?[2]usize = null, // the n-gram tables' width when not 6/32
    scale: ?ple_scale.Scale = null, // the n-gram tables' weight_scale when not 1
};

/// Every intermediate a window of up to MAXR rows writes, and the per-window values the kernels read.
pub const Tmp = struct {
    h: [2]Buf,
    ssp: Buf,
    part: Buf,
    mixed: Buf,
    inj_a: Buf,
    inj_m: Buf,
    xs: Buf,
    p: Buf,
    gout: Buf,
    branch: Buf,
    lg: Buf,
    act: Buf,
    pick: Buf,
    wts: Buf,
    ydown: Buf,
    q: Buf,
    kout: Buf,
    iq: Buf,
    po: Buf,
    pm: Buf,
    aout: Buf,
    emb: Buf,
    kvp: Buf,
    gated: Buf,
    hout: Buf,
    logits: Buf,
    picks: Buf,
    rows: Buf,
    mdims: Buf,
    eps: Buf,
    ids8: Buf,
    pos8: Buf,
    nk8: Buf,
    zero8: Buf,
    ids81: Buf,
    scale: Buf,
    log2base: Buf,
    ple_ids: Buf,
    ple_meta: Buf,
    kvmeta: Buf,
    vocab: Buf,
};

/// The rotary's constants every rotary kernel reads: [log2(base), YaRN on, YaRN's cos and sin scale, 32 frequencies].
pub fn ropeBuf(r: *Run, log2base: f32, factor: f64) !Buf {
    const rotary = @import("rotary_config.zig");
    const b = r.rotary_buffer orelse try r.buffer(rotary.words * 4);
    @memcpy(b.slice(f32, rotary.words), &rotary.table(log2base, factor, TRAINED));
    return .{ .b = b };
}

pub const yarn = @import("rotary_config.zig").yarn;

pub fn f32Buf(r: *Run, v: f32) !Buf {
    const b = try r.buffer(4);
    b.slice(f32, 1)[0] = v;
    return .{ .b = b };
}

pub fn i32Buf(r: *Run, vals: []const i32) !Buf {
    const b = try r.buffer(vals.len * 4);
    @memcpy(b.slice(i32, vals.len), vals);
    return .{ .b = b };
}

/// One MTP call's per-row values (every call in a command buffer reads its own).
pub const Slot = struct { rows: Buf, md: Buf, md4: Buf, ids8: Buf, pos8: Buf, nk8: Buf, kvmeta: Buf, n_add: Buf };

/// How many token ids the MTP head scores, from the pack's `mtp.draft_ids` bytes, refused unless whole 64-row tiles...
pub fn draftCount(bytes: usize) !usize {
    const n = bytes / 4;
    if (n == 0 or n % 64 != 0 or n > VOCAB) return error.DraftVocab;
    return n;
}

test "the MTP head takes any draft list of whole 64-row tiles up to the vocabulary" {
    try std.testing.expectEqual(@as(usize, 79_616), try draftCount(79_616 * 4)); // the shipped list, padded to 64
    try std.testing.expectEqual(@as(usize, 135_040), try draftCount(135_040 * 4)); // it and every CJK id
    try std.testing.expectError(error.DraftVocab, draftCount(79_591 * 4));
    try std.testing.expectError(error.DraftVocab, draftCount((VOCAB + 64) * 4));
    try std.testing.expectError(error.DraftVocab, draftCount(0));
}

/// The MTP head: its decoder layer and mixer, the input projections, the cut head, its own attention cache.
pub const Mtp = struct {
    ahc: Hc,
    mhc: Hc,
    mix: Hc,
    proj: Lane,
    out: Lane,
    fce: Lane,
    fch: Lane,
    draft: Lane,
    qn: Buf,
    kn: Buf,
    iqn: Buf,
    enorm: Buf,
    hnorm: Buf,
    router: Buf,
    ids: Buf,
    ids_n: usize,
    ex: [18]Buf,
    xw: ?variants.Experts = null,
    keys: Buf,
    vals: Buf,
    raw: Buf,
    pos: usize = 0,
    drafted: usize = 0,
    h: [2]Buf,
    emb: Buf,
    en: Buf,
    e: Buf,
    hn: Buf,
    hs: Buf,
    logits: Buf,
    pick: Buf,
    md1: Buf,
    n_ids: Buf,
    slots: [2 * MAXR]Slot, // 0: host calls; GPU-side rounds: chained draft j at j, an absorb of n rows at MAXR - 1 + n
    last: Buf = undefined, // the last call's output streams (its kept row)
    pool: Buf = undefined, // its indexer's block pooling (the head's layer is a sparse-attention layer)
    pooled: Buf = undefined,
    pooled_n: usize = 0,
    gsel: ?GSelect = null, // its block selection in GPU-side rounds
};

/// Marks a prompt call can take its DeltaNet states at, their slots, and a layer's index among the DeltaNet layers...
pub const MARKS = mark_slots.MARKS;
pub const Marks = mark_slots.Slots;
pub const linearIndex = mark_slots.linearIndex;
