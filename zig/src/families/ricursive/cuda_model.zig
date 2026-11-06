//! Ripple's forward on CUDA: Metal forward.zig's layer order; prompt chunks of one slot, and rounds of rows over slots.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const glue = @import("cuda_kernels.zig");
const rp = @import("cuda_prep.zig");
const attn = @import("cuda_attn.zig");
const wts = @import("cuda_weights.zig");

const wa = cuda.window_attention;
const Stream = cuda.Stream;
const DeviceBuffer = cuda.DeviceBuffer;
const Config = cfg.Config;
const Weight = cuda.qmmf.Weight;

/// A prompt forward's rows at most (a chunk).
pub const prompt_rows = 2048;
/// A sliding layer's ring: the window's keys before a chunk and the chunk's own (window - 1 + 2,048, to 128s).
pub const ring_rows = 3072;
/// Positions a global layer's page holds.
pub const page_tokens = 512;
/// Rows the head writes at once for a chunk's every-row logits.
pub const head_rows = 128;
/// A one-row round's rows: the round maximum of the CLI and the one-stream backend (M1's one-row chain decode).
pub const decode_rows = 1;
/// Device offsets start on 256 bytes (the row kernels read 16-byte chunks).
const buffer_align = 256;

/// prompt: a chunk by the prompt kernels; decode: a round of one chain by the lane matmul and the decode attention.
pub const Mode = enum { prompt, decode };

/// One stream's rows in a round: `tokens` consecutive positions from p0 in its slot.
pub const Chain = struct { slot: usize, p0: usize, tokens: usize };

pub const Kernels = @import("cuda_model_kernels.zig").Kernels;

/// The forward's scratch for `prompt_rows` rows (a round's at most), each at its own 256-byte boundary.
const Bufs = struct {
    resid: u64 = 0, // bf16 [R, H]: the residual
    v0: u64 = 0, // bf16 [R, H]: the value residual's source (the scaled embedding)
    x: u64 = 0, // bf16 [R, H]: the next norm's rows
    vx: u64 = 0, // bf16 [R, H]: lerp(v0, x, blend), v's input
    y: u64 = 0, // bf16 [R, H]: o's and down's output
    qkg: u64 = 0, // bf16 [R, q + kv + q]: the fused projection
    v: u64 = 0, // bf16 [R, kv]
    q: u64 = 0, // bf16 [R, heads, 128]: prep's normed, rotated queries
    att: u64 = 0, // bf16 [R, q]: the attention, then gated in place
    g: u64 = 0, // bf16 [R, q]: the gate's rows contiguous (several rows)
    gu: u64 = 0, // bf16 [R, 2I]: gate then up a row, as tf_rc_silu_clamp reads them
    act: u64 = 0, // bf16 [R, I]: the planar gate, then silu_clamp's output
    up: u64 = 0, // bf16 [R, I]: the planar up
    ids: u64 = 0, // u32 [R]
    pos: u64 = 0, // int32 [round]: a round's rows' positions
    slot: u64 = 0, // int32 [round]: a round's rows' slots
    inv: u64 = 0, // f32 [64]: RoPE's inverse frequencies
    wscr: u64 = 0, // the sliding decode's partials (16 bytes at head dim 128)
    part: u64 = 0, // the lane matmul's slice partials at a round's rows
    logits: u64 = 0, // bf16 [V]: the last row's
    rows: u64 = 0, // bf16 [max(head_rows, round), V]: a chunk's or a round's rows
    ctab: u64 = 0, // int32 [round, pages a slot]: a round's call table, row r row r's slot's page ids
};

/// The scratch for rounds of at most `round` rows over slots of `per_slot` pages.
fn bufSizes(c: Config, part: usize, round: usize, per_slot: usize) Bufs {
    const r = prompt_rows;
    const h = c.hidden;
    const q = c.queryWidth();
    const f = c.intermediate;
    return .{
        .resid = r * h * 2,
        .v0 = r * h * 2,
        .x = r * h * 2,
        .vx = r * h * 2,
        .y = r * h * 2,
        .qkg = r * wts.qkgWidth(c) * 2,
        .v = r * c.kvWidth() * 2,
        .q = r * q * 2,
        .att = r * q * 2,
        .g = r * q * 2,
        .gu = r * 2 * f * 2,
        .act = r * f * 2,
        .up = r * f * 2,
        .ids = r * 4,
        .pos = round * 4,
        .slot = round * 4,
        .inv = cfg.head_dim / 2 * 4,
        .wscr = wa.WindowAttention.scratchBytes(slidingWindow(c), round),
        .part = part,
        .logits = c.vocab * 2,
        .rows = @max(head_rows, round) * c.vocab * 2,
        .ctab = round * per_slot * 4,
    };
}

/// Each buffer's offset from `base` in field order; the bytes they span.
fn place(sizes: Bufs, base: u64, out: *Bufs) usize {
    var at: usize = 0;
    inline for (@typeInfo(Bufs).@"struct".field_names) |name| {
        @field(out, name) = base + at;
        at = std.mem.alignForward(usize, at + @field(sizes, name), buffer_align);
    }
    return at;
}

fn slidingWindow(c: Config) wa.Window {
    return attn.sliding(c.heads, c.kv_heads, c.window, ring_rows, c.scale());
}

/// The pages a slot's global layer holds: the context in whole pages.
pub fn pagesOf(context: usize) usize {
    return (context + page_tokens - 1) / page_tokens;
}

/// A layer's caches from slot 0: a sliding layer's rings [slots][ring][kv_heads][128], or (`pg`) a global's pages.
pub const Cache = struct { kc: u64, vc: u64, pg: ?wa.Pages = null };

/// One slot's rows of a layer as a snapshot copies them: position p at ring row p % ring_rows, or global row p.
pub const Span = struct { k: u64, v: u64, rows: usize, row_bytes: usize, ring: bool };

/// One slot's caches: a sliding layer's K and V rings, a global layer's K and V pages for `context`.
pub fn slotBytes(c: Config, context: usize) usize {
    const kvw = c.kvWidth();
    const g = c.globals();
    return (c.layers - g) * 2 * ring_rows * kvw * 2 + g * 2 * pagesOf(context) * page_tokens * kvw * 2;
}

/// Every slot's caches in one buffer, a layer at a time (each slot's rows, K then V), then the page table.
pub const Arena = struct {
    gpa: std.mem.Allocator,
    kv: DeviceBuffer,
    caches: []Cache, // a layer each, at slot 0
    ids: []i32, // the page table's host copy [slots][per_slot]: the call tables gather its rows
    slots: usize,
    per_slot: usize, // pages a slot's global layer holds
    kvw: usize,

    /// `slots` slots' caches for `context` positions and the int32 table (one slot: M1's State.bytes).
    pub fn bytes(c: Config, context: usize, slots: usize) usize {
        return slots * slotBytes(c, context) + std.mem.alignForward(usize, slots * pagesOf(context) * 4, buffer_align);
    }

    /// Zeroed caches (no stale bits under a masked key), the identity table: slot s's page j is page s * per_slot + j.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: Stream, c: Config, context: usize, slots: usize) !Arena {
        const kvw = c.kvWidth();
        const per = pagesOf(context);
        var kv = try DeviceBuffer.alloc(d, bytes(c, context, slots));
        errdefer kv.free();
        try d.check(d.api.cuMemsetD8Async(kv.ptr, 0, kv.len, s.handle), "cuMemsetD8Async");
        const caches = try gpa.alloc(Cache, c.layers);
        errdefer gpa.free(caches);
        const ids = try gpa.alloc(i32, slots * per);
        errdefer gpa.free(ids);
        for (ids, 0..) |*x, i| x.* = @intCast(i);
        const table_at = kv.len - std.mem.alignForward(usize, ids.len * 4, buffer_align);
        try cuda.copy_source.enqueue(d, kv.ptr + table_at, std.mem.sliceAsBytes(ids), s);
        var at: u64 = kv.ptr;
        for (caches, 0..) |*cache, i| {
            const half: usize = slots * if (c.global[i]) per * page_tokens * kvw * 2 else ring_rows * kvw * 2;
            cache.* = .{ .kc = at, .vc = at + half };
            if (c.global[i]) cache.pg = .{ .table = kv.ptr + table_at, .page_elems = page_tokens * kvw, .tokens = page_tokens, .stride = per };
            at += 2 * half;
        }
        return .{ .gpa = gpa, .kv = kv, .caches = caches, .ids = ids, .slots = slots, .per_slot = per, .kvw = kvw };
    }

    pub fn deinit(a: *Arena) void {
        a.gpa.free(a.ids);
        a.gpa.free(a.caches);
        a.kv.free();
        a.* = undefined;
    }

    /// Layer i's caches of one slot as a prompt chunk reads them: its rings, or the pools and its table row (stride 0).
    pub fn slotCache(a: *const Arena, i: usize, slot: usize) Cache {
        const c = a.caches[i];
        if (c.pg) |pg| {
            var row = pg;
            row.table += slot * a.per_slot * 4;
            row.stride = 0;
            return .{ .kc = c.kc, .vc = c.vc, .pg = row };
        }
        const ring = ring_rows * a.kvw * 2;
        return .{ .kc = c.kc + slot * ring, .vc = c.vc + slot * ring };
    }

    /// Layer i's rows of one slot.
    pub fn span(a: *const Arena, i: usize, slot: usize) Span {
        const c = a.caches[i];
        const row = a.kvw * 2;
        const rows: usize = if (c.pg != null) a.per_slot * page_tokens else ring_rows;
        return .{ .k = c.kc + slot * rows * row, .v = c.vc + slot * rows * row, .rows = rows, .row_bytes = row, .ring = c.pg == null };
    }

    /// The V pools' offset from the K pools in bf16 elements: the same in every global layer.
    fn vOff(a: *const Arena) usize {
        for (a.caches) |c| if (c.pg != null) return (c.vc - c.kc) / 2;
        return 0;
    }
};

/// A round's rows on the host: positions and slots (int32 for the kernels, usize for the plan), the call table.
const Host = struct { pos: []i32, slot: []i32, at: []usize, ctab: []i32 };

/// Checks a round's chains against `ids` and fills the host rows: chain j's rows in order, each slot at most once.
fn fill(h: Host, table: []const i32, per_slot: usize, slots: usize, context: usize, chains: []const Chain, n: usize) !void {
    if (n == 0 or n > h.pos.len) return error.ForwardRows;
    var r: usize = 0;
    for (chains, 0..) |ch, j| {
        if (ch.slot >= slots) return error.NoSuchSlot;
        if (ch.tokens == 0 or ch.tokens > n - r) return error.ForwardRows;
        if (ch.p0 > context or ch.tokens > context - ch.p0) return error.PastContext;
        for (chains[0..j]) |o| if (o.slot == ch.slot) return error.SlotTwice; // its rows would race in the caches
        for (r..r + ch.tokens, ch.p0..) |row, p| {
            h.pos[row] = @intCast(p);
            h.slot[row] = @intCast(ch.slot);
            h.at[row] = p;
            @memcpy(h.ctab[row * per_slot ..][0..per_slot], table[ch.slot * per_slot ..][0..per_slot]);
        }
        r += ch.tokens;
    }
    if (r != n) return error.ForwardRows;
}

/// One slot's stream as the CLI and the one-stream backend drive it: its slot and the positions its caches hold.
pub const Seq = struct { slot: usize = 0, len: usize = 0 };

/// A forward's rows: a prompt chunk of one chain, or a round (its global layers' plan, null without global layers).
const Pass = union(enum) { chunk: Chain, round: ?attn.Planned };

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: Stream,
    w: *const wts.Weights,
    k: *const Kernels,
    c: Config,
    context: usize,
    /// A round's rows at most (a caller's choice, for example 16 rows for each of its streams; at most prompt_rows).
    round_rows: usize,
    scratch: DeviceBuffer,
    b: Bufs = .{},
    gd: attn.GlobalDecode,
    arena: Arena,
    host: Host,
    /// Rounds replay CUDA graphs (round's `.graph`); false: eager launches with the round's own key groups (B1's).
    graphs: bool = true,
    /// Captured rounds by row count ([n], n to round_rows), each the layers' launches over the staged rows.
    execs: []?cuda.graph.Exec,

    /// The lane's slice partials of every projection and the head at up to `rows` rows (they reduce below fused_rows).
    fn partBytes(w: *const wts.Weights, k: *const Kernels, rows: usize) usize {
        var most: usize = 0;
        for ([_]usize{ rows, @min(rows, cuda.qmmf.fused_rows - 1) }) |m| {
            most = @max(most, k.lane.partBytes(m, w.head));
            for (w.layers) |l| for ([_]Weight{ l.qkg, l.v, l.o, l.gate, l.up, l.down }) |x| {
                most = @max(most, k.lane.partBytes(m, x));
            };
        }
        return most;
    }

    /// The device bytes init allocates beside the weights: scratch, `slots` slots' caches, global decode for `round`.
    pub fn openBytes(w: *const wts.Weights, k: *const Kernels, context: usize, slots: usize, round_rows: usize) usize {
        const c = w.config;
        var b: Bufs = .{};
        const scratch = place(bufSizes(c, partBytes(w, k, round_rows), round_rows, pagesOf(context)), 0, &b);
        return scratch + Arena.bytes(c, context, slots) + attn.GlobalDecode.bytes(round_rows, context);
    }

    /// Scratch for `round_rows`-row rounds and `slots` slots of `context` positions; refuses a shape no ring holds.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: Stream, w: *const wts.Weights, k: *const Kernels, context: usize, slots: usize, round_rows: usize) !Model {
        const c = w.config;
        if (c.window == 0 or c.window - 1 + prompt_rows > ring_rows) return error.UnsupportedRicursiveCudaShape;
        if (context == 0 or context > cfg.max_context) return error.PastContext;
        // a round's rows fit the chunk's scratch (and a chain's rows never lap the ring); page ids are int32
        if (round_rows == 0 or round_rows > prompt_rows or round_rows > context or slots == 0 or slots * pagesOf(context) > std.math.maxInt(i32)) return error.ForwardRows;
        const sizes = bufSizes(c, partBytes(w, k, round_rows), round_rows, pagesOf(context));
        var b: Bufs = .{};
        var scratch = try DeviceBuffer.alloc(d, place(sizes, 0, &b));
        errdefer scratch.free();
        _ = place(sizes, scratch.ptr, &b);
        var inv: [cfg.head_dim / 2]f32 = undefined;
        c.inverseFrequencies(&inv);
        try cuda.copy_source.enqueue(d, b.inv, std.mem.sliceAsBytes(&inv), s);
        var gd = try attn.GlobalDecode.init(d, gpa, round_rows, context);
        errdefer gd.deinit();
        var arena = try Arena.init(gpa, d, s, c, context, slots);
        errdefer arena.deinit();
        try gd.stage(s, arena.vOff());
        const execs = try gpa.alloc(?cuda.graph.Exec, round_rows + 1);
        errdefer gpa.free(execs);
        @memset(execs, null);
        const h: Host = .{
            .pos = try gpa.alloc(i32, round_rows),
            .slot = try gpa.alloc(i32, round_rows),
            .at = try gpa.alloc(usize, round_rows),
            .ctab = try gpa.alloc(i32, round_rows * arena.per_slot),
        };
        return .{ .gpa = gpa, .d = d, .s = s, .w = w, .k = k, .c = c, .context = context, .round_rows = round_rows, .scratch = scratch, .b = b, .gd = gd, .arena = arena, .host = h, .execs = execs };
    }

    pub fn deinit(m: *Model) void {
        m.dropGraphs();
        m.gpa.free(m.execs);
        m.gpa.free(m.host.ctab);
        m.gpa.free(m.host.at);
        m.gpa.free(m.host.slot);
        m.gpa.free(m.host.pos);
        m.arena.deinit();
        m.gd.deinit();
        m.scratch.free();
        m.* = undefined;
    }

    /// Zeroes a slot's positions [0, written) (a ring past its rows whole): a fresh arena's bytes, for another stream.
    pub fn reset(m: *const Model, slot: usize, written: usize) !void {
        if (slot >= m.arena.slots) return error.NoSuchSlot;
        for (0..m.c.layers) |i| {
            const sp = m.arena.span(i, slot);
            const n = @min(written, sp.rows) * sp.row_bytes;
            for ([_]u64{ sp.k, sp.v }) |x| try m.d.check(m.d.api.cuMemsetD8Async(x, 0, n, m.s.handle), "cuMemsetD8Async");
        }
    }

    fn linear(m: *const Model, x: u64, ldx: usize, n: usize, w: Weight, y: u64, mode: Mode) !void {
        if (mode == .decode) return m.k.lane.matmulOut(m.s, x, ldx, n, w, y, .bf16, m.b.part);
        return m.k.prompt.matmulOut(m.s, x, ldx, n, w, y, .bf16);
    }

    fn copy2D(m: *const Model, src: u64, src_pitch: usize, dst: u64, dst_pitch: usize, width: usize, height: usize) !void {
        const p: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = src_pitch, .dst = dst, .dst_pitch = dst_pitch, .width = width, .height = height };
        try m.d.check(m.d.api.cuMemcpy2DAsync_v2(&p, m.s.handle), "cuMemcpy2DAsync");
    }

    /// Gate and up into gu's rows (gate at column 0, up at column I): one row straight in, else planar then copied.
    fn mlpIn(m: *const Model, l: *const wts.Layer, n: usize, mode: Mode) !void {
        const h = m.c.hidden;
        const f2 = m.c.intermediate * 2;
        const b = m.b;
        if (n == 1) {
            try m.linear(b.x, h, 1, l.gate, b.gu, mode);
            return m.linear(b.x, h, 1, l.up, b.gu + f2, mode);
        }
        try m.linear(b.x, h, n, l.gate, b.act, mode);
        try m.linear(b.x, h, n, l.up, b.up, mode);
        try m.copy2D(b.act, f2, b.gu, 2 * f2, f2, n);
        try m.copy2D(b.up, f2, b.gu + f2, 2 * f2, f2, n);
    }

    /// Layer i's q, k, v prep and attention: a chunk's prep and prompt kernels, or a round's prep_rows and decode.
    fn attention(m: *const Model, l: *const wts.Layer, i: usize, n: usize, pass: Pass) !void {
        const c = m.c;
        const b = m.b;
        const k = m.k;
        const qw = c.queryWidth();
        const kvw = c.kvWidth();
        const wq = wts.qkgWidth(c);
        const scale = c.scale();
        const xq = rp.Rows.of(b.qkg, wq);
        const xk = rp.Rows.of(b.qkg + qw * 2, wq);
        const xv = rp.Rows.of(b.v, kvw);
        switch (pass) {
            .chunk => |ch| {
                const cache = m.arena.slotCache(i, ch.slot);
                try rp.prep(&k.prep, m.s, .{ .xq = xq, .xk = xk, .xv = xv, .inv = b.inv, .qw = l.q_norm, .kw = l.k_norm, .q = b.q, .kc = cache.kc, .vc = cache.vc, .rows = n, .heads = c.heads, .kv_heads = c.kv_heads, .pos = ch.p0, .ring = if (cache.pg == null) ring_rows else 0, .rope = !c.global[i], .eps = c.eps, .pg = cache.pg });
                const w = if (cache.pg) |pg| attn.global(c.heads, c.kv_heads, scale, pg) else slidingWindow(c);
                return attn.prompt(&k.attn, m.s, b.q, cache.kc, cache.vc, b.att, ch.p0, n, w);
            },
            .round => |planned| {
                const cache = m.arena.caches[i];
                try rp.prepRows(&k.prep, m.s, .{ .xq = xq, .xk = xk, .xv = xv, .inv = b.inv, .qw = l.q_norm, .kw = l.k_norm, .q = b.q, .kc = cache.kc, .vc = cache.vc, .rows = n, .heads = c.heads, .kv_heads = c.kv_heads, .pos = b.pos, .slot = b.slot, .ring = if (cache.pg == null) ring_rows else 0, .rope = !c.global[i], .eps = c.eps, .pg = cache.pg });
                if (cache.pg) |pg| {
                    var call = pg; // tree_attention reads its streams' rows: the round's gathered table
                    call.table = b.ctab;
                    return attn.runRows(&m.gd, m.s, planned.?, b.q, cache.kc, b.att, c.heads, call, scale);
                }
                return k.win.decode(m.s, b.q, cache.kc, cache.vc, b.att, b.pos, b.slot, slidingWindow(c), b.wscr, n);
            },
        }
    }

    /// The forward's n rows (ids uploaded): every layer, then the final norm into x (the heads read it).
    fn layers(m: *const Model, n: usize, pass: Pass) !void {
        const c = m.c;
        const s = m.s;
        const b = m.b;
        const k = m.k;
        const h = c.hidden;
        const qw = c.queryWidth();
        const kvw = c.kvWidth();
        const wq = wts.qkgWidth(c);
        const mode: Mode = if (pass == .chunk) .prompt else .decode;
        const ls = m.w.layers;
        try glue.embed(&k.glue, s, b.ids, m.w.embed, b.resid, b.v0, n, h, c.embed_scale);
        try glue.addNorm(&k.glue, s, .{ .resid = b.resid, .next = ls[0].input_norm, .v0 = b.v0, .x = b.x, .vx = b.vx, .rows = n, .width = h, .eps = c.eps, .blend = ls[0].blend });
        for (ls, 0..) |*l, i| {
            try m.linear(b.x, h, n, l.qkg, b.qkg, mode);
            try m.linear(b.vx, h, n, l.v, b.v, mode);
            try m.attention(l, i, n, pass);
            const g_at = b.qkg + (qw + kvw) * 2;
            const g = if (n == 1) g_at else blk: { // tf_rc_gate reads G rows contiguous
                try m.copy2D(g_at, wq * 2, b.g, qw * 2, qw * 2, n);
                break :blk b.g;
            };
            try glue.gate(&k.glue, s, b.att, g, n, qw);
            try m.linear(b.att, qw, n, l.o, b.y, mode);
            try glue.addNorm(&k.glue, s, .{ .resid = b.resid, .branch = b.y, .post = l.post_attn, .next = l.pre_ffn, .x = b.x, .rows = n, .width = h, .eps = c.eps });
            try m.mlpIn(l, n, mode);
            try glue.siluClamp(&k.glue, s, b.gu, b.act, n, c.intermediate, c.clamp);
            try m.linear(b.act, c.intermediate, n, l.down, b.y, mode);
            if (i + 1 < ls.len) {
                const next = &ls[i + 1];
                try glue.addNorm(&k.glue, s, .{ .resid = b.resid, .branch = b.y, .post = l.post_ffn, .next = next.input_norm, .v0 = b.v0, .x = b.x, .vx = b.vx, .rows = n, .width = h, .eps = c.eps, .blend = next.blend });
            } else try glue.addNorm(&k.glue, s, .{ .resid = b.resid, .branch = b.y, .post = l.post_ffn, .next = m.w.norm, .x = b.x, .rows = n, .width = h, .eps = c.eps });
        }
    }

    /// A prompt chunk of one slot: `ids` at p0, p0 + 1, ... (at most prompt_rows), M1's prompt path and bits.
    pub fn prompt(m: *const Model, slot: usize, p0: usize, ids: []const u32) !void {
        const n = ids.len;
        if (n == 0 or n > prompt_rows) return error.ForwardRows;
        if (slot >= m.arena.slots) return error.NoSuchSlot;
        if (p0 > m.context or n > m.context - p0) return error.PastContext;
        try cuda.copy_source.enqueue(m.d, m.b.ids, std.mem.sliceAsBytes(ids), m.s);
        try m.layers(n, .{ .chunk = .{ .slot = slot, .p0 = p0, .tokens = n } });
    }

    /// One round: chain j's `tokens` rows at p0, p0 + 1, ... of its slot, each seeing its slot's keys to its position.
    pub fn round(m: *const Model, chains: []const Chain, ids: []const u32) !void {
        return m.roundAs(chains, ids, if (m.graphs) .graph else .exact);
    }

    /// Launches: graph (captured, the context's key groups), exact (eager, own key groups), fixed (eager, graph's).
    pub const Run = enum { graph, exact, fixed };

    /// A round's rows launched as `run` says; ids, positions, slots, table rows and committed keys upload first.
    pub fn roundAs(m: *const Model, chains: []const Chain, ids: []const u32, run: Run) !void {
        const n = ids.len;
        const h = m.host;
        try fill(h, m.arena.ids, m.arena.per_slot, m.arena.slots, m.context, chains, n);
        const b = m.b;
        try cuda.copy_source.enqueue(m.d, b.ids, std.mem.sliceAsBytes(ids), m.s);
        try cuda.copy_source.enqueue(m.d, b.pos, std.mem.sliceAsBytes(h.pos[0..n]), m.s);
        try cuda.copy_source.enqueue(m.d, b.slot, std.mem.sliceAsBytes(h.slot[0..n]), m.s);
        const planned: ?attn.Planned = if (m.c.globals() == 0) null else blk: {
            try cuda.copy_source.enqueue(m.d, b.ctab, std.mem.sliceAsBytes(h.ctab[0 .. n * m.arena.per_slot]), m.s);
            break :blk try attn.planRows(&m.gd, m.s, h.at[0..n], run != .exact);
        };
        if (run != .graph) return m.layers(n, .{ .round = planned });
        const g = &m.execs[n];
        if (g.* == null) g.* = try m.capture(n, planned);
        try g.*.?.launchOn(m.s);
    }

    /// The layers' launches for a round of n rows as a graph: they read every per-round value from the staged buffers.
    fn capture(m: *const Model, n: usize, planned: ?attn.Planned) !cuda.graph.Exec {
        try m.s.synchronize();
        try cuda.graph.beginCapture(m.s, .thread_local);
        m.layers(n, .{ .round = planned }) catch |e| {
            if (cuda.graph.endCapture(m.s)) |g| {
                var gg = g;
                gg.deinit();
            } else |_| {}
            return e;
        };
        var g = try cuda.graph.endCapture(m.s);
        defer g.deinit();
        return g.instantiate();
    }

    /// Every captured graph dropped (the next round of each row count captures again).
    pub fn dropGraphs(m: *const Model) void {
        for (m.execs) |*g| if (g.*) |*e| {
            e.deinit();
            g.* = null;
        };
    }

    /// The row counts whose rounds have a graph.
    pub fn captured(m: *const Model) usize {
        var n: usize = 0;
        for (m.execs) |g| n += @intFromBool(g != null);
        return n;
    }

    /// `ids` at seq.len, seq.len + 1, ...: a prompt chunk, or (decode) a round of one chain; seq.len moves past them.
    pub fn forward(m: *const Model, seq: *Seq, ids: []const u32, mode: Mode) !void {
        switch (mode) {
            .prompt => try m.prompt(seq.slot, seq.len, ids),
            .decode => try m.round(&.{.{ .slot = seq.slot, .p0 = seq.len, .tokens = ids.len }}, ids),
        }
        seq.len += ids.len;
    }

    /// The last forward's row `row` through the bf16 head (the lane matmul) into `logits` [V].
    pub fn head(m: *const Model, row: usize) !void {
        const h = m.c.hidden;
        try m.k.lane.matmulOut(m.s, m.b.x + row * h * 2, h, 1, m.w.head, m.b.logits, .bf16, m.b.part);
    }

    /// The last forward's rows [r0, r0 + rows) (at most head_rows) through the head (the prompt matmul) into `rows`.
    pub fn headRows(m: *const Model, r0: usize, rows: usize) !void {
        if (rows == 0 or rows > head_rows) return error.ForwardRows;
        const h = m.c.hidden;
        try m.k.prompt.matmulOut(m.s, m.b.x + r0 * h * 2, h, rows, m.w.head, m.b.rows, .bf16);
    }

    /// The last round's rows [0, rows) through the head (the lane matmul, as head's) into `rows` [rows, V].
    pub fn rowLogits(m: *const Model, rows: usize) !void {
        if (rows == 0 or rows > m.round_rows) return error.ForwardRows;
        try m.k.lane.matmulOut(m.s, m.b.x, m.c.hidden, rows, m.w.head, m.b.rows, .bf16, m.b.part);
    }

    /// The greedy draw of `logits` into `out` (int32).
    pub fn draw(m: *const Model, out: u64) !void {
        try m.k.argmaxRows(m.s, m.b.logits, 1, m.c.vocab, m.c.vocab, out);
    }

    /// The greedy draws of rowLogits' rows into `out` (int32 [rows]).
    pub fn drawRows(m: *const Model, rows: usize, out: u64) !void {
        try m.k.argmaxRows(m.s, m.b.rows, rows, m.c.vocab, m.c.vocab, out);
    }

    /// Device bytes at `ptr` into `out`, after the stream's work.
    pub fn download(m: *const Model, ptr: u64, out: []u8) !void {
        try m.s.synchronize();
        try m.d.check(m.d.api.cuMemcpyDtoH_v2(out.ptr, ptr, out.len), "cuMemcpyDtoH");
    }
};

const testing = std.testing;

fn release() !Config {
    var c = try cfg.parse(testing.allocator, @embedFile("fixtures/config.json"));
    c.fp4_mlp = true;
    return c;
}

test "ricursive cuda caches: 64 rings of 3,072 rows, 16 layers' pages for the context; slots side by side" {
    const c = try release();
    try testing.expectEqual(@as(usize, 16), c.globals());
    const ring = 64 * 2 * 3072 * 1024 * 2;
    try testing.expectEqual(ring + 16 * 2 * 64 * 512 * 1024 * 2 + 256, Arena.bytes(c, 32768, 1)); // M1's State.bytes
    try testing.expectEqual(ring + 16 * 2 * 9 * 512 * 1024 * 2 + 256, Arena.bytes(c, 4096 + 1, 1)); // a part page is a page
    try testing.expectEqual(4 * slotBytes(c, 32768) + 4 * 64 * 4, Arena.bytes(c, 32768, 4)); // the table: 4 rows of 64 ids
    try testing.expect(c.window - 1 + prompt_rows <= ring_rows);
}

test "ricursive cuda scratch: every buffer on 256 bytes, about 1 GiB for 2,048 rows" {
    const c = try release();
    var b: Bufs = .{};
    const total = place(bufSizes(c, 1 << 20, 1, 64), 4096, &b);
    inline for (@typeInfo(Bufs).@"struct".field_names) |name| try testing.expect(@field(b, name) % buffer_align == 0);
    try testing.expectEqual(@as(u64, 4096), b.resid);
    try testing.expect(b.rows > b.logits and b.logits > b.part);
    try testing.expect(total > 1_000_000_000 and total < 1_200_000_000);
    var wide: Bufs = .{};
    const sizes = bufSizes(c, 0, 256, 64);
    try testing.expectEqual(@as(usize, 256 * c.vocab * 2), sizes.rows); // a round past head_rows: its rows' logits
    try testing.expectEqual(@as(usize, 256 * 64 * 4), sizes.ctab);
    _ = place(sizes, 0, &wide);
}

test "ricursive cuda round rows: chains in order, each row its slot's table row; a slot twice or a short round refused" {
    const per = 3;
    var table: [4 * per]i32 = undefined;
    for (&table, 0..) |*x, i| x.* = @intCast(i);
    var pos: [8]i32 = undefined;
    var slot: [8]i32 = undefined;
    var at: [8]usize = undefined;
    var ctab: [8 * per]i32 = undefined;
    const h: Host = .{ .pos = &pos, .slot = &slot, .at = &at, .ctab = &ctab };
    try fill(h, &table, per, 4, 1536, &.{ .{ .slot = 2, .p0 = 510, .tokens = 3 }, .{ .slot = 0, .p0 = 7, .tokens = 1 } }, 4);
    try testing.expectEqualSlices(i32, &.{ 510, 511, 512, 7 }, pos[0..4]);
    try testing.expectEqualSlices(i32, &.{ 2, 2, 2, 0 }, slot[0..4]);
    try testing.expectEqualSlices(usize, &.{ 510, 511, 512, 7 }, at[0..4]);
    try testing.expectEqualSlices(i32, &.{ 6, 7, 8, 6, 7, 8, 6, 7, 8, 0, 1, 2 }, ctab[0 .. 4 * per]);
    try testing.expectError(error.SlotTwice, fill(h, &table, per, 4, 1536, &.{ .{ .slot = 1, .p0 = 0, .tokens = 1 }, .{ .slot = 1, .p0 = 1, .tokens = 1 } }, 2));
    try testing.expectError(error.ForwardRows, fill(h, &table, per, 4, 1536, &.{.{ .slot = 1, .p0 = 0, .tokens = 2 }}, 3));
    try testing.expectError(error.ForwardRows, fill(h, &table, per, 4, 1536, &.{.{ .slot = 1, .p0 = 0, .tokens = 9 }}, 9));
    try testing.expectError(error.NoSuchSlot, fill(h, &table, per, 4, 1536, &.{.{ .slot = 4, .p0 = 0, .tokens = 1 }}, 1));
    try testing.expectError(error.PastContext, fill(h, &table, per, 4, 1536, &.{.{ .slot = 3, .p0 = 1535, .tokens = 2 }}, 2));
}

test "ricursive cuda forward type-checks against the kernels' wrappers" {
    testing.refAllDecls(Model);
    testing.refAllDecls(Kernels);
    testing.refAllDecls(Arena);
}
