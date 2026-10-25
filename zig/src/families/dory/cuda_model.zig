//! Dory's forward on CUDA over cache slots: per_loop KV caches a stream, prompt chunks and rounds of decode rows.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const schedule = @import("schedule.zig");
const memory = @import("memory.zig");
const wts = @import("cuda_weights.zig");
const pages = @import("cuda_pages.zig");

const Config = config.Config;
const DeviceBuffer = cuda.DeviceBuffer;
const WindowAttention = cuda.window_attention.WindowAttention;
pub const Kernels = @import("cuda_kernels.zig").Kernels;

pub const eps: f32 = 1e-6;
/// Prompt rows a forward takes at most: one stream's chunk.
pub const prompt_rows = 2048;
/// Decode rows a forward takes over every stream's window.
pub const round_rows = 128;
/// The widest verify window of one stream (one-row steps and every wider window give a row the same bits).
pub const window_rows = 16;
/// Decode rows an attention call takes: its partials' scratch is sized for this many.
const attend_rows = 16;
const key_block = 64;

/// Keys a sliding cache keeps: a chunk's rows written before its first row reads back its window.
pub fn ringSize(rows: usize, window: usize) usize {
    return (rows + window + key_block - 1) / key_block * key_block;
}

/// A stream's rows in one forward: its cache slot, first position, tokens.
pub const Chain = struct { slot: usize, p0: usize, tokens: []const u32 };

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *const wts.Weights,
    c: Config,
    sched: schedule.Schedule,
    context: usize,
    ring: usize,
    slots: usize,
    k: []u64 = &.{}, // per cache: [slots, len, HK, D] bf16 (len: context for full caches, the ring for sliding ones)
    v: []u64 = &.{}, // paged: a full cache's K and V regions of the arena, page id p at rows [p * tokens, (p + 1) * tokens)
    paged: bool = false, // full caches live in pages (attach makes them); sliding rings stay a buffer a slot
    paging: ?pages.Paging = null,
    rings: std.ArrayList(DeviceBuffer) = .empty, // the sliding caches' buffers, made again for the slots attach names
    inv: [2]u64 = .{ 0, 0 }, // fp32 [D / 2] inverse frequencies: sliding, full
    s: Scratch = undefined,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    planned: ?usize = null, // a plan: the bytes counted, nothing allocated
    /// Decode forwards of n rows as CUDA graphs at this model's loop count, captured at first use ([n]).
    graphs: [round_rows + 1]?cuda.graph.Exec = @splat(null),
    /// Paged: decode forwards of n rows whose streams all hold a window of pages (the contiguous kernels).
    flat_graphs: [round_rows + 1]?cuda.graph.Exec = @splat(null),
    graphed: bool = true,

    const Scratch = struct { ids: u64, pos: u64, slot: u64, fslot: u64, h: u64, h2: u64, state: u64, input: u64, out: u64, q: u64, kk: u64, vv: u64, g: u64, qn: u64, kn: u64, vn: u64, att: u64, gated: u64, mg: u64, mu: u64, m: u64, part: u64, win: u64, logits: u64 };

    fn alloc(m: *Model, bytes: usize) !u64 {
        if (m.planned) |*p| {
            p.* += @max(bytes, 256);
            return 0;
        }
        const b = try DeviceBuffer.alloc(m.d, @max(bytes, 256));
        errdefer {
            var x = b;
            x.free();
        }
        try m.buffers.append(m.gpa, b);
        return b.ptr;
    }

    fn zeros(m: *Model, s: cuda.Stream, bytes: usize) !u64 {
        const p = try m.alloc(bytes);
        if (m.planned == null) try m.d.check(m.d.api.cuMemsetD8Async(p, 0, bytes, s.handle), "cuMemsetD8Async");
        return p;
    }

    /// A zeroed buffer for a sliding cache (paged models make these again for the slots attach names).
    fn ringZeros(m: *Model, s: cuda.Stream, bytes: usize) !u64 {
        if (m.planned) |*p| {
            p.* += @max(bytes, 256);
            return 0;
        }
        var b = try DeviceBuffer.alloc(m.d, @max(bytes, 256));
        errdefer b.free();
        try m.d.check(m.d.api.cuMemsetD8Async(b.ptr, 0, bytes, s.handle), "cuMemsetD8Async");
        try m.rings.append(m.gpa, b);
        return b.ptr;
    }

    /// Cache `i`'s positions a slot: the context (full) or the ring (sliding).
    pub fn cacheLen(m: *const Model, i: usize) usize {
        return if (m.sched.cache_kind[i] == .full) m.context else m.ring;
    }

    /// Bytes one stream's caches take at `context` positions (the loop count is the config's).
    pub fn streamBytes(c: Config, context: usize) usize {
        return memory.streamBytes(c, context, ringSize(prompt_rows, c.window));
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, k: *const Kernels, context: usize, slots: usize) !Model {
        var m = blank(gpa, d, w, context, slots);
        errdefer m.deinit();
        try m.build(s, k);
        return m;
    }

    /// A model whose full caches are pages: one slot's rings now; `attach` makes the arena, the other slots' rings.
    pub fn initPaged(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, k: *const Kernels, context: usize) !Model {
        var m = blank(gpa, d, w, context, 1);
        m.paged = true;
        errdefer m.deinit();
        try m.build(s, k);
        return m;
    }

    /// The device bytes `initPaged` allocates.
    pub fn openBytesPaged(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, k: *const Kernels, context: usize) !usize {
        var m = blank(gpa, d, w, context, 1);
        defer m.deinit();
        m.paged = true;
        m.planned = 0;
        try m.build(s, k);
        return m.planned.?;
    }

    /// What the host's pool plans for this model's pages (see cuda_pages.layout).
    pub fn layout(m: *const Model) @import("core").kv_pool.Layout {
        return pages.layout(m.c, m.sched.cacheCount(.full), m.sched.cacheCount(.sliding), m.context, m.ring);
    }

    /// Slots a backend holds: the model's now, or (paged) as many as a round takes.
    pub fn slotCap(m: *const Model) usize {
        return if (m.paged) pages.max_slots else m.slots;
    }

    /// Makes the arena of `pool` pages to lease, the tables and the rings of `extra + 1` slots.
    pub fn attach(m: *Model, s: cuda.Stream, extra: usize, pool: usize) !void {
        if (!m.paged or m.paging != null or extra >= pages.max_slots) return error.BadAttach;
        const c = m.c;
        const per_slot = pages.pagesOf(m.context);
        const elems = pages.pageElems(c, m.sched.cacheCount(.full));
        const ids = try m.gpa.alloc(u32, pages.max_slots * per_slot);
        errdefer m.gpa.free(ids);
        @memset(ids, 0);
        for (m.rings.items) |*b| b.free();
        m.rings.clearRetainingCapacity();
        m.slots = extra + 1;
        try m.rings.ensureTotalCapacity(m.gpa, 2 * m.sched.caches);
        var p: pages.Paging = .{ .pool = pool, .per_slot = per_slot, .page_elems = elems, .arena = 0, .table = 0, .ids = ids };
        p.arena = try m.alloc(pool * elems * 2);
        p.table = try m.alloc(pages.max_slots * per_slot * 4);
        try m.d.check(m.d.api.cuMemsetD8Async(p.arena, 0, pool * elems * 2, s.handle), "cuMemsetD8Async");
        try m.upload(s, p.table, std.mem.sliceAsBytes(ids));
        const region = pool * pages.tokens * c.kvDim() * 2; // one full cache's K (or V) rows over every page
        var block: usize = 0;
        for (m.sched.cache_kind[0..m.sched.caches], m.k, m.v) |kind, *kc, *vc| {
            if (kind == .full) {
                kc.* = p.arena + block * region;
                vc.* = kc.* + region;
                block += 2;
            } else {
                const bytes = m.slots * m.ring * c.kvDim() * 2;
                kc.* = try m.ringZeros(s, bytes);
                vc.* = try m.ringZeros(s, bytes);
            }
        }
        m.paging = p;
        m.dropGraphs();
        try s.synchronize();
    }

    /// Slot `slot`'s pages are `ids` (arena page numbers, in position order from entry `from`): table and host copy.
    pub fn mapPages(m: *Model, s: cuda.Stream, slot: usize, ids: []const u32, from: usize) !void {
        const p = if (m.paging) |*x| x else return error.NotPaged;
        if (slot >= m.slots or from + ids.len > p.per_slot) return error.BadChain;
        const at = slot * p.per_slot + from;
        @memcpy(p.ids[at..][0..ids.len], ids);
        try m.upload(s, p.table + at * 4, std.mem.sliceAsBytes(ids));
        p.caps[slot] = (from + ids.len) * pages.tokens;
        p.wins[slot] = pages.windowOf(p.ids[slot * p.per_slot ..][0 .. from + ids.len], p.per_slot);
    }

    /// Slot `slot` holds no pages (its table entries stay; no forward may reach them).
    pub fn unmapPages(m: *Model, slot: usize) void {
        if (m.paging) |*p| {
            p.caps[slot] = 0;
            p.wins[slot] = null;
        }
    }

    /// The device bytes `init` allocates for these sizes, counted by the same path without allocating.
    pub fn openBytes(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, k: *const Kernels, context: usize, slots: usize) !usize {
        var m = blank(gpa, d, w, context, slots);
        defer m.deinit();
        m.planned = 0;
        try m.build(s, k);
        return m.planned.?;
    }

    fn blank(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const wts.Weights, context: usize, slots: usize) Model {
        const c = w.config;
        return .{ .gpa = gpa, .d = d, .w = w, .c = c, .sched = schedule.build(c), .context = context, .ring = ringSize(prompt_rows, c.window), .slots = slots };
    }

    fn build(m: *Model, s: cuda.Stream, k: *const Kernels) !void {
        const c = m.c;
        m.k = try m.gpa.alloc(u64, m.sched.caches);
        m.v = try m.gpa.alloc(u64, m.sched.caches);
        @memset(m.k, 0);
        @memset(m.v, 0);
        for (m.k, m.v, 0..) |*kc, *vc, i| {
            if (m.paged and m.sched.cache_kind[i] == .full) continue;
            const bytes = m.slots * m.cacheLen(i) * c.kvDim() * 2;
            kc.* = if (m.paged) try m.ringZeros(s, bytes) else try m.zeros(s, bytes);
            vc.* = if (m.paged) try m.ringZeros(s, bytes) else try m.zeros(s, bytes);
        }
        for ([_]f64{ c.theta_sliding, c.theta_full }, &m.inv) |theta, *p| {
            var inv: [256]f32 = undefined; // vLLM's inv_freq in fp32: 1 / theta^(2j / D)
            const half = c.head_dim / 2;
            for (inv[0..half], 0..) |*v, j| v.* = 1 / std.math.pow(f32, @floatCast(theta), @as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(c.head_dim)));
            p.* = try m.alloc(half * 4);
            if (m.planned == null) try m.upload(s, p.*, std.mem.sliceAsBytes(inv[0..half]));
        }
        try m.scratch(k);
        if (m.planned == null) try s.synchronize();
    }

    pub fn deinit(m: *Model) void {
        m.dropGraphs();
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
        for (m.rings.items) |*b| b.free();
        m.rings.deinit(m.gpa);
        if (m.paging) |p| m.gpa.free(p.ids);
        m.gpa.free(m.k);
        m.gpa.free(m.v);
    }

    /// Every captured graph dropped (the next decode forward of each row count captures again).
    pub fn dropGraphs(m: *Model) void {
        for (&m.graphs) |*g| if (g.*) |*e| {
            e.deinit();
            g.* = null;
        };
        for (&m.flat_graphs) |*g| if (g.*) |*e| {
            e.deinit();
            g.* = null;
        };
    }

    fn upload(m: *const Model, s: cuda.Stream, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(m.d, dst, bytes, s);
    }

    fn copy(m: *const Model, s: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, s.handle), "cuMemcpyDtoDAsync");
    }

    /// A call's window; full caches of a paged model read through the table (a prompt's slot: its own row).
    fn window(m: *const Model, full: bool, prompt_slot: ?usize) cuda.window_attention.Window {
        const c = m.c;
        var w: cuda.window_attention.Window = .{ .h = c.heads, .hk = c.kv_heads, .d = c.head_dim, .win = if (full) 0 else c.window, .ring = if (full) m.context else m.ring, .scale = c.factors().attn };
        if (full) if (m.paging) |p| {
            w.pg = .{ .table = p.table + (prompt_slot orelse 0) * p.per_slot * 4, .page_elems = pages.tokens * c.kvDim(), .tokens = pages.tokens, .stride = p.per_slot };
        };
        return w;
    }

    fn scratch(m: *Model, k: *const Kernels) !void {
        const c = m.c;
        const r = @max(prompt_rows, round_rows);
        const d = c.hidden;
        var part: usize = 0;
        for (m.w.subs) |sub| switch (sub) {
            .attn => |a| for ([_]cuda.qlinear.Weight{ a.q, a.k, a.v, a.g, a.o }) |q| for (1..round_rows + 1) |rows| {
                part = @max(part, k.lane.partBytes(rows, q.nvfp4));
            },
            .mlp => |p| for ([_]cuda.qlinear.Weight{ p.gate, p.up, p.down }) |q| for (1..round_rows + 1) |rows| {
                part = @max(part, k.lane.partBytes(rows, q.nvfp4));
            },
        };
        const win = @max(WindowAttention.scratchBytes(m.window(true, null), attend_rows), WindowAttention.scratchBytes(m.window(false, null), attend_rows));
        m.s = .{
            .ids = try m.alloc(r * 4),
            .pos = try m.alloc(r * 4),
            .slot = try m.alloc(r * 4),
            .fslot = if (m.paged) try m.alloc(r * 4) else 0, // a row's window of pages, as a contiguous slot
            .h = try m.alloc(r * d * 2),
            .h2 = try m.alloc(r * d * 2),
            .state = try m.alloc(r * d * 2),
            .input = try m.alloc(r * d * 2),
            .out = try m.alloc(r * d * 2),
            .q = try m.alloc(r * c.qDim() * 2),
            .kk = try m.alloc(r * c.kvDim() * 2),
            .vv = try m.alloc(r * c.kvDim() * 2),
            .g = try m.alloc(r * c.qDim() * 2),
            .qn = try m.alloc(r * c.qDim() * 2),
            .kn = try m.alloc(r * c.kvDim() * 2),
            .vn = try m.alloc(r * c.kvDim() * 2),
            .att = try m.alloc(r * c.qDim() * 2),
            .gated = try m.alloc(r * c.qDim() * 2),
            .mg = try m.alloc(r * c.intermediate * 2),
            .mu = try m.alloc(r * c.intermediate * 2),
            .m = try m.alloc(r * c.intermediate * 2),
            .part = try m.alloc(part),
            .win = try m.alloc(win),
            .logits = try m.alloc(round_rows * c.vocab * 4),
        };
    }

    /// Every chain's rows (one prompt chunk, or decode windows); rows [head0, head0 + heads)'s fp32 logits at s.logits.
    pub fn forward(m: *Model, k: *const Kernels, s: cuda.Stream, chains: []const Chain, prompt: bool, head0: usize, heads: usize) !void {
        var rows: usize = 0;
        for (chains) |ch| {
            if (ch.tokens.len == 0 or ch.slot >= m.slots or ch.p0 + ch.tokens.len > m.context) return error.BadChain;
            if (m.paging) |p| if (ch.p0 + ch.tokens.len > p.caps[ch.slot]) return error.PastPages;
            if (!prompt and ch.tokens.len > window_rows) return error.WindowTooWide;
            rows += ch.tokens.len;
        }
        if (rows == 0 or rows > @as(usize, if (prompt) prompt_rows else round_rows) or (prompt and chains.len != 1)) return error.TooManyRows;
        if (heads > round_rows or head0 + heads > rows) return error.TooManyRows;
        // paged: when every stream holds a window of pages, its full caches are main's contiguous slot of that window
        const flat = if (m.paging) |p| for (chains) |ch| {
            if (p.wins[ch.slot] == null) break false;
        } else true else false;
        const ids = try m.gpa.alloc(i32, 4 * rows);
        defer m.gpa.free(ids);
        var at: usize = 0;
        for (chains) |ch| {
            for (ch.tokens, 0..) |t, i| {
                ids[at + i] = @intCast(t);
                ids[rows + at + i] = @intCast(ch.p0 + i);
                ids[2 * rows + at + i] = @intCast(ch.slot);
                if (flat) ids[3 * rows + at + i] = @intCast(m.paging.?.wins[ch.slot].?);
            }
            at += ch.tokens.len;
        }
        try m.upload(s, m.s.ids, std.mem.sliceAsBytes(ids[0..rows]));
        try m.upload(s, m.s.pos, std.mem.sliceAsBytes(ids[rows .. 2 * rows]));
        try m.upload(s, m.s.slot, std.mem.sliceAsBytes(ids[2 * rows .. 3 * rows]));
        if (flat) try m.upload(s, m.s.fslot, std.mem.sliceAsBytes(ids[3 * rows ..]));
        const p0 = if (prompt) chains[0].p0 else 0;
        const slot = if (prompt) chains[0].slot else 0;
        const fslot = if (flat and prompt) m.paging.?.wins[slot].? else slot;
        // a decode forward's launches depend on its row count and `flat` only (the rest is read on the device)
        if (prompt or !m.graphed or head0 != 0 or heads != rows) return m.launch(k, s, rows, prompt, p0, slot, fslot, flat, head0, heads);
        const g = if (flat) &m.flat_graphs[rows] else &m.graphs[rows];
        if (g.* == null) g.* = try m.capture(k, s, rows, flat);
        try g.*.?.launchOn(s);
    }

    /// A decode forward of `rows` rows, every row's logits, captured as a graph.
    fn capture(m: *Model, k: *const Kernels, s: cuda.Stream, rows: usize, flat: bool) !cuda.graph.Exec {
        try s.synchronize();
        try cuda.graph.beginCapture(s, .thread_local);
        m.launch(k, s, rows, false, 0, 0, 0, flat, 0, rows) catch |e| {
            if (cuda.graph.endCapture(s)) |g| {
                var gg = g;
                gg.deinit();
            } else |_| {}
            return e;
        };
        var g = try cuda.graph.endCapture(s);
        defer g.deinit();
        return g.instantiate();
    }

    /// The forward's launches over staged ids, positions and slots: no host copies, so a graph can hold them.
    fn launch(m: *Model, k: *const Kernels, s: cuda.Stream, n: usize, prompt: bool, p0: usize, slot: usize, fslot: usize, flat: bool, head0: usize, heads: usize) !void {
        const c = m.c;
        const lin: cuda.qlinear.Linear = .{ .lane = &k.lane, .nvfp4_prompt = &k.prompt };
        const d = c.hidden;
        const f = c.factors();
        const x = m.s;
        try k.embed.bf16(s, x.ids, m.w.embed, x.h, n, d);
        var h = x.h;
        var h2 = x.h2;
        for (m.sched.slice()) |step| {
            if (step.enter) {
                if (step.loop == 0) {
                    try m.copy(s, x.input, h, n * d * 2);
                    try m.copy(s, x.state, h, n * d * 2);
                }
                try k.norms.l2(s, x.state, x.input, h, n, d, eps);
            }
            const alpha = switch (m.w.subs[step.sub]) {
                .attn => |a| blk: {
                    try m.attention(k, s, lin, a, step, h, n, prompt, p0, slot, fslot, flat);
                    break :blk a.alpha;
                },
                .mlp => |p| blk: {
                    try m.proj(lin, s, p.gate, h, x.mg, n, prompt);
                    try m.proj(lin, s, p.up, h, x.mu, n, prompt);
                    try k.act.swigluL2(s, x.mg, c.intermediate, x.mu, c.intermediate, p.su_g, p.su_u, x.m, n, c.intermediate, eps);
                    try m.proj(lin, s, p.down, x.m, x.out, n, prompt);
                    break :blk p.alpha;
                },
            };
            try k.norms.slerp(s, h, x.out, alpha, f.alpha, h2, n, d, eps);
            std.mem.swap(u64, &h, &h2);
            if (step.leave) {
                try k.norms.l2(s, h, null, x.state, n, d, eps);
                try m.copy(s, h, x.state, n * d * 2);
            }
        }
        if (heads == 0) return;
        try k.dense.rowsScaled(s, h + head0 * d * 2, d, m.w.head, m.w.logit, x.logits, c.vocab, heads, c.vocab, d);
    }

    fn attention(m: *Model, k: *const Kernels, s: cuda.Stream, lin: cuda.qlinear.Linear, a: wts.Attn, step: schedule.Step, h: u64, n: usize, prompt: bool, p0: usize, slot: usize, fslot: usize, flat: bool) !void {
        const c = m.c;
        const x = m.s;
        try m.proj(lin, s, a.q, h, x.q, n, prompt);
        try m.proj(lin, s, a.k, h, x.kk, n, prompt);
        try m.proj(lin, s, a.v, h, x.vv, n, prompt);
        try m.proj(lin, s, a.g, h, x.g, n, prompt);
        const full = step.kind == .full;
        const cache = step.cache.?;
        const dh: i64 = @intCast(c.head_dim);
        const src: cuda.qk_rows.Src = .{ .q = x.q, .q_ld = @intCast(c.qDim()), .q_hs = dh, .k = x.kk, .k_ld = @intCast(c.kvDim()), .k_hs = dh, .v = x.vv, .v_ld = @intCast(c.kvDim()), .v_hs = dh };
        const shape: cuda.qk_rows.Shape = .{ .l2 = true, .q_scale = a.qs, .inv = m.inv[@intFromBool(full)], .rot = c.head_dim, .eps = eps };
        const len = m.cacheLen(cache);
        const flat_full = full and flat; // a full cache's window of pages w is main's slot w: its kernels and addresses
        const pg = full and m.paging != null and !flat; // written by `put` into the pages, not by the rows kernel
        const sl = if (flat_full) x.fslot else x.slot;
        const out: cuda.qk_rows.Out = .{ .q = x.qn, .k = x.kn, .v = x.vn, .kc = if (pg) null else m.k[cache], .vc = if (pg) null else m.v[cache], .len = len };
        try k.qk.rowsFrom(s, src, .{ .h = c.heads, .hk = c.kv_heads, .dim = c.head_dim }, shape, x.pos, sl, out, n);
        var w = m.window(full, slot);
        if (flat_full) w.pg = null;
        if (pg) try k.win.put(s, x.kn, x.vn, m.k[cache], m.v[cache], x.pos, x.slot, m.window(true, null), n); // the whole table
        if (prompt) {
            const off = if (pg) 0 else (if (flat_full) fslot else slot) * len * c.kvDim() * 2;
            try k.win.promptRows(s, x.qn, m.k[cache] + off, m.v[cache] + off, x.att, p0, n, w, null);
        } else {
            const qw = c.qDim() * 2;
            var r: usize = 0;
            while (r < n) : (r += attend_rows) {
                const rows = @min(attend_rows, n - r);
                try k.win.decode(s, x.qn + r * qw, m.k[cache], m.v[cache], x.att + r * qw, x.pos + r * 4, sl + r * 4, w, x.win, rows);
            }
        }
        try k.act.gateL2(s, x.att, x.g, c.qDim(), a.sg, x.gated, n, c.qDim(), eps);
        try m.proj(lin, s, a.o, x.gated, x.out, n, prompt);
    }

    fn proj(m: *const Model, lin: cuda.qlinear.Linear, s: cuda.Stream, w: cuda.qlinear.Weight, in: u64, out: u64, n: usize, prompt: bool) !void {
        const rows: cuda.qlinear.Rows = .{ .x = in, .part = m.s.part };
        try (if (prompt) lin.prompt(s, w, rows, out, n) else lin.decode(s, w, rows, out, n));
    }
};

test "the forward's launches type-check against the shared operators" {
    std.testing.refAllDecls(Model);
    std.testing.refAllDecls(wts);
}

test "a sliding ring holds a prompt chunk and its window in whole key blocks" {
    try std.testing.expectEqual(@as(usize, 3072), ringSize(prompt_rows, 1024));
    try std.testing.expectEqual(@as(usize, 1088), ringSize(window_rows, 1024));
}

test "Dory queued upload: Model inputs survive caller mutation until stream completion" {
    const Send = struct {
        fn send(d: *const cuda.Driver, s: cuda.Stream, bytes: []const u8) !void {
            var x: Model = undefined;
            x.d = d;
            try x.upload(s, 0x1000, bytes);
        }
    };
    try @import("cuda_upload_test.zig").check(Send.send);
}
