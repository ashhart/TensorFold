//! Gemma 4 on CUDA: prompt chunks and decode windows through the shared operators in the Metal family's steps.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const cfg = @import("cuda_config.zig");
const wts = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Window = cuda.window_attention.Window;
const Pages = cuda.window_attention.Pages;

/// Rows a prompt chunk takes (Metal's chunk_rows: cut where Metal cuts, so fresh and resumed passes keep their bits).
pub const prompt_rows = 512;
/// Rows a decode forward takes over every stream's window (Metal's prompt_rows for windows).
pub const round_rows = 128;
/// Rows a decode attention call takes: its partials' scratch is sized for this many (Metal's attend_rows).
const attend_rows = 16;
/// A sliding layer's ring: the window and a prompt chunk, so a chunk's writes never reach keys its first row reads.
pub const ring = cfg.window + prompt_rows;
/// Positions a global layer's page holds: window_rows.cu's decode chunk, so a chunk and a prompt tile stay in a page.
pub const page_tokens = cuda.window_attention.page_unit;

/// A stream's rows in one forward: its cache slot, first position, tokens.
pub const Chain = struct { slot: usize, p0: usize, tokens: []const u32, images: []const Inject = &.{} };

/// An image's rows in a chunk: rows [row, row + rows) take `src`'s (unscaled), seeing all of them on sliding layers.
pub const Inject = struct { row: usize, rows: usize, src: u64 };

/// sqrt(hidden) as MLX multiplies by it: a weak-typed scalar, rounded to bf16 first (Metal forward.embedScale).
pub fn embedScale(hidden: usize) f32 {
    const x: f32 = @sqrt(@as(f32, @floatFromInt(hidden)));
    const b: u32 = @bitCast(x);
    return @bitCast((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000);
}

/// fp32 inverse RoPE frequencies [dim / 2]: sliding as Metal's host table; global 1 / f32(1e6^(2i/512)), 0 past 64.
pub fn inverseFrequencies(is_global: bool, out: []f32) void {
    if (!is_global) return cfg.inverseFrequencies(false, out);
    for (out, 0..) |*f, i| {
        if (i >= cfg.global_rotated) {
            f.* = 0;
            continue;
        }
        const p: f32 = @floatCast(std.math.pow(f64, cfg.global_theta, @as(f64, @as(f32, @floatFromInt(2 * i)) / 512.0)));
        f.* = @floatCast(1.0 / @as(f64, p));
    }
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *const wts.Weights,
    geo: cfg.Geometry,
    layers: usize,
    context: usize,
    slots: usize,
    kc: []u64, // per layer: keys [slots, len, kv heads, head dim] bf16 (len: the ring, or the context if global)
    vc: []u64, // paged: a global layer's K or V region of the arena, page id p at rows [p * page_tokens, ...)
    paged: bool = false, // global caches are pages of the core pool (attach makes the arena); sliding stay rings
    pages: usize = 0, // pages in the arena
    table: u64 = 0, // paged: int32 [slots][perSlot()], slot s's page ids in position order
    ids: []u32 = &.{}, // paged: the tables on the host
    caps: []usize = &.{}, // paged: positions slot s's pages cover
    inv_sliding: u64, // fp32 [128]
    inv_global: u64, // fp32 [256]
    s: Scratch,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    planned: ?usize = null, // a plan: the bytes counted, nothing allocated
    /// Decode forwards of n rows as CUDA graphs, captured at their first use ([n]; null: not yet, or graphs off).
    graphs: [round_rows + 1]?cuda.graph.Exec = @splat(null),
    graphed: bool = true,
    /// A drafter's tapped layers: each one's output rows (before the next norm) at taps_buf [rows, taps, hidden].
    taps: [8]u32 = undefined,
    tap_count: usize = 0,
    taps_buf: ?DeviceBuffer = null,

    const Scratch = struct { ids: u64, pos: u64, slot: u64, hi: u64, h: [2]u64, n: u64, xs: u64, q: u64, k: u64, v: u64, qr: u64, kr: u64, vr: u64, att: u64, o: u64, gate: u64, up: u64, act: u64, y: u64, part: u64, logits: u64 };

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
        if (m.planned == null) try (DeviceBuffer{ .d = m.d, .ptr = p, .len = @max(bytes, 256), .borrowed = true }).fill8(0, s.handle);
        return p;
    }

    /// A layer's cache positions: the ring, or every position of the context (global layers).
    pub fn cacheLen(m: *const Model, i: usize) usize {
        return if (cfg.global(i)) m.context else ring;
    }

    /// Bytes one stream's caches take.
    pub fn streamBytes(geo: cfg.Geometry, layers: usize, context: usize) usize {
        var total: usize = 0;
        for (0..layers) |i| total += 2 * (if (cfg.global(i)) context else ring) * geo.kvHeads(i) * cfg.headDim(i) * 2;
        return total;
    }

    /// `slots` streams with linear caches; paged: one slot's rings now, `attach` makes the pages and the other slots.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, context: usize, slots: usize, paged: bool) !Model {
        var m = blank(gpa, d, w, context, if (paged) 1 else slots, paged);
        errdefer m.deinit();
        try m.build(s);
        return m;
    }

    /// The device bytes `init` allocates for these sizes, counted by the same path without allocating.
    pub fn openBytes(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, context: usize, slots: usize, paged: bool) !usize {
        var m = blank(gpa, d, w, context, if (paged) 1 else slots, paged);
        defer m.deinit();
        m.planned = 0;
        try m.build(s);
        return m.planned.?;
    }

    fn blank(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const wts.Weights, context: usize, slots: usize, paged: bool) Model {
        return .{ .gpa = gpa, .d = d, .w = w, .geo = w.config.geo, .layers = w.layers.len, .context = context, .slots = slots, .kc = &.{}, .vc = &.{}, .paged = paged, .inv_sliding = 0, .inv_global = 0, .s = undefined };
    }

    /// Table entries a slot: the pages its context takes.
    pub fn perSlot(m: *const Model) usize {
        return (m.context + page_tokens - 1) / page_tokens;
    }

    /// bf16 elements a page id steps in one global layer's K (or V) region: page_tokens rows of its kv heads.
    pub fn pageElems(m: *const Model) usize {
        return page_tokens * m.geo.global_kv_heads * cfg.global_dim;
    }

    fn globalLayers(m: *const Model) usize {
        var n: usize = 0;
        for (0..m.layers) |i| n += @intFromBool(cfg.global(i));
        return n;
    }

    /// Bytes a page holds over every global layer's K and V.
    pub fn pageBytes(m: *const Model) u64 {
        return m.globalLayers() * 2 * m.pageElems() * 2;
    }

    /// What the host's pool plans: a stream's sliding rings and table row fixed, a token its global layers' rows.
    pub fn layout(m: *const Model) kv.Layout {
        var fixed: u64 = m.perSlot() * 4;
        for (0..m.layers) |i| if (!cfg.global(i)) {
            fixed += 2 * ring * m.geo.kvHeads(i) * cfg.headDim(i) * 2;
        };
        return .{ .fixed_bytes = fixed, .token_bytes = m.pageBytes() / page_tokens, .page_tokens = page_tokens, .snap_bytes = 0 };
    }

    /// Slots a backend holds: the model's, or (paged) as many as a round takes.
    pub fn slotCap(m: *const Model) usize {
        return if (m.paged) round_rows else m.slots;
    }

    /// The global layers' pages as a window reads them (a prompt's slot: its own table row); null: linear caches.
    pub fn pagesFor(m: *const Model, prompt_slot: ?usize) ?Pages {
        if (!m.paged) return null;
        return .{ .table = m.table + (prompt_slot orelse 0) * m.perSlot() * 4, .page_elems = m.pageElems(), .tokens = page_tokens, .stride = m.perSlot() };
    }

    /// Makes the arena of `pool` pages to lease, the rings and tables of `extra + 1` slots (no pages mapped).
    pub fn attach(m: *Model, s: cuda.Stream, extra: usize, pool: usize) !void {
        if (!m.paged or m.pages != 0 or pool == 0 or extra >= round_rows) return error.BadAttach;
        for (0..m.layers) |i| if (!cfg.global(i)) {
            m.forget(m.kc[i]);
            m.forget(m.vc[i]);
        };
        m.forget(m.table);
        m.gpa.free(m.ids);
        m.gpa.free(m.caps);
        m.ids = &.{};
        m.caps = &.{};
        m.slots = extra + 1;
        m.pages = pool;
        const region = pool * m.pageElems() * 2; // one global layer's K (or V) rows over every page
        const arena = try m.zeros(s, m.globalLayers() * 2 * region);
        var block: usize = 0;
        for (0..m.layers) |i| {
            if (cfg.global(i)) {
                m.kc[i] = arena + block * region;
                m.vc[i] = m.kc[i] + region;
                block += 2;
            } else try m.ringOf(s, i);
        }
        try m.makeTables(s);
        m.dropGraphs();
        try s.synchronize();
    }

    /// Slot `slot`'s pages from table entry `from` on are `ids` (arena page numbers in position order).
    pub fn mapPages(m: *Model, s: cuda.Stream, slot: usize, ids: []const u32, from: usize) !void {
        const per = m.perSlot();
        if (!m.paged or slot >= m.slots or from + ids.len > per) return error.BadChain;
        for (ids) |id| if (id >= m.pages) return error.BadChain;
        const at = slot * per + from;
        @memcpy(m.ids[at..][0..ids.len], ids);
        m.caps[slot] = (from + ids.len) * page_tokens;
        try m.upload(s, m.table + at * 4, std.mem.sliceAsBytes(m.ids[at..][0..ids.len]));
    }

    /// Slot `slot` holds no pages: its table entries stay, no forward may reach them.
    pub fn unmapPages(m: *Model, slot: usize) void {
        if (m.paged) m.caps[slot] = 0;
    }

    /// Frees one buffer `alloc` made (a plan counted nothing to free).
    fn forget(m: *Model, ptr: u64) void {
        for (m.buffers.items, 0..) |*b, i| if (b.ptr == ptr) {
            b.free();
            _ = m.buffers.swapRemove(i);
            return;
        };
    }

    fn dropGraphs(m: *Model) void {
        for (&m.graphs) |*g| if (g.*) |*e| {
            e.deinit();
            g.* = null;
        };
    }

    /// Layer i's caches for every slot: a ring, or (linear) a global layer's whole context.
    fn ringOf(m: *Model, s: cuda.Stream, i: usize) !void {
        const bytes = m.slots * m.cacheLen(i) * m.geo.kvHeads(i) * cfg.headDim(i) * 2;
        m.kc[i] = try m.zeros(s, bytes);
        m.vc[i] = try m.zeros(s, bytes);
    }

    /// The slots' page tables on the device and the host, no page mapped.
    fn makeTables(m: *Model, s: cuda.Stream) !void {
        m.table = try m.alloc(m.slots * m.perSlot() * 4);
        if (m.planned != null) return;
        m.ids = try m.gpa.alloc(u32, m.slots * m.perSlot());
        @memset(m.ids, 0);
        m.caps = try m.gpa.alloc(usize, m.slots);
        @memset(m.caps, 0);
        try m.upload(s, m.table, std.mem.sliceAsBytes(m.ids));
    }

    fn build(m: *Model, s: cuda.Stream) !void {
        const geo = m.geo;
        m.kc = try m.gpa.alloc(u64, m.layers);
        m.vc = try m.gpa.alloc(u64, m.layers);
        @memset(m.kc, 0);
        @memset(m.vc, 0);
        for (0..m.layers) |i| if (!m.paged or !cfg.global(i)) try m.ringOf(s, i); // paged global layers: attach's arena
        if (m.paged) try m.makeTables(s);
        var sl: [cfg.sliding_dim / 2]f32 = undefined;
        var gl: [cfg.global_dim / 2]f32 = undefined;
        inverseFrequencies(false, &sl);
        inverseFrequencies(true, &gl);
        m.inv_sliding = try m.alloc(sl.len * 4);
        m.inv_global = try m.alloc(gl.len * 4);
        if (m.planned == null) {
            try (DeviceBuffer{ .d = m.d, .ptr = m.inv_sliding, .len = sl.len * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(&sl));
            try (DeviceBuffer{ .d = m.d, .ptr = m.inv_global, .len = gl.len * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(&gl));
        }
        const r = @max(prompt_rows, round_rows);
        const h = geo.hidden;
        const qw = geo.queryWidth();
        const kw = geo.kvWidth();
        const most = @max(h, geo.intermediate);
        // the widest decode partials: attend_rows rows of a global layer over the context in 256-key chunks
        const wide: Window = .{ .h = geo.query_heads, .hk = geo.global_kv_heads, .d = cfg.global_dim, .win = 0, .ring = m.context, .scale = 1 };
        const narrow: Window = .{ .h = geo.query_heads, .hk = geo.sliding_kv_heads, .d = cfg.sliding_dim, .win = cfg.window, .ring = ring, .scale = 1 };
        const part = @max(cuda.window_attention.WindowAttention.scratchBytes(wide, attend_rows), cuda.window_attention.WindowAttention.scratchBytes(narrow, attend_rows));
        m.s = .{
            .ids = try m.alloc(r * 4),
            .hi = try m.alloc(prompt_rows * 4),
            .pos = try m.alloc(r * 4),
            .slot = try m.alloc(r * 4),
            .h = .{ try m.alloc(r * h * 2), try m.alloc(r * h * 2) },
            .n = try m.alloc(r * h * 2),
            .xs = try m.alloc(r * most / 64 * 4),
            .q = try m.alloc(r * qw * 2),
            .k = try m.alloc(r * kw * 2),
            .v = try m.alloc(r * kw * 2),
            .qr = try m.alloc(r * qw * 2),
            .kr = try m.alloc(r * kw * 2),
            .vr = try m.alloc(r * kw * 2),
            .att = try m.alloc(r * qw * 2),
            .o = try m.alloc(r * h * 2),
            .gate = try m.alloc(r * geo.intermediate * 2),
            .up = try m.alloc(r * geo.intermediate * 2),
            .act = try m.alloc(r * geo.intermediate * 2),
            .y = try m.alloc(r * h * 2),
            .part = try m.alloc(part),
            .logits = try m.alloc(round_rows * cfg.vocab * 2),
        };
        if (m.planned == null) try s.synchronize();
    }

    pub fn deinit(m: *Model) void {
        m.dropGraphs();
        if (m.taps_buf) |*b| b.free();
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
        m.gpa.free(m.kc);
        m.gpa.free(m.vc);
        m.gpa.free(m.ids);
        m.gpa.free(m.caps);
    }

    fn upload(m: *const Model, s: cuda.Stream, dst: u64, bytes: []const u8) !void {
        try m.d.check(m.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, s.handle), "cuMemcpyHtoDAsync");
    }

    /// The layers whose outputs every forward keeps for a drafter (before any graph is captured with the old set).
    pub fn setTaps(m: *Model, taps: []const u32) !void {
        if (taps.len == 0 or taps.len > m.taps.len) return error.GemmaTapNotServed;
        for (taps, 0..) |t, i| if (t >= m.layers or (i > 0 and t <= taps[i - 1])) return error.GemmaTapNotServed;
        if (m.taps_buf) |*b| b.free();
        m.taps_buf = null;
        m.taps_buf = try DeviceBuffer.alloc(m.d, prompt_rows * taps.len * m.geo.hidden * 2);
        @memcpy(m.taps[0..taps.len], taps);
        m.tap_count = taps.len;
        for (&m.graphs) |*g| if (g.*) |*e| { // captured without the tap copies
            e.deinit();
            g.* = null;
        };
    }

    /// x [rows, k] through `l` into out [rows, n]: a prompt chunk's GEMM, or decode rows on their group sums `xs`.
    fn project(m: *const Model, k: *const kern.Kernels, s: cuda.Stream, l: wts.Proj, x: u64, xs: u64, out: u64, rows: usize, prompt: bool) !void {
        _ = m;
        if (prompt) return switch (l) {
            .a4 => |q| k.affine.prompt(s, x, q, out, rows),
            .an => |q| k.wide.prompt(s, x, q, out, rows),
        };
        const tile: usize = switch (l) {
            .a4 => k.affine.verifyRows(),
            .an => 16,
        };
        var at: usize = 0;
        while (at < rows) : (at += tile) {
            const n = @min(tile, rows - at);
            const xa = x + at * l.k() * 2;
            const sa = xs + at * (l.k() / 64) * 4;
            const oa = out + at * l.n() * 2;
            switch (l) {
                .a4 => |q| try k.affine.verify(s, xa, sa, q, oa, n),
                .an => |q| try k.wide.decode(s, xa, sa, q, oa, n),
            }
        }
    }

    /// Every chain's rows (a prompt chunk, or decode windows); rows [head0, head0 + heads)'s capped logits at s.logits.
    pub fn forward(m: *Model, k: *const kern.Kernels, s: cuda.Stream, chains: []const Chain, prompt: bool, head0: usize, heads: usize) !void {
        var rows: usize = 0;
        for (chains) |c| {
            if (c.tokens.len == 0 or c.slot >= m.slots or c.p0 + c.tokens.len > m.context) return error.BadChain;
            if (m.paged and c.p0 + c.tokens.len > m.caps[c.slot]) return error.PastPages; // rows past the slot's pages
            rows += c.tokens.len;
        }
        if (rows > @as(usize, if (prompt) prompt_rows else round_rows) or (prompt and chains.len != 1)) return error.BadWindow;
        if (heads > round_rows or head0 + heads > rows) return error.BadWindow;
        const ids = try m.gpa.alloc(i32, 3 * rows);
        defer m.gpa.free(ids);
        var at: usize = 0;
        for (chains) |c| {
            for (c.tokens, 0..) |t, i| {
                ids[at + i] = @intCast(t);
                ids[rows + at + i] = @intCast(c.p0 + i);
                ids[2 * rows + at + i] = @intCast(c.slot);
            }
            at += c.tokens.len;
        }
        try m.upload(s, m.s.ids, std.mem.sliceAsBytes(ids[0..rows]));
        try m.upload(s, m.s.pos, std.mem.sliceAsBytes(ids[rows .. 2 * rows]));
        try m.upload(s, m.s.slot, std.mem.sliceAsBytes(ids[2 * rows ..]));
        if (prompt and chains[0].images.len > 0) { // each row's last key: its own, an image row's the image's last
            const c = chains[0];
            const hi = ids[rows .. 2 * rows];
            for (c.images) |im| {
                if (im.rows == 0 or im.row + im.rows > rows) return error.BadWindow;
                @memset(hi[im.row..][0..im.rows], @intCast(c.p0 + im.row + im.rows - 1));
            }
            try m.upload(s, m.s.hi, std.mem.sliceAsBytes(hi));
        }
        // a decode forward's launches depend on its row count only (positions and slots are read on the device)
        if (prompt or !m.graphed or head0 != 0 or heads != rows) return m.launch(k, s, chains, rows, prompt, head0, heads);
        if (m.graphs[rows] == null) m.graphs[rows] = try m.capture(k, s, rows);
        try m.graphs[rows].?.launchOn(s);
    }

    /// A decode forward of `rows` rows, every row's logits, captured as a graph.
    fn capture(m: *Model, k: *const kern.Kernels, s: cuda.Stream, rows: usize) !cuda.graph.Exec {
        try s.synchronize();
        try cuda.graph.beginCapture(s, .thread_local);
        m.launch(k, s, &.{}, rows, false, 0, rows) catch |e| {
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
    fn launch(m: *Model, k: *const kern.Kernels, s: cuda.Stream, chains: []const Chain, rows: usize, prompt: bool, head0: usize, heads: usize) !void {
        const h = m.geo.hidden;
        const x = m.s;
        try k.embed.affine(s, x.ids, m.w.embed_w, m.w.embed_s, m.w.embed_b, x.h[0], rows, h, .{ .bits = @intCast(m.w.embed_bits), .group = cfg.group, .scale = embedScale(h) });
        if (prompt) for (chains[0].images) |im| {
            try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(x.h[0] + im.row * h * 2, im.src, im.rows * h * 2, s.handle), "cuMemcpyDtoDAsync");
        };
        // layer 0's input norm alone; every later one is the previous layer's dense tail
        try k.norms.addRmsTail(s, x.h[0], null, m.w.layers[0].input_norm, x.h[1], x.n, x.xs, rows, h, cfg.eps, .{ .round = true });
        for (0..m.layers) |i| try m.layer(k, s, chains, i, rows, prompt);
        if (heads == 0) return;
        try m.project(k, s, m.w.head, x.n + head0 * h * 2, x.xs + head0 * (h / 64) * 4, x.logits, heads, false);
        try k.act.softcap(s, x.logits, x.logits, heads, cfg.vocab, cfg.softcap);
    }

    /// Layer i on the rows: residual in s.h[0], its normed input (and sums) in s.n (s.xs); the same on return.
    fn layer(m: *Model, k: *const kern.Kernels, s: cuda.Stream, chains: []const Chain, i: usize, rows: usize, prompt: bool) !void {
        const geo = m.geo;
        const l = m.w.layers[i];
        const x = m.s;
        const h = geo.hidden;
        const global = cfg.global(i);
        const dh = cfg.headDim(i);
        const kvh = geo.kvHeads(i);
        const qw = geo.query_heads * dh;
        const kw = kvh * dh;
        try m.project(k, s, l.q, x.n, x.xs, x.q, rows, prompt);
        try m.project(k, s, l.k, x.n, x.xs, x.k, rows, prompt);
        if (l.v) |v| try m.project(k, s, v, x.n, x.xs, x.v, rows, prompt);
        const vsrc = if (global) x.k else x.v; // global layers: v is k_proj's output, unit-normed and not rotated
        const src: cuda.qk_rows.Src = .{ .q = x.q, .q_ld = @intCast(qw), .q_hs = @intCast(dh), .k = x.k, .k_ld = @intCast(kw), .k_hs = @intCast(dh), .v = vsrc, .v_ld = @intCast(kw), .v_hs = @intCast(dh) };
        const shape: cuda.qk_rows.Shape = .{ .norms = l.q_norm, .norm_k = l.k_norm, .norms_bf16 = true, .inv = if (global) m.inv_global else m.inv_sliding, .rot = dh, .eps = cfg.eps, .round = true, .v_norm = true };
        const len = m.cacheLen(i);
        const pg = if (global) m.pagesFor(null) else null; // paged: the rows kernel writes no cache
        try k.qk.rowsFrom(s, src, .{ .h = geo.query_heads, .hk = kvh, .dim = dh }, shape, x.pos, x.slot, .{ .q = x.qr, .k = x.kr, .v = x.vr, .kc = if (pg != null) null else m.kc[i], .vc = if (pg != null) null else m.vc[i], .len = len }, rows);
        const w: Window = .{ .h = geo.query_heads, .hk = kvh, .d = dh, .win = if (global) 0 else cfg.window, .ring = len, .scale = 1, .p32 = true, .pg = pg };
        // paged: `put` writes into the pages the bits the rows kernel would have written into a linear cache
        if (pg != null) try k.win.put(s, x.kr, x.vr, m.kc[i], m.vc[i], x.pos, x.slot, w, rows);
        if (prompt) {
            const c = chains[0];
            const off = if (pg != null) 0 else c.slot * len * kw * 2; // pages: the layer's region, the slot's row
            const hi: ?u64 = if (!global and c.images.len > 0) x.hi else null;
            try k.win.promptRows(s, x.qr, m.kc[i] + off, m.vc[i] + off, x.att, c.p0, rows, if (global) .{ .h = w.h, .hk = w.hk, .d = w.d, .win = 0, .ring = 0, .scale = 1, .p32 = true, .pg = m.pagesFor(c.slot) } else w, hi);
        } else {
            var r: usize = 0;
            while (r < rows) : (r += attend_rows) {
                const n = @min(attend_rows, rows - r);
                try k.win.decode(s, x.qr + r * qw * 2, m.kc[i], m.vc[i], x.att + r * qw * 2, x.pos + r * 4, x.slot + r * 4, w, x.part, n);
            }
        }
        try k.act.groupSums(s, x.att, qw, x.xs, rows, qw);
        try m.project(k, s, l.o, x.att, x.xs, x.o, rows, prompt);
        try k.norms.addRmsTail(s, x.o, x.h[0], l.pre_feedforward, x.h[1], x.n, x.xs, rows, h, cfg.eps, .{ .round = true, .branch = l.post_attention });
        try m.project(k, s, l.gate, x.n, x.xs, x.gate, rows, prompt);
        try m.project(k, s, l.up, x.n, x.xs, x.up, rows, prompt);
        try k.act.geglu(s, x.gate, geo.intermediate, x.up, geo.intermediate, x.act, x.xs, rows, geo.intermediate);
        try m.project(k, s, l.down, x.act, x.xs, x.y, rows, prompt);
        const next = if (i + 1 < m.layers) m.w.layers[i + 1].input_norm else m.w.final_norm;
        try k.norms.addRmsTail(s, x.y, x.h[1], next, x.h[0], x.n, x.xs, rows, h, cfg.eps, .{ .round = true, .branch = l.post_feedforward, .scalar = l.scalar });
        if (std.mem.indexOfScalar(u32, m.taps[0..m.tap_count], @intCast(i))) |j| {
            const p: cuda.abi.Memcpy2D = .{ .src = x.h[0], .src_pitch = h * 2, .dst = m.taps_buf.?.ptr + j * h * 2, .dst_pitch = m.tap_count * h * 2, .width = h * 2, .height = rows };
            try m.d.check(m.d.api.cuMemcpy2DAsync_v2(&p, s.handle), "cuMemcpy2DAsync");
        }
    }
};

test "the embedding scale is bf16(sqrt(hidden)): 62 for the 12B, 73.5 for the 31B" {
    try std.testing.expectEqual(@as(f32, 62.0), embedScale(3840));
    try std.testing.expectEqual(@as(f32, 73.5), embedScale(5376));
}

test "RoPE tables: sliding as Metal's host table, global zero past 64 pairs and decreasing before" {
    var sl: [128]f32 = undefined;
    var ref: [128]f32 = undefined;
    inverseFrequencies(false, &sl);
    cfg.inverseFrequencies(false, &ref);
    try std.testing.expectEqualSlices(f32, &ref, &sl);
    var gl: [256]f32 = undefined;
    inverseFrequencies(true, &gl);
    try std.testing.expectEqual(@as(f32, 1), gl[0]);
    for (1..64) |i| try std.testing.expect(gl[i] < gl[i - 1] and gl[i] > 0);
    for (gl[64..]) |v| try std.testing.expectEqual(@as(f32, 0), v);
}
