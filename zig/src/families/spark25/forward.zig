//! Spark2.5 on CUDA: prompt chunks and decode windows through the shared operators, the reference's fp32 residual.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const kern = @import("kernels.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Window = cuda.window_attention.Window;
const Pages = cuda.window_attention.Pages;

/// Rows a prompt chunk takes.
pub const prompt_rows = 512;
/// Rows a decode forward takes over every stream's window.
pub const round_rows = 128;
/// Rows a decode attention call takes: its partials' scratch is sized for this many.
const attend_rows = 16;
/// A sliding layer's ring: the window and a prompt chunk, so a chunk's writes never reach keys its first row reads.
pub const ring = cfg.window + prompt_rows;
/// Positions a full layer's page holds: window_rows.cu's decode chunk, so a chunk and a prompt tile stay in a page.
pub const page_tokens = cuda.window_attention.page_unit;

/// A stream's rows in one forward: its cache slot, first position, tokens.
pub const Chain = struct { slot: usize, p0: usize, tokens: []const u32 };

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *const wts.Weights,
    geo: cfg.Geometry,
    layers: usize,
    context: usize,
    slots: usize,
    kc: []u64, // per layer: keys [slots, len, kv heads, head dim] bf16 (len: the ring, or the context if full)
    vc: []u64, // paged: a full layer's K or V region of the arena, page id p at rows [p * page_tokens, ...)
    paged: bool = false, // full layers' caches are pages of the core pool (attach makes the arena); sliding stay rings
    pages: usize = 0, // pages in the arena
    table: u64 = 0, // paged: int32 [slots][perSlot()], slot s's page ids in position order
    ids: []u32 = &.{}, // paged: the tables on the host
    caps: []usize = &.{}, // paged: positions slot s's pages cover
    inv_sliding: u64, // fp32 [128]
    inv_full: u64, // fp32 [128], zero past the rotated 32 pairs
    s: Scratch,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    planned: ?usize = null, // a plan: the bytes counted, nothing allocated
    /// Decode forwards of n rows as CUDA graphs, captured at their first use ([n]; null: not yet, or graphs off).
    graphs: [round_rows + 1]?cuda.graph.Exec = @splat(null),
    graphed: bool = true,
    /// Counts dropGraphs calls: attach moves the caches every captured graph reads.
    graph_gen: u32 = 0,

    const Scratch = struct { ids: u64, pos: u64, slot: u64, res: u64, e: u64, n: u64, xs: u64, qkvg: u64, qr: u64, kr: u64, vr: u64, att: u64, ag: u64, o: u64, gate: u64, up: u64, act: u64, y: u64, part: u64, logits: u64 };

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

    /// A layer's cache positions: the ring, or every position of the context (full layers).
    pub fn cacheLen(m: *const Model, i: usize) usize {
        return if (cfg.full(i)) m.context else ring;
    }

    /// Bytes one stream's caches take.
    pub fn streamBytes(geo: cfg.Geometry, layers: usize, context: usize) usize {
        var total: usize = 0;
        for (0..layers) |i| total += 2 * (if (cfg.full(i)) context else ring) * geo.kvWidth() * 2;
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
        return .{ .gpa = gpa, .d = d, .w = w, .geo = w.config.geo, .layers = w.layers.len, .context = context, .slots = slots, .kc = &.{}, .vc = &.{}, .paged = paged, .inv_sliding = 0, .inv_full = 0, .s = undefined };
    }

    /// Table entries a slot: the pages its context takes.
    pub fn perSlot(m: *const Model) usize {
        return (m.context + page_tokens - 1) / page_tokens;
    }

    /// bf16 elements a page id steps in one full layer's K (or V) region: page_tokens rows of its kv heads.
    pub fn pageElems(m: *const Model) usize {
        return page_tokens * m.geo.kvWidth();
    }

    fn fullLayers(m: *const Model) usize {
        var n: usize = 0;
        for (0..m.layers) |i| n += @intFromBool(cfg.full(i));
        return n;
    }

    /// Bytes a page holds over every full layer's K and V.
    pub fn pageBytes(m: *const Model) u64 {
        return m.fullLayers() * 2 * m.pageElems() * 2;
    }

    /// What the host's pool plans: a stream's sliding rings and table row fixed, a token its full layers' rows.
    pub fn layout(m: *const Model) kv.Layout {
        const fixed: u64 = m.perSlot() * 4 + (m.layers - m.fullLayers()) * 2 * ring * m.geo.kvWidth() * 2;
        return .{ .fixed_bytes = fixed, .token_bytes = m.pageBytes() / page_tokens, .page_tokens = page_tokens, .snap_bytes = 0 };
    }

    /// Slots a backend holds: the model's, or (paged) as many as a round takes.
    pub fn slotCap(m: *const Model) usize {
        return if (m.paged) round_rows else m.slots;
    }

    /// The full layers' pages as a window reads them (a prompt's slot: its own table row); null: linear caches.
    pub fn pagesFor(m: *const Model, prompt_slot: ?usize) ?Pages {
        if (!m.paged) return null;
        return .{ .table = m.table + (prompt_slot orelse 0) * m.perSlot() * 4, .page_elems = m.pageElems(), .tokens = page_tokens, .stride = m.perSlot() };
    }

    /// Makes the arena of `pool` pages to lease, the rings and tables of `extra + 1` slots (no pages mapped).
    pub fn attach(m: *Model, s: cuda.Stream, extra: usize, pool: usize) !void {
        if (!m.paged or m.pages != 0 or pool == 0 or extra >= round_rows) return error.BadAttach;
        for (0..m.layers) |i| { // the rings, and the full layers' first pages
            m.forget(m.kc[i]);
            m.forget(m.vc[i]);
        }
        m.forget(m.table);
        m.gpa.free(m.ids);
        m.gpa.free(m.caps);
        m.ids = &.{};
        m.caps = &.{};
        m.slots = extra + 1;
        m.pages = pool;
        const region = pool * m.pageElems() * 2; // one full layer's K (or V) rows over every page
        const arena = try m.zeros(s, m.fullLayers() * 2 * region);
        var block: usize = 0;
        for (0..m.layers) |i| {
            if (cfg.full(i)) {
                m.kc[i] = arena + block * region;
                m.vc[i] = m.kc[i] + region;
                block += 2;
            } else try m.ringOf(s, i);
        }
        try m.makeTables(s);
        m.dropGraphs(); // the captured forwards read the caches attach just replaced
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
        m.graph_gen +%= 1;
        for (&m.graphs) |*g| if (g.*) |*e| {
            e.deinit();
            g.* = null;
        };
    }

    /// Layer i's caches for every slot: a ring, or (linear) a full layer's whole context.
    fn ringOf(m: *Model, s: cuda.Stream, i: usize) !void {
        const bytes = m.slots * m.cacheLen(i) * m.geo.kvWidth() * 2;
        m.kc[i] = try m.zeros(s, bytes);
        m.vc[i] = try m.zeros(s, bytes);
    }

    /// A paged full layer before attach: one page (id 0) for slot 0, so a forward before attach has its rows.
    fn firstPage(m: *Model, s: cuda.Stream, i: usize) !void {
        m.kc[i] = try m.zeros(s, m.pageElems() * 2);
        m.vc[i] = try m.zeros(s, m.pageElems() * 2);
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
        for (0..m.layers) |i| if (!m.paged or !cfg.full(i)) try m.ringOf(s, i) else try m.firstPage(s, i);
        if (m.paged) {
            try m.makeTables(s);
            if (m.planned == null) m.caps[0] = page_tokens; // before attach: slot 0 reads page 0 (a warm-up's rows)
        }
        var sl: [cfg.head_dim / 2]f32 = undefined;
        var fl: [cfg.head_dim / 2]f32 = undefined;
        cfg.inverseFrequencies(false, &sl);
        cfg.inverseFrequencies(true, &fl);
        m.inv_sliding = try m.alloc(sl.len * 4);
        m.inv_full = try m.alloc(fl.len * 4);
        if (m.planned == null) {
            try (DeviceBuffer{ .d = m.d, .ptr = m.inv_sliding, .len = sl.len * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(&sl));
            try (DeviceBuffer{ .d = m.d, .ptr = m.inv_full, .len = fl.len * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(&fl));
        }
        const r = @max(prompt_rows, round_rows);
        const h = geo.hidden;
        const qw = geo.queryWidth();
        const kw = geo.kvWidth();
        const most = @max(h, geo.intermediate);
        const wide: Window = .{ .h = geo.query_heads, .hk = geo.kv_heads, .d = cfg.head_dim, .win = 0, .ring = m.context, .scale = cfg.scale };
        const narrow: Window = .{ .h = geo.query_heads, .hk = geo.kv_heads, .d = cfg.head_dim, .win = cfg.window, .ring = ring, .scale = cfg.scale };
        const part = @max(cuda.window_attention.WindowAttention.scratchBytes(wide, attend_rows), cuda.window_attention.WindowAttention.scratchBytes(narrow, attend_rows));
        m.s = .{
            .ids = try m.alloc(r * 4),
            .pos = try m.alloc(r * 4),
            .slot = try m.alloc(r * 4),
            .res = try m.alloc(r * h * 4),
            .e = try m.alloc(r * h * 2),
            .n = try m.alloc(r * h * 2),
            .xs = try m.alloc(r * most / 64 * 4),
            .qkvg = try m.alloc(r * geo.qkvgWidth() * 2),
            .qr = try m.alloc(r * qw * 2),
            .kr = try m.alloc(r * kw * 2),
            .vr = try m.alloc(r * kw * 2),
            .att = try m.alloc(r * qw * 2),
            .ag = try m.alloc(r * qw * 2),
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

    /// Same-input affine-4 native projections share the GB10 grid; prompt and mixed widths keep their launches.
    pub fn projectMany(m: *const Model, k: *const kern.Kernels, s: cuda.Stream, ls: []const wts.Proj, x: u64, xs: u64, outs: []const u64, rows: usize, prompt: bool) !void {
        if (ls.len == 0 or ls.len != outs.len) return error.ShapeMismatch;
        const A = cuda.qlinear.Affine4Kernels;
        if (k.affine.pdl and !prompt and ls.len >= 2 and ls.len <= A.segs_max) grouped: {
            var many: [A.segs_max]A.Out = undefined;
            for (ls, outs, many[0..ls.len]) |l, out, *v| {
                if (l != .a4 or l.k() != ls[0].k()) break :grouped;
                v.* = .{ .q = l.a4, .out = out };
            }
            return k.affine.verifyManyRows(s, x, xs, many[0..ls.len], rows);
        }
        for (ls, outs) |l, out| try m.project(k, s, l, x, xs, out, rows, prompt);
    }

    /// Every chain's rows (a prompt chunk, or decode windows); rows [head0, head0 + heads)'s logits at s.logits.
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
        try k.embed.affine(s, x.ids, m.w.embed_w, m.w.embed_s, m.w.embed_b, x.e, rows, h, .{ .bits = @intCast(m.w.config.bits), .group = cfg.group });
        // the reference's residual is fp32 from the embedding on: res = 0 + e, then layer 0's input norm
        try (DeviceBuffer{ .d = m.d, .ptr = x.res, .len = rows * h * 4, .borrowed = true }).fill8(0, s.handle);
        try m.residual(k, s, x.e, m.w.layers[0].input_norm, rows);
        for (0..m.layers) |i| try m.layer(k, s, chains, i, rows, prompt);
        if (heads == 0) return;
        try m.project(k, s, m.w.head, x.n + head0 * h * 2, x.xs + head0 * (h / 64) * 4, x.logits, heads, false);
    }

    /// res += x (bf16 [rows, hidden]); s.n = bf16(norm(res) w), s.xs its 64-value sums.
    fn residual(m: *Model, k: *const kern.Kernels, s: cuda.Stream, x: u64, w: u64, rows: usize) !void {
        const h = m.geo.hidden;
        try k.norms.addRms(s, x, .bf16, m.s.res, null, w, m.s.n, .bf16, rows, h, cfg.eps);
        try k.act.groupSums(s, m.s.n, h, m.s.xs, rows, h);
    }

    /// Layer i on the rows: residual in s.res, its normed input (and sums) in s.n (s.xs); the same on return.
    fn layer(m: *Model, k: *const kern.Kernels, s: cuda.Stream, chains: []const Chain, i: usize, rows: usize, prompt: bool) !void {
        const geo = m.geo;
        const l = m.w.layers[i];
        const x = m.s;
        const is_full = cfg.full(i);
        const dh = cfg.head_dim;
        const qw = geo.queryWidth();
        const kw = geo.kvWidth();
        const ld: i64 = @intCast(geo.qkvgWidth());
        try m.project(k, s, l.qkvg, x.n, x.xs, x.qkvg, rows, prompt);
        const src: cuda.qk_rows.Src = .{ .q = x.qkvg, .q_ld = ld, .q_hs = dh, .k = x.qkvg + qw * 2, .k_ld = ld, .k_hs = dh, .v = x.qkvg + (qw + kw) * 2, .v_ld = ld, .v_hs = dh };
        const shape: cuda.qk_rows.Shape = .{ .inv = if (is_full) m.inv_full else m.inv_sliding, .rot = if (is_full) cfg.full_rotated else dh, .eps = cfg.eps };
        const len = m.cacheLen(i);
        const pg = if (is_full) m.pagesFor(null) else null; // paged: the rows kernel writes no cache
        try k.qk.rowsFrom(s, src, .{ .h = geo.query_heads, .hk = geo.kv_heads, .dim = dh }, shape, x.pos, x.slot, .{ .q = x.qr, .k = x.kr, .v = x.vr, .kc = if (pg != null) null else m.kc[i], .vc = if (pg != null) null else m.vc[i], .len = len }, rows);
        const w: Window = .{ .h = geo.query_heads, .hk = geo.kv_heads, .d = dh, .win = if (is_full) 0 else cfg.window, .ring = len, .scale = cfg.scale, .p32 = true, .pg = pg };
        // paged: `put` writes into the pages the bits the rows kernel would have written into a linear cache
        if (pg != null) try k.win.put(s, x.kr, x.vr, m.kc[i], m.vc[i], x.pos, x.slot, w, rows);
        if (prompt) {
            const c = chains[0];
            const off = if (pg != null) 0 else c.slot * len * kw * 2; // pages: the layer's region, the slot's row
            const pw: Window = if (is_full) .{ .h = w.h, .hk = w.hk, .d = w.d, .win = 0, .ring = 0, .scale = cfg.scale, .p32 = true, .pg = m.pagesFor(c.slot) } else w;
            try k.win.promptRows(s, x.qr, m.kc[i] + off, m.vc[i] + off, x.att, c.p0, rows, pw, null);
        } else {
            var r: usize = 0;
            while (r < rows) : (r += attend_rows) {
                const n = @min(attend_rows, rows - r);
                try k.win.decode(s, x.qr + r * qw * 2, m.kc[i], m.vc[i], x.att + r * qw * 2, x.pos + r * 4, x.slot + r * 4, w, x.part, n);
            }
        }
        try k.act.headGate(s, x.att, x.qkvg + (qw + 2 * kw) * 2, @intCast(ld), x.ag, x.xs, rows, geo.query_heads, dh);
        try m.project(k, s, l.out, x.ag, x.xs, x.o, rows, prompt);
        try m.residual(k, s, x.o, l.post_attention, rows);
        try m.projectMany(k, s, &.{ l.gate, l.up }, x.n, x.xs, &.{ x.gate, x.up }, rows, prompt);
        try k.act.gegluErf(s, x.gate, geo.intermediate, x.up, geo.intermediate, x.act, x.xs, rows, geo.intermediate);
        try m.project(k, s, l.down, x.act, x.xs, x.y, rows, prompt);
        const next = if (i + 1 < m.layers) m.w.layers[i + 1].input_norm else m.w.final_norm;
        try m.residual(k, s, x.y, next, rows);
    }
};

test "a stream's caches: three rings of 1,024 a full layer's context" {
    try std.testing.expectEqual(@as(usize, 1024), ring);
    const one = Model.streamBytes(cfg.spark4b, 4, 4096);
    try std.testing.expectEqual(@as(usize, 2 * (3 * 1024 + 4096) * 1024 * 2), one);
}
