//! Kolibri-1's layer graph over streams' serial chains of rows: prompt chunks, decode rows, one-row step graphs.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const Config = @import("config.zig").Config;
const wts = @import("weights.zig");
const kern = @import("kernels.zig");
const tree = @import("tree.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const grouped = cuda.grouped;
const experts = cuda.experts;
const TreeAttention = cuda.tree_attention.TreeAttention;
const Pages = cuda.window_attention.Pages;

/// Prompt rows a forward takes: the chunk grid, from position 0.
pub const prompt_rows = 8192;
/// Decode rows a forward takes over every stream's window.
pub const round_rows = 64;
const key_block = 64; // the window kernels' key tile
/// Positions a full layer's page holds: tree attention's key group (512), whole prompt tiles (64) and step rows (1).
pub const page_tokens = cuda.tree_attention.span;

/// Keys a sliding layer keeps: a chunk's rows written before its first row reads back its window.
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
    context: usize,
    slots: usize,
    ring: usize,
    max_rows: usize,
    groups: usize, // tree attention's key groups at the context and a full chain
    k: []u64, // per layer: the cache [slots, len, HK, D] bf16 (len: context for full layers, the ring for sliding)
    v: []u64, // paged: a full layer's K or V region of the arena, page id p at rows [p * page_tokens, ...)
    paged: bool = false, // full layers' caches are pages of the core pool (attach makes the arena); sliding stay rings
    pages: usize = 0, // pages in the arena
    table: u64 = 0, // paged: int32 [slots][perSlot()], slot s's page ids in position order
    ids: []u32 = &.{}, // paged: the tables on the host
    caps: []usize = &.{}, // paged: positions slot s's pages cover
    norms: []u64, // per layer: q_norm then k_norm, fp32 [2, D]
    cos: u64,
    sin: u64,
    origin: u64, // the bf16 base tree attention's cache offsets count from
    s: Scratch,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    planned: ?usize = null, // a plan: the bytes counted, nothing allocated

    const Scratch = struct { ids: u64, pos: u64, slot: u64, res: u64, x: u64, qkv: u64, q: u64, kn: u64, vn: u64, att: u64, o: u64, part: u64, logits_e: u64, pick: u64, wts: u64, plan: grouped.Plan, act: u64, y: u64, mo: u64, flat: u64, paths: u64, depths: u64, offs: u64, sel: u64, win: u64, tree: u64, ctab: u64 };

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
        if (m.planned != null) return p;
        try m.d.check(m.d.api.cuMemsetD8Async(p, 0, @max(bytes, 256), s.handle), "cuMemsetD8Async");
        return p;
    }

    /// Bytes one stream's caches take: full layers hold `context` positions, sliding layers the ring.
    pub fn streamBytes(c: Config, context: usize) usize {
        var total: usize = 0;
        for (c.full[0..c.layers]) |full| total += 2 * (if (full) context else ringSize(prompt_rows, c.window)) * c.kv_heads * c.head_dim * 2;
        return total;
    }

    /// Caches for `slots` streams of `context` positions, RoPE tables, scratch (paged: one slot until `attach`).
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, k: *const kern.Kernels, context: usize, slots: usize, max_rows: usize, paged: bool) !Model {
        var m = blank(gpa, d, w, context, if (paged) 1 else slots, max_rows, paged);
        errdefer m.deinit();
        try m.build(s, k);
        return m;
    }

    /// The device bytes `init` allocates for these sizes, counted by the same path without allocating.
    pub fn openBytes(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, k: *const kern.Kernels, context: usize, slots: usize, max_rows: usize, paged: bool) !usize {
        var m = blank(gpa, d, w, context, if (paged) 1 else slots, max_rows, paged);
        defer m.deinit();
        m.planned = 0;
        try m.build(s, k);
        return m.planned.?;
    }

    fn blank(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const wts.Weights, context: usize, slots: usize, max_rows: usize, paged: bool) Model {
        var c = w.config;
        c.layers = w.layers.len; // a model loaded with fewer layers (checks) runs those alone
        return .{ .gpa = gpa, .d = d, .w = w, .c = c, .context = context, .slots = slots, .ring = ringSize(prompt_rows, c.window), .max_rows = max_rows, .groups = TreeAttention.groups(context + tree.max_rows), .k = &.{}, .v = &.{}, .norms = &.{}, .cos = 0, .sin = 0, .origin = 0, .s = undefined, .paged = paged };
    }

    /// Table entries a slot: the pages its context takes.
    pub fn perSlot(m: *const Model) usize {
        return (m.context + page_tokens - 1) / page_tokens;
    }

    /// bf16 elements a page id steps in one full layer's K (or V) region: page_tokens rows of its kv heads.
    pub fn pageElems(m: *const Model) usize {
        return page_tokens * m.c.kv_heads * m.c.head_dim;
    }

    fn fullLayers(m: *const Model) usize {
        var n: usize = 0;
        for (m.c.full[0..m.c.layers]) |full| n += @intFromBool(full);
        return n;
    }

    /// Bytes a page holds over every full layer's K and V.
    pub fn pageBytes(m: *const Model) u64 {
        return m.fullLayers() * 2 * m.pageElems() * 2;
    }

    /// What the host's pool plans: a stream's sliding rings and table row fixed, a token its full layers' rows.
    pub fn layout(m: *const Model) kv.Layout {
        const c = m.c;
        var fixed: u64 = m.perSlot() * 4;
        for (c.full[0..c.layers]) |full| if (!full) {
            fixed += 2 * m.ring * c.kv_heads * c.head_dim * 2;
        };
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

    /// Tree attention's pages over `table`: a row of page ids for each stream of the call, in its order.
    fn treePages(m: *const Model, table: u64) ?cuda.tree_attention.Pages {
        if (!m.paged) return null;
        return .{ .table = table, .page = m.pageElems(), .tok = page_tokens, .stride = m.perSlot() };
    }

    /// Makes the arena of `pool` pages to lease, the rings and tables of `extra + 1` slots (no pages mapped).
    pub fn attach(m: *Model, s: cuda.Stream, extra: usize, pool: usize) !void {
        const c = m.c;
        if (!m.paged or m.pages != 0 or pool == 0 or extra >= round_rows) return error.BadAttach;
        for (c.full[0..c.layers], 0..) |full, i| if (!full) {
            m.forget(m.k[i]);
            m.forget(m.v[i]);
        };
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
        for (c.full[0..c.layers], 0..) |full, i| {
            if (full) {
                m.k[i] = arena + block * region;
                m.v[i] = m.k[i] + region;
                block += 2;
            } else try m.cacheOf(s, i);
        }
        try m.makeTables(s);
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

    /// Frees one buffer `alloc` made.
    fn forget(m: *Model, ptr: u64) void {
        for (m.buffers.items, 0..) |*b, i| if (b.ptr == ptr) {
            b.free();
            _ = m.buffers.swapRemove(i);
            return;
        };
    }

    /// Layer i's cache for every slot: a full layer's context, or a sliding layer's ring.
    fn cacheOf(m: *Model, s: cuda.Stream, i: usize) !void {
        const len = if (m.c.full[i]) m.context else m.ring;
        const bytes = m.slots * len * m.c.kv_heads * m.c.head_dim * 2;
        m.k[i] = try m.zeros(s, bytes);
        m.v[i] = try m.zeros(s, bytes);
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

    fn build(m: *Model, s: cuda.Stream, k: *const kern.Kernels) !void {
        const gpa = m.gpa;
        const d = m.d;
        const c = m.c;
        const w = m.w;
        const dh = c.head_dim;
        m.k = try gpa.alloc(u64, c.layers);
        m.v = try gpa.alloc(u64, c.layers);
        m.norms = try gpa.alloc(u64, c.layers);
        @memset(m.k, 0);
        @memset(m.v, 0);
        for (0..c.layers) |i| {
            if (!m.paged or !c.full[i]) try m.cacheOf(s, i); // paged full layers: attach's arena
            m.norms[i] = try m.alloc(2 * dh * 4);
            if (m.planned != null) continue;
            const L = w.layers[i];
            try d.check(d.api.cuMemcpyDtoDAsync_v2(m.norms[i], L.q_norm, dh * 4, s.handle), "cuMemcpyDtoDAsync");
            try d.check(d.api.cuMemcpyDtoDAsync_v2(m.norms[i] + dh * 4, L.k_norm, dh * 4, s.handle), "cuMemcpyDtoDAsync");
        }
        if (m.paged) try m.makeTables(s);
        try m.rope(s);
        m.origin = try m.zeros(s, 256);
        try m.scratch(k);
        if (m.planned == null) try s.synchronize();
    }

    pub fn deinit(m: *Model) void {
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
        m.gpa.free(m.k);
        m.gpa.free(m.v);
        m.gpa.free(m.norms);
        m.gpa.free(m.ids);
        m.gpa.free(m.caps);
    }

    /// RoPE tables: angle p / theta^(2j / D) in float64, cos and sin rounded to fp32, the halves repeated (neox).
    fn rope(m: *Model, s: cuda.Stream) !void {
        const dh = m.c.head_dim;
        if (m.planned != null) {
            m.cos = try m.alloc(m.context * dh * 4);
            m.sin = try m.alloc(m.context * dh * 4);
            return;
        }
        const cos = try m.gpa.alloc(f32, m.context * dh);
        defer m.gpa.free(cos);
        const sin = try m.gpa.alloc(f32, m.context * dh);
        defer m.gpa.free(sin);
        const theta: f64 = m.c.rope_theta;
        for (0..m.context) |p| for (0..dh / 2) |j| {
            const frac = @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(dh));
            const ang = @as(f64, @floatFromInt(p)) / std.math.pow(f64, theta, frac);
            for ([_]usize{ j, j + dh / 2 }) |col| {
                cos[p * dh + col] = @floatCast(@cos(ang));
                sin[p * dh + col] = @floatCast(@sin(ang));
            }
        };
        m.cos = try m.alloc(cos.len * 4);
        m.sin = try m.alloc(sin.len * 4);
        try m.upload(s, m.cos, std.mem.sliceAsBytes(cos));
        try m.upload(s, m.sin, std.mem.sliceAsBytes(sin));
        try s.synchronize(); // the host tables are freed on return
    }

    pub fn upload(m: *Model, s: cuda.Stream, dst: u64, bytes: []const u8) !void {
        try m.d.check(m.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, s.handle), "cuMemcpyHtoDAsync");
    }

    /// The lane's slice partials at most, over every row count a forward takes.
    fn partBytes(m: *const Model, k: *const kern.Kernels) usize {
        var most: usize = 0;
        for (m.w.layers) |L| for ([_]cuda.qlinear.Weight{ L.qkv, L.o }) |q| {
            var rows: usize = 1;
            while (rows <= m.max_rows) : (rows += 1) most = @max(most, k.lane.partBytes(rows, q.fp8g));
        };
        return most;
    }

    fn scratch(m: *Model, k: *const kern.Kernels) !void {
        const c = m.c;
        const r = m.max_rows;
        const dh = c.head_dim;
        const pairs = r * (c.top_k + 1);
        const count = c.experts + 1;
        const wide = pairs > grouped.small;
        const items = @max(grouped.maxItems(pairs, count, 16), grouped.maxItems(pairs, count, experts.promptTile(.fp8g)));
        const cap: usize = @min(r, round_rows);
        m.s = .{
            .ids = try m.alloc(r * 4),
            .pos = try m.alloc(r * 4),
            .slot = try m.alloc(r * 4),
            .res = try m.alloc(r * c.hidden * 4),
            .x = try m.alloc(r * c.hidden * 2),
            .qkv = try m.alloc(r * c.qkvDim() * 2),
            .q = try m.alloc(r * c.heads * dh * 2),
            .kn = try m.alloc(r * c.kv_heads * dh * 2),
            .vn = try m.alloc(r * c.kv_heads * dh * 2),
            .att = try m.alloc(r * c.heads * dh * 2),
            .o = try m.alloc(r * c.hidden * 2),
            .part = try m.alloc(m.partBytes(k)),
            .logits_e = try m.alloc(r * c.experts * 4),
            .pick = try m.alloc(pairs * 4),
            .wts = try m.alloc(pairs * 4),
            .plan = .{
                .members = try m.alloc(pairs * 4),
                .items = try m.alloc(items * 12),
                .counts = try m.alloc(8),
                .rank = try m.alloc((if (wide) pairs else 1) * 4),
                .hist = try m.alloc((if (wide) (pairs + 1023) / 1024 * count else 1) * 4),
            },
            .act = try m.alloc(pairs * c.moe_width * 2),
            .y = try m.alloc(pairs * c.hidden * 4),
            .mo = try m.alloc(r * c.hidden * 2),
            .flat = try m.alloc((2 * cap + 4 * m.slotCap()) * 4),
            .paths = try m.alloc(cap * tree.max_rows * 4),
            .depths = try m.alloc(cap * 4),
            .offs = try m.alloc(c.layers * m.slotCap() * 2 * 8),
            .sel = try m.alloc(r * c.hidden * 2),
            .tree = try m.alloc(TreeAttention.scratchBytes(cap, c.heads, c.kv_heads, m.groups, dh)),
            .win = try m.alloc(cuda.window_attention.WindowAttention.scratchBytes(m.window(), cap)),
            .ctab = if (m.paged) try m.alloc(m.slotCap() * m.perSlot() * 4) else 0,
        };
    }

    /// A sliding layer's keys: (pos - window, pos], key p at ring row p % ring.
    fn window(m: *const Model) cuda.window_attention.Window {
        const c = m.c;
        return .{ .h = c.heads, .hk = c.kv_heads, .d = c.head_dim, .win = c.window, .ring = m.ring, .scale = scale(c) };
    }

    fn scale(c: Config) f32 {
        return @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(c.head_dim))));
    }

    /// fp32 logits [rows.len, V] at `out` of the chains' rows `rows` (default: each chain's last).
    pub fn forward(m: *Model, o: kern.Ops, chains: []const Chain, prompt: bool, rows: ?[]const usize, out: u64) !void {
        const gpa = m.gpa;
        const s = o.s;
        if (prompt and chains.len != 1) return error.PromptTakesOneStream;
        if (!prompt and chains.len > m.slots) return error.TooManyStreams;
        var n: usize = 0;
        for (chains) |ch| {
            if (ch.tokens.len == 0) return error.EmptyChain;
            if (ch.tokens.len > m.ring - m.c.window) return error.ChainOverrunsRing;
            if (ch.p0 + ch.tokens.len > m.context) return error.PastContext;
            if (ch.slot >= m.slots) return error.NoSuchSlot;
            if (m.paged and ch.p0 + ch.tokens.len > m.caps[ch.slot]) return error.PastPages; // past its pages
            n += ch.tokens.len;
        }
        if (n > m.max_rows or (!prompt and n > round_rows)) return error.TooManyRows;
        const host = try gpa.alloc(i32, 3 * n);
        defer gpa.free(host);
        var at: usize = 0;
        for (chains) |ch| for (ch.tokens, 0..) |t, i| {
            host[at] = @intCast(t);
            host[n + at] = @intCast(ch.p0 + i);
            host[2 * n + at] = @intCast(ch.slot);
            at += 1;
        };
        try m.upload(s, m.s.ids, std.mem.sliceAsBytes(host[0..n]));
        try m.upload(s, m.s.pos, std.mem.sliceAsBytes(host[n .. 2 * n]));
        try m.upload(s, m.s.slot, std.mem.sliceAsBytes(host[2 * n ..]));
        var t: ?cuda.tree_attention.Rows = null;
        var ng: usize = 0;
        var plan: ?tree.Plan = null;
        defer if (plan) |p| gpa.free(p.flat);
        if (!prompt) {
            const windows = try gpa.alloc(tree.Window, chains.len);
            defer gpa.free(windows);
            var most: usize = 0;
            for (chains, windows) |ch, *win| {
                win.* = .{ .rows = ch.tokens.len, .committed = ch.p0 };
                most = @max(most, ch.p0 + ch.tokens.len);
            }
            plan = try tree.build(gpa, windows);
            try m.upload(s, m.s.flat, std.mem.sliceAsBytes(plan.?.flat));
            t = m.treeRows(m.s.flat, plan.?, m.s.paths, m.s.depths, m.s.offs, m.treePages(m.s.ctab));
            if (m.paged) try m.callTable(s, chains);
            try o.k.tree.paths(s, m.s.flat + plan.?.parentsAt(), m.s.paths, m.s.depths, n);
            try m.offsets(s, chains, m.s.offs);
            ng = TreeAttention.groups(most);
        }
        var ends: std.ArrayList(usize) = .empty;
        defer ends.deinit(gpa);
        const pick = rows orelse blk: {
            var e: usize = 0;
            for (chains) |ch| {
                e += ch.tokens.len;
                try ends.append(gpa, e - 1);
            }
            break :blk ends.items;
        };
        for (pick) |r| if (r >= n) return error.NoSuchRow;
        try m.launch(o, n, if (prompt) chains[0] else null, t, chains.len, ng, @min(m.max_rows, round_rows), pick, out);
    }

    fn treeRows(m: *const Model, base: u64, p: tree.Plan, paths: u64, depths: u64, offs: u64, pages: ?cuda.tree_attention.Pages) cuda.tree_attention.Rows {
        return .{ .row_stream = base, .streams = base + p.streamsAt(), .offs = offs, .origin = m.origin, .paths = paths, .depths = depths, .pages = pages };
    }

    /// Paged: the chains' slots' rows of page ids into the call's table, in chain order (tree attention's streams).
    fn callTable(m: *Model, s: cuda.Stream, chains: []const Chain) !void {
        const per = m.perSlot();
        const rows = try m.gpa.alloc(u32, chains.len * per);
        defer m.gpa.free(rows);
        for (chains, 0..) |ch, j| @memcpy(rows[j * per ..][0..per], m.ids[ch.slot * per ..][0..per]);
        try m.upload(s, m.s.ctab, std.mem.sliceAsBytes(rows));
    }

    /// Each full layer's cache offsets a chain, K then V, in bf16 elements from `origin`: [layer][chain][2].
    fn offsets(m: *Model, s: cuda.Stream, chains: []const Chain, dst: u64) !void {
        const c = m.c;
        const per = m.context * c.kv_heads * c.head_dim * 2;
        const offs = try m.gpa.alloc(i64, c.layers * 2 * chains.len);
        defer m.gpa.free(offs);
        for (0..c.layers) |i| for (chains, 0..) |ch, j| {
            const at = if (m.paged and c.full[i]) 0 else ch.slot * per; // paged: the layer's region, the table's rows
            offs[(i * chains.len + j) * 2] = @intCast((m.k[i] + at - m.origin) / 2);
            offs[(i * chains.len + j) * 2 + 1] = @intCast((m.v[i] + at - m.origin) / 2);
        };
        try m.upload(s, dst, std.mem.sliceAsBytes(offs));
    }

    /// The layers over staged inputs, no host copies (a graph can hold them); `prompt_chain` set: a prompt chunk.
    fn launch(m: *Model, o: kern.Ops, n: usize, prompt_chain: ?Chain, t: ?cuda.tree_attention.Rows, streams: usize, ng: usize, cap: usize, pick: []const usize, out: u64) !void {
        const c = m.c;
        const s = o.s;
        const prompt = prompt_chain != null;
        const w = m.w;
        const h = c.heads;
        const hk = c.kv_heads;
        const dh = c.head_dim;
        const lin: cuda.qlinear.Linear = .{ .lane = &o.k.lane };
        try o.embed(w.embed, m.s.ids, n, c.hidden, m.s.x, m.s.res);
        try o.rms(m.s.res, w.layers[0].input_norm, m.s.x, n, c.hidden, c.eps);
        const slots_e = c.top_k + 1;
        const pairs = n * slots_e;
        const tile = if (prompt) experts.promptTile(.fp8g) else 16;
        const items = grouped.maxItems(pairs, c.experts + 1, tile);
        const gx: experts.Grouped = .{ .fp8g = &o.k.fp8x };
        for (w.layers, 0..) |L, i| {
            try (if (prompt) lin.prompt(s, L.qkv, .{ .x = m.s.x, .part = m.s.part }, m.s.qkv, n) else lin.decode(s, L.qkv, .{ .x = m.s.x, .part = m.s.part }, m.s.qkv, n));
            const len = if (c.full[i]) m.context else m.ring;
            const paged = c.full[i] and m.paged; // the rows kernel writes no cache: `put` copies its k and v rows' bits
            try o.qkv(.{ .rows = m.s.qkv, .norms = m.norms[i], .cos = m.cos, .sin = m.sin, .pos = m.s.pos, .slots = m.s.slot, .q = m.s.q, .k = m.s.kn, .v = m.s.vn, .kc = if (paged) 0 else m.k[i], .vc = if (paged) 0 else m.v[i], .len = len }, n, h, hk, dh, c.eps, !c.full[i]);
            if (paged) {
                var pw = m.window();
                pw.win = 0;
                pw.pg = m.pagesFor(null);
                try o.put(m.s.kn, m.s.vn, m.k[i], m.v[i], m.s.pos, m.s.slot, pw, n);
            }
            if (prompt_chain) |ch| {
                const per = if (paged) 0 else len * hk * dh * 2; // a stream's rows (paged: pages)
                var win = m.window();
                if (c.full[i]) {
                    win.win = 0; // every key before the row
                    win.ring = m.context;
                    win.pg = m.pagesFor(ch.slot);
                }
                try o.promptRows(m.s.q, m.k[i] + ch.slot * per, m.v[i] + ch.slot * per, m.s.att, ch.p0, n, win);
            } else if (!c.full[i]) {
                try o.sliding(m.s.q, m.k[i], m.v[i], m.s.att, m.s.pos, m.s.slot, n, m.window(), m.s.win);
            } else {
                var rows = t.?;
                rows.offs += i * streams * 16;
                try o.full(m.s.q, m.s.kn, m.s.vn, rows, m.s.tree, cap, ng, m.s.att, n, h, hk, dh, scale(c));
            }
            try (if (prompt) lin.prompt(s, L.o, .{ .x = m.s.att, .part = m.s.part }, m.s.o, n) else lin.decode(s, L.o, .{ .x = m.s.att, .part = m.s.part }, m.s.o, n));
            try o.addRms(m.s.o, m.s.res, L.post_attn_norm, L.pre_moe_norm, m.s.x, n, c.hidden, c.eps);
            // routed experts by sigmoid + bias, the shared expert as one more slot of weight 1, then the weighted sum
            try o.dense(m.s.x, c.hidden, L.router, m.s.logits_e, n, c.hidden, c.experts);
            try o.topk(m.s.logits_e, L.bias, m.s.pick, m.s.wts, n, c.experts, c.top_k);
            try o.k.router.route(s, m.s.pick, pairs, c.experts + 1, tile, m.s.plan);
            const up_in: experts.Rows = .{ .x = m.s.x, .stride = c.hidden, .slots = slots_e };
            const down_in: experts.Rows = .{ .x = m.s.act, .stride = c.moe_width };
            if (prompt) {
                try gx.prompt(s, .up, up_in, L.experts, m.s.plan, m.s.act, items, -1);
                try gx.prompt(s, .down_bf16, down_in, L.experts, m.s.plan, m.s.y, items, -1);
            } else {
                try gx.decode(s, .up, up_in, L.experts, m.s.plan, m.s.act, items, -1);
                try gx.decode(s, .down_f32, down_in, L.experts, m.s.plan, m.s.y, items, -1);
            }
            try o.combine(m.s.y, prompt, m.s.wts, m.s.mo, n, slots_e, c.hidden);
            const after = if (i + 1 < w.layers.len) w.layers[i + 1].input_norm else w.norm;
            try o.addRms(m.s.mo, m.s.res, L.post_moe_norm, after, m.s.x, n, c.hidden, c.eps);
        }
        for (pick, 0..) |r, j| try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(m.s.sel + j * c.hidden * 2, m.s.x + r * c.hidden * 2, c.hidden * 2, s.handle), "cuMemcpyDtoDAsync");
        try o.dense(m.s.sel, c.hidden, w.head, out, pick.len, c.hidden, c.vocab);
    }

    /// One stream's one-row step as a CUDA graph: its stream row, paths and offsets fixed; a step writes token and row.
    pub const Step = struct { exec: cuda.graph.Exec, slot: usize, flat: u64, logits: u64 };

    pub fn captureStep(m: *Model, o: kern.Ops, slot: usize) !Step {
        const c = m.c;
        const s = o.s;
        const p = try tree.build(m.gpa, &.{.{ .rows = 1, .committed = 0 }});
        defer m.gpa.free(p.flat);
        const flat = try m.alloc(p.flat.len * 4);
        try m.upload(s, flat, std.mem.sliceAsBytes(p.flat));
        const paths = try m.alloc(tree.max_rows * 4);
        const depths = try m.alloc(4);
        try o.k.tree.paths(s, flat + p.parentsAt(), paths, depths, 1); // a lone root: its path never changes
        const offs = try m.alloc(c.layers * 16);
        try m.offsets(s, &.{.{ .slot = slot, .p0 = 0, .tokens = &.{} }}, offs);
        const pages = if (m.pagesFor(slot)) |pg| m.treePages(pg.table) else null; // the slot's own row: stream 0's
        const logits = try m.alloc(c.vocab * 4);
        try s.synchronize();
        try cuda.graph.beginCapture(s, .thread_local);
        m.launch(o, 1, null, m.treeRows(flat, p, paths, depths, offs, pages), 1, m.groups, 1, &.{0}, logits) catch |e| {
            if (cuda.graph.endCapture(s)) |g| {
                var gg = g;
                gg.deinit();
            } else |_| {}
            return e;
        };
        var g = try cuda.graph.endCapture(s);
        defer g.deinit();
        return .{ .exec = try g.instantiate(), .slot = slot, .flat = flat, .logits = logits };
    }

    /// The step: `token` at position `p0` (the cache length before it) through the captured graph.
    pub fn step(m: *Model, o: kern.Ops, g: *const Step, token: u32, p0: usize) !void {
        const s = o.s;
        if (p0 + 1 > m.context) return error.PastContext;
        if (m.paged and p0 + 1 > m.caps[g.slot]) return error.PastPages;
        const staged = [3]i32{ @intCast(token), @intCast(p0), @intCast(g.slot) };
        try m.upload(s, m.s.ids, std.mem.asBytes(&staged[0]));
        try m.upload(s, m.s.pos, std.mem.asBytes(&staged[1]));
        try m.upload(s, m.s.slot, std.mem.asBytes(&staged[2]));
        const row = tree.stepRow(p0);
        try m.upload(s, g.flat + 4, std.mem.asBytes(&row));
        try g.exec.launchOn(s);
    }
};

test "the ring holds a prompt chunk and a window, in whole key tiles" {
    try std.testing.expectEqual(@as(usize, 8768), ringSize(prompt_rows, 513));
    try std.testing.expect(ringSize(prompt_rows, 513) - 513 >= prompt_rows);
}
