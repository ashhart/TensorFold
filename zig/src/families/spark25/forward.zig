//! Spark2.5 on CUDA: prompt chunks and decode windows through the shared operators, the reference's fp32 residual.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const kern = @import("kernels.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Window = cuda.window_attention.Window;

/// Rows a prompt chunk takes.
pub const prompt_rows = 512;
/// Rows a decode forward takes over every stream's window.
pub const round_rows = 128;
/// Rows a decode attention call takes: its partials' scratch is sized for this many.
const attend_rows = 16;
/// A sliding layer's ring: the window and a prompt chunk, so a chunk's writes never reach keys its first row reads.
pub const ring = cfg.window + prompt_rows;

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
    vc: []u64,
    inv_sliding: u64, // fp32 [128]
    inv_full: u64, // fp32 [128], zero past the rotated 32 pairs
    s: Scratch,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    planned: ?usize = null, // a plan: the bytes counted, nothing allocated
    /// Decode forwards of n rows as CUDA graphs, captured at their first use ([n]; null: not yet, or graphs off).
    graphs: [round_rows + 1]?cuda.graph.Exec = @splat(null),
    graphed: bool = true,

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

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, context: usize, slots: usize) !Model {
        var m = blank(gpa, d, w, context, slots);
        errdefer m.deinit();
        try m.build(s);
        return m;
    }

    /// The device bytes `init` allocates for these sizes, counted by the same path without allocating.
    pub fn openBytes(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, context: usize, slots: usize) !usize {
        var m = blank(gpa, d, w, context, slots);
        defer m.deinit();
        m.planned = 0;
        try m.build(s);
        return m.planned.?;
    }

    fn blank(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const wts.Weights, context: usize, slots: usize) Model {
        return .{ .gpa = gpa, .d = d, .w = w, .geo = w.config.geo, .layers = w.layers.len, .context = context, .slots = slots, .kc = &.{}, .vc = &.{}, .inv_sliding = 0, .inv_full = 0, .s = undefined };
    }

    fn build(m: *Model, s: cuda.Stream) !void {
        const geo = m.geo;
        m.kc = try m.gpa.alloc(u64, m.layers);
        m.vc = try m.gpa.alloc(u64, m.layers);
        for (0..m.layers) |i| {
            const bytes = m.slots * m.cacheLen(i) * geo.kvWidth() * 2;
            m.kc[i] = try m.zeros(s, bytes);
            m.vc[i] = try m.zeros(s, bytes);
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
        for (&m.graphs) |*g| if (g.*) |*e| e.deinit();
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
        m.gpa.free(m.kc);
        m.gpa.free(m.vc);
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

    /// Every chain's rows (a prompt chunk, or decode windows); rows [head0, head0 + heads)'s logits at s.logits.
    pub fn forward(m: *Model, k: *const kern.Kernels, s: cuda.Stream, chains: []const Chain, prompt: bool, head0: usize, heads: usize) !void {
        var rows: usize = 0;
        for (chains) |c| {
            if (c.tokens.len == 0 or c.slot >= m.slots or c.p0 + c.tokens.len > m.context) return error.BadChain;
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
        try k.qk.rowsFrom(s, src, .{ .h = geo.query_heads, .hk = geo.kv_heads, .dim = dh }, shape, x.pos, x.slot, .{ .q = x.qr, .k = x.kr, .v = x.vr, .kc = m.kc[i], .vc = m.vc[i], .len = len }, rows);
        const w: Window = .{ .h = geo.query_heads, .hk = geo.kv_heads, .d = dh, .win = if (is_full) 0 else cfg.window, .ring = len, .scale = cfg.scale, .p32 = true };
        if (prompt) {
            const c = chains[0];
            const off = c.slot * len * kw * 2;
            const pw: Window = if (is_full) .{ .h = w.h, .hk = w.hk, .d = w.d, .win = 0, .ring = 0, .scale = cfg.scale, .p32 = true } else w;
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
        try m.project(k, s, l.gate, x.n, x.xs, x.gate, rows, prompt);
        try m.project(k, s, l.up, x.n, x.xs, x.up, rows, prompt);
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
