//! Qwen3.8-27B decode on device: one token through all layers (bf16 residual stream) to a greedy token id.

const std = @import("std");
const rt = @import("xpu").rt;
const ld = @import("xpu").loader;
const cq = @import("xpu_config.zig");
const qb = @import("xpu_blocks.zig");
const ql = @import("xpu_load.zig");
const qw = @import("xpu_win.zig");
const stop = @import("xpu").stop;

const Layer = qw.Layer;

/// Size of the largest recurrent state (GDN delta-rule state, fp32); a zero buffer this long clears any of them.
pub const state_bytes: usize = @as(usize, qb.gdn_heads) * 128 * 128 * 4;

pub const Model = struct {
    r: *rt.Runtime,
    cfg: cq.Config,
    ops: *qb.Ops,
    layers: []Layer,
    /// Multi-row window state (see qwen_win.zig); created at load.
    win: *qw.Win,
    emb: qb.Table,
    norm_f: qb.Buf,
    head: qb.Table,
    pos: u32 = 0,
    trace: bool = false,
    /// Print the per-layer residual statistics of a traced token.
    print_trace: bool = false,
    /// Round the lm_head logits to bf16 before argmax and readback, as upstream.
    bf16_logits: bool = false,
    /// Receives the residual stream (bf16) of a traced token when set: row 0 the embedding, row i + 1 after layer i.
    dump: ?[]u16 = null,

    pub fn load(gpa: std.mem.Allocator, r: *rt.Runtime, l: *ld.Loader, cfg: cq.Config, cap: u32) !Model {
        stop.install();
        const ops = try gpa.create(qb.Ops);
        ops.* = try qb.Ops.init(r, l, cfg, cap);
        try ql.attach(gpa, ops, l, cfg);
        const t = cfg.text_config;
        const m: Model = .{
            .win = undefined,
            .r = r,
            .cfg = cfg,
            .ops = ops,
            .layers = try gpa.alloc(Layer, t.num_hidden_layers),
            .emb = try ql.loadEmbed(gpa, ops, l, t.vocab_size),
            .norm_f = try ql.loadFinalNorm(gpa, ops, l),
            .head = try ql.loadHead(gpa, ops, l, t.vocab_size),
        };
        for (m.layers, 0..) |*ly, i| {
            ly.mixer = if (cfg.isAttention(i)) .{ .attn = try ql.loadAttn(gpa, ops, l, i, cap) } else .{ .gdn = try ql.loadGdn(gpa, ops, l, i) };
            ly.mlp = try ql.loadMlp(gpa, ops, l, i);
            std.debug.print("\rloaded layer {d}/{d}, {d:.2} GB on device", .{ i + 1, t.num_hidden_layers, @as(f64, @floatFromInt(l.total)) / 1e9 });
        }
        std.debug.print("\n", .{});
        const win = try gpa.create(qw.Win);
        win.* = try qw.Win.init(gpa, ops, l, cfg);
        var mm = m;
        mm.win = win;
        return mm;
    }

    /// Prints finiteness and range of the residual stream after a layer, and copies it to `dump` when set.
    fn traceLayer(m: *Model, i: usize) !void {
        try m.ops.flush();
        var x: [qb.hidden]u16 = undefined;
        try m.r.download(std.mem.sliceAsBytes(&x), m.ops.s.x);
        try m.r.sync();
        var bad: u32 = 0;
        var xmax: f32 = 0;
        var ss: f64 = 0;
        for (x) |xv| {
            const xf: f32 = @bitCast(@as(u32, xv) << 16);
            if (!std.math.isFinite(xf)) bad += 1 else {
                xmax = @max(xmax, @abs(xf));
                ss += @as(f64, xf) * xf;
            }
        }
        const kind: []const u8 = if (m.cfg.isAttention(i)) "attention" else "deltanet";
        if (m.print_trace) std.debug.print("  layer {d:>2} {s:<9} x max {d:>10.3} rms {d:>9.3}  nonfinite {d}\n", .{ i, kind, xmax, @sqrt(ss / @as(f64, qb.hidden)), bad });
        if (m.dump) |d| @memcpy(d[(i + 1) * qb.hidden ..][0..qb.hidden], &x);
    }

    fn dumpEmbedding(m: *Model) !void {
        try m.r.download(std.mem.sliceAsBytes(m.dump.?[0..qb.hidden]), m.ops.s.x);
        try m.r.sync();
    }

    /// Queues one token through every layer (and the head when `logits`); the caller syncs. Advances the position.
    pub fn forward(m: *Model, token: u32, logits: bool) !void {
        if (stop.requested()) return m.interrupted();
        if (m.pos >= m.ops.cap) return error.ContextFull;
        const o = m.ops;
        try o.embed(m.emb, token);
        try o.setPosition(m.pos);
        if (m.trace and m.dump != null) try m.dumpEmbedding();
        for (m.layers, 0..) |ly, i| {
            switch (ly.mixer) {
                .gdn => |w| try o.gdn(w),
                .attn => |w| try o.attention(w, m.pos),
            }
            try o.mlp(ly.mlp);
            if (m.trace) try m.traceLayer(i);
        }
        if (logits) try o.head(m.norm_f, m.head, m.cfg.text_config.vocab_size, m.bf16_logits);
        m.pos += 1;
    }

    /// A graceful stop was asked for (signal or stop file): drain the queue so device memory is freed on exit.
    fn interrupted(m: *Model) error{Interrupted}!void {
        m.r.sync() catch {};
        return error.Interrupted;
    }

    /// Back to an empty context: position 0, GDN states zeroed (KV read only up to pos); `zeros` >= `state_bytes`.
    pub fn reset(m: *Model, zeros: []const u8) !void {
        for (m.layers) |ly| switch (ly.mixer) {
            .gdn => |w| {
                try m.r.upload(w.cstate, zeros[0..w.cstate.len]);
                try m.r.upload(w.sstate, zeros[0..w.sstate.len]);
            },
            .attn => {},
        };
        m.pos = 0;
        try m.r.sync();
    }

    /// Runs `tokens` (1..16) as one window (qwen_win.Win.forward); last `head_rows` rows get logits; else commitRows.
    pub fn forwardRows(m: *Model, tokens: []const u32, head_rows: u32, inplace: bool) !void {
        if (stop.requested()) return m.interrupted();
        if (m.pos + tokens.len > m.ops.cap) return error.ContextFull;
        try m.win.forward(m.layers, m.emb, m.norm_f, m.head, tokens, m.pos, inplace, head_rows, m.bf16_logits);
        if (inplace) m.pos += @intCast(tokens.len);
    }

    /// lm_head over n <= 16 hidden rows from row0 of the last window run with qwen_win.keep_hidden.
    pub fn headBatch(m: *Model, row0: u32, n: u32) !void {
        try m.win.headBatch(m.head, row0, n, m.bf16_logits);
    }

    /// Keeps the first c rows of the last non-inplace window and advances the position by c.
    pub fn commitRows(m: *Model, c: u32) !void {
        try m.win.commit(m.layers, c);
        m.pos += c;
    }

    /// Greedy token of each head row of the last window (syncs).
    pub fn rowArgmax(m: *Model, out: []i32) !void {
        try m.r.download(std.mem.sliceAsBytes(out[0..m.win.head_rows]), m.win.am_out);
        try m.r.sync();
    }

    /// fp32 logits [head_rows][vocab] of the last window (syncs).
    pub fn fetchRowLogits(m: *Model, out: []f32) !void {
        try m.r.download(std.mem.sliceAsBytes(out[0 .. m.win.head_rows * m.win.vocab]), m.win.logits);
        try m.r.sync();
    }

    /// The greedy token of the last forward with logits (syncs).
    pub fn argmax(m: *Model) !u32 {
        return m.ops.argmax();
    }

    /// Copies the fp32 logits of the last forward with logits to `out` (syncs).
    pub fn fetchLogits(m: *Model, out: []f32) !void {
        try m.r.download(std.mem.sliceAsBytes(out), m.ops.s.logits);
        try m.r.sync();
    }
};
