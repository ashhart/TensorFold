//! LFM3 forward on the Intel GPU: MLX 4-bit weights on the device, a conv state and KV rings, windows of rows, no DPAS.

const std = @import("std");
const xpu = @import("xpu");
const cfg = @import("config.zig");
const names = @import("names.zig");
const rt = xpu.rt;
const ld = xpu.loader;

const Buf = rt.Buffer;
const H: u32 = cfg.hidden;
const FF: u32 = cfg.intermediate;
const V: u32 = cfg.vocab;
const NQ: u32 = cfg.query_heads;
const NKV: u32 = cfg.kv_heads;
const HD: u32 = cfg.head_dim;
const QKV: u32 = (NQ + 2 * NKV) * HD;
const chunk: u32 = 128; // keys a split-K work-group scans
const argmax_parts: u32 = 64;
/// Without DPAS (Gen12LP) prompt projections take nem_prow_gen12: 4 outputs a sub-group, as 8 x 8 spills on Gen12LP.
const gen12 = !xpu.kernels.dpas;
const tile_rows: u32 = 8;
const tile_outs: u32 = if (gen12) 4 else 8;

pub const Table = struct { w: Buf, s: Buf, b: Buf };
const Conv = struct { in: Table, out: Table, cw: Buf };
const Attn = struct { q: Table, k: Table, v: Table, g: Table, o: Table, qn: Buf, kn: Buf, ring: u32, window: u32 };
const Mixer = union(enum) { conv: Conv, attn: Attn };
const Layer = struct { op_norm: Buf, post_op: Buf, ffn_norm: Buf, post_ffn: Buf, w1: Table, w3: Table, w2: Table, mixer: Mixer };

const Kernels = struct {
    embed: rt.Kernel,
    embed_tok: rt.Kernel,
    conv_hist: rt.Kernel,
    qmv: rt.Kernel,
    head: rt.Kernel,
    am_part: rt.Kernel,
    am_fin: rt.Kernel,
    pdense: rt.Kernel,
    rms: rt.Kernel,
    pan: rt.Kernel,
    conv: rt.Kernel,
    rope: rt.Kernel,
    vst: rt.Kernel,
    part: rt.Kernel,
    merge: rt.Kernel,
    swiglu: rt.Kernel,
    qkvg: rt.Kernel,
    w13: rt.Kernel,
    rope_store: rt.Kernel,
    pgemm: rt.Kernel,
};

/// One stream's state: each conv layer's last six B*x rows, each attention layer's KV slots, and its position.
pub const Seq = struct {
    conv: [cfg.layers]Buf = undefined,
    kc: [cfg.layers]Buf = undefined,
    vc: [cfg.layers]Buf = undefined,
    pos: u32 = 0,
    /// Each conv layer's state before the last verified window, then its rows' B * x: keepRows' source.
    ext: [cfg.layers]Buf = undefined,
    /// The last verified window still to keep: its first position and rows (0: none).
    win_start: u32 = 0,
    win_rows: u32 = 0,
    /// The engine's slot for this stream (0 is the engine's own).
    slot: u32 = 0,
};

/// Rows of a verified window at most (all streams' rows of a shared forward), and streams a shared forward takes.
pub const window_rows: u32 = 16;
pub const max_streams: u32 = 8;

/// A stream's rows in one forward: `ids` at the stream's own position (its window, or one row of several streams).
pub const Seg = struct { q: *Seq, ids: []const u32 };
/// What a forward leaves: nothing, the last row's logits and argmax (a prompt), or every row's (verified windows).
pub const Out = enum { none, last, rows };

/// A kept prompt state at `at`: conv rows, the full layers' first `at` KV slots, the sliding rings, in one buffer.
pub const Snap = struct { at: u32, buf: Buf };

const conv_bytes: usize = 6 * H * 2;
const ext_bytes: usize = (6 + window_rows) * H * 2;
const slot_bytes: usize = NKV * HD * 2;

/// `b` from byte `off` on.
fn from(b: Buf, off: usize) Buf {
    return .{ .rt = b.rt, .ptr = @ptrFromInt(@intFromPtr(b.ptr.?) + off), .len = b.len - off };
}

/// Device scratch for a window of up to `max_rows` rows.
const Scratch = struct { ids: Buf, x: Buf, xn: Buf, d: Buf, proj: Buf, qkv: Buf, g: Buf, q: Buf, att: Buf, a: Buf, b: Buf, act: Buf, po: Buf, pm: Buf, pl: Buf, logits: Buf, am_scratch: Buf, am_out: Buf };

/// Called after each layer (index) and after the final norm (-1) with the window's rows, bf16 [n][2048].
pub const Hook = struct { ctx: *anyopaque, f: *const fn (ctx: *anyopaque, layer: i32, rows: []const u16) void };

fn run(k: *rt.Kernel, groups: [3]u32, a: anytype) !void {
    inline for (a, 0..) |v, i| {
        const T = @TypeOf(v);
        if (T == Buf) {
            try k.setBuffer(i, v);
        } else if (T == f32) {
            try k.setF32(i, v);
        } else try k.setU32(i, @as(u32, v));
    }
    try k.launch(groups);
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    r: *rt.Runtime,
    layers: []Layer,
    emb: Table,
    norm_f: Buf,
    inv_full: Buf,
    inv_slide: Buf,
    k: Kernels,
    s: Scratch,
    host_rows: []u16,
    cap: u32,
    max_rows: u32,
    maxc: u32,
    pos: u32 = 0,
    device_bytes: u64 = 0,
    /// The bound stream's state (`own` until another is bound); `pos` is the bound stream's position.
    seq: *Seq = undefined,
    own: *Seq = undefined,
    /// Fused launches by bit (1 w1+w3+SwiGLU, 2 q/k/v/gate, 4 rope and KV store); LFM3_FUSE sets them for A/B.
    fuse: u32 = 7,
    /// In forwardPrompt: projections take pgemm, whose rows are the same however the prompt is cut.
    prompt: bool = false,
    /// The MLX group size of the projections.
    qg: u32 = 64,

    /// The affine tensors of module `m` (in layer `layer`, or a model tensor) by names.zig in namespace `ns`.
    fn table(l: *ld.Loader, ns: names.Namespace, layer: ?usize, m: names.Module) !Table {
        var a: [160]u8 = undefined;
        var b: [176]u8 = undefined;
        const base = try names.moduleName(&a, ns, layer, m);
        return .{
            .w = try l.load(try std.fmt.bufPrint(&b, "{s}.weight", .{base})),
            .s = try l.load(try std.fmt.bufPrint(&b, "{s}.scales", .{base})),
            .b = try l.load(try std.fmt.bufPrint(&b, "{s}.biases", .{base})),
        };
    }

    fn vec(l: *ld.Loader, ns: names.Namespace, layer: ?usize, m: names.Module) !Buf {
        var a: [160]u8 = undefined;
        return l.load(try names.moduleName(&a, ns, layer, m));
    }

    /// The checkpoint's namespace (bare `model.` or wrapped `language_model.model.`) from the embedding it holds.
    fn namespace(l: *const ld.Loader) !names.Namespace {
        var a: [64]u8 = undefined;
        var b: [64]u8 = undefined;
        return names.detect(l.map.contains(try names.embedName(&a, .bare)), l.map.contains(try names.embedName(&b, .wrapped)));
    }

    /// Uploads the checkpoint: `cap` positions on full-attention layers, windows of at most `max_rows` rows.
    pub fn load(gpa: std.mem.Allocator, r: *rt.Runtime, l: *ld.Loader, c: *const cfg.Config, cap: u32, max_rows: u32) !Model {
        if (max_rows == 0 or max_rows > 256 or cap < max_rows) return error.Invalid;
        var bm = try r.module(xpu.kernels.basic);
        var am = try r.module(xpu.kernels.attn);
        var pm = try r.module(if (gen12) xpu.kernels.nem_prow_gen12 else xpu.kernels.nem_prow);
        var fm = try r.module(xpu.kernels.lfm3);
        var vm = try r.module(xpu.kernels.lfm3_mv);
        const k: Kernels = .{
            .embed = try bm.kernel("embed4", .{ 64, 1, 1 }),
            .embed_tok = try bm.kernel("embed4_tok", .{ 64, 1, 1 }),
            .conv_hist = try fm.kernel("lfm3_conv_hist", .{ 64, 1, 1 }),
            .qmv = try am.kernel("qmv4_bf", .{ 64, 1, 1 }),
            .head = try am.kernel("qmv4_f32", .{ 64, 1, 1 }),
            .am_part = try am.kernel("argmax_partial", .{ 256, 1, 1 }),
            .am_fin = try am.kernel("argmax_final", .{ 256, 1, 1 }),
            .pdense = try pm.kernel("pdense", .{ 16 * (16 / tile_outs), 1, 1 }),
            .rms = try fm.kernel("lfm3_rmsnorm", .{ 256, 1, 1 }),
            .pan = try fm.kernel("lfm3_post_add_norm", .{ 256, 1, 1 }),
            .conv = try fm.kernel("lfm3_conv_rows", .{ 64, 1, 1 }),
            .rope = try fm.kernel("lfm3_qk_norm_rope", .{ 64, 1, 1 }),
            .vst = try fm.kernel("lfm3_v_store", .{ 64, 1, 1 }),
            .part = try fm.kernel("lfm3_attn_partial", .{ 64, 1, 1 }),
            .merge = try fm.kernel("lfm3_attn_merge_gate", .{ 16, 1, 1 }),
            .swiglu = try fm.kernel("lfm3_swiglu", .{ 64, 1, 1 }),
            .qkvg = try vm.kernel("qmv4_bf_x4", .{ 64, 1, 1 }),
            .w13 = try vm.kernel("qmv4_swiglu", .{ 64, 1, 1 }),
            .rope_store = try fm.kernel("lfm3_rope_store", .{ 64, 1, 1 }),
            .pgemm = try vm.kernel("pgemm", .{ 16, 16, 1 }),
        };
        const maxc = (cap + chunk - 1) / chunk;
        const w: usize = max_rows;
        const s: Scratch = .{
            .ids = try l.empty(w * 4),
            .x = try l.empty(w * H * 2),
            .xn = try l.empty(w * H * 2),
            .d = try l.empty(w * H * 2),
            .proj = try l.empty(w * 3 * H * 2),
            .qkv = try l.empty(w * QKV * 2),
            .g = try l.empty(w * NQ * 2),
            .q = try l.empty(w * NQ * HD * 2),
            .att = try l.empty(w * NQ * HD * 2),
            .a = try l.empty(w * FF * 2),
            .b = try l.empty(w * FF * 2),
            .act = try l.empty(w * FF * 2),
            .po = try l.empty(w * NQ * maxc * HD * 4),
            .pm = try l.empty(w * NQ * maxc * 4),
            .pl = try l.empty(w * NQ * maxc * 4),
            .logits = try l.empty(@as(usize, window_rows) * V * 4),
            .am_scratch = try l.empty(window_rows * argmax_parts * 8),
            .am_out = try l.empty(window_rows * 4),
        };
        const ns = try namespace(l);
        const layers = try gpa.alloc(Layer, cfg.layers);
        for (layers, 0..) |*ly, i| {
            const kind = c.layer_types[i];
            ly.* = .{
                .op_norm = try vec(l, ns, i, .operator_norm),
                .post_op = try vec(l, ns, i, .post_operator_norm),
                .ffn_norm = try vec(l, ns, i, .ffn_norm),
                .post_ffn = try vec(l, ns, i, .post_ffn_norm),
                .w1 = try table(l, ns, i, .w1),
                .w3 = try table(l, ns, i, .w3),
                .w2 = try table(l, ns, i, .w2),
                .mixer = if (kind == .conv) .{ .conv = .{
                    .in = try table(l, ns, i, .conv_in),
                    .out = try table(l, ns, i, .conv_out),
                    .cw = try vec(l, ns, i, .conv),
                } } else blk: {
                    const ring: u32 = if (kind == .sliding_attention) @intCast(cfg.ring(max_rows)) else cap;
                    break :blk .{ .attn = .{
                        .q = try table(l, ns, i, .q_proj),
                        .k = try table(l, ns, i, .k_proj),
                        .v = try table(l, ns, i, .v_proj),
                        .g = try table(l, ns, i, .gate_proj),
                        .o = try table(l, ns, i, .o_proj),
                        .qn = try vec(l, ns, i, .q_norm),
                        .kn = try vec(l, ns, i, .k_norm),
                        .ring = ring,
                        .window = @intCast(kind.window()),
                    } };
                },
            };
        }
        // the family's RoPE frequencies (torch's float32 values), as every backend's
        const inv_full = try l.empty(HD / 2 * 4);
        try r.upload(inv_full, std.mem.sliceAsBytes(&c.full_rotary.inverseFrequencies()));
        const inv_slide = try l.empty(HD / 2 * 4);
        try r.upload(inv_slide, std.mem.sliceAsBytes(&c.sliding_rotary.inverseFrequencies()));
        try r.sync();
        var m: Model = .{
            .gpa = gpa,
            .r = r,
            .layers = layers,
            .emb = try table(l, ns, null, .embed_tokens),
            .norm_f = try vec(l, ns, null, .embedding_norm),
            .inv_full = inv_full,
            .inv_slide = inv_slide,
            .k = k,
            .s = s,
            .host_rows = try gpa.alloc(u16, w * H),
            .cap = cap,
            .max_rows = max_rows,
            .maxc = maxc,
            .device_bytes = l.total,
            .qg = c.quant.global.group,
        };
        if (std.c.getenv("LFM3_FUSE")) |v| m.fuse = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 7;
        m.own = try m.newSeq();
        m.seq = m.own;
        return m;
    }

    /// A new stream state at position 0 (zeroed conv rows; KV slots are only read once written).
    pub fn newSeq(m: *Model) !*Seq {
        const q = try m.gpa.create(Seq);
        errdefer m.gpa.destroy(q);
        q.* = .{};
        for (m.layers, 0..) |ly, i| switch (ly.mixer) {
            .conv => {
                q.conv[i] = try m.r.alloc(conv_bytes);
                q.ext[i] = try m.r.alloc(ext_bytes);
            },
            .attn => |a| {
                q.kc[i] = try m.r.alloc(@as(usize, a.ring) * slot_bytes);
                q.vc[i] = try m.r.alloc(@as(usize, a.ring) * slot_bytes);
            },
        };
        try m.zeroConv(q);
        return q;
    }

    pub fn freeSeq(m: *Model, q: *Seq) void {
        if (q == m.seq) { // `own` bound again with its own position, not the freed stream's
            m.seq = m.own;
            m.pos = m.own.pos;
        }
        for (m.layers, 0..) |ly, i| switch (ly.mixer) {
            .conv => {
                q.conv[i].free();
                q.ext[i].free();
            },
            .attn => {
                q.kc[i].free();
                q.vc[i].free();
            },
        };
        m.gpa.destroy(q);
    }

    /// Makes `q` the stream the next forwards read and write.
    pub fn bind(m: *Model, q: *Seq) void {
        m.seq.pos = m.pos;
        m.seq = q;
        m.pos = q.pos;
    }

    fn zeroConv(m: *Model, q: *Seq) !void {
        const zero: [6 * 256]u16 = @splat(0);
        for (m.layers, 0..) |ly, i| if (ly.mixer == .conv) {
            var off: usize = 0;
            while (off < conv_bytes) : (off += zero.len * 2) try m.r.upload(from(q.conv[i], off), std.mem.sliceAsBytes(&zero));
        };
        try m.r.sync();
    }

    /// The bound stream back to position 0.
    pub fn reset(m: *Model) !void {
        try m.zeroConv(m.seq);
        m.pos = 0;
    }

    /// Bytes of a kept state at position `at`: conv rows, `at` slots of each full layer, each sliding ring whole.
    pub fn snapBytes(m: *const Model, at: u32) u64 {
        var n: u64 = 0;
        for (m.layers) |ly| n += switch (ly.mixer) {
            .conv => conv_bytes,
            .attn => |a| 2 * @as(u64, if (a.window == 0) at else a.ring) * slot_bytes,
        };
        return n;
    }

    /// One piece of a kept state at byte `off` of its buffer, to (save) or from it; the next piece's offset.
    fn piece(m: *Model, sn: *const Snap, save: bool, b: Buf, bytes: usize, off: usize) !usize {
        if (bytes > 0) {
            if (save) try m.r.copy(from(sn.buf, off), b, bytes) else try m.r.copy(b, from(sn.buf, off), bytes);
        }
        return off + bytes;
    }

    /// Copies the bound stream's state to (save) or from a kept state's buffer, on the device.
    fn snapCopy(m: *Model, sn: *const Snap, save: bool) !void {
        var off: usize = 0;
        for (m.layers, 0..) |ly, i| switch (ly.mixer) {
            .conv => off = try m.piece(sn, save, m.seq.conv[i], conv_bytes, off),
            .attn => |a| {
                const bytes = @as(usize, if (a.window == 0) sn.at else a.ring) * slot_bytes;
                off = try m.piece(sn, save, m.seq.kc[i], bytes, off);
                off = try m.piece(sn, save, m.seq.vc[i], bytes, off);
            },
        };
        try m.r.sync();
    }

    /// The bound stream's state at its position, kept.
    pub fn snapSave(m: *Model) !*Snap {
        const sn = try m.gpa.create(Snap);
        errdefer m.gpa.destroy(sn);
        sn.* = .{ .at = m.pos, .buf = try m.r.alloc(@intCast(@max(m.snapBytes(m.pos), 1))) };
        try m.snapCopy(sn, true);
        return sn;
    }

    /// The bound stream continues from a kept state.
    pub fn snapRestore(m: *Model, sn: *const Snap) !void {
        m.pos = sn.at;
        try m.snapCopy(sn, false);
    }

    pub fn snapDrop(m: *Model, sn: *Snap) void {
        sn.buf.free();
        m.gpa.destroy(sn);
    }

    /// y[r * y_stride + y_off + o] = x[r] . W[o] for n rows: pgemm in a prompt window, else qmv (one row) or pdense.
    fn proj(m: *Model, t: Table, x: Buf, y: Buf, in_dim: u32, rows: u32, n: u32, y_stride: u32, y_off: u32) !void {
        if (m.prompt) return run(&m.k.pgemm, .{ (rows + 63) / 64, (n + 63) / 64, 1 }, .{ t.w, t.s, t.b, x, y, in_dim, rows, n, y_stride, y_off, m.qg });
        if (n == 1) return run(&m.k.qmv, .{ (rows + 3) / 4, 1, 1 }, .{ t.w, t.s, t.b, x, y, in_dim, @as(u32, 0), y_off, rows });
        try run(&m.k.pdense, .{ (rows + 15) / 16, (n + tile_rows - 1) / tile_rows, 1 }, .{ t.w, t.s, t.b, x, y, in_dim, rows, n, y_stride, y_off, @as(u32, 0) });
    }

    /// The token mixer of every segment's rows: matvecs over all rows, conv and attention per stream.
    fn mixer(m: *Model, ly: *const Layer, li: usize, segs: []const Seg, n: u32, hist: bool) !void {
        const s = &m.s;
        switch (ly.mixer) {
            .conv => |c| {
                try m.proj(c.in, s.xn, s.proj, H, 3 * H, n, 3 * H, 0);
                var r0: usize = 0;
                for (segs) |g| {
                    const k: u32 = @intCast(g.ids.len);
                    const pr = from(s.proj, r0 * 3 * H * 2);
                    const y = from(s.att, r0 * H * 2);
                    if (hist) {
                        try run(&m.k.conv_hist, .{ H / 64, 1, 1 }, .{ pr, 3 * H, g.q.conv[li], g.q.conv[li], c.cw, y, k, H, g.q.ext[li] });
                    } else try run(&m.k.conv, .{ H / 64, 1, 1 }, .{ pr, 3 * H, g.q.conv[li], g.q.conv[li], c.cw, y, k, H });
                    r0 += k;
                }
                try m.proj(c.out, s.att, s.d, H, H, n, H, 0);
            },
            .attn => |a| {
                if (n == 1 and m.fuse & 2 != 0) { // the four decode matvecs in one launch, each row as qmv4_bf sums it
                    try run(&m.k.qkvg, .{ (QKV + NQ + 3) / 4, 1, 1 }, .{ a.q.w, a.q.s, a.q.b, s.qkv, NQ * HD, @as(u32, 0), a.k.w, a.k.s, a.k.b, s.qkv, NKV * HD, NQ * HD, a.v.w, a.v.s, a.v.b, s.qkv, NKV * HD, (NQ + NKV) * HD, a.g.w, a.g.s, a.g.b, s.g, NQ, @as(u32, 0), s.xn, H });
                } else {
                    try m.proj(a.q, s.xn, s.qkv, H, NQ * HD, n, QKV, 0);
                    try m.proj(a.k, s.xn, s.qkv, H, NKV * HD, n, QKV, NQ * HD);
                    try m.proj(a.v, s.xn, s.qkv, H, NKV * HD, n, QKV, (NQ + NKV) * HD);
                    try m.proj(a.g, s.xn, s.g, H, NQ, n, NQ, 0);
                }
                const inv = if (a.window == 0) m.inv_full else m.inv_slide;
                const scale: f32 = 1.0 / @sqrt(@as(f32, HD));
                var r0: usize = 0;
                for (segs) |g| {
                    const k: u32 = @intCast(g.ids.len);
                    const pos = g.q.pos;
                    const qkv = from(s.qkv, r0 * QKV * 2);
                    const q = from(s.q, r0 * NQ * HD * 2);
                    if (m.fuse & 4 != 0) { // q, k and v of every row in one launch
                        try run(&m.k.rope_store, .{ NQ + NKV + NKV * HD / 64, k, 1 }, .{ qkv, QKV, q, g.q.kc[li], g.q.vc[li], a.qn, a.kn, inv, NQ, NKV, pos, a.ring, cfg.eps });
                    } else {
                        try run(&m.k.rope, .{ NQ, k, 1 }, .{ qkv, QKV, @as(u32, 0), q, NQ * HD, a.qn, inv, pos, @as(u32, 0), cfg.eps });
                        try run(&m.k.rope, .{ NKV, k, 1 }, .{ qkv, QKV, NQ * HD, g.q.kc[li], NKV * HD, a.kn, inv, pos, a.ring, cfg.eps });
                        try run(&m.k.vst, .{ NKV * HD / 64, k, 1 }, .{ qkv, QKV, (NQ + NKV) * HD, g.q.vc[li], NKV * HD, pos, a.ring });
                    }
                    const span = if (a.window == 0) pos + k else @min(pos + k, a.window);
                    try run(&m.k.part, .{ NKV, (span + chunk - 1) / chunk, k }, .{ q, g.q.kc[li], g.q.vc[li], s.po, s.pm, s.pl, pos, chunk, a.ring, a.window, NKV, m.maxc, scale });
                    try run(&m.k.merge, .{ NQ, k, 1 }, .{ s.po, s.pm, s.pl, from(s.g, r0 * NQ * 2), from(s.att, r0 * NQ * HD * 2), pos, chunk, a.window, NQ, m.maxc });
                    r0 += k;
                }
                try m.proj(a.o, s.att, s.d, NQ * HD, H, n, H, 0);
            },
        }
    }

    fn hook(m: *Model, h: ?Hook, layer: i32, b: Buf, n: u32) !void {
        const k = h orelse return;
        const rows = m.host_rows[0 .. n * H];
        try m.r.sync();
        try m.r.download(std.mem.sliceAsBytes(rows), b);
        try m.r.sync();
        k.f(k.ctx, layer, rows);
    }

    /// The tokens as one window of the bound stream (1 .. max_rows rows); `logits`: the last row's logits and argmax.
    pub fn forward(m: *Model, ids: []const u32, logits: bool, h: ?Hook) !void {
        try m.forwardSegs(&.{.{ .q = m.seq, .ids = ids }}, if (logits) .last else .none, false, h);
    }

    /// A prompt window of the bound stream, every projection in pgemm (verified windows and decode keep theirs).
    pub fn forwardPrompt(m: *Model, ids: []const u32, logits: bool, h: ?Hook) !void {
        const fuse = m.fuse;
        m.prompt = true;
        m.fuse = fuse & 4; // a one-row window too: not the decode matvecs
        defer {
            m.prompt = false;
            m.fuse = fuse;
        }
        try m.forward(ids, logits, h);
    }

    /// One forward over every segment's rows, each at its stream's position; `hist` keeps a window's rollback source.
    pub fn forwardSegs(m: *Model, segs: []const Seg, out: Out, hist: bool, h: ?Hook) !void {
        var n: u32 = 0;
        m.seq.pos = m.pos;
        for (segs) |g| {
            if (g.ids.len == 0 or (hist and g.ids.len > window_rows)) return error.Invalid;
            if (g.q.pos + g.ids.len > m.cap) return error.ContextFull;
            n += @intCast(g.ids.len);
        }
        if (n > m.max_rows or (out == .rows and n > window_rows)) return error.Invalid;
        const s = &m.s;
        if (n == 1 or out == .rows) { // ids as kernel arguments: the queue reads no host memory the caller may reuse
            var r: usize = 0;
            for (segs) |g| for (g.ids) |id| {
                try run(&m.k.embed_tok, .{ 1, 1, 1 }, .{ m.emb.w, m.emb.s, m.emb.b, id, from(s.x, r * H * 2), H });
                r += 1;
            };
        } else { // a prompt window: its ids live as long as its stream
            try m.r.upload(s.ids, std.mem.sliceAsBytes(segs[0].ids));
            try run(&m.k.embed, .{ n, 1, 1 }, .{ m.emb.w, m.emb.s, m.emb.b, s.ids, s.x, H });
        }
        try run(&m.k.rms, .{ n, 1, 1 }, .{ s.x, m.layers[0].op_norm, s.xn, H, cfg.eps });
        for (m.layers, 0..) |*ly, i| {
            try m.mixer(ly, i, segs, n, hist);
            try run(&m.k.pan, .{ n, 1, 1 }, .{ s.d, s.x, ly.post_op, ly.ffn_norm, s.xn, H, cfg.eps });
            if (n == 1 and m.fuse & 1 != 0) { // w1, w3 and their SwiGLU in one launch, each value as the three launches round it
                try run(&m.k.w13, .{ FF / 4, 1, 1 }, .{ ly.w1.w, ly.w1.s, ly.w1.b, ly.w3.w, ly.w3.s, ly.w3.b, s.xn, s.act, H, FF });
            } else {
                try m.proj(ly.w1, s.xn, s.a, H, FF, n, FF, 0);
                try m.proj(ly.w3, s.xn, s.b, H, FF, n, FF, 0);
                try run(&m.k.swiglu, .{ (n * FF + 63) / 64, 1, 1 }, .{ s.a, s.b, s.act, n * FF });
            }
            try m.proj(ly.w2, s.act, s.d, FF, H, n, H, 0);
            const next = if (i + 1 < m.layers.len) m.layers[i + 1].op_norm else m.norm_f;
            try run(&m.k.pan, .{ n, 1, 1 }, .{ s.d, s.x, ly.post_ffn, next, s.xn, H, cfg.eps });
            try m.hook(h, @intCast(i), s.x, n);
        }
        try m.hook(h, -1, s.xn, n);
        for (segs) |g| {
            if (hist) {
                g.q.win_start = g.q.pos;
                g.q.win_rows = @intCast(g.ids.len);
            }
            g.q.pos += @intCast(g.ids.len);
        }
        m.pos = m.seq.pos;
        const per = (V + argmax_parts - 1) / argmax_parts;
        switch (out) {
            .none => {},
            .last => {
                try run(&m.k.head, .{ V / 4, 1, 1 }, .{ m.emb.w, m.emb.s, m.emb.b, s.xn, s.logits, H, (n - 1) * H, @as(u32, 0), V });
                try run(&m.k.am_part, .{ argmax_parts, 1, 1 }, .{ s.logits, s.am_scratch, V, per });
                try run(&m.k.am_fin, .{ 1, 1, 1 }, .{ s.am_scratch, s.am_out, argmax_parts });
            },
            .rows => {
                // every row's logits: pdense's fp32 epilogue sums each as qmv4_f32 does (tf-xpu-lfm3_rows-test)
                try run(&m.k.pdense, .{ V / 16, (n + tile_rows - 1) / tile_rows, 1 }, .{ m.emb.w, m.emb.s, m.emb.b, s.xn, s.logits, H, V, n, V, @as(u32, 0), @as(u32, 1) });
                try run(&m.k.am_part, .{ argmax_parts, n, 1 }, .{ s.logits, s.am_scratch, V, per });
                try run(&m.k.am_fin, .{ n, 1, 1 }, .{ s.am_scratch, s.am_out, argmax_parts });
            },
        }
    }

    /// Keeps the first c rows of the bound stream's last verified window: its position, and its conv rows from ext.
    pub fn keepRows(m: *Model, c: u32) !void {
        const q = m.seq;
        if (q.win_rows == 0) return;
        if (c > q.win_rows) return error.Invalid;
        if (c < q.win_rows) for (m.layers, 0..) |ly, i| if (ly.mixer == .conv) try m.r.copy(q.conv[i], from(q.ext[i], c * H * 2), conv_bytes);
        m.pos = q.win_start + c;
        q.win_rows = 0;
    }

    /// The greedy tokens of the last forward's rows (Out.rows), in row order.
    pub fn rowTokens(m: *Model, out: []i32) !void {
        try m.r.sync();
        try m.r.download(std.mem.sliceAsBytes(out), m.s.am_out);
        try m.r.sync();
    }

    /// The logits of the last forward's rows (Out.rows), [rows][vocab].
    pub fn rowLogits(m: *Model, out: []f32) !void {
        try m.r.sync();
        try m.r.download(std.mem.sliceAsBytes(out), m.s.logits);
        try m.r.sync();
    }

    /// Prefill of a prompt in windows of max_rows; the last window leaves its logits and argmax.
    pub fn prefill(m: *Model, ids: []const u32, h: ?Hook) !void {
        var i: usize = 0;
        while (i < ids.len) {
            const n = @min(ids.len - i, m.max_rows);
            try m.forwardPrompt(ids[i .. i + n], i + n == ids.len, h);
            i += n;
        }
    }

    /// The greedy token of the last forward with logits.
    pub fn argmax(m: *Model) !u32 {
        var out: [1]i32 = undefined;
        try m.r.sync();
        try m.r.download(std.mem.sliceAsBytes(&out), m.s.am_out);
        try m.r.sync();
        return @intCast(out[0]);
    }

    pub fn fetchLogits(m: *Model, out: []f32) !void {
        try m.r.sync();
        try m.r.download(std.mem.sliceAsBytes(out[0..V]), m.s.logits);
        try m.r.sync();
    }
};
