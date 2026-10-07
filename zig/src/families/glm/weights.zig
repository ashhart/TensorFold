//! GLM-5.3-Flash's weights in the Python family's layout (stacked projections and experts, repacked mixes), read by parallel preads.
const std = @import("std");
const mtl = @import("metal");
const st = @import("../../core/safetensors.zig");
const cfg = @import("config.zig");

pub const Ref = struct {
    buf: mtl.Buffer,
    off: usize = 0,

    pub fn at(r: Ref, bytes: usize) Ref {
        return .{ .buf = r.buf, .off = r.off + bytes };
    }

    pub fn addr(r: Ref) [*]u8 {
        return r.buf.contents() + r.off;
    }
};

/// A 4-bit affine matrix in groups of 64: w [n, k/8] u32, s and b [n, k/64] bf16 (stacked: [e, n, ...]).
pub const Q4 = struct {
    w: Ref,
    s: Ref,
    b: Ref,
    n: u32,
    k: u32,

    pub fn wBytes(n: usize, k: usize) usize {
        return n * k / 2;
    }

    pub fn sBytes(n: usize, k: usize) usize {
        return n * (k / 64) * 2;
    }
};

pub const Hc = struct { fnp: Ref, base: Ref, scale: Ref };

pub const Kda = struct { in_proj: Q4, f_b: Q4, g_b: Q4, o_proj: Q4, conv_w: Ref, a_log: Ref, a: Ref, dt_bias: Ref, o_norm: Ref };

pub const Mla = struct { x_proj: Q4, qr_proj: Q4, kv_b: Q4, o_proj: Q4, q_norm: Ref, kv_norm: Ref, k_norm_w: Ref, k_norm_b: Ref, ape: Ref, igate: Ref }; // igate packed as a router

pub const Dense = struct { gate_up: Q4, down: Q4 };

pub const Moe = struct { router: Ref, bias: Ref, gate: Q4, up: Q4, down: Q4, sh_gate_up: Q4, sh_down: Q4 };

pub const Layer = struct {
    hc: ?[2]Hc, // attention's and the MLP's hyper-connections (none on the MTP layer)
    in_norm: Ref,
    post_norm: Ref,
    attn: union(cfg.Kind) { kda: Kda, mla: Mla },
    mlp: union(enum) { dense: Dense, moe: Moe },
};

pub const Mtp = struct { eh_proj: Q4, enorm: Ref, hnorm: Ref, norm: Ref };

pub const Weights = struct {
    layers: [cfg.max_layers]Layer = undefined, // the backbone's, then the MTP layer at index c.layers
    mtp: ?Mtp = null,
    embed: Q4 = undefined,
    head: Q4 = undefined,
    norm: Ref = undefined,
    buffers: std.ArrayList(mtl.Buffer) = .empty,
    bytes: usize = 0,
    gpa: std.mem.Allocator,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |b| b.deinit();
        w.buffers.deinit(w.gpa);
    }
};

const prefix = "model.language_model.";

const load_fix = @import("weights_fix.zig");
const Src = load_fix.Src;
const Fix = load_fix.Fix;
const Transform = load_fix.Transform;
const readAll = load_fix.readAll;

const Copy = struct { shard: u32, off: u64, len: u64, dst: [*]u8 };

const Loader = struct {
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    device: mtl.Device,
    c: *const cfg.Config,
    w: *Weights,
    dir: []const u8,
    shards: std.ArrayList([:0]const u8) = .empty,
    names: std.StringHashMapUnmanaged(Src) = .empty,
    copies: std.ArrayList(Copy) = .empty,
    transforms: std.ArrayList(Transform) = .empty,
    dry: bool = false, // plan and check only: no buffers, no reads
    // the buffer being filled and its next free byte
    cur: ?mtl.Buffer = null,
    cur_len: usize = 0,
    cur_at: usize = 0,

    fn src(l: *Loader, comptime fmt: []const u8, args: anytype) !Src {
        var name: [200]u8 = undefined;
        const full = try std.fmt.bufPrint(&name, fmt, args);
        return l.names.get(full) orelse {
            std.log.err("glm: the checkpoint has no tensor {s}", .{full});
            return error.MissingTensor;
        };
    }

    fn expect(s: Src, name: []const u8, dtype: st.DType, shape: []const usize) !void {
        if (s.dtype == dtype and s.rank == shape.len and std.mem.eql(usize, s.shape[0..s.rank], shape)) return;
        std.log.err("glm: {s} is {t} {any}; the kernels read {t} {any}", .{ name, s.dtype, s.shape[0..s.rank], dtype, shape });
        return error.UnexpectedTensor;
    }

    /// A new buffer of `len` bytes that the next `take`s fill.
    fn begin(l: *Loader, len: usize) !void {
        const b: ?mtl.Buffer = if (l.dry) null else try l.device.buffer(@max(len, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        if (b) |x| l.w.buffers.append(l.gpa, x) catch |err| {
            x.deinit();
            return err;
        };
        l.cur = b;
        l.cur_len = len;
        l.cur_at = 0;
        l.w.bytes += len;
    }

    fn take(l: *Loader, len: usize) Ref {
        const at = l.cur_at;
        l.cur_at = std.mem.alignForward(usize, at + len, 256);
        if (l.cur_at > l.cur_len) std.debug.panic("glm: a weights buffer of {d} bytes is too small for {d}", .{ l.cur_len, l.cur_at });
        return .{ .buf = l.cur orelse .{ .id = undefined }, .off = at };
    }

    fn copyTo(l: *Loader, s: Src, dst: Ref) !void {
        if (l.dry) return;
        try l.copies.append(l.gpa, .{ .shard = s.shard, .off = s.off, .len = s.len, .dst = dst.addr() });
    }

    /// A tensor as stored (bf16 norms, fp32 biases), checked.
    fn plain(l: *Loader, comptime fmt: []const u8, args: anytype, dtype: st.DType, shape: []const usize) !Ref {
        const s = try l.src(fmt, args);
        var name: [200]u8 = undefined;
        try expect(s, try std.fmt.bufPrint(&name, fmt, args), dtype, shape);
        const r = l.take(s.len);
        try l.copyTo(s, r);
        return r;
    }

    fn fixed(l: *Loader, s: Src, bytes: usize, fix: Fix) !Ref {
        const r = l.take(bytes);
        if (!l.dry) try l.transforms.append(l.gpa, .{ .src = s, .dst = r.addr(), .fix = fix });
        return r;
    }

    /// 4-bit projections that read one input, stacked by rows in order; each part checked [n_i, k].
    fn q4(l: *Loader, k: usize, comptime fmt: []const u8, parts: []const []const u8, ns: []const usize, args: anytype) !Q4 {
        const zeros: [8]usize = @splat(0);
        return l.q4Rows(k, fmt, parts, zeros[0..parts.len], ns, ns, 1, args);
    }

    /// 4-bit rows [lo, lo + n) of each part stored [all, k], stacked; room for `pad`-row tiles.
    fn q4Rows(l: *Loader, k: usize, comptime fmt: []const u8, parts: []const []const u8, los: []const usize, ns: []const usize, alls: []const usize, pad: usize, args: anytype) !Q4 {
        var n: usize = 0;
        for (ns) |x| n += x;
        const room = std.mem.alignForward(usize, n, pad);
        const out: Q4 = .{ .w = l.take(Q4.wBytes(room, k)), .s = l.take(Q4.sBytes(room, k)), .b = l.take(Q4.sBytes(room, k)), .n = @intCast(n), .k = @intCast(k) };
        var row: usize = 0;
        for (parts, los, ns, alls) |p, lo, rows, all| {
            const comps = [_][]const u8{ "weight", "scales", "biases" };
            const dsts = [_]Ref{ out.w.at(Q4.wBytes(row, k)), out.s.at(Q4.sBytes(row, k)), out.b.at(Q4.sBytes(row, k)) };
            for (comps, dsts, 0..) |comp, dst, j| {
                var name: [200]u8 = undefined;
                const base = try std.fmt.bufPrint(&name, fmt, args);
                var full: [240]u8 = undefined;
                const nm = try std.fmt.bufPrint(&full, "{s}{s}.{s}", .{ base, p, comp });
                const s = l.names.get(nm) orelse {
                    std.log.err("glm: the checkpoint has no tensor {s}", .{nm});
                    return error.MissingTensor;
                };
                try expect(s, nm, if (j == 0) .u32 else .bf16, &.{ all, if (j == 0) k / 8 else k / 64 });
                const row_bytes: usize = if (j == 0) k / 2 else k / 64 * 2;
                try l.copyTo(.{ .shard = s.shard, .off = s.off + lo * row_bytes, .len = rows * row_bytes, .dtype = s.dtype, .shape = s.shape, .rank = s.rank }, dst);
            }
            row += rows;
        }
        return out;
    }

    /// One 4-bit matrix stored [n, k_all]: its input columns [lo, lo + k) (TP2: o_proj over one Mac's heads).
    fn q4Cols(l: *Loader, comptime fmt: []const u8, args: anytype, n: usize, k_all: usize, lo: usize, k: usize) !Q4 {
        const out: Q4 = .{ .w = l.take(Q4.wBytes(n, k)), .s = l.take(Q4.sBytes(n, k)), .b = l.take(Q4.sBytes(n, k)), .n = @intCast(n), .k = @intCast(k) };
        const comps = [_][]const u8{ "weight", "scales", "biases" };
        for (comps, [_]Ref{ out.w, out.s, out.b }, 0..) |comp, dst, j| {
            var name: [200]u8 = undefined;
            const base = try std.fmt.bufPrint(&name, fmt, args);
            var full: [240]u8 = undefined;
            const nm = try std.fmt.bufPrint(&full, "{s}.{s}", .{ base, comp });
            const s = l.names.get(nm) orelse return error.MissingTensor;
            try expect(s, nm, if (j == 0) .u32 else .bf16, &.{ n, if (j == 0) k_all / 8 else k_all / 64 });
            const row_all: usize = if (j == 0) k_all / 2 else k_all / 64 * 2;
            const span: [2]usize = if (j == 0) .{ lo / 2, k / 2 } else .{ lo / 64 * 2, k / 64 * 2 };
            if (!l.dry) try l.transforms.append(l.gpa, .{ .src = s, .dst = dst.addr(), .fix = .cols, .arg = .{ row_all, span[0], span[1] } });
        }
        return out;
    }

    /// Elements [lo, lo + n) of a vector stored with `all` elements of `size` bytes (TP2: one Mac's heads).
    fn plainRange(l: *Loader, comptime fmt: []const u8, args: anytype, dtype: st.DType, all: usize, lo: usize, n: usize, size: usize) !Ref {
        const s = try l.src(fmt, args);
        var name: [200]u8 = undefined;
        try expect(s, try std.fmt.bufPrint(&name, fmt, args), dtype, &.{all});
        const r = l.take(n * size);
        try l.copyTo(.{ .shard = s.shard, .off = s.off + lo * size, .len = n * size, .dtype = s.dtype, .shape = s.shape, .rank = s.rank }, r);
        return r;
    }

    /// This Mac's routed experts' `proj` stacked [own, n, k]; by rows, gate/up keep rows `inter` and down keeps input columns `inter`.
    fn experts(l: *Loader, i: usize, proj: []const u8, n_all: usize, k_all: usize, part: enum { whole, rows, cols }) !Q4 {
        const every = i == l.c.layers; // the MTP layer: every expert whole on every Mac (its drafts need no exchange)
        const lo = if (every) 0 else l.c.own[0];
        const e = (if (every) l.c.experts else l.c.own[1]) - lo;
        const in_lo: usize = l.c.inter[0];
        const in_n: usize = l.c.inter[1] - l.c.inter[0];
        const n = if (part == .rows) in_n else n_all;
        const k = if (part == .cols) in_n else k_all;
        const out: Q4 = .{ .w = l.take(e * Q4.wBytes(n, k)), .s = l.take(e * Q4.sBytes(n, k)), .b = l.take(e * Q4.sBytes(n, k)), .n = @intCast(n), .k = @intCast(k) };
        for (0..e) |x| {
            const comps = [_][]const u8{ "weight", "scales", "biases" };
            const dsts = [_]Ref{ out.w.at(x * Q4.wBytes(n, k)), out.s.at(x * Q4.sBytes(n, k)), out.b.at(x * Q4.sBytes(n, k)) };
            for (comps, dsts, 0..) |comp, dst, j| {
                const s = try l.src(prefix ++ "layers.{d}.mlp.experts.{d}.{s}.{s}", .{ i, lo + x, proj, comp });
                var name: [200]u8 = undefined;
                try expect(s, try std.fmt.bufPrint(&name, "experts.{d}.{d}.{s}.{s}", .{ i, x, proj, comp }), if (j == 0) .u32 else .bf16, &.{ n_all, if (j == 0) k_all / 8 else k_all / 64 });
                const row_all: usize = if (j == 0) k_all / 2 else k_all / 64 * 2; // a stored row's bytes
                switch (part) {
                    .whole => try l.copyTo(s, dst),
                    .rows => try l.copyTo(.{ .shard = s.shard, .off = s.off + in_lo * row_all, .len = in_n * row_all, .dtype = s.dtype, .shape = s.shape, .rank = s.rank }, dst),
                    .cols => if (!l.dry) try l.transforms.append(l.gpa, .{ .src = s, .dst = dst.addr(), .fix = .cols, .arg = .{ row_all, if (j == 0) in_lo / 2 else in_lo / 64 * 2, if (j == 0) in_n / 2 else in_n / 64 * 2 } }),
                }
            }
        }
        return out;
    }

    fn hc(l: *Loader, i: usize, site: []const u8) !Hc {
        const D: usize = l.c.hidden;
        const f = try l.src(prefix ++ "layers.{d}.hc_{s}_fn", .{ i, site });
        try expect(f, "hc fn", .bf16, &.{ 24, 4 * D });
        return .{
            .fnp = try l.fixed(f, f.len, .hc_pack),
            .base = try l.plain(prefix ++ "layers.{d}.hc_{s}_base", .{ i, site }, .f32, &.{24}),
            .scale = try l.plain(prefix ++ "layers.{d}.hc_{s}_scale", .{ i, site }, .f32, &.{3}),
        };
    }

    fn kda(l: *Loader, i: usize) !Kda {
        const c = l.c;
        const D: usize = c.hidden;
        const W: usize = c.kdaWidth(); // this Mac's heads' channels
        const H: usize = c.kda_heads;
        const dk: usize = c.kda_dim;
        const h0: usize = c.tp_rank * H; // TP2: this Mac's first head (0 on one Mac)
        const ch0 = h0 * dk;
        const Wa = W * c.tp; // every head's channels, as stored
        const Ha = H * c.tp;
        const a = prefix ++ "layers.{d}.self_attn.";
        const conv_q = try l.src(a ++ "q_conv1d.weight", .{i});
        try expect(conv_q, "q_conv1d", .bf16, &.{ Wa, 1, c.conv });
        const conv = try l.fixed(conv_q, 3 * W * c.conv * 4, .conv);
        const extra: [3]?Src = .{ try l.src(a ++ "k_conv1d.weight", .{i}), try l.src(a ++ "v_conv1d.weight", .{i}), null };
        if (!l.dry) {
            const t = &l.transforms.items[l.transforms.items.len - 1];
            t.extra = extra;
            t.arg = .{ ch0, W, 0 };
        }
        const o_norm = try l.src(a ++ "o_norm.weight", .{i});
        try expect(o_norm, "o_norm", .bf16, &.{dk});
        return .{
            .in_proj = try l.q4Rows(D, a, &.{ "q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj" }, &.{ ch0, ch0, ch0, 0, 0, h0 }, &.{ W, W, W, dk, dk, H }, &.{ Wa, Wa, Wa, dk, dk, Ha }, 64, .{i}),
            .f_b = try l.q4Rows(dk, a, &.{"f_b_proj"}, &.{ch0}, &.{W}, &.{Wa}, 1, .{i}),
            .g_b = try l.q4Rows(dk, a, &.{"g_b_proj"}, &.{ch0}, &.{W}, &.{Wa}, 1, .{i}),
            .o_proj = if (c.tp == 1) try l.q4(W, a, &.{"o_proj"}, &.{D}, .{i}) else try l.q4Cols(a ++ "o_proj", .{i}, D, Wa, ch0, W),
            .conv_w = conv,
            .a_log = try l.plainRange(a ++ "A_log", .{i}, .f32, Ha, h0, H, 4),
            .a = l.take(H * 4),
            .dt_bias = try l.plainRange(a ++ "dt_bias", .{i}, .f32, Wa, ch0, W, 4),
            .o_norm = try l.fixed(o_norm, dk * 4, .to_f32),
        };
    }

    fn mla(l: *Loader, i: usize) !Mla {
        const c = l.c;
        const D: usize = c.hidden;
        const a = prefix ++ "layers.{d}.self_attn.";
        const H: usize = c.mla_heads; // this Mac's heads
        const h0: usize = c.tp_rank * H; // TP2: its first head (0 on one Mac)
        const Ha = H * c.tp; // every head, as stored
        const gate = try l.src(a ++ "indexer.index_kpool_compress_gate", .{i});
        try expect(gate, "index_kpool_compress_gate", .bf16, &.{ c.i_dim, D });
        return .{
            .x_proj = try l.q4Rows(D, a, &.{ "q_a_proj", "kv_a_proj_with_mqa", "indexer.wk", "indexer.weights_proj" }, &.{ 0, 0, 0, 0 }, &.{ c.q_lora, c.kv_lora, c.i_dim, c.i_heads }, &.{ c.q_lora, c.kv_lora, c.i_dim, c.i_heads }, 64, .{i}),
            .qr_proj = try l.q4Rows(c.q_lora, a, &.{ "q_b_proj", "indexer.wq_b" }, &.{ h0 * c.nope, 0 }, &.{ H * c.nope, c.i_heads * c.i_dim }, &.{ Ha * c.nope, c.i_heads * c.i_dim }, 1, .{i}),
            .kv_b = try l.q4Rows(c.kv_lora, a, &.{"kv_b_proj"}, &.{h0 * (c.nope + c.v_dim)}, &.{H * (c.nope + c.v_dim)}, &.{Ha * (c.nope + c.v_dim)}, 1, .{i}),
            .o_proj = if (c.tp == 1) try l.q4(H * c.v_dim, a, &.{"o_proj"}, &.{D}, .{i}) else try l.q4Cols(a ++ "o_proj", .{i}, D, Ha * c.v_dim, h0 * c.v_dim, H * c.v_dim),
            .q_norm = try l.plain(a ++ "q_a_layernorm.weight", .{i}, .bf16, &.{c.q_lora}),
            .kv_norm = try l.plain(a ++ "kv_a_layernorm.weight", .{i}, .bf16, &.{c.kv_lora}),
            .k_norm_w = try l.plain(a ++ "indexer.k_norm.weight", .{i}, .bf16, &.{c.i_dim}),
            .k_norm_b = try l.plain(a ++ "indexer.k_norm.bias", .{i}, .bf16, &.{c.i_dim}),
            .ape = try l.plain(a ++ "indexer.index_kpool_compress_ape", .{i}, .bf16, &.{ c.kpool, c.i_dim }),
            .igate = try l.fixed(gate, gate.len, .router_pack), // core/moe_route.zig's layout: the gate is the same gemv_t
        };
    }

    /// Bytes of layer i's dense buffer (everything but routed experts), generously rounded.
    fn layerBytes(l: *Loader, i: usize) usize {
        const c = l.c;
        const D: usize = c.hidden;
        var n: usize = 0;
        const q = struct {
            fn b(rows: usize, k: usize) usize {
                return Q4.wBytes(rows, k) + 2 * Q4.sBytes(rows, k) + 3 * 256;
            }
        };
        n += 2 * (24 * 4 * D * 2 + 1024) + 2 * (D * 2 + 256);
        switch (c.kind(@intCast(i))) {
            .kda => n += q.b(std.mem.alignForward(usize, c.kdaProj(), 64), D) + 2 * q.b(c.kdaWidth(), c.kda_dim) + q.b(D, c.kdaWidth()) + 3 * c.kdaWidth() * c.conv * 4 + 8 * 1024 + c.kdaWidth() * 4,
            .mla => n += q.b(std.mem.alignForward(usize, c.xProj(), 64), D) + q.b(c.qrProj(), c.q_lora) + q.b(c.mla_heads * (c.nope + c.v_dim), c.kv_lora) + q.b(D, c.mla_heads * c.v_dim) + 16 * 1024 + c.i_dim * D * 2,
        }
        if (c.isMoe(@intCast(i))) {
            n += c.experts * D * 2 + c.experts * 4 + q.b(2 * c.moe_inter, D) + q.b(D, c.moe_inter) + 2048;
        } else n += q.b(2 * c.dense_inter, D) + q.b(D, c.dense_inter);
        if (i == c.layers) n += q.b(D, 2 * D) + 3 * (D * 2 + 256);
        return n;
    }

    fn expertBytes(l: *Loader, i: usize) usize {
        const c = l.c;
        const D: usize = c.hidden;
        const every = i == c.layers;
        const m: usize = if (every) c.moe_inter else c.inter[1] - c.inter[0];
        const per = 2 * (Q4.wBytes(m, D) + 2 * Q4.sBytes(m, D)) + Q4.wBytes(D, m) + 2 * Q4.sBytes(D, m);
        return (if (every) c.experts else c.own[1] - c.own[0]) * per + 9 * 256;
    }

    fn layer(l: *Loader, i: usize) !Layer {
        const c = l.c;
        const D: usize = c.hidden;
        const plain_layer = i == c.layers; // the MTP block: no hyper-connections
        try l.begin(l.layerBytes(i));
        var out: Layer = .{
            .hc = if (plain_layer) null else .{ try l.hc(i, "attn"), try l.hc(i, "ffn") },
            .in_norm = try l.plain(prefix ++ "layers.{d}.input_layernorm.weight", .{i}, .bf16, &.{D}),
            .post_norm = try l.plain(prefix ++ "layers.{d}.post_attention_layernorm.weight", .{i}, .bf16, &.{D}),
            .attn = switch (c.kind(@intCast(i))) {
                .kda => .{ .kda = try l.kda(i) },
                .mla => .{ .mla = try l.mla(i) },
            },
            .mlp = undefined,
        };
        if (plain_layer) { // the MTP head's own tensors, in the layer's buffer (before the experts begin theirs)
            const p = prefix ++ "layers.{d}.";
            l.w.mtp = .{
                .eh_proj = try l.q4(2 * D, p, &.{"eh_proj"}, &.{D}, .{i}),
                .enorm = try l.plain(p ++ "enorm.weight", .{i}, .bf16, &.{D}),
                .hnorm = try l.plain(p ++ "hnorm.weight", .{i}, .bf16, &.{D}),
                .norm = try l.plain(p ++ "shared_head.norm.weight", .{i}, .bf16, &.{D}),
            };
        }
        const m = prefix ++ "layers.{d}.mlp.";
        if (c.isMoe(@intCast(i))) {
            const router = try l.src(m ++ "gate.weight", .{i});
            try expect(router, "mlp.gate.weight", .bf16, &.{ c.experts, D });
            var moe: Moe = .{
                .router = try l.fixed(router, router.len, .router_pack),
                .bias = try l.plain(m ++ "gate.e_score_correction_bias", .{i}, .f32, &.{c.experts}),
                .sh_gate_up = try l.q4(D, m ++ "shared_experts.", &.{ "gate_proj", "up_proj" }, &.{ c.moe_inter, c.moe_inter }, .{i}),
                .sh_down = try l.q4(c.moe_inter, m ++ "shared_experts.", &.{"down_proj"}, &.{D}, .{i}),
                .gate = undefined,
                .up = undefined,
                .down = undefined,
            };
            try l.begin(l.expertBytes(i));
            const by_rows = c.byRows() and i != c.layers;
            moe.gate = try l.experts(i, "gate_proj", c.moe_inter, D, if (by_rows) .rows else .whole);
            moe.up = try l.experts(i, "up_proj", c.moe_inter, D, if (by_rows) .rows else .whole);
            moe.down = try l.experts(i, "down_proj", D, c.moe_inter, if (by_rows) .cols else .whole);
            out.mlp = .{ .moe = moe };
        } else {
            const n: usize = c.dense_inter; // this Mac's rows; TP2: rank r's half [r n, (r + 1) n)
            const lo = c.tp_rank * n;
            out.mlp = .{ .dense = .{
                .gate_up = try l.q4Rows(D, m, &.{ "gate_proj", "up_proj" }, &.{ lo, lo }, &.{ n, n }, &.{ n * c.tp, n * c.tp }, 1, .{i}),
                .down = if (c.tp == 1) try l.q4(n, m, &.{"down_proj"}, &.{D}, .{i}) else try l.q4Cols(m ++ "down_proj", .{i}, D, n * c.tp, lo, n),
            } };
        }
        return out;
    }

    /// The index and every shard's header: each tensor's shard and byte range.
    fn index(l: *Loader) !void {
        const path = try std.fmt.allocPrintSentinel(l.arena, "{s}/model.safetensors.index.json", .{l.dir}, 0);
        const f = try mtl.MappedFile.open(path);
        defer f.deinit();
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, l.arena, f.bytes[0..f.size], .{ .allocate = .alloc_always });
        var files: std.StringArrayHashMapUnmanaged(void) = .empty;
        var it = doc.object.get("weight_map").?.object.iterator();
        while (it.next()) |kv| try files.put(l.arena, kv.value_ptr.string, {});
        for (files.keys(), 0..) |name, si| {
            const full = try std.fmt.allocPrintSentinel(l.arena, "{s}/{s}", .{ l.dir, name }, 0);
            try l.shards.append(l.gpa, full);
            const fd = std.c.open(full, .{ .ACCMODE = .RDONLY });
            if (fd < 0) return error.OpenFailed;
            defer _ = std.c.close(fd);
            var head: [8]u8 = undefined;
            try readAll(fd, &head, 0);
            const hlen = std.mem.readInt(u64, &head, .little);
            const header = try l.arena.alloc(u8, hlen);
            try readAll(fd, header, 8);
            const end = std.c.lseek(fd, 0, std.c.SEEK.END);
            const data = 8 + hlen;
            const doc2 = try std.json.parseFromSliceLeaky(std.json.Value, l.arena, header, .{});
            var eit = doc2.object.iterator();
            while (eit.next()) |kv| {
                const tensor = kv.key_ptr.*;
                // the language model's tensors only: the vision tower's (5-D patch kernels among them) are never read
                if (!std.mem.startsWith(u8, tensor, prefix) and !std.mem.startsWith(u8, tensor, "lm_head.")) continue;
                const o = kv.value_ptr.object;
                const dtype = st.DType.parse(o.get("dtype").?.string) orelse return error.UnsupportedDType;
                const dims = o.get("shape").?.array.items;
                if (dims.len > st.max_rank) return error.RankTooHigh;
                var shape: [st.max_rank]usize = @splat(1);
                for (dims, 0..) |d, j| shape[j] = @intCast(d.integer);
                const offs = o.get("data_offsets").?.array.items;
                const b: u64 = @intCast(offs[0].integer);
                const e2: u64 = @intCast(offs[1].integer);
                if (e2 < b or data + e2 > @as(u64, @intCast(end))) return error.BadSafetensors;
                try l.names.put(l.arena, tensor, .{ .shard = @intCast(si), .off = data + b, .len = e2 - b, .dtype = dtype, .shape = shape, .rank = @intCast(dims.len) });
            }
        }
    }
};

const Pool = struct {
    fds: []std.c.fd_t,
    jobs: []const Copy,
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),

    fn run(p: *Pool) void {
        while (true) {
            const i = p.next.fetchAdd(1, .monotonic);
            if (i >= p.jobs.len) return;
            const j = p.jobs[i];
            readAll(p.fds[j.shard], j.dst[0..j.len], j.off) catch p.failed.store(true, .release);
        }
    }
};

/// Every copy on `threads` threads, the shards opened uncached (182 GB would only churn the page cache).
fn runCopies(l: *Loader, threads: usize) !void {
    const fds = try l.gpa.alloc(std.c.fd_t, l.shards.items.len);
    defer l.gpa.free(fds);
    for (l.shards.items, fds) |path, *fd| {
        fd.* = std.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd.* < 0) return error.OpenFailed;
        _ = std.c.fcntl(fd.*, 48, @as(c_int, 1)); // F_NOCACHE
    }
    defer for (fds) |fd| {
        _ = std.c.close(fd);
    };
    var pool: Pool = .{ .fds = fds, .jobs = l.copies.items };
    const workers = try l.gpa.alloc(?std.Thread, threads);
    defer l.gpa.free(workers);
    for (workers) |*t| t.* = std.Thread.spawn(.{}, Pool.run, .{&pool}) catch null;
    pool.run();
    for (workers) |t| if (t) |th| th.join();
    if (pool.failed.load(.acquire)) return error.ShortRead;
    // the transforms on the same threads (each reads its own tensors whole)
    const Tx = struct {
        l: *Loader,
        fds: []std.c.fd_t,
        next: std.atomic.Value(usize) = .init(0),
        failed: std.atomic.Value(bool) = .init(false),
        fn run(x: *@This()) void {
            while (true) {
                const i = x.next.fetchAdd(1, .monotonic);
                if (i >= x.l.transforms.items.len) return;
                load_fix.transform(x.l.gpa, x.fds, x.l.transforms.items[i]) catch x.failed.store(true, .release);
            }
        }
    };
    var tx: Tx = .{ .l = l, .fds = fds };
    for (workers) |*t| t.* = std.Thread.spawn(.{}, Tx.run, .{&tx}) catch null;
    tx.run();
    for (workers) |t| if (t) |th| th.join();
    if (tx.failed.load(.acquire)) return error.TransformFailed;
}

/// The first `c.run` layers, the MTP layer (when stored) and the head; `dry`: check names, dtypes and shapes, read nothing.
pub fn load(gpa: std.mem.Allocator, device: mtl.Device, dir: []const u8, c: *const cfg.Config, threads: usize, dry: bool) !*Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const w = try gpa.create(Weights);
    w.* = .{ .gpa = gpa };
    errdefer {
        w.deinit();
        gpa.destroy(w);
    }
    var l: Loader = .{ .gpa = gpa, .arena = arena_state.allocator(), .device = device, .c = c, .w = w, .dir = dir, .dry = dry };
    defer l.shards.deinit(gpa);
    defer l.copies.deinit(gpa);
    defer l.transforms.deinit(gpa);
    try l.index();
    const D: usize = c.hidden;
    const V: usize = c.vocab;
    try l.begin(2 * (Q4.wBytes(V, D) + 2 * Q4.sBytes(V, D)) + D * 2 + 8 * 256);
    w.embed = try l.q4(D, prefix ++ "embed_tokens", &.{""}, &.{V}, .{});
    const vp = c.vocabPart(); // TP2: this Mac's half of the head's rows
    w.head = try l.q4Rows(D, "lm_head", &.{""}, &.{vp[0]}, &.{vp[1] - vp[0]}, &.{V}, 1, .{});
    w.norm = try l.plain(prefix ++ "norm.weight", .{}, .bf16, &.{D});
    for (0..c.run) |i| w.layers[i] = try l.layer(i);
    const has_mtp = c.mtp > 0 and l.names.contains(prefix ++ "layers.45.eh_proj.weight");
    if (has_mtp) w.layers[c.layers] = try l.layer(c.layers);
    if (dry) {
        std.debug.print("glm plan: {d} of {d} layers{s}, {d:.2} GB in buffers, every tensor's name, dtype and shape checked\n", .{ c.run, c.layers, if (has_mtp) " and the MTP layer" else "", @as(f64, @floatFromInt(w.bytes)) / 1e9 });
        return w;
    }
    try runCopies(&l, threads);
    return w;
}
