//! Fused DFlash2 decode passes on Metal: stacked KV/QKV/gate+up projections, KV ring scatter, QKV prep, conv+norm.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const sources = @import("kernel_sources");
const op = @import("../qwen27/dflash/operators.zig");
const w = @import("../qwen27/dflash/weights.zig");
const Config = @import("../qwen27/dflash/config.zig").Config;
const execution = @import("../qwen27/dflash/execution.zig");
const projection = @import("../qwen27/projection.zig");
const backend_mod = @import("metal_dflash_backend.zig");
const ring = @import("metal_dflash_context.zig").ring;

pub const StreamPos = extern struct { first: u64, row0: u32, rows: u32 };
pub const KvAppendArgs = extern struct { total: u32, kv_heads: u32, dim: u32, kv_stride: u32, kv_width: u32, ring: u32, streams: u32, eps: f32, theta: f32 };
pub const QkvArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, dim: u32, qkv_stride: u32, eps: f32, theta: f32 };
pub const ConvNormArgs = extern struct { rows: u32, width: u32, group: u32, block: u32, branch: u32, eps: f32 };

pub const LayerOpt = struct {
    kv: core.affine4_lane.Adapter,
    qkv: core.affine4_lane.Adapter,
    gate_up: core.affine4_lane.Adapter,

    fn deinit(l: LayerOpt) void {
        l.kv.deinit();
        l.qkv.deinit();
        l.gate_up.deinit();
    }
};

/// Stacked 4-bit lane projections and fused Metal kernels for GLM DFlash2 decode.
pub const Opt = struct {
    gpa: std.mem.Allocator,
    layers: []LayerOpt,
    kv_append: mtl.Pipeline,
    qkv_prep: mtl.Pipeline,
    conv_norm: mtl.Pipeline,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, checkpoint: *core.checkpoint_metal.Checkpoint, c: Config) !?Opt {
        const kv_sk = try projection.split(c.kvWidth(), c.hidden);
        if (!device.tensorUnits() or (try projection.split(c.qWidth(), c.hidden)) != kv_sk or
            c.head_dim > 128 or c.hidden % 128 != 0 or c.hidden > 4096) return null;
        const gu_sk = try projection.split(c.intermediate, c.hidden);
        const order = try core.lane_projection.RegOrder.init(device);
        defer order.deinit();
        const lib = try mtl.Library.fromSource(device, sources.glm_dflash_opt, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const kv_append = try mtl.Pipeline.init(device, lib, "glm_dflash_kv_append", false);
        errdefer kv_append.deinit();
        const qkv_prep = try mtl.Pipeline.init(device, lib, "glm_dflash_qkv_prep", false);
        errdefer qkv_prep.deinit();
        const conv_norm = try mtl.Pipeline.init(device, lib, "glm_dflash_conv_norm", false);
        errdefer conv_norm.deinit();
        const layers = try gpa.alloc(LayerOpt, c.layers);
        var built: usize = 0;
        errdefer {
            for (layers[0..built]) |l| l.deinit();
            gpa.free(layers);
        }
        for (layers, 0..) |*l, i| {
            l.* = try buildLayer(gpa, device, order, checkpoint, c, i, kv_sk, gu_sk);
            built += 1;
        }
        return .{ .gpa = gpa, .layers = layers, .kv_append = kv_append, .qkv_prep = qkv_prep, .conv_norm = conv_norm };
    }

    pub fn deinit(o: *Opt) void {
        for (o.layers) |l| l.deinit();
        o.gpa.free(o.layers);
        o.kv_append.deinit();
        o.qkv_prep.deinit();
        o.conv_norm.deinit();
    }

    pub fn bytes(o: *const Opt) usize {
        var n: usize = 0;
        for (o.layers) |l| {
            n += l.kv.words.length() + l.kv.metadata.length();
            n += l.qkv.words.length() + l.qkv.metadata.length();
            n += l.gate_up.words.length() + l.gate_up.metadata.length();
        }
        return n;
    }

    pub fn appendBuffers(o: *const Opt, out: *std.ArrayList(mtl.Buffer)) !void {
        for (o.layers) |l| try out.appendSlice(o.gpa, &.{ l.kv.words, l.kv.metadata, l.qkv.words, l.qkv.metadata, l.gate_up.words, l.gate_up.metadata });
    }

    /// Fused taps -> KV absorb: one KV projection per layer + single k_norm/RoPE/ring-scatter dispatch.
    pub fn absorbFromTaps(o: *const Opt, b: *backend_mod.Backend, ops: op.Ops, g: *const w.Graph, frame: u64, taps: op.View, streams: []const StreamPos, caches: []const []backend_mod.Cache, total: u32) !void {
        if (streams.len == 0 or streams.len > 8) return error.BadDraftContext;
        const c = g.config;
        const fusion = try ops.vtable.project(ops.ptr, frame, taps, g.fusion, .prepared_q4_reference);
        const hidden = try ops.vtable.norm(ops.ptr, frame, fusion, .{ .weight = g.hidden_norm, .eps = c.eps, .heads = 1, .head_dim = c.hidden });
        const href = try b.frame.ref(hidden);
        var st_buf: [8]StreamPos = @splat(.{ .first = 0, .row0 = total, .rows = 0 });
        @memcpy(st_buf[0..streams.len], streams);
        for (g.layers, 0..) |layer, i| {
            const kv_view = try b.frame.alloc(total, 2 * c.kvWidth(), .bf16);
            const kv_ref = try b.frame.ref(kv_view);
            try encodeAdapter(o.layers[i].kv, try b.encoder(frame), hidden, href, kv_view, kv_ref);
            const e = try b.encoder(frame);
            const kw = try b.weight(layer.attention.k_norm);
            e.setPipeline(o.kv_append);
            e.setBuffer(kv_ref.buf, kv_ref.off, 0);
            e.setBuffer(kw.buf, kw.off, 1);
            e.setValue(KvAppendArgs{ .total = total, .kv_heads = c.kv_heads, .dim = c.head_dim, .kv_stride = kv_view.stride, .kv_width = c.kvWidth(), .ring = ring(c), .streams = @intCast(streams.len), .eps = c.eps, .theta = @floatCast(c.theta) }, 2);
            e.setBytes(std.mem.sliceAsBytes(&st_buf), 3);
            for (0..8) |s| {
                const sc = caches[if (s < caches.len) s else 0][i];
                e.setBuffer(sc.keys, 0, 4 + s);
                e.setBuffer(sc.values, 0, 12 + s);
            }
            e.dispatchGroups(mtl.Size.of(c.kv_heads, total, 1), mtl.Size.of(32, 1, 1));
            e.barrier();
        }
    }

    /// Fused 5-layer DFlash2 forward: stacked QKV + gate/up, fused qkv_prep, and fused branch=1 conv+norm.
    pub fn encodeLayers(o: *const Opt, b: *backend_mod.Backend, ops: op.Ops, g: *const w.Graph, streams: []const execution.Stream, frame: u64, block: u32) !op.View {
        const c = g.config;
        const n: u32 = @intCast(streams.len);
        const rows = n * block;
        if (n == 0 or rows > 128 or n > 8) return error.BadRuntimeBlock;
        var ids: [128]u32 = @splat(c.mask);
        var positions: [128]u64 = undefined;
        var contexts: [8]op.Context = undefined;
        for (streams, 0..) |s, k| {
            contexts[k] = s.context;
            ids[k * block] = s.anchor;
            for (positions[k * block ..][0..block], 0..) |*p, j| p.* = try std.math.add(u64, s.context.end, j);
        }
        var hidden = try ops.vtable.embed(ops.ptr, frame, g.embed, ids[0..rows]);
        try hidden.check(rows, c.hidden, .bf16);
        const pos_ref = if (b.follow) |f| f.positions else blk: {
            const p = try b.frame.ref(try b.frame.alloc(rows, 1, .u64));
            @memcpy(@as([*]u64, @ptrCast(@alignCast(p.buf.contents() + p.off)))[0..rows], positions[0..rows]);
            break :blk p;
        };
        var xn = try ops.vtable.norm(ops.ptr, frame, hidden, .{ .weight = g.layers[0].input_norm, .eps = c.eps, .heads = 1, .head_dim = c.hidden });
        for (g.layers, 0..) |layer, i| {
            const residual = hidden;
            const dynamic = try ops.vtable.project(ops.ptr, frame, xn, layer.attention_conv.projection, .prepared_q4_reference);
            const input = try ops.vtable.conv(ops.ptr, frame, xn, .{ .dynamic = dynamic, .base = layer.attention_conv.base, .branch = 0, .group = c.group, .block = block, .residual = null });
            const qkv = try b.frame.alloc(rows, c.qWidth() + 2 * c.kvWidth(), .bf16);
            const qkv_ref = try b.frame.ref(qkv);
            try encodeAdapter(o.layers[i].qkv, try b.encoder(frame), input, try b.frame.ref(input), qkv, qkv_ref);
            const query = try b.frame.alloc(rows, c.qWidth(), .bf16);
            const keys = try b.frame.alloc(rows, c.kvWidth(), .bf16);
            const values = try b.frame.alloc(rows, c.kvWidth(), .bf16);
            try o.encodeQkvPrep(b, frame, qkv, qkv_ref, pos_ref, layer.attention.q_norm, layer.attention.k_norm, query, keys, values, c);
            const attended = try ops.vtable.attention(ops.ptr, frame, .{ .context = streams[0].context, .config = c, .layer = @intCast(i), .query = query, .keys = keys, .values = values, .positions = positions[0..rows], .streams = if (n > 1) contexts[0..n] else &.{}, .block = block });
            const o_out = try ops.vtable.project(ops.ptr, frame, attended, layer.attention.o, .prepared_q4_reference);
            const res2 = try b.frame.alloc(rows, c.hidden, .bf16);
            const post = try b.frame.alloc(rows, c.hidden, .bf16);
            try o.encodeConvNorm(b, frame, o_out, dynamic, layer.attention_conv.base, residual, layer.post_norm, res2, post, c, block);
            const dyn_mlp = try ops.vtable.project(ops.ptr, frame, post, layer.mlp_conv.projection, .prepared_q4_reference);
            const mlp_input = try ops.vtable.conv(ops.ptr, frame, post, .{ .dynamic = dyn_mlp, .base = layer.mlp_conv.base, .branch = 0, .group = c.group, .block = block, .residual = null });
            const gu = try b.frame.alloc(rows, 2 * c.intermediate, .bf16);
            const gu_ref = try b.frame.ref(gu);
            try encodeAdapter(o.layers[i].gate_up, try b.encoder(frame), mlp_input, try b.frame.ref(mlp_input), gu, gu_ref);
            const gate = try b.frame.add(gu_ref, rows, c.intermediate, gu.stride, .bf16);
            const up = try b.frame.add(.{ .buf = gu_ref.buf, .off = gu_ref.off + @as(usize, c.intermediate) * 2 }, rows, c.intermediate, gu.stride, .bf16);
            const activated = try ops.vtable.swiglu(ops.ptr, frame, gate, up);
            const down = try ops.vtable.project(ops.ptr, frame, activated, layer.mlp.down, .prepared_q4_reference);
            if (i + 1 < g.layers.len) {
                const next_h = try b.frame.alloc(rows, c.hidden, .bf16);
                const next_xn = try b.frame.alloc(rows, c.hidden, .bf16);
                try o.encodeConvNorm(b, frame, down, dyn_mlp, layer.mlp_conv.base, res2, g.layers[i + 1].input_norm, next_h, next_xn, c, block);
                hidden = next_h;
                xn = next_xn;
            } else {
                hidden = try ops.vtable.conv(ops.ptr, frame, down, .{ .dynamic = dyn_mlp, .base = layer.mlp_conv.base, .branch = 1, .group = c.group, .block = block, .residual = res2 });
            }
        }
        return hidden;
    }

    fn encodeQkvPrep(o: *const Opt, b: *backend_mod.Backend, frame: u64, qkv: op.View, qkv_ref: core.draft_ops.Ref, pos_ref: backend_mod.Ref, q_norm: w.Tensor, k_norm: w.Tensor, query: op.View, keys: op.View, values: op.View, c: Config) !void {
        const e = try b.encoder(frame);
        const qr = try b.frame.ref(query);
        const kr = try b.frame.ref(keys);
        const vr = try b.frame.ref(values);
        const qw = try b.weight(q_norm);
        const kw = try b.weight(k_norm);
        e.setPipeline(o.qkv_prep);
        e.setBuffer(qkv_ref.buf, qkv_ref.off, 0);
        e.setBuffer(pos_ref.buf, pos_ref.off, 1);
        e.setBuffer(qw.buf, qw.off, 2);
        e.setBuffer(kw.buf, kw.off, 3);
        e.setBuffer(qr.buf, qr.off, 4);
        e.setBuffer(kr.buf, kr.off, 5);
        e.setBuffer(vr.buf, vr.off, 6);
        e.setValue(QkvArgs{ .rows = qkv.rows, .heads = c.heads, .kv_heads = c.kv_heads, .dim = c.head_dim, .qkv_stride = qkv.stride, .eps = c.eps, .theta = @floatCast(c.theta) }, 7);
        e.dispatchGroups(mtl.Size.of(c.heads + c.kv_heads, qkv.rows, 1), mtl.Size.of(32, 1, 1));
        e.barrier();
    }

    fn encodeConvNorm(o: *const Opt, b: *backend_mod.Backend, frame: u64, x: op.View, dynamic: op.View, base: w.Tensor, residual: op.View, norm_w: w.Tensor, hidden_out: op.View, norm_out: op.View, c: Config, block: u32) !void {
        const e = try b.encoder(frame);
        const xr = try b.frame.ref(x);
        const dr = try b.frame.ref(dynamic);
        const br = try b.weight(base);
        const rr = try b.frame.ref(residual);
        const nw = try b.weight(norm_w);
        const hr = try b.frame.ref(hidden_out);
        const nr = try b.frame.ref(norm_out);
        e.setPipeline(o.conv_norm);
        e.setBuffer(xr.buf, xr.off, 0);
        e.setBuffer(dr.buf, dr.off, 1);
        e.setBuffer(br.buf, br.off, 2);
        e.setBuffer(rr.buf, rr.off, 3);
        e.setBuffer(nw.buf, nw.off, 4);
        e.setBuffer(hr.buf, hr.off, 5);
        e.setBuffer(nr.buf, nr.off, 6);
        e.setValue(ConvNormArgs{ .rows = x.rows, .width = c.hidden, .group = c.group, .block = block, .branch = 1, .eps = c.eps }, 7);
        const tg = ((c.hidden + 3) / 4 + 31) / 32 * 32;
        e.dispatchGroups(mtl.Size.of(x.rows, 1, 1), mtl.Size.of(tg, 1, 1));
        e.barrier();
    }
};

fn encodeAdapter(a: core.affine4_lane.Adapter, e: mtl.ComputeEncoder, x: op.View, xr: core.draft_ops.Ref, y: op.View, yr: core.draft_ops.Ref) !void {
    const most = a.projection.max_rows;
    var first: u32 = 0;
    while (first < x.rows) : (first += most) {
        try a.encode(e, .{ .buf = xr.buf, .off = xr.off + @as(usize, first) * x.stride * 2 }, .{ .buf = yr.buf, .off = yr.off + @as(usize, first) * y.stride * 2 }, @min(x.rows - first, most));
    }
    e.barrier();
}

fn readRows(checkpoint: *core.checkpoint_metal.Checkpoint, name: []const u8, dst: []u16) !void {
    if (checkpoint.tensors.get(name)) |t| {
        @memcpy(dst, t.host(u16));
    } else {
        try checkpoint.read(name, 0, std.mem.sliceAsBytes(dst));
    }
}

fn stackAdapter(gpa: std.mem.Allocator, device: mtl.Device, order: core.lane_projection.RegOrder, checkpoint: *core.checkpoint_metal.Checkpoint, names: []const []const u8, member_rows: []const usize, k: usize, sk: u32) !core.affine4_lane.Adapter {
    var total_n: usize = 0;
    for (member_rows) |nr| total_n += nr;
    const buf = try gpa.alloc(u16, total_n * k);
    defer gpa.free(buf);
    var off: usize = 0;
    for (names, member_rows) |name, nr| {
        const need = nr * k;
        try readRows(checkpoint, name, buf[off..][0..need]);
        off += need;
    }
    var prep = try core.affine4.prepare(gpa, buf, total_n, k);
    defer prep.deinit();
    return core.affine4_lane.Adapter.init(gpa, device, prep, sk, order);
}

fn buildLayer(gpa: std.mem.Allocator, device: mtl.Device, order: core.lane_projection.RegOrder, checkpoint: *core.checkpoint_metal.Checkpoint, c: Config, i: usize, kv_sk: u32, gu_sk: u32) !LayerOpt {
    var bq: [64]u8 = undefined;
    var bk: [64]u8 = undefined;
    var bv: [64]u8 = undefined;
    var bg: [64]u8 = undefined;
    var bu: [64]u8 = undefined;
    const nq = try std.fmt.bufPrint(&bq, "layers.{d}.self_attn.q_proj.weight", .{i});
    const nk = try std.fmt.bufPrint(&bk, "layers.{d}.self_attn.k_proj.weight", .{i});
    const nv = try std.fmt.bufPrint(&bv, "layers.{d}.self_attn.v_proj.weight", .{i});
    const ng = try std.fmt.bufPrint(&bg, "layers.{d}.mlp.gate_proj.weight", .{i});
    const nu = try std.fmt.bufPrint(&bu, "layers.{d}.mlp.up_proj.weight", .{i});
    const kv = try stackAdapter(gpa, device, order, checkpoint, &.{ nk, nv }, &.{ c.kvWidth(), c.kvWidth() }, c.hidden, kv_sk);
    errdefer kv.deinit();
    const qkv = try stackAdapter(gpa, device, order, checkpoint, &.{ nq, nk, nv }, &.{ c.qWidth(), c.kvWidth(), c.kvWidth() }, c.hidden, kv_sk);
    errdefer qkv.deinit();
    const gate_up = try stackAdapter(gpa, device, order, checkpoint, &.{ ng, nu }, &.{ c.intermediate, c.intermediate }, c.hidden, gu_sk);
    return .{ .kv = kv, .qkv = qkv, .gate_up = gate_up };
}
