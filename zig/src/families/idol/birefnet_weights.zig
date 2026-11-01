//! BiRefNet's weights (speridlabs/idol's birefnet.safetensors, fp16) as fp32 GEMM operands in one Metal buffer.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
pub const Ref = @import("../../core/qwen_vl/weights.zig").Ref;

const cfg = @import("birefnet_config.zig");

/// The released geometry (birefnet_config.zig): Swin-L, the decoder's widths, the checkpoint's tensors.
pub const embed = cfg.embed;
pub const depths = cfg.depths;
pub const heads = cfg.heads;
pub const window = cfg.window;
pub const widths = cfg.widths;
pub const channels = cfg.channels;
pub const inter = cfg.inter;
pub const squeeze_in = cfg.squeeze_in;
pub const aspp_planes = cfg.aspp_planes;
pub const tensor_count = cfg.tensor_count;
pub const ln_eps = cfg.ln_eps;
pub const bn_eps = cfg.bn_eps;

/// A conv or linear as a GEMM: w [k * k * cin, ldw] by tap, bias [ldw], post [mean | scale | shift] after a BN.
pub const Conv = struct { w: Ref, b: Ref, post: ?Ref = null, cin: usize, cout: usize, ldw: usize, k: usize };
pub const Norm = struct { g: Ref, b: Ref };
pub const Block = struct { norm1: Norm, qkv: Conv, table: Ref, proj: Conv, norm2: Norm, fc1: Conv, fc2: Conv };
pub const Layer = struct { blocks: [18]Block, merge: ?Norm, reduction: ?Conv };
/// A modulated deformable conv: offsets (2 k^2) and modulators (k^2) as one conv, then the conv over its columns.
pub const Deform = struct { offs: Conv, regular: Conv };
pub const Aspp = struct { branches: [4]Deform, pool: Conv, out: Conv };
pub const DecBlock = struct { conv_in: Conv, aspp: Aspp, conv_out: Conv };
pub const Simple = struct { conv1: Conv, conv_out: Conv };
pub const Dot = struct { w: Ref, b: Ref };

pub const branch_k = cfg.branch_k;

pub const Weights = struct {
    buf: mtl.Buffer,
    patch: Conv,
    patch_norm: Norm,
    layers: [4]Layer,
    norms: [4]Norm,
    squeeze: DecBlock,
    /// ipt_blk5 .. ipt_blk1; ipt_blk1's conv1 is [27, 64] for br_conv3.
    ipt: [5]Simple,
    /// decoder_block4 .. decoder_block1, then lateral_block4 .. 2, gdt_convs_4 .. 2 and their attention 1x1s.
    dec: [4]DecBlock,
    lateral: [3]Conv,
    gdt: [3]Conv,
    gdt_attn: [3]Dot,
    head: Dot,

    pub fn deinit(w: *Weights) void {
        w.buf.deinit();
    }
};

/// `path` (birefnet.safetensors) checked against the geometry and packed; Swin's position indices must match.
pub fn load(gpa: std.mem.Allocator, device: mtl.Device, path: [:0]const u8) !Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    var c = ck.Checkpoint.init(gpa);
    defer c.deinit();
    try c.addFile(device, path, "");
    if (c.tensors.count() != tensor_count) {
        std.log.err("idol birefnet: {s} has {d} tensors; BiRefNet's Swin-L release has {d}", .{ path, c.tensors.count(), tensor_count });
        return error.UnexpectedTensors;
    }
    var w: Weights = undefined;
    var p: Packer = .{ .c = &c, .arena = arena_state.allocator() };
    try layout(&w, &p);
    w.buf = try device.buffer(p.at * 4, mtl.ResourceOptions.shared);
    errdefer w.buf.deinit();
    p.out = @as([*]f32, @ptrCast(@alignCast(w.buf.contents())))[0..p.at];
    p.at = 0;
    p.read = 0;
    p.buf = w.buf;
    try layout(&w, &p);
    if (p.read != tensor_count) {
        std.log.err("idol birefnet: read {d} of the {d} tensors", .{ p.read, tensor_count });
        return error.UnexpectedTensors;
    }
    return w;
}

fn layout(w: *Weights, p: *Packer) !void {
    var buf: [128]u8 = undefined;
    w.patch = try p.conv("bb.patch_embed.proj", 3, embed, 4, .{ .cin_pad = 64, .tap_major = false });
    w.patch_norm = try p.norm("bb.patch_embed.norm", embed);
    for (&w.layers, 0..) |*l, i| {
        const C = widths[i];
        for (l.blocks[0..depths[i]], 0..) |*b, j| {
            const pre = try std.fmt.bufPrint(&buf, "bb.layers.{d}.blocks.{d}", .{ i, j });
            var nb: [160]u8 = undefined;
            b.norm1 = try p.norm(try cat(&nb, pre, ".norm1"), C);
            b.qkv = try p.conv(try cat(&nb, pre, ".attn.qkv"), C, 3 * C, 1, .{});
            b.table = try p.copy(try cat(&nb, pre, ".attn.relative_position_bias_table"), &.{ 529, heads[i] });
            try p.index(try cat(&nb, pre, ".attn.relative_position_index"));
            b.proj = try p.conv(try cat(&nb, pre, ".attn.proj"), C, C, 1, .{});
            b.norm2 = try p.norm(try cat(&nb, pre, ".norm2"), C);
            b.fc1 = try p.conv(try cat(&nb, pre, ".mlp.fc1"), C, 4 * C, 1, .{});
            b.fc2 = try p.conv(try cat(&nb, pre, ".mlp.fc2"), 4 * C, C, 1, .{});
        }
        if (i < 3) {
            l.merge = try p.norm(try std.fmt.bufPrint(&buf, "bb.layers.{d}.downsample.norm", .{i}), 4 * C);
            l.reduction = try p.conv(try std.fmt.bufPrint(&buf, "bb.layers.{d}.downsample.reduction", .{i}), 4 * C, 2 * C, 1, .{ .bias = false });
        } else {
            l.merge = null;
            l.reduction = null;
        }
        w.norms[i] = try p.norm(try std.fmt.bufPrint(&buf, "bb.norm{d}", .{i}), C);
    }
    w.squeeze = try p.decBlock("squeeze_module.0", squeeze_in, channels[0]);
    const ipt_in = [5]usize{ 3 << 10, 3 << 8, 3 << 6, 3 << 4, 3 };
    const ipt_out = [5]usize{ channels[0] / 8, channels[0] / 8, channels[1] / 8, channels[2] / 8, channels[3] / 8 };
    for (&w.ipt, 0..) |*s, i| {
        const n = 5 - i;
        s.conv1 = if (n == 1)
            try p.conv("decoder.ipt_blk1.conv1", 3, inter, 3, .{ .direct = true })
        else
            try p.conv(try std.fmt.bufPrint(&buf, "decoder.ipt_blk{d}.conv1", .{n}), ipt_in[i], inter, 3, .{});
        s.conv_out = try p.conv(try std.fmt.bufPrint(&buf, "decoder.ipt_blk{d}.conv_out", .{n}), inter, ipt_out[i], 3, .{});
    }
    const dec_in = [4]usize{ channels[0] + channels[0] / 8, channels[1] + channels[0] / 8, channels[2] + channels[1] / 8, channels[3] + channels[2] / 8 };
    const dec_out = [4]usize{ channels[1], channels[2], channels[3], channels[3] / 2 };
    for (&w.dec, 0..) |*d, i| d.* = try p.decBlock(try std.fmt.bufPrint(&buf, "decoder.decoder_block{d}", .{4 - i}), dec_in[i], dec_out[i]);
    for (0..3) |i| {
        const n = 4 - i;
        const C = channels[i + 1];
        w.lateral[i] = try p.conv(try std.fmt.bufPrint(&buf, "decoder.lateral_block{d}.conv", .{n}), C, C, 1, .{});
        w.gdt[i] = try p.conv(try std.fmt.bufPrint(&buf, "decoder.gdt_convs_{d}.0", .{n}), C, 16, 3, .{});
        w.gdt[i].post = try p.batchNorm(try std.fmt.bufPrint(&buf, "decoder.gdt_convs_{d}.1", .{n}), 16, w.gdt[i].ldw);
        w.gdt_attn[i] = try p.dot(try std.fmt.bufPrint(&buf, "decoder.gdt_convs_attn_{d}.0", .{n}), 16);
    }
    w.head = try p.dot("decoder.conv_out1.0", channels[3] / 2 + channels[3] / 8);
}

fn cat(buf: []u8, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
}

const Options = struct { bias: bool = true, cin_pad: ?usize = null, tap_major: bool = true, direct: bool = false };

/// Packs the tensors in one fp32 buffer: sizes on the first pass (out null), copies on the second.
const Packer = struct {
    c: *const ck.Checkpoint,
    arena: std.mem.Allocator,
    buf: mtl.Buffer = undefined,
    out: ?[]f32 = null,
    at: usize = 0,
    read: usize = 0,

    fn alloc(p: *Packer, n: usize) Ref {
        const r: Ref = .{ .buf = p.buf, .off = p.at * 4 };
        p.at = std.mem.alignForward(usize, p.at + n, 64);
        return r;
    }

    fn dst(p: *Packer, r: Ref, n: usize) ?[]f32 {
        const out = p.out orelse return null;
        return out[r.off / 4 ..][0..n];
    }

    /// Tensor `name` of `shape`, fp16 as the file holds it, widened to fp32 exactly (as the SDK's load does).
    fn tensor(p: *Packer, name: []const u8, shape: []const usize) ![]const f32 {
        const t = p.c.get(name) catch {
            std.log.err("idol birefnet: {s} is missing", .{name});
            return error.MissingTensor;
        };
        if (t.dtype != .f16 or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
            std.log.err("idol birefnet: {s} is {t} {any}; BiRefNet's is f16 {any}", .{ name, t.dtype, t.shape[0..t.rank], shape });
            return error.UnexpectedTensor;
        }
        p.read += 1;
        if (p.out == null) return &.{};
        const v = try p.arena.alloc(f32, t.count());
        for (v, t.host(f16)) |*o, h| o.* = h;
        return v;
    }

    fn copy(p: *Packer, name: []const u8, shape: []const usize) !Ref {
        var n: usize = 1;
        for (shape) |s| n *= s;
        const v = try p.tensor(name, shape);
        const r = p.alloc(n);
        if (p.dst(r, n)) |d| @memcpy(d, v);
        return r;
    }

    fn norm(p: *Packer, name: []const u8, width: usize) !Norm {
        var buf: [160]u8 = undefined;
        return .{ .g = try p.copy(try cat(&buf, name, ".weight"), &.{width}), .b = try p.copy(try cat(&buf, name, ".bias"), &.{width}) };
    }

    /// relative_position_index must be (dy + 11) * 23 + dx + 11, as br_window computes it.
    fn index(p: *Packer, name: []const u8) !void {
        const t = try p.c.get(name);
        if (t.dtype != .i64 or t.rank != 2 or t.shape[0] != 144 or t.shape[1] != 144) return error.UnexpectedTensor;
        p.read += 1;
        if (p.out == null) return;
        for (t.host(i64), 0..) |v, at| {
            const i = at / 144;
            const j = at % 144;
            const want: i64 = @intCast((i / 12 + 11 - j / 12) * 23 + (i % 12 + 11 - j % 12));
            if (v != want) return error.UnexpectedRelativeIndex;
        }
    }

    /// Batch norm `name` as post [mean | weight / sqrt(var + eps) | bias]; its counter is read and ignored.
    fn batchNorm(p: *Packer, name: []const u8, n: usize, ldw: usize) !Ref {
        var buf: [160]u8 = undefined;
        const mean = try p.tensor(try cat(&buf, name, ".running_mean"), &.{n});
        const var_ = try p.tensor(try cat(&buf, name, ".running_var"), &.{n});
        const g = try p.tensor(try cat(&buf, name, ".weight"), &.{n});
        const b = try p.tensor(try cat(&buf, name, ".bias"), &.{n});
        const counter = try p.c.get(try cat(&buf, name, ".num_batches_tracked"));
        if (counter.dtype != .i64 or counter.rank != 0) return error.UnexpectedTensor;
        p.read += 1;
        const r = p.alloc(3 * ldw);
        if (p.dst(r, 3 * ldw)) |d| {
            @memset(d, 0);
            for (0..n) |i| {
                d[i] = mean[i];
                d[ldw + i] = @floatCast(@as(f64, g[i]) / @sqrt(@as(f64, var_[i]) + bn_eps));
                d[2 * ldw + i] = b[i];
            }
        }
        return r;
    }

    /// Conv `name` as w [taps * cin_pad, ldw] (row tap * cin_pad + c); tap_major false: torch's order; direct: [27, n].
    fn conv(p: *Packer, name: []const u8, cin: usize, cout: usize, k: usize, o: Options) !Conv {
        var buf: [160]u8 = undefined;
        const taps = k * k;
        const cin_pad = o.cin_pad orelse if (o.direct) cin else std.mem.alignForward(usize, cin, 32);
        const ldw = if (o.direct) cout else std.mem.alignForward(usize, cout, 64);
        const linear = std.mem.indexOf(u8, name, "bb.layers") != null and k == 1;
        const shape: []const usize = if (linear) &.{ cout, cin } else &.{ cout, cin, k, k };
        const g = try p.tensor(try cat(&buf, name, ".weight"), shape);
        const rows = if (o.tap_major) taps * cin_pad else cin_pad;
        const r = p.alloc(rows * ldw);
        if (p.dst(r, rows * ldw)) |d| {
            @memset(d, 0);
            for (0..cout) |co| for (0..cin) |ci| for (0..taps) |tap| {
                const row = if (o.tap_major) tap * cin_pad + ci else ci * taps + tap;
                d[row * ldw + co] = g[(co * cin + ci) * taps + tap];
            };
        }
        const b = p.alloc(ldw);
        if (o.bias) {
            const v = try p.tensor(try cat(&buf, name, ".bias"), &.{cout});
            if (p.dst(b, ldw)) |d| {
                @memset(d, 0);
                @memcpy(d[0..cout], v);
            }
        } else if (p.dst(b, ldw)) |d| @memset(d, 0);
        return .{ .w = r, .b = b, .cin = cin_pad, .cout = cout, .ldw = ldw, .k = if (o.tap_major) k else 1 };
    }

    /// A one-output 1x1 [1, n, 1, 1] and its bias, as plain vectors.
    fn dot(p: *Packer, name: []const u8, n: usize) !Dot {
        var buf: [160]u8 = undefined;
        return .{ .w = try p.copy(try cat(&buf, name, ".weight"), &.{ 1, n, 1, 1 }), .b = try p.copy(try cat(&buf, name, ".bias"), &.{1}) };
    }

    /// One deformable branch `name` (atrous_conv's three convolutions and bn) with kernel side k over 64 channels.
    fn deform(p: *Packer, name: []const u8, k: usize) !Deform {
        var buf: [160]u8 = undefined;
        var nb: [160]u8 = undefined;
        const taps = k * k;
        const ldo = std.mem.alignForward(usize, 3 * taps, 64);
        const rows = taps * inter;
        const offs: Conv = .{ .w = p.alloc(rows * ldo), .b = p.alloc(ldo), .cin = inter, .cout = 3 * taps, .ldw = ldo, .k = k };
        if (p.dst(offs.w, rows * ldo)) |d| @memset(d, 0);
        if (p.dst(offs.b, ldo)) |d| @memset(d, 0);
        for ([_][]const u8{ ".atrous_conv.offset_conv", ".atrous_conv.modulator_conv" }, [_]usize{ 0, 2 * taps }, [_]usize{ 2 * taps, taps }) |part, first, n| {
            const pre = try cat(&nb, name, part);
            const g = try p.tensor(try cat(&buf, pre, ".weight"), &.{ n, inter, k, k });
            const b = try p.tensor(try cat(&buf, pre, ".bias"), &.{n});
            if (p.dst(offs.w, rows * ldo)) |d| for (0..n) |co| for (0..inter) |ci| for (0..taps) |tap| {
                d[(tap * inter + ci) * ldo + first + co] = g[(co * inter + ci) * taps + tap];
            };
            if (p.dst(offs.b, ldo)) |d| @memcpy(d[first..][0..n], b);
        }
        var regular = try p.conv(try cat(&buf, name, ".atrous_conv.regular_conv"), inter, aspp_planes, k, .{ .bias = false });
        regular.cin = taps * inter;
        regular.k = 1;
        regular.post = try p.batchNorm(try cat(&buf, name, ".bn"), aspp_planes, regular.ldw);
        return .{ .offs = offs, .regular = regular };
    }

    fn decBlock(p: *Packer, name: []const u8, cin: usize, cout: usize) !DecBlock {
        var buf: [160]u8 = undefined;
        var d: DecBlock = undefined;
        d.conv_in = try p.conv(try cat(&buf, name, ".conv_in"), cin, inter, 3, .{});
        d.conv_in.post = try p.batchNorm(try cat(&buf, name, ".bn_in"), inter, d.conv_in.ldw);
        const names = [4][]const u8{ ".dec_att.aspp1", ".dec_att.aspp_deforms.0", ".dec_att.aspp_deforms.1", ".dec_att.aspp_deforms.2" };
        for (&d.aspp.branches, names, branch_k) |*b, n, k| b.* = try p.deform(try cat(&buf, name, n), k);
        d.aspp.pool = try p.conv(try cat(&buf, name, ".dec_att.global_avg_pool.1"), inter, aspp_planes, 1, .{ .bias = false });
        d.aspp.pool.post = try p.batchNorm(try cat(&buf, name, ".dec_att.global_avg_pool.2"), aspp_planes, d.aspp.pool.ldw);
        d.aspp.out = try p.conv(try cat(&buf, name, ".dec_att.conv1"), 5 * aspp_planes, inter, 1, .{ .bias = false });
        d.aspp.out.post = try p.batchNorm(try cat(&buf, name, ".dec_att.bn1"), inter, d.aspp.out.ldw);
        d.conv_out = try p.conv(try cat(&buf, name, ".conv_out"), inter, cout, 3, .{});
        d.conv_out.post = try p.batchNorm(try cat(&buf, name, ".bn_out"), cout, d.conv_out.ldw);
        return d;
    }
};
