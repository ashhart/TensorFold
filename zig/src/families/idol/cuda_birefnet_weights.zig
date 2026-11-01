//! BiRefNet's weights on CUDA: birefnet.safetensors (fp16) widened exactly to fp32, one device buffer.
const std = @import("std");
const cuda = @import("cuda");
const st = @import("../../core/safetensors.zig");
const cfg = @import("birefnet_config.zig");

/// A conv or linear as a GEMM: w [k * k * cin, ldw] by tap, bias [ldw], post [mean | scale | shift] (0: none).
pub const Conv = struct { w: u64, b: u64, post: u64 = 0, cin: usize, cout: usize, ldw: usize, k: usize };
pub const Norm = struct { g: u64, b: u64 };
pub const Block = struct { norm1: Norm, qkv: Conv, table: u64, proj: Conv, norm2: Norm, fc1: Conv, fc2: Conv };
pub const Layer = struct { blocks: [18]Block, merge: ?Norm, reduction: ?Conv };
pub const Deform = struct { offs: Conv, regular: Conv };
pub const Aspp = struct { branches: [4]Deform, pool: Conv, out: Conv };
pub const DecBlock = struct { conv_in: Conv, aspp: Aspp, conv_out: Conv };
pub const Simple = struct { conv1: Conv, conv_out: Conv };
pub const Dot = struct { w: u64, b: u64 };

pub const Weights = struct {
    buf: cuda.DeviceBuffer,
    patch: Conv,
    patch_norm: Norm,
    layers: [4]Layer,
    norms: [4]Norm,
    squeeze: DecBlock,
    /// ipt_blk5 .. ipt_blk1; ipt_blk1's conv1 is [27, 64] for tf_br_conv3.
    ipt: [5]Simple,
    /// decoder_block4 .. decoder_block1, then lateral_block4 .. 2, gdt_convs_4 .. 2 and their attention 1x1s.
    dec: [4]DecBlock,
    lateral: [3]Conv,
    gdt: [3]Conv,
    gdt_attn: [3]Dot,
    head: Dot,

    pub fn deinit(w: *Weights) void {
        w.buf.free();
    }
};

/// `path` (birefnet.safetensors) checked against the geometry and packed; Swin's position indices must match.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, path: []const u8) !Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    var f = try st.File.open(gpa, io, path);
    defer f.close(io);
    if (f.names.count() != cfg.tensor_count) {
        std.log.err("idol birefnet: {s} has {d} tensors; BiRefNet's Swin-L release has {d}", .{ path, f.names.count(), cfg.tensor_count });
        return error.UnexpectedTensors;
    }
    var w: Weights = undefined;
    var p: Packer = .{ .f = &f, .arena = arena_state.allocator() };
    try layout(&w, &p);
    w.buf = try cuda.DeviceBuffer.alloc(d, p.at * 4);
    errdefer w.buf.free();
    const image = try gpa.alloc(f32, p.at);
    defer gpa.free(image);
    p.out = image;
    p.base = w.buf.ptr;
    p.at = 0;
    p.read = 0;
    try layout(&w, &p);
    if (p.read != cfg.tensor_count) {
        std.log.err("idol birefnet: read {d} of the {d} tensors", .{ p.read, cfg.tensor_count });
        return error.UnexpectedTensors;
    }
    try w.buf.upload(0, std.mem.sliceAsBytes(image));
    return w;
}

fn cat(buf: []u8, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
}

fn layout(w: *Weights, p: *Packer) !void {
    var buf: [128]u8 = undefined;
    w.patch = try p.conv("bb.patch_embed.proj", 3, cfg.embed, 4, .{ .cin_pad = 64, .tap_major = false });
    w.patch_norm = try p.norm("bb.patch_embed.norm", cfg.embed);
    for (&w.layers, 0..) |*l, i| {
        const C = cfg.widths[i];
        for (l.blocks[0..cfg.depths[i]], 0..) |*b, j| {
            const pre = try std.fmt.bufPrint(&buf, "bb.layers.{d}.blocks.{d}", .{ i, j });
            var nb: [160]u8 = undefined;
            b.norm1 = try p.norm(try cat(&nb, pre, ".norm1"), C);
            b.qkv = try p.conv(try cat(&nb, pre, ".attn.qkv"), C, 3 * C, 1, .{});
            b.table = try p.copy(try cat(&nb, pre, ".attn.relative_position_bias_table"), &.{ 529, cfg.heads[i] });
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
    const ch = cfg.channels;
    w.squeeze = try p.decBlock("squeeze_module.0", cfg.squeeze_in, ch[0]);
    const ipt_in = [5]usize{ 3 << 10, 3 << 8, 3 << 6, 3 << 4, 3 };
    const ipt_out = [5]usize{ ch[0] / 8, ch[0] / 8, ch[1] / 8, ch[2] / 8, ch[3] / 8 };
    for (&w.ipt, 0..) |*s, i| {
        const n = 5 - i;
        s.conv1 = if (n == 1)
            try p.conv("decoder.ipt_blk1.conv1", 3, cfg.inter, 3, .{ .direct = true })
        else
            try p.conv(try std.fmt.bufPrint(&buf, "decoder.ipt_blk{d}.conv1", .{n}), ipt_in[i], cfg.inter, 3, .{});
        s.conv_out = try p.conv(try std.fmt.bufPrint(&buf, "decoder.ipt_blk{d}.conv_out", .{n}), cfg.inter, ipt_out[i], 3, .{});
    }
    for (&w.dec, 0..) |*dd, i| dd.* = try p.decBlock(try std.fmt.bufPrint(&buf, "decoder.decoder_block{d}", .{4 - i}), cfg.level_in[i], cfg.level_out[i]);
    for (0..3) |i| {
        const n = 4 - i;
        const C = ch[i + 1];
        w.lateral[i] = try p.conv(try std.fmt.bufPrint(&buf, "decoder.lateral_block{d}.conv", .{n}), C, C, 1, .{});
        w.gdt[i] = try p.conv(try std.fmt.bufPrint(&buf, "decoder.gdt_convs_{d}.0", .{n}), C, 16, 3, .{});
        w.gdt[i].post = try p.batchNorm(try std.fmt.bufPrint(&buf, "decoder.gdt_convs_{d}.1", .{n}), 16, w.gdt[i].ldw);
        w.gdt_attn[i] = try p.dot(try std.fmt.bufPrint(&buf, "decoder.gdt_convs_attn_{d}.0", .{n}), 16);
    }
    w.head = try p.dot("decoder.conv_out1.0", ch[3] / 2 + ch[3] / 8);
}

const Options = struct { bias: bool = true, cin_pad: ?usize = null, tap_major: bool = true, direct: bool = false };

/// Packs the tensors in one fp32 image: sizes on the first pass (out null), values on the second at `base`.
const Packer = struct {
    f: *const st.File,
    arena: std.mem.Allocator,
    base: u64 = 0,
    out: ?[]f32 = null,
    at: usize = 0,
    read: usize = 0,

    /// n floats, 64-float aligned: their device address.
    fn alloc(p: *Packer, n: usize) u64 {
        const r = p.base + p.at * 4;
        p.at = std.mem.alignForward(usize, p.at + n, 64);
        return r;
    }

    fn dst(p: *Packer, r: u64, n: usize) ?[]f32 {
        const out = p.out orelse return null;
        return out[(r - p.base) / 4 ..][0..n];
    }

    /// Tensor `name` of `shape`, fp16 as the file holds it, widened to fp32 exactly (as the SDK's load does).
    fn tensor(p: *Packer, name: []const u8, shape: []const usize) ![]const f32 {
        const t = p.f.get(name) orelse {
            std.log.err("idol birefnet: {s} is missing", .{name});
            return error.MissingTensor;
        };
        if (!t.is(.f16, shape)) {
            std.log.err("idol birefnet: {s} is {t} {any}; BiRefNet's is f16 {any}", .{ name, t.dtype, t.shape[0..t.rank], shape });
            return error.UnexpectedTensor;
        }
        p.read += 1;
        if (p.out == null) return &.{};
        const v = try p.arena.alloc(f32, t.numel());
        for (v, 0..) |*o, i| o.* = @as(f16, @bitCast(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)));
        return v;
    }

    fn copy(p: *Packer, name: []const u8, shape: []const usize) !u64 {
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

    /// relative_position_index must be (dy + 11) * 23 + dx + 11, as tf_br_window computes it.
    fn index(p: *Packer, name: []const u8) !void {
        const t = p.f.get(name) orelse return error.MissingTensor;
        if (!t.is(.i64, &.{ 144, 144 })) return error.UnexpectedTensor;
        p.read += 1;
        if (p.out == null) return;
        for (0..144 * 144) |at| {
            const v = std.mem.readInt(i64, t.bytes[8 * at ..][0..8], .little);
            const i = at / 144;
            const j = at % 144;
            const want: i64 = @intCast((i / 12 + 11 - j / 12) * 23 + (i % 12 + 11 - j % 12));
            if (v != want) return error.UnexpectedRelativeIndex;
        }
    }

    /// Batch norm `name` as post [mean | weight / sqrt(var + eps) | bias]; its counter is read and ignored.
    fn batchNorm(p: *Packer, name: []const u8, n: usize, ldw: usize) !u64 {
        var buf: [160]u8 = undefined;
        const mean = try p.tensor(try cat(&buf, name, ".running_mean"), &.{n});
        const var_ = try p.tensor(try cat(&buf, name, ".running_var"), &.{n});
        const g = try p.tensor(try cat(&buf, name, ".weight"), &.{n});
        const b = try p.tensor(try cat(&buf, name, ".bias"), &.{n});
        const counter = p.f.get(try cat(&buf, name, ".num_batches_tracked")) orelse return error.MissingTensor;
        if (counter.dtype != .i64 or counter.rank != 0) return error.UnexpectedTensor;
        p.read += 1;
        const r = p.alloc(3 * ldw);
        if (p.dst(r, 3 * ldw)) |d| {
            @memset(d, 0);
            for (0..n) |i| {
                d[i] = mean[i];
                d[ldw + i] = @floatCast(@as(f64, g[i]) / @sqrt(@as(f64, var_[i]) + cfg.bn_eps));
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
        const rows = taps * cfg.inter;
        const offs: Conv = .{ .w = p.alloc(rows * ldo), .b = p.alloc(ldo), .cin = cfg.inter, .cout = 3 * taps, .ldw = ldo, .k = k };
        if (p.dst(offs.w, rows * ldo)) |d| @memset(d, 0);
        if (p.dst(offs.b, ldo)) |d| @memset(d, 0);
        for ([_][]const u8{ ".atrous_conv.offset_conv", ".atrous_conv.modulator_conv" }, [_]usize{ 0, 2 * taps }, [_]usize{ 2 * taps, taps }) |part, first, n| {
            const pre = try cat(&nb, name, part);
            const g = try p.tensor(try cat(&buf, pre, ".weight"), &.{ n, cfg.inter, k, k });
            const b = try p.tensor(try cat(&buf, pre, ".bias"), &.{n});
            if (p.dst(offs.w, rows * ldo)) |d| for (0..n) |co| for (0..cfg.inter) |ci| for (0..taps) |tap| {
                d[(tap * cfg.inter + ci) * ldo + first + co] = g[(co * cfg.inter + ci) * taps + tap];
            };
            if (p.dst(offs.b, ldo)) |d| @memcpy(d[first..][0..n], b);
        }
        var regular = try p.conv(try cat(&buf, name, ".atrous_conv.regular_conv"), cfg.inter, cfg.aspp_planes, k, .{ .bias = false });
        regular.cin = taps * cfg.inter;
        regular.k = 1;
        regular.post = try p.batchNorm(try cat(&buf, name, ".bn"), cfg.aspp_planes, regular.ldw);
        return .{ .offs = offs, .regular = regular };
    }

    fn decBlock(p: *Packer, name: []const u8, cin: usize, cout: usize) !DecBlock {
        var buf: [160]u8 = undefined;
        var d: DecBlock = undefined;
        d.conv_in = try p.conv(try cat(&buf, name, ".conv_in"), cin, cfg.inter, 3, .{});
        d.conv_in.post = try p.batchNorm(try cat(&buf, name, ".bn_in"), cfg.inter, d.conv_in.ldw);
        const names = [4][]const u8{ ".dec_att.aspp1", ".dec_att.aspp_deforms.0", ".dec_att.aspp_deforms.1", ".dec_att.aspp_deforms.2" };
        for (&d.aspp.branches, names, cfg.branch_k) |*b, n, k| b.* = try p.deform(try cat(&buf, name, n), k);
        d.aspp.pool = try p.conv(try cat(&buf, name, ".dec_att.global_avg_pool.1"), cfg.inter, cfg.aspp_planes, 1, .{ .bias = false });
        d.aspp.pool.post = try p.batchNorm(try cat(&buf, name, ".dec_att.global_avg_pool.2"), cfg.aspp_planes, d.aspp.pool.ldw);
        d.aspp.out = try p.conv(try cat(&buf, name, ".dec_att.conv1"), 5 * cfg.aspp_planes, cfg.inter, 1, .{ .bias = false });
        d.aspp.out.post = try p.batchNorm(try cat(&buf, name, ".dec_att.bn1"), cfg.inter, d.aspp.out.ldw);
        d.conv_out = try p.conv(try cat(&buf, name, ".conv_out"), cfg.inter, cout, 3, .{});
        d.conv_out.post = try p.batchNorm(try cat(&buf, name, ".bn_out"), cout, d.conv_out.ldw);
        return d;
    }
};

test "the packer's addresses: 64-float aligned from the buffer's base, the first pass only sizes" {
    var p: Packer = .{ .f = undefined, .arena = std.testing.allocator, .base = 0x1000 };
    try std.testing.expectEqual(@as(u64, 0x1000), p.alloc(3));
    try std.testing.expectEqual(@as(u64, 0x1000 + 64 * 4), p.alloc(65));
    try std.testing.expectEqual(@as(usize, 192), p.at);
    try std.testing.expect(p.dst(0x1000, 3) == null);
}
