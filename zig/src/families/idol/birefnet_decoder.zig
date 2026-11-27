//! BiRefNet's squeeze and decoder on Metal: the blocks, image patches, gradient gates, laterals and the head.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("birefnet_kernels.zig");
const wts = @import("birefnet_weights.zig");
const bn = @import("birefnet.zig");
const Ref = ks.Ref;
const Net = bn.Net;
const C = wts.channels;

/// Each level's input channels (the previous level's output plus its image patches' convolution) and grid side.
pub const level_in = [4]usize{ C[0] + C[0] / 8, C[1] + C[0] / 8, C[2] + C[1] / 8, C[3] + C[2] / 8 };
pub const level_side = [4]usize{ 32, 64, 128, 256 };
/// Each level's output channels (decoder_block4 .. 1), and the image patches' tile side at each level.
pub const level_out = [4]usize{ C[1], C[2], C[3], C[3] / 2 };
const ipt_tile = [4]usize{ 32, 16, 8, 4 };

/// squeeze_module: the encoder's [32^2, 5,760] to the first level's input [32^2, 3,456], its first 3,072 channels.
pub fn squeeze(n: *Net, e: mtl.ComputeEncoder) void {
    block(n, e, &n.w.squeeze, n.xs[3], wts.squeeze_in, 32, n.lin[0], level_in[0], 0);
}

/// _BasicDecBlk on x [side^2, cin]: conv_in, BN, ReLU; ASPP; conv_out, BN into y [side^2, ldy] from column yoff.
pub fn block(n: *Net, e: mtl.ComputeEncoder, d: *const wts.DecBlock, x: Ref, cin: usize, side: usize, y: Ref, ldy: usize, yoff: usize) void {
    n.conv(e, d.conv_in, x, n.a64, .{ .rows = side, .cols = side, .ldy = wts.inter, .flags = ks.flag_relu, .ldx = cin });
    aspp(n, e, &d.aspp, side);
    n.conv(e, d.conv_out, n.gdt, y, .{ .rows = side, .cols = side, .ldy = ldy, .yoff = yoff, .ldx = wts.inter });
}

/// _ASPPDeformable, a64 into gdt [side^2, 64]: four deformable branches and the pooled one, concatenated, conv1.
fn aspp(n: *Net, e: mtl.ComputeEncoder, a: *const wts.Aspp, side: usize) void {
    const pixels = side * side;
    const ld = 5 * wts.aspp_planes;
    const g: ks.Grid = .{ .h = side, .w = side };
    for (a.branches, wts.branch_k, 0..) |b, k, i| {
        n.conv(e, b.offs, n.a64, n.offs, .{ .rows = side, .cols = side, .ldy = b.offs.ldw });
        var first: usize = 0;
        while (first < pixels) : (first += bn.chunk) {
            const count = @min(bn.chunk, pixels - first);
            ks.deform(e, n.k.get("br_deform"), n.a64, n.offs, n.cols, g, k, first, count, b.offs.ldw);
            n.conv(e, b.regular, n.cols, n.cat, .{ .rows = count, .ldy = ld, .yoff = i * wts.aspp_planes, .flags = ks.flag_relu, .row0 = first });
        }
    }
    ks.pool(e, n.k.get("br_pool"), n.a64, n.pooled, pixels, wts.inter);
    const pooled_out = n.pooled.at(wts.inter * 4);
    n.conv(e, a.pool, n.pooled, pooled_out, .{ .rows = 1, .ldy = wts.aspp_planes, .flags = ks.flag_relu });
    ks.resize(e, n.k.get("br_resize"), pooled_out, n.cat, .{ .h = 1, .w = 1 }, g, wts.aspp_planes, wts.aspp_planes, 0, ld, 4 * wts.aspp_planes);
    n.conv(e, a.out, n.cat, n.gdt, .{ .rows = pixels, .ldy = wts.inter, .flags = ks.flag_relu });
}

/// ipt_blk at level l: x's tiles of the level's side, conv1 and conv_out into the level's input past its channels.
fn patches(n: *Net, e: mtl.ComputeEncoder, l: usize) void {
    const side = level_side[l];
    const t = ipt_tile[l];
    const s = &n.w.ipt[l];
    const ch = 3 * t * t;
    ks.tiles(e, n.k.get("br_tiles"), n.x, n.patches, .{ .h = side, .w = side }, bn.size, 3, t, t, true, s.conv1.cin);
    n.conv(e, s.conv1, n.patches, n.a64, .{ .rows = side, .cols = side, .ldy = wts.inter });
    std.debug.assert(ch <= s.conv1.cin);
    n.conv(e, s.conv_out, n.a64, n.lin[l], .{ .rows = side, .cols = side, .ldy = level_in[l], .yoff = level_in[l] - s.conv_out.cout });
}

/// The decoder from lin[0]'s first 3,072 channels (the squeezed x4), xs[0..3] and x: the mask logits [1024^2].
pub fn decode(n: *Net, e: mtl.ComputeEncoder) void {
    for (0..4) |l| {
        if (l > 0) {
            // _p = resize(p, this grid) + lateral(x_l), beside this level's patches
            const prev = level_side[l - 1];
            const side = level_side[l];
            const cl = level_out[l - 1];
            ks.resize(e, n.k.get("br_resize"), n.out[l - 1], n.lin[l], .{ .h = prev, .w = prev }, .{ .h = side, .w = side }, cl, cl, 0, level_in[l], 0);
            n.convWith(e, n.w.lateral[l - 1], n.xs[3 - l], n.lin[l], n.lin[l], null, .{ .rows = side * side, .ldy = level_in[l], .flags = ks.flag_add });
        }
        patches(n, e, l);
        level(n, e, l);
    }
    head(n, e);
}

/// decoder_block at level l on lin[l] into out[l], then (levels 0 to 2) gated by its gradient attention.
pub fn level(n: *Net, e: mtl.ComputeEncoder, l: usize) void {
    const side = level_side[l];
    block(n, e, &n.w.dec[l], n.lin[l], level_in[l], side, n.out[l], level_out[l], 0);
    if (l == 3) return;
    n.conv(e, n.w.gdt[l], n.out[l], n.gdt, .{ .rows = side, .cols = side, .ldy = 16, .flags = ks.flag_relu, .ldx = level_out[l] });
    ks.gate(e, n.k.get("br_gate"), n.gdt, n.w.gdt_attn[l].w, n.w.gdt_attn[l].b, n.out[l], side * side, level_out[l]);
}

/// ipt_blk1 on the picture (3 to 64 to 48 channels at 1024^2) and conv_out1 over [resize(out[3]) | that]: the logits.
pub fn head(n: *Net, e: mtl.ComputeEncoder) void {
    const s = &n.w.ipt[4];
    const full: ks.Grid = .{ .h = bn.size, .w = bn.size };
    ks.conv3(e, n.k.get("br_conv3"), n.x, s.conv1.w, s.conv1.b, n.cat, full, wts.inter);
    n.conv(e, s.conv_out, n.cat, n.q48, .{ .rows = bn.size, .cols = bn.size, .ldy = s.conv_out.cout, .ldx = wts.inter });
    const side = level_side[3];
    ks.head(e, n.k.get("br_head"), n.out[3], n.q48, n.w.head.w, n.w.head.b, n.logits, .{ .h = side, .w = side }, full, level_out[3], s.conv_out.cout);
}

test "the levels' widths are the decoder blocks' inputs" {
    try std.testing.expectEqual([4]usize{ 3456, 1920, 960, 480 }, level_in);
    try std.testing.expectEqual(@as(usize, 240), level_out[3] + C[3] / 8);
}
