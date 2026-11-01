//! BiRefNet's squeeze and decoder on CUDA (blocks, patches, gates, laterals, head), as birefnet_decoder.zig.
const std = @import("std");
const cfg = @import("birefnet_config.zig");
const ks = @import("cuda_birefnet_kernels.zig");
const wts = @import("cuda_birefnet_weights.zig");
const Net = @import("cuda_birefnet.zig").Net;

const level_in = cfg.level_in;
const level_side = cfg.level_side;
const level_out = cfg.level_out;

/// squeeze_module: the encoder's [32^2, 5,760] to the first level's input [32^2, 3,456], its first 3,072 channels.
pub fn squeeze(n: *Net) !void {
    try block(n, &n.w.squeeze, n.xs[3], cfg.squeeze_in, 32, n.lin[0], level_in[0], 0);
}

/// _BasicDecBlk on x [side^2, cin]: conv_in, BN, ReLU; ASPP; conv_out, BN into y [side^2, ldy] from column yoff.
pub fn block(n: *Net, d: *const wts.DecBlock, x: u64, cin: usize, side: usize, y: u64, ldy: usize, yoff: usize) !void {
    try n.conv(d.conv_in, x, n.a64, .{ .rows = side, .cols = side, .ldy = cfg.inter, .flags = ks.flag_relu, .ldx = cin });
    try aspp(n, &d.aspp, side);
    try n.conv(d.conv_out, n.gdt, y, .{ .rows = side, .cols = side, .ldy = ldy, .yoff = yoff, .ldx = cfg.inter });
}

/// _ASPPDeformable, a64 into gdt [side^2, 64]: four deformable branches and the pooled one, concatenated, conv1.
fn aspp(n: *Net, a: *const wts.Aspp, side: usize) !void {
    const pixels = side * side;
    const ld = 5 * cfg.aspp_planes;
    const g: ks.Grid = .{ .h = side, .w = side };
    for (a.branches, cfg.branch_k, 0..) |b, k, i| {
        try n.conv(b.offs, n.a64, n.offs, .{ .rows = side, .cols = side, .ldy = b.offs.ldw });
        var first: usize = 0;
        while (first < pixels) : (first += cfg.chunk) {
            const count = @min(cfg.chunk, pixels - first);
            try ks.deform(&n.k, n.stream, n.a64, n.offs, n.cols, g, k, first, count, b.offs.ldw);
            try n.conv(b.regular, n.cols, n.cat, .{ .rows = count, .ldy = ld, .yoff = i * cfg.aspp_planes, .flags = ks.flag_relu, .row0 = first });
        }
    }
    try ks.pool(&n.k, n.stream, n.a64, n.pooled, pixels, cfg.inter);
    const pooled_out = n.pooled + cfg.inter * 4;
    try n.conv(a.pool, n.pooled, pooled_out, .{ .rows = 1, .ldy = cfg.aspp_planes, .flags = ks.flag_relu });
    try ks.resize(&n.k, n.stream, pooled_out, n.cat, .{ .h = 1, .w = 1 }, g, cfg.aspp_planes, cfg.aspp_planes, 0, ld, 4 * cfg.aspp_planes);
    try n.conv(a.out, n.cat, n.gdt, .{ .rows = pixels, .ldy = cfg.inter, .flags = ks.flag_relu });
}

/// ipt_blk at level l: x's tiles of the level's side, conv1 and conv_out into the level's input past its channels.
fn patches(n: *Net, l: usize) !void {
    const side = level_side[l];
    const t = cfg.ipt_tile[l];
    const s = &n.w.ipt[l];
    std.debug.assert(3 * t * t <= s.conv1.cin);
    try ks.tiles(&n.k, n.stream, n.x, n.patches, .{ .h = side, .w = side }, cfg.size, 3, t, t, true, s.conv1.cin);
    try n.conv(s.conv1, n.patches, n.a64, .{ .rows = side, .cols = side, .ldy = cfg.inter });
    try n.conv(s.conv_out, n.a64, n.lin[l], .{ .rows = side, .cols = side, .ldy = level_in[l], .yoff = level_in[l] - s.conv_out.cout });
}

/// The decoder from lin[0]'s first 3,072 channels (the squeezed x4), xs[0..3] and x: the mask logits [1024^2].
pub fn decode(n: *Net) !void {
    for (0..4) |l| {
        if (l > 0) {
            // _p = resize(p, this grid) + lateral(x_l), beside this level's patches
            const prev = level_side[l - 1];
            const side = level_side[l];
            const cl = level_out[l - 1];
            try ks.resize(&n.k, n.stream, n.out[l - 1], n.lin[l], .{ .h = prev, .w = prev }, .{ .h = side, .w = side }, cl, cl, 0, level_in[l], 0);
            try n.convWith(n.w.lateral[l - 1], n.xs[3 - l], n.lin[l], n.lin[l], 0, .{ .rows = side * side, .ldy = level_in[l], .flags = ks.flag_add });
        }
        try patches(n, l);
        try level(n, l);
    }
    try head(n);
}

/// decoder_block at level l on lin[l] into out[l], then (levels 0 to 2) gated by its gradient attention.
pub fn level(n: *Net, l: usize) !void {
    const side = level_side[l];
    try block(n, &n.w.dec[l], n.lin[l], level_in[l], side, n.out[l], level_out[l], 0);
    if (l == 3) return;
    try n.conv(n.w.gdt[l], n.out[l], n.gdt, .{ .rows = side, .cols = side, .ldy = 16, .flags = ks.flag_relu, .ldx = level_out[l] });
    try ks.gate(&n.k, n.stream, n.gdt, n.w.gdt_attn[l].w, n.w.gdt_attn[l].b, n.out[l], side * side, level_out[l]);
}

/// ipt_blk1 on the picture (3 to 64 to 48 channels at 1024^2) and conv_out1 over [resize(out[3]) | that]: the logits.
pub fn head(n: *Net) !void {
    const s = &n.w.ipt[4];
    const full: ks.Grid = .{ .h = cfg.size, .w = cfg.size };
    try ks.conv3(&n.k, n.stream, n.x, s.conv1.w, s.conv1.b, n.cat, full, cfg.inter);
    try n.conv(s.conv_out, n.cat, n.q48, .{ .rows = cfg.size, .cols = cfg.size, .ldy = s.conv_out.cout, .ldx = cfg.inter });
    const side = level_side[3];
    try ks.head(&n.k, n.stream, n.out[3], n.q48, n.w.head.w, n.w.head.b, n.logits, .{ .h = side, .w = side }, full, level_out[3], s.conv_out.cout);
}
