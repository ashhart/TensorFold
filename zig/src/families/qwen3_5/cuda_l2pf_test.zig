//! Real projection unions feed the optional prefetch without consulting scratch or embedding views.
const std = @import("std");
const cuda = @import("cuda");
const pf = @import("cuda_l2pf.zig");
const layers = @import("cuda_layers.zig");
const t = std.testing;
fn bf(p: u64) layers.Proj {
    return .{ .bf = .{ .w = p, .n = 64, .k = 64 } };
}
test "L2 prefetch: Qwen dense and routed sources select only resident projections" {
    var l: [2]layers.Layer = undefined;
    l[0].mixer = .{ .delta = .{ .qkv = bf(0x1000), .z = bf(0x2000), .a = bf(0x3000), .b = bf(0x4000), .out = bf(0x5000), .conv = 0, .norm = 0, .a_log = 0, .dt_bias = 0 } };
    l[0].mlp = .{ .dense = .{ .gate = bf(0x6000), .up = bf(0x7000), .down = bf(0x8000) } };
    l[1].mixer = .{ .attention = .{ .q = bf(0x9000), .k = bf(0xa000), .v = bf(0xb000), .o = bf(0xc000), .q_norm = 0, .k_norm = 0 } };
    const n = pf.sources(&l, 0, .n);
    try t.expectEqualSlices(u64, &.{ 0x9000, 0xa000, 0xb000 }, n.ptrs[0..n.len]);
    const f = pf.sources(&l, 0, .f);
    try t.expectEqualSlices(u64, &.{ 0x6000, 0x7000 }, f.ptrs[0..f.len]);
    const o = pf.sources(&l, 0, .o);
    try t.expectEqualSlices(u64, &.{0x5000}, o.ptrs[0..o.len]);
    try t.expectEqual(@as(usize, 0), pf.sources(&l, 1, .n).len);
    l[0].mlp = .{ .moe = undefined };
    l[0].mlp.moe.router = 0xd000;
    const route = pf.sources(&l, 0, .f);
    try t.expectEqualSlices(u64, &.{0xd000}, route.ptrs[0..route.len]);
    l[0].mixer.delta.out = .{ .ql = .{ .nvfp4 = .{ .mode = .fp4, .codes = 0xe000, .scales = 0xf000, .n = 64, .k = 64, .npad = 64 } } };
    const lane_source = pf.sources(&l, 0, .o);
    try t.expectEqualSlices(u64, &.{ 0xf000, 0xe000 }, lane_source.ptrs[0..lane_source.len]);
}
