//! The decode schedule's Mamba, attention and router views, independent of prompt chunks.
const std = @import("std");
const pf = @import("cuda_l2pf.zig");
const weights = @import("cuda_weights.zig");
const t = std.testing;
test "L2 prefetch: Nemotron next-block and output sites exclude experts and final head" {
    var b: [3]weights.Block = undefined;
    b[0].kind = .mamba;
    b[0].mamba = undefined;
    b[0].mamba.out_proj = .{ .nvfp4 = .{ .mode = .fp4, .codes = 0x1000, .scales = 0x2000, .n = 64, .k = 64, .npad = 64 } };
    b[1].kind = .attention;
    b[1].attn = undefined;
    b[1].attn.qkv = .{ .affine4 = .{ .w = 0x3000, .s = 0x4000, .b = 0x5000, .n = 64, .k = 64, .npad = 64 } };
    b[2].kind = .moe;
    b[2].moe = undefined;
    b[2].moe.router = 0x6000;
    const n = pf.sources(&b, 0, .n);
    try t.expectEqualSlices(u64, &.{ 0x4000, 0x5000, 0x3000 }, n.ptrs[0..n.len]);
    const o = pf.sources(&b, 0, .o);
    try t.expectEqualSlices(u64, &.{ 0x2000, 0x1000 }, o.ptrs[0..o.len]);
    const route = pf.sources(&b, 1, .n);
    try t.expectEqualSlices(u64, &.{0x6000}, route.ptrs[0..route.len]);
    try t.expectEqual(@as(usize, 0), pf.sources(&b, 2, .n).len);
    try t.expectEqual(@as(usize, 0), pf.sources(&b, 2, .o).len);
    try t.expectEqual(@as(usize, 0), pf.sources(&b, 0, .f).len);
}
