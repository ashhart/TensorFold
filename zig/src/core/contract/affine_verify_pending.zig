//! The wide-row-tile affine launches keep their resource gates pending until compiled receipts exist.
const R = @import("registry.zig").CudaRecord;
const none = @import("registry.zig").none;
const symbol = "_ZN12tf_qmm_group12group_kernelILi64ELi32ELi64ELi1ELi4ELi5ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const symbol64 = "_ZN12tf_qmm_group12group_kernelILi64ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const symbol128 = "_ZN12tf_qmm_group12group_kernelILi64ELi128ELi64ELi1ELi4ELi3ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
pub const records = [_]R{
    .{ .launch = .{ .name = "affine4 verify32 one CTA a tile", .symbol = symbol, .image = "qmm_group", .threads = 128, .dynamic = 32640, .opt_in = true, .min_sm = 80 }, .usage = none },
    .{ .launch = .{ .name = "affine4 verify32 cluster of 8", .symbol = symbol, .image = "qmm_group", .threads = 128, .dynamic = 32640, .cluster = 8, .opt_in = true, .min_sm = 90 }, .usage = none },
    .{ .launch = .{ .name = "affine4 verify64 one CTA a tile", .symbol = symbol64, .image = "qmm_group", .threads = 128, .dynamic = 43008, .opt_in = true, .min_sm = 80 }, .usage = none },
    .{ .launch = .{ .name = "affine4 verify64 cluster of 8", .symbol = symbol64, .image = "qmm_group", .threads = 128, .dynamic = 43008, .cluster = 8, .opt_in = true, .min_sm = 90 }, .usage = none },
    .{ .launch = .{ .name = "affine4 verify128 one CTA a tile", .symbol = symbol128, .image = "qmm_group", .threads = 128, .dynamic = 57600, .opt_in = true, .min_sm = 80 }, .usage = none },
    .{ .launch = .{ .name = "affine4 verify128 cluster of 8", .symbol = symbol128, .image = "qmm_group", .threads = 128, .dynamic = 57600, .cluster = 8, .opt_in = true, .min_sm = 90 }, .usage = none },
};
