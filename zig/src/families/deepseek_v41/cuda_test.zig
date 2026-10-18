const std = @import("std");
const dk = @import("dsv41_kernels");
test {
    _ = @import("catalog_test.zig");
    inline for (.{ dk, dk.Kernels, dk.dense, dk.dense.Functions, dk.dense.Ops, dk.exl3, dk.exl3.Functions, dk.exl3.Ops, dk.ops, dk.ops.Functions, dk.ops.Ops, dk.replay, dk.kv_glue.Ops, dk.sampling }) |m| std.testing.refAllDecls(m);
}
