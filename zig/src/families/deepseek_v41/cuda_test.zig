const std = @import("std");
const dk = @import("dsv41_kernels");
test {
    _ = @import("load_test.zig");
    _ = @import("named.zig");
    inline for (.{ @import("cuda_weights.zig"), @import("cuda_weights.zig").Weights, @import("forms.zig"), @import("forms_inplace.zig"), @import("ex_stacks.zig"), @import("triton_capture.zig"), @import("host_stage.zig").Stage }) |m| std.testing.refAllDecls(m);
    _ = @import("block_test.zig");
    inline for (.{ @import("block.zig").Emitter, @import("block_attn.zig"), @import("block_index.zig"), @import("block_moe.zig"), @import("block_prefetch.zig"), @import("block_wide.zig"), @import("buffers.zig") }) |m| std.testing.refAllDecls(m);
    _ = @import("catalog_test.zig");
    _ = @import("launch_bounds_test.zig");
    inline for (.{ dk, dk.Kernels, dk.dense, dk.dense.Functions, dk.dense.Ops, dk.exl3, dk.exl3.Functions, dk.exl3.Ops, dk.ops, dk.ops.Functions, dk.ops.Ops, dk.replay, dk.kv_glue.Ops, dk.sampling }) |m| std.testing.refAllDecls(m);
}
