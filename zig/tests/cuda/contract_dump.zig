//! One finite prepared launch per registered served form, separate from the unchanged contract verdicts.
const std = @import("std");
const contract = @import("core").contract;
const dump = @import("byte_dump.zig");

pub fn run(a: std.mem.Allocator, kernels: []const contract.gates.Kernel) !void {
    if (!dump.active()) return;
    for (kernels) |kernel| {
        if (kernel.control != null) continue;
        for (kernel.spec.served, 0..) |shape, form| {
            if (!try kernel.vt.prepare(kernel.ctx, shape, .random, 0x6b63)) return error.DumpShapeRefused;
            try kernel.vt.run(kernel.ctx);
            if (!try kernel.vt.intact(kernel.ctx)) return error.DumpGuard;
            if (kernel.vt.overlap(kernel.ctx) and !kernel.spec.in_place) return error.DumpOverlap;
            const bytes = try kernel.vt.output(kernel.ctx);
            const count = if (kernel.spec.out == .raw) bytes.len else kernel.vt.count(kernel.ctx);
            const dtype: dump.DType = switch (kernel.spec.out) {
                .bf16 => .bf16,
                .f16 => .f16,
                .f32 => .f32,
                .raw => .raw,
            };
            try dump.output(a, "kernel-contract", kernel.spec.name, "outputs", &.{count}, dtype, .{ .form = form, .served = shape, .seed = @as(u64, 0x6b63) }, bytes);
        }
    }
}
