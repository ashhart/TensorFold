//! BF16 prompt launches on sm_12x: nvcc 13.0's usage on sm_120 and sm_121; later releases remain pending.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

fn usage(comptime regs: u32) [5]limits.Usage {
    const off: limits.Usage = .{ .regs = 0, .static = 0 };
    const on: limits.Usage = .{ .regs = regs, .static = 0, .bound = 256 };
    return .{ off, off, off, on, on };
}

fn record(comptime wide: bool, comptime fused: bool) registry.CudaRecord {
    return .{
        .launch = .{
            .name = "qmmf BF16 prompt 128x128 " ++ (if (wide) "fp32" else "bf16") ++ (if (fused) " fused slices" else " unsplit"),
            .symbol = std.fmt.comptimePrint("_ZN7tf_qmmf11qmmf_kernelILi4ELi128ELi128ELi2ELi4ELi3ELb{d}ELb0ELb{d}EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .{ @intFromBool(wide), @intFromBool(fused) }),
            .image = "qmmf",
            .threads = 256,
            .dynamic = 98432,
            .opt_in = true,
            .min_sm = 120,
        },
        .usage = usage(if (fused) 255 else 95),
        .later_pending = true,
    };
}

pub const records = [_]registry.CudaRecord{ record(false, false), record(false, true), record(true, false), record(true, true) };

test "qmmf prompt launch records hold nvcc 13.0's usage on sm_12x, later releases pending" {
    try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &records, true));
    for (records) |r| {
        try std.testing.expectEqual(@as(u32, 256), r.launch.threads);
        try std.testing.expectEqual(@as(u32, 98432), r.launch.dynamic);
        try std.testing.expect(r.launch.opt_in and !r.geometry_pending);
        try std.testing.expect(registry.usageFor(r, .{ .major = 13, .minor = 0 })[4].regs > 0);
        for ([_]limits.Nvcc{ .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 } }) |rel|
            try std.testing.expectEqual(registry.none, registry.usageFor(r, rel));
    }
}
