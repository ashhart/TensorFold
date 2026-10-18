//! Release CUDA images carry no source line information.
const std = @import("std");
const catalog = @import("cuda_kernels.zig");
const deepseek = @import("dsv41_units.zig");

test "release CUDA flags omit source line information in every image" {
    for (catalog.kernels ++ deepseek.units ++ deepseek.optional) |kernel| {
        for (kernel.flags) |flag| {
            if (std.mem.eql(u8, flag, "-lineinfo") or std.mem.eql(u8, flag, "--generate-line-info")) {
                std.debug.print("source line information in release image: {s}\n", .{kernel.name});
                return error.ReleaseImageContainsLineInfo;
            }
        }
    }
}
