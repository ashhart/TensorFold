//! Embed the production arithmetic dependencies in the C++ translation unit so every header edit invalidates the check.
const std = @import("std");
pub fn wire(b: *std.Build, m: *std.Build.Module) void {
    const first = std.mem.replaceOwned(u8, b.allocator, @embedFile("../tests/qwen4/living_rows_host.cpp"), "#include \"../../kernels/common/living_rows.h\"", @embedFile("../kernels/common/living_rows.h")) catch @panic("OOM");
    const second = std.mem.replaceOwned(u8, b.allocator, first, "#include \"../../kernels/common/living_gdn.h\"", @embedFile("../kernels/common/living_gdn.h")) catch @panic("OOM");
    const text = std.mem.replaceOwned(u8, b.allocator, second, "#include \"../../kernels/common/living_qsa.h\"", @embedFile("../kernels/common/living_qsa.h")) catch @panic("OOM");
    const reference = @import("living_reference_host.zig").generate(b.allocator) catch @panic("native learner reference source changed");
    const combined = std.mem.concat(b.allocator, u8, &.{ text, "\n", reference }) catch @panic("OOM");
    const source = b.addWriteFiles().add("living_native_host.cpp", combined);
    m.addCSourceFile(.{ .file = source, .flags = &.{ "-std=c++20", "-O1", "-fno-fast-math", "-ffp-contract=off" } });
}
