//! Compile the actual runtime source libraries before a device gate, without opening a GPU.
const std = @import("std");
pub fn steps(b: *std.Build) *std.Build.Step {
    const emit = b.step("emit-qwen4-metal-sources", "Write Qwen4 runtime libraries and controls without compiling shaders");
    const check = b.step("test-qwen4-metal-sources", "Compile generated Qwen4 Metal libraries without a device run");
    const files = b.addWriteFiles();
    const entries = @import("../src/families/qwen4/metal_source_inventory.zig").generate(b.allocator) catch |err| std.debug.panic("Qwen4 source inventory changed: {t}", .{err});
    const version = @import("metal_targets.zig").mlxMetalStd(b);
    for (entries) |entry| {
        const file = files.add(b.fmt("{s}.metal", .{entry.name}), entry.text);
        emit.dependOn(&b.addInstallFile(file, b.fmt("qwen4-metal/{s}.metal", .{entry.name})).step);
        const cc = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal", b.fmt("-std={s}", .{version}), "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=fast", "-c" });
        cc.addFileArg(file);
        cc.addArg("-o");
        _ = cc.addOutputFileArg(b.fmt("{s}.air", .{entry.name}));
        check.dependOn(&cc.step);
    }
    return check;
}
