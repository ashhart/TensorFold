//! The model-free kernels on a real GPU: every kernel resolves, and each launch matches its host reference.
const std = @import("std");
const Runtime = @import("runtime.zig").Runtime;
const Context = @import("context.zig").Context;
const kernels = @import("kernels.zig");
const Launcher = @import("launches.zig").Launcher;

test "the code objects load on this device and every kernel the launchers use resolves" {
    var r = try Runtime.open();
    defer r.close();
    var ctx = try Context.init(&r, 0);
    defer ctx.deinit();
    var arch_buffer: [256]u8 = undefined;
    const arch = try @import("device_arch.zig").query(&r, 0, &arch_buffer);
    try std.testing.expectEqualStrings(kernels.arch, arch);
    var l = try Launcher.load(&r, .{}, kernels.images);
    defer l.unload();
}

test {
    _ = @import("kernel_tests/gdn.zig");
    _ = @import("kernel_tests/decode.zig");
    _ = @import("kernel_tests/chain.zig");
    _ = @import("kernel_tests/attention.zig");
    _ = @import("kernel_tests/ops.zig");
}
