const std = @import("std");
const driver = @import("driver.zig");
const fixtures = @import("hip_fixtures");

test {
    _ = @import("affine.zig");
    _ = @import("args.zig");
    _ = @import("caps.zig");
    _ = @import("code_object.zig");
    _ = @import("abi.zig");
    _ = @import("runtime.zig");
    _ = @import("launch.zig");
    _ = @import("graph.zig");
    _ = @import("context.zig").Context.init;
    _ = @import("stream.zig").Stream.init;
    _ = @import("memory.zig").DeviceBuffer.alloc;
    _ = @import("module.zig").Module.load;
    _ = @import("module.zig").Module.loadForArchitecture;
    _ = @import("launch.zig").launch;
}

test "admission-only fixture cannot satisfy complete runtime ABI" {
    try std.testing.expectError(error.MissingSymbol, @import("runtime.zig").Runtime.openPath(fixtures.success));
}

test "real dynamic loading admits the mock HIP ABI" {
    var d = try driver.Driver.openPath(fixtures.success);
    defer d.close();
    try std.testing.expectEqual(@as(c_int, 2), try d.deviceCount());
}

test "real dynamic loading refuses missing symbols and failed initialization" {
    try std.testing.expectError(error.MissingSymbol, driver.Driver.openPath(fixtures.missing));
    try std.testing.expectError(error.HipFailed, driver.Driver.openPath(fixtures.failed));
}
