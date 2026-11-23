//! The real engine core and CUDA runtime share one process-global route-tap session.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
extern "c" fn setenv([*:0]const u8, [*:0]const u8, c_int) c_int;
extern "c" fn unsetenv([*:0]const u8) c_int;
test "route tap core: engine start is visible to CUDA copy live" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [160]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, ".zig-cache/tmp/{s}/shared-core.tfr1", .{tmp.sub_path}, 0);
    try std.testing.expectEqual(@as(c_int, 0), setenv(core.route_tap.env_name, path, 1));
    defer _ = unsetenv(core.route_tap.env_name);
    try std.testing.expect(!cuda.route_tap_copy.live());
    try core.route_tap.start(a, 0);
    defer core.route_tap.stop();
    try std.testing.expect(cuda.route_tap_copy.live());
}
