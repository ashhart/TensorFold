//! Every registry name walks the actual launch resolver without compiling or loading a shader.
const std = @import("std");
const registry = @import("contract").registry;
const launches = @import("metal_limits.zig");

test "Metal launch map: every registered launch and control has a resolver branch" {
    var missing: usize = 0;
    for ([_][]const @import("contract").limits.MetalLaunch{ &registry.all_metal, &registry.metal_controls }) |list| {
        for (list) |d| launches.requireMapped(d) catch |err| {
            std.debug.print("UNMAPPED {s}: {t}\n", .{ d.name, err });
            missing += 1;
        };
    }
    std.debug.print("Metal launch map: {d} registered launches, {d} controls, {d} missing\n", .{ registry.all_metal.len, registry.metal_controls.len, missing });
    try std.testing.expectEqual(@as(usize, 0), missing);
}

test "Metal launch map: unknown registrations stay errors on every chip" {
    try std.testing.expectError(error.UnknownLaunch, launches.requireMapped(.{ .name = "unregistered family missing_kernel", .threads = 32, .tg_memory = 0 }));
}

test {
    _ = @import("metal_support.zig");
}
