//! CPU-only asset planning; neither command opens a CUDA driver or executes model code.
const std = @import("std");

const Command = union(enum) { help, needs: struct { sigs: []const u8, output: []const u8 } };

fn parse(args: []const []const u8) !Command {
    if (args.len == 2 and std.mem.eql(u8, args[1], "--help")) return .help;
    if (args.len == 4 and std.mem.eql(u8, args[1], "aot-needs")) {
        if (args[2].len == 0 or args[3].len == 0) return error.BadArguments;
        return .{ .needs = .{ .sigs = args[2], .output = args[3] } };
    }
    return error.BadArguments;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const command = parse(args) catch {
        std.debug.print("usage: tf-dsv41-assets aot-needs SIGS OUT\n", .{});
        return 2;
    };
    switch (command) {
        .help => std.debug.print("tf-dsv41-assets aot-needs SIGS OUT (CPU only; serving settings from the environment)\n", .{}),
        .needs => |paths| {
            var buffer: [4096]u8 = undefined;
            var log = std.Io.File.stderr().writer(init.io, &buffer);
            try @import("aot_needs.zig").main(init.gpa, init.io, paths.sigs, paths.output, &log.interface);
        },
    }
    return 0;
}

test "asset commands cannot select a GPU operation" {
    try std.testing.expectError(error.BadArguments, parse(&.{ "assets", "follow", "pack", "assets" }));
    try std.testing.expectError(error.BadArguments, parse(&.{ "assets", "generate", "pack", "trace" }));
    try std.testing.expectError(error.BadArguments, parse(&.{ "assets", "aot-needs", "sigs" }));
    try std.testing.expectError(error.BadArguments, parse(&.{ "assets", "aot-needs", "", "out" }));
    const got = try parse(&.{ "assets", "aot-needs", "sigs.json", "needs.json" });
    try std.testing.expectEqualStrings("sigs.json", got.needs.sigs);
    try std.testing.expectEqualStrings("needs.json", got.needs.output);
}

test {
    _ = @import("config.zig");
}
