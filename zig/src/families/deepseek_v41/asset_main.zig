//! CPU-only asset planning; no command opens a CUDA driver or executes model code.
const std = @import("std");

const Command = union(enum) {
    help,
    needs: struct { sigs: []const u8, output: []const u8 },
    engram_list: struct { index: []const u8, prefix: ?[]const u8 },
};

fn parse(args: []const []const u8) !Command {
    if (args.len == 2 and std.mem.eql(u8, args[1], "--help")) return .help;
    if (args.len == 4 and std.mem.eql(u8, args[1], "aot-needs")) {
        if (args[2].len == 0 or args[3].len == 0) return error.BadArguments;
        return .{ .needs = .{ .sigs = args[2], .output = args[3] } };
    }
    if ((args.len == 3 or args.len == 4) and std.mem.eql(u8, args[1], "engram-list")) {
        if (args[2].len == 0) return error.BadArguments;
        return .{ .engram_list = .{ .index = args[2], .prefix = if (args.len == 4) args[3] else null } };
    }
    return error.BadArguments;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const command = parse(args) catch {
        std.debug.print("usage: tf-dsv41-assets aot-needs SIGS OUT | engram-list INDEX [BARE_SHARD_PREFIX]\n", .{});
        return 2;
    };
    switch (command) {
        .engram_list => |paths| {
            var buffer: [4096]u8 = undefined;
            var out = std.Io.File.stdout().writer(init.io, &buffer);
            @import("engram_discovery.zig").run(init.arena.allocator(), init.io, paths.index, paths.prefix, &out.interface) catch |err| {
                std.debug.print("Engram discovery: {s}; download model.safetensors.index.json first; bare names may need an explicit repo directory prefix\n", .{@errorName(err)});
                return 2;
            };
        },
        .help => std.debug.print("tf-dsv41-assets aot-needs SIGS OUT | engram-list INDEX [BARE_SHARD_PREFIX] (CPU only)\n", .{}),
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
    try std.testing.expectError(error.BadArguments, parse(&.{ "assets", "engram-list" }));
    try std.testing.expectError(error.BadArguments, parse(&.{ "assets", "engram-list", "" }));
    const discovery = try parse(&.{ "assets", "engram-list", "index.json", "shards" });
    try std.testing.expectEqualStrings("index.json", discovery.engram_list.index);
    try std.testing.expectEqualStrings("shards", discovery.engram_list.prefix.?);
    const got = try parse(&.{ "assets", "aot-needs", "sigs.json", "needs.json" });
    try std.testing.expectEqualStrings("sigs.json", got.needs.sigs);
    try std.testing.expectEqualStrings("needs.json", got.needs.output);
}

test {
    _ = @import("config.zig");
    _ = @import("engram_discovery.zig");
}
