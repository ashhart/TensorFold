//! Clef's joint schema head on the GPU: `tf-clef-test <command> ...`, the commands listed in `usage`.
const std = @import("std");

/// No per-thread alternate signal stack: its thread-local leaves HIP's threads too little stack to start.
pub const std_options: std.Options = .{ .signal_stack_size = null };

const usage = "usage: tf-clef-test tiny <scratch dir> | head <model dir> <cases dir> <out dir> [--piece N|whole]\n";

const commands = [_]struct { name: []const u8, run: *const fn (std.mem.Allocator, std.Io, []const [:0]const u8) anyerror!void }{
    .{ .name = "tiny", .run = @import("tiny_run.zig").run },
    .{ .name = "head", .run = @import("head_run.zig").run },
};

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len >= 2) for (commands) |c| if (std.mem.eql(u8, args[1], c.name)) {
        c.run(init.gpa, init.io, args[2..]) catch |e| {
            std.debug.print("FAIL {t}\n", .{e});
            return 1;
        };
        return 0;
    };
    std.debug.print(usage, .{});
    return 2;
}
