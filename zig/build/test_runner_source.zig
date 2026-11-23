//! Selects zero-frame allocation tracking in a generated copy of the pinned compiler's unchanged test protocol.
const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) return error.BadArguments;
    const source = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(1024 * 1024));
    const before = "testing.allocator_instance = .init(std.heap.page_allocator, .{";
    const after = before ++ " .stack_trace_frames = 0,";
    if (std.mem.count(u8, source, before) != 5) return error.UnsupportedTestRunner;
    const output = try std.mem.replaceOwned(u8, a, source, before, after);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[2], .data = output });
}
