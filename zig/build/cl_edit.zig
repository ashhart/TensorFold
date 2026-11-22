//! cl_edit SRC OUT FROM TO [FROM TO ...]: a control's source, each FROM (exactly once) replaced by its TO.
const std = @import("std");

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 5 or (args.len - 3) % 2 != 0) {
        std.debug.print("usage: cl_edit SRC OUT FROM TO [FROM TO ...]\n", .{});
        return 2;
    }
    var text: []const u8 = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(1 << 24));
    var i: usize = 3;
    while (i < args.len) : (i += 2) {
        const n = std.mem.count(u8, text, args[i]);
        if (n != 1) {
            std.debug.print("cl_edit: {s} holds \"{s}\" {d} times, not once: update the control\n", .{ args[1], args[i], n });
            return 1;
        }
        text = try std.mem.replaceOwned(u8, a, text, args[i], args[i + 1]);
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[2], .data = text });
    return 0;
}
