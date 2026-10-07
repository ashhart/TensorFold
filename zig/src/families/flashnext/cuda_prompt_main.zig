//! Prompt tokens in, greedy token ids out. The ids file is whitespace or comma separated.

const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");

const want = [_]u32{ 1596, 1144, 310, 5707, 310, 1156, 25, 328 };

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 5) {
        std.debug.print("usage: flashnext-prompt <model-dir> <kernel-dir> <ids-file> <count>\n", .{});
        return 2;
    }
    const count = try std.fmt.parseInt(usize, args[4], 10);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[3], gpa, .limited(1 << 20));
    defer gpa.free(text);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \t\r\n");
    while (it.next()) |part| try ids.append(gpa, try std.fmt.parseInt(u32, part, 10));
    if (ids.items.len == 0 or count == 0) return 2;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var set = try cuda.aot.Set.load(gpa, io, &driver, ctx.device, args[2]);
    defer set.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const tri: flash.Tri = .{ .set = &set, .s = stream };
    const got = try gpa.alloc(u32, count);
    defer gpa.free(got);
    try flash.prompt.run(flash.Tri, gpa, io, &driver, &stream, tri, args[1], ids.items, count, got);
    std.debug.print("tokens", .{});
    for (got) |id| std.debug.print(" {d}", .{id});
    std.debug.print("\n", .{});
    if (count == want.len and ids.items.len == 58) {
        var same = true;
        for (got, &want) |g, w| if (g != w) {
            same = false;
        };
        std.debug.print("{s}\n", .{if (same) "match" else "differ"});
        if (!same) return 1;
    }
    return 0;
}
