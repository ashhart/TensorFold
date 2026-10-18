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
        std.debug.print("usage: flashnext-prompt <model-dir> <kernel-dir|native> <ids-file> <count>\n", .{});
        return 2;
    }
    const count = try std.fmt.parseInt(usize, args[4], 10);
    // TF_FLASHNEXT_DRAFTS: the MTP check (cuda_mtp.zig) over every comma-separated ids file, one load
    if (std.c.getenv("TF_FLASHNEXT_DRAFTS") != null) return drafted(gpa, io, args[1], args[2], args[3], count);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[3], gpa, .limited(1 << 24));
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
    var set: ?cuda.aot.Set = if (std.mem.eql(u8, args[2], "native")) null else try cuda.aot.Set.load(gpa, io, &driver, ctx.device, args[2]);
    defer if (set) |*x| x.deinit();
    var glue = try flash.Glue.load(&driver);
    defer glue.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const tri: flash.Tri = .{ .set = if (set) |*x| x else null, .own = &glue, .s = stream };
    const got = try gpa.alloc(u32, count);
    defer gpa.free(got);
    try flash.row_path.run(flash.Tri, gpa, io, &driver, &stream, tri, args[1], ids.items, count, got);
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

/// Serial and drafted decoding of each prompt compared token for token (flash.mtp.check), with their speeds.
fn drafted(gpa: std.mem.Allocator, io: std.Io, model: []const u8, kernels: []const u8, files: []const u8, count: usize) !u8 {
    var prompts: std.ArrayList([]u32) = .empty;
    defer {
        for (prompts.items) |p| gpa.free(p);
        prompts.deinit(gpa);
    }
    var names = std.mem.tokenizeScalar(u8, files, ',');
    while (names.next()) |name| {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, name, gpa, .limited(1 << 20));
        defer gpa.free(text);
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(gpa);
        var it = std.mem.tokenizeAny(u8, text, ", \t\r\n");
        while (it.next()) |part| try ids.append(gpa, try std.fmt.parseInt(u32, part, 10));
        try prompts.append(gpa, try ids.toOwnedSlice(gpa));
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var set: ?cuda.aot.Set = if (std.mem.eql(u8, kernels, "native")) null else try cuda.aot.Set.load(gpa, io, &driver, ctx.device, kernels);
    defer if (set) |*x| x.deinit();
    var glue = try flash.Glue.load(&driver);
    defer glue.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const tri: flash.Tri = .{ .set = if (set) |*x| x else null, .own = &glue, .s = stream };
    const same = try flash.mtp.check(flash.Tri, gpa, io, &driver, &stream, tri, model, prompts.items, count);
    std.debug.print("{s}\n", .{if (same) "drafted == serial on every prompt" else "drafted DIFFERS from serial"});
    return if (same) 0 else 1;
}
