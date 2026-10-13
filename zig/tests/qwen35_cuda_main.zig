//! `tf-qwen35-cuda <model dir> <step|window|prompt> <ids> <out>`: Qwen3.5-2B's CUDA forward over comma-separated token ids,
//! one token a forward (step: every position's logits) or the ids as one prompt (prompt: the last position's), as
//! little-endian bf16 rows into <out> for a reference to compare. Also prints the step mode's decode speed.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const E = q.engine;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 5) {
        std.debug.print("usage: tf-qwen35-cuda <model dir> <step|window|prompt> <ids> <out>\n", .{});
        return 2;
    }
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, args[3], ", ");
    while (it.next()) |t| try ids.append(a, try std.fmt.parseInt(u32, t, 10));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const e = try E.Engine.init(init.gpa, init.io, &ctx, args[1]);
    defer e.deinit();
    var cache = try E.Cache.init(init.gpa, e, 4096);
    defer cache.deinit(init.gpa);
    var out: std.ArrayList(u8) = .empty;
    if (std.mem.eql(u8, args[2], "step")) {
        const t0 = std.Io.Clock.awake.now(init.io).toNanoseconds();
        for (ids.items) |id| {
            try e.forward(&.{.{ .cache = &cache, .rows = 1 }}, &.{id}, .logits);
            @memcpy(cache.logits, e.rowLogits(0));
            try out.appendSlice(a, std.mem.sliceAsBytes(cache.logits));
        }
        const ns: u64 = @intCast(std.Io.Clock.awake.now(init.io).toNanoseconds() - t0);
        std.debug.print("{d} steps in {d:.1} ms: {d:.1} tok/s\n", .{ ids.items.len, @as(f64, @floatFromInt(ns)) / 1e6, @as(f64, @floatFromInt(ids.items.len)) * 1e9 / @as(f64, @floatFromInt(ns)) });
    } else if (std.mem.eql(u8, args[2], "window")) {
        // windows of 1, 2, 3, ... rows: every row's logits, which must equal step's bit for bit
        var at: usize = 0;
        var w: usize = 1;
        while (at < ids.items.len) : (w = w % E.window_rows + 1) {
            const n = @min(w, ids.items.len - at);
            try e.forward(&.{.{ .cache = &cache, .rows = n }}, ids.items[at..][0..n], .logits);
            for (0..n) |r| try out.appendSlice(a, std.mem.sliceAsBytes(e.rowLogits(r)));
            at += n;
        }
    } else {
        var at: usize = 0;
        while (at < ids.items.len) {
            const n = @min(E.max_rows, ids.items.len - at);
            try e.forward(&.{.{ .cache = &cache, .rows = n }}, ids.items[at..][0..n], .last);
            at += n;
        }
        try out.appendSlice(a, std.mem.sliceAsBytes(cache.logits));
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = out.items });
    return 0;
}
