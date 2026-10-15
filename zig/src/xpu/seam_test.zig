//! Device test: one prompt fed whole, cut at 64 and cut at 1024 gives the same last logits, bit for bit.
const std = @import("std");
const q = @import("qwen27.zig");

const LEN = 1100; // a chunk of 1024 and a rest of 76: cut at 64, the rest lands in a different place

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.c_allocator;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    const dir = if (argv.len > 1) argv[1] else if (std.c.getenv("TF_XPU_QWEN27")) |v| std.mem.span(v) else {
        std.debug.print("skipped: TF_XPU_QWEN27 not set\n", .{});
        return;
    };
    _ = q.ze.Gpu.open(0xe223) catch {
        std.debug.print("skipped: no Level Zero device\n", .{});
        return;
    };
    const r = try q.Runner.init(dir, 4096);
    var prng = std.Random.DefaultPrng.init(0x5ea4);
    var ids: [LEN]u32 = undefined;
    for (&ids) |*t| t.* = 1000 + prng.random().uintLessThan(u32, 100000);

    var first: []f32 = &.{};
    for ([_]usize{ 0, 64, 1024 }) |cut| {
        try r.reset();
        if (cut > 0) try r.prompt(ids[0..cut], false);
        try r.prompt(ids[cut..], true);
        const l = try r.logits();
        if (cut == 0) {
            first = try gpa.dupe(f32, l);
            continue;
        }
        var differ: usize = 0;
        for (first, l) |x, y| differ += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
        std.debug.print("cut at {d}: {d} of {d} logits differ from the whole prompt\n", .{ cut, differ, l.len });
        if (differ != 0) return error.PromptSeam;
    }
    std.debug.print("ok\n", .{});
}
