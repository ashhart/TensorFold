//! Device test: a runner opened and closed three times, with a second slot and a snapshot, leaves nothing behind.
const std = @import("std");
const q = @import("qwen27.zig");

/// Free memory a round may lose: the driver keeps a reuse pool of freed memory, once, not per round.
const SLACK: u64 = 64 << 20;

fn freeMemory() !u64 {
    var g = try q.ze.Gpu.open(0xe223);
    defer g.close();
    return (try g.memory()).free;
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    const dir = if (argv.len > 1) argv[1] else if (std.c.getenv("TF_XPU_QWEN27")) |v| std.mem.span(v) else {
        std.debug.print("skipped: TF_XPU_QWEN27 not set\n", .{});
        return;
    };
    const before = freeMemory() catch {
        std.debug.print("skipped: no Level Zero device\n", .{});
        return;
    };
    var first: u64 = 0;
    for (0..3) |round| {
        const r = try q.Runner.init(dir, 4096);
        var ids: [128]u32 = undefined;
        for (&ids, 0..) |*t, i| t.* = 1000 + @as(u32, @intCast(i)) * 7;
        try r.select(1);
        try r.reset();
        try r.prompt(&ids, false);
        r.dropSnap(try r.saveSlot(1, ids.len));
        r.deinit();
        const after = try freeMemory();
        std.debug.print("round {d}: {d} MiB free before the first runner, {d} MiB after this one closed\n", .{ round, before >> 20, after >> 20 });
        if (round == 0) first = after else if (after + SLACK < first) return error.CardNotGivenBack;
    }
    std.debug.print("ok\n", .{});
}
