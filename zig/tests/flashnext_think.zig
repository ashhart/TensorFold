//! tf-flashnext-think [CASES] [SEED]: Flash Next's verdict kernels under thinking budgets against the lane core.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const cases = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 2000;
    const seed = if (args.len > 2) try std.fmt.parseInt(u64, args[2], 10) else 1;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const dev = try mtl.Device.init();
    defer dev.deinit();
    var g = try tf.flashnext_think_check.Rig.init(dev);
    defer g.deinit();
    const t0 = std.c.mach_absolute_time();
    const sum = try tf.flashnext_think_check.run(init.gpa, &g, cases, seed);
    const s = @as(f64, @floatFromInt(std.c.mach_absolute_time() - t0)) * 125 / 3 / 1e9; // mach ticks are 125/3 ns on Apple silicon
    std.debug.print("{s}: {d} cases (seed {d}), plain and drafted == the lane core in each; closed at the budget {d}, think end first {d}; {d:.1} s\n", .{ std.mem.span(dev.name()), sum.cases, seed, sum.closed, sum.ended_first, s });
}
