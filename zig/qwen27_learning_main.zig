//! Run only the bounded local backward panel without opening a checkpoint or trainer.
const std = @import("std");
const mtl = @import("metal");
const checks = @import("checks");
pub fn main(init: std.process.Init) !u8 {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var tally: checks.Tally = .{};
    try checks.run(@import("tests/contract/rig.zig"), init.gpa, device, queue, null, &tally);
    std.debug.print("Q27_LOCAL_SUMMARY controls={d} red={d} missed={d} failures={d}\n", .{ tally.controls, tally.caught, tally.missed, tally.failed });
    return if (tally.problems() == 0) 0 else 1;
}
