//! Focused source-only entry point for the same cases registered in metal_main; not full backend qualification.
const std = @import("std");
const metal = @import("metal");
const ct = @import("contract");
const suite = @import("metal_masked_values.zig");
pub fn main(init: std.process.Init) !void {
    const a = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const only: ?[]const u8 = if (args.len == 3 and std.mem.eql(u8, args[1], "--only")) args[2] else if (args.len == 1) null else return error.Usage;
    const device = try metal.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    std.debug.print("MASKED_DEVICE {s} tensor_units={any}\n", .{ device.name(), device.tensorUnits() });
    var kernels: std.ArrayList(ct.gates.Kernel) = .empty;
    defer kernels.deinit(a);
    var cases = suite.Suite{ .a = a, .device = device, .queue = queue, .kernels = &kernels };
    defer cases.deinit();
    try cases.collect(only);
    if (kernels.items.len == 0) return error.NoMaskedValueCases;
    var tally: ct.gates.Tally = .{};
    try ct.gates.runAll(a, kernels.items, 0x5541009, &tally);
    std.debug.print("MASKED_SUMMARY kernels={d} passed={d} failed={d} known={d} controls={d}/{d}\n", .{ tally.kernels, tally.passed, tally.failed, tally.known, tally.caught, tally.controls });
    if (tally.problems() > 0) return error.MaskedValueFindings;
}
