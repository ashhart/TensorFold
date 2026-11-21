//! The registered EXL3 chain gates without unrelated bundled shader or checkpoint dependencies.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const ex = @import("exl3_projection");
const Suite = @import("chain_contract").For(ex);

test "EXL3 chain contract: fused outputs, FP32 accumulators, immutable inputs and fault controls" {
    const a = std.testing.allocator;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var kernels: std.ArrayList(contract.gates.Kernel) = .empty;
    defer kernels.deinit(a);
    const suite = try Suite.create(a, device, queue, &kernels, null);
    defer suite.destroy();
    var tally = contract.gates.Tally{};
    try contract.gates.runAll(a, kernels.items, 0x6b63, &tally);
    std.debug.print("SUMMARY {d} kernels: {d} pass gates 1-4, {d} fail, {d} known; {d} controls: {d} red, {d} missed\n", .{
        tally.kernels, tally.passed, tally.failed, tally.known, tally.controls, tally.caught, tally.missed,
    });
    try std.testing.expectEqual(@as(usize, 0), tally.problems());
}
