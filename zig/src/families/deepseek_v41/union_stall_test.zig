//! The actual split-union stream wait remains owned while the native call is blocked, before union mutation.
const std = @import("std");
const t = std.testing;
const fixture = @import("stall_wait_fixture.zig");
const Kv = @import("cuda_kv.zig").Kv;
const Runner = @import("run.zig").Runner;
const calls = @import("block_calls.zig");
test "KV union stall: an actual blocked union wait is supervised before table or union mutation" {
    var driver = @import("load_mock.zig").init();
    var gate: fixture.Wait = .{};
    gate.install(&driver);
    var progress: @import("round_progress.zig").Progress = .{ .io = t.io };
    var runner: Runner = .{ .gpa = t.allocator, .d = &driver, .stream = .{ .d = &driver, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined, .progress = &progress };
    defer runner.deinit();
    try runner.external("fixture", 0x1000);
    var kv: Kv = undefined;
    kv.gpa = t.allocator;
    kv.blocks = .empty;
    kv.hist = false;
    kv.stats = .{};
    const tensor: calls.Tensor = .{ .role = .{ .buf = "fixture" }, .dt = .i32, .shape = &.{ 1, 1 }, .stride = &.{ 1, 1 } };
    var args: [7]calls.Named = undefined;
    for (args[0..6]) |*arg| arg.* = .{ .arg = .{ .t = tensor } };
    args[6] = .{ .arg = .{ .i = 8 } };
    const call: calls.Call = .{ .name = "kx_union", .triton = false, .glue = true, .args = &args };
    const Work = struct {
        kv: *Kv,
        runner: *Runner,
        call: *const calls.Call,
        result: ?anyerror = null,
        fn run(w: *@This()) void {
            w.kv.glue(w.runner, w.call, "kx_union") catch |err| {
                w.result = err;
            };
        }
    };
    var work: Work = .{ .kv = &kv, .runner = &runner, .call = &call };
    const thread = try std.Thread.spawn(.{}, Work.run, .{&work});
    var joined = false;
    defer if (!joined) {
        gate.release.store(true, .release);
        thread.join();
    };
    try fixture.wait(&gate.entered);
    try t.expectEqual(@import("round_progress.zig").Phase.kv_union, progress.stalled(1).?);
    var failure: fixture.Failure = .{};
    var supervisor: @import("model_supervisor.zig").Supervisor = .{ .io = t.io, .ctx = &failure, .healthy = fixture.Failure.healthy, .fail = fixture.Failure.fail, .progress = &progress, .stall_ns = 1 };
    try supervisor.start();
    defer supervisor.stop();
    try fixture.wait(&failure.done);
    try t.expectEqual(error.StalledRound, failure.err.?);
    try t.expect(supervisor.failed.load(.acquire));
    gate.release.store(true, .release);
    thread.join();
    joined = true;
    try t.expectEqual(error.NotReady, work.result.?);
    try t.expectEqual(@as(u32, 0), progress.active);
    try t.expectEqual(@as(u64, 0), kv.stats.unions);
    try t.expectEqual(@as(usize, 0), kv.blocks.items.len);
    gate.result = 0;
    try runner.stream.synchronize();
}
