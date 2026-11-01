//! The actual snapshot stream wait stays supervised before any bounded-state allocation or copy can occur.
const std = @import("std");
const t = std.testing;
const fixture = @import("stall_wait_fixture.zig");
const Sessions = @import("sessions_gpu.zig").Sessions;
test "snapshot stall: an actual blocked snapshot wait fails independently and leaves no partial snapshot" {
    var driver = @import("load_mock.zig").init();
    var gate: fixture.Wait = .{};
    gate.install(&driver);
    var progress: @import("round_progress.zig").Progress = .{ .io = t.io };
    var runner: @import("run.zig").Runner = .{ .gpa = t.allocator, .d = &driver, .stream = .{ .d = &driver, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined, .progress = &progress };
    defer runner.deinit();
    var forward: @import("forward.zig").Forward = undefined;
    forward.runner = &runner;
    forward.slot = .{};
    var sessions: Sessions = undefined;
    sessions.f = &forward;
    sessions.slot_set = null;
    const Work = struct {
        sessions: *Sessions,
        result: ?anyerror = null,
        fn run(w: *@This()) void {
            _ = w.sessions.snapshot() catch |err| {
                w.result = err;
                return;
            };
        }
    };
    var work: Work = .{ .sessions = &sessions };
    const thread = try std.Thread.spawn(.{}, Work.run, .{&work});
    var joined = false;
    defer if (!joined) {
        gate.release.store(true, .release);
        thread.join();
    };
    try fixture.wait(&gate.entered);
    try t.expectEqual(@import("round_progress.zig").Phase.snapshot, progress.stalled(1).?);
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
    try t.expectEqual(@as(u64, 0), forward.slot.pos);
    gate.result = 0;
    try runner.stream.synchronize();
}
