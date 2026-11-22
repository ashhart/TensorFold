const std = @import("std");
const cuda = @import("cuda");
const batch = @import("cuda_batch.zig");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const ops_ = @import("cuda_mtp_ops.zig");

const Ops = ops_.Ops;
const Draft = mtp.Draft;
const max_rows = mtp.max_rows;

test "exl3: common verification scopes TP1 and TP2 validation and delegation without joins" {
    const Tri = @import("cuda_launch.zig").Tri;
    const Fake = @import("exl3/model/cuda_exl3_fake_driver_test.zig").Fake;
    const Phase = @import("exl3/model/exl3_telemetry.zig").Phase;
    const Delegate = struct {
        engine: *prompt.Engine,
        calls: usize = 0,
        fail: bool = false,

        fn verify(ptr: *anyopaque, _: *prompt.Seq, _: *Draft, _: []const u32) !void {
            const delegate: *@This() = @ptrCast(@alignCast(ptr));
            delegate.calls += 1;
            try std.testing.expectEqual(Phase.verify, delegate.engine.telemetry.forward_phase);
            if (delegate.fail) return error.VerificationFailed;
        }

        fn forward(_: *anyopaque, _: *prompt.Seq, _: []const u32, _: batch.Hooks) !void {
            return error.UnexpectedDelegation;
        }

        fn rollback(_: *anyopaque, _: *prompt.Seq, _: *Draft, _: usize) !void {
            return error.UnexpectedDelegation;
        }
    };
    var driver = Fake.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var engine: prompt.Engine = undefined;
    engine.stream = &stream;
    engine.round_graph = .{};
    var sequence: prompt.Seq = undefined;
    sequence.cap = max_rows;
    var window: Draft = undefined;
    var delegate: Delegate = .{ .engine = &engine };
    var ops: Ops(Tri) = .{ .e = &engine, .tri = .{ .s = stream }, .h = undefined };
    const too_wide: [max_rows + 1]u32 = @splat(1);
    const syncs = Fake.sync_calls;
    for ([_]usize{ 1, 2 }) |world| {
        for ([_]bool{ false, true }) |enabled| {
            for ([_]Phase{ .unknown, .prefill, .decode, .verify }) |previous| {
                engine.telemetry = .{ .enabled = enabled, .rank = .{ .rank = 0, .world = world }, .forward_phase = previous };
                sequence.fed = 0;
                ops.main = null;
                try std.testing.expectError(error.WindowTooWide, ops.verify(&sequence, &window, &.{}));
                try std.testing.expectEqual(previous, engine.telemetry.forward_phase);
                try std.testing.expectError(error.WindowTooWide, ops.verify(&sequence, &window, &too_wide));
                try std.testing.expectEqual(previous, engine.telemetry.forward_phase);
                sequence.fed = sequence.cap;
                try std.testing.expectError(error.PromptTooLong, ops.verify(&sequence, &window, &.{1}));
                try std.testing.expectEqual(previous, engine.telemetry.forward_phase);
                ops.main = .{ .ptr = &delegate, .forward = Delegate.forward, .verify = Delegate.verify, .rollback = Delegate.rollback };
                const calls = delegate.calls;
                delegate.fail = false;
                try ops.verify(&sequence, &window, &.{1});
                try std.testing.expectEqual(previous, engine.telemetry.forward_phase);
                delegate.fail = true;
                try std.testing.expectError(error.VerificationFailed, ops.verify(&sequence, &window, &.{1}));
                try std.testing.expectEqual(previous, engine.telemetry.forward_phase);
                try std.testing.expectEqual(calls + 2, delegate.calls);
                try std.testing.expectEqual(enabled, engine.telemetry.enabled);
                try std.testing.expectEqual(syncs, Fake.sync_calls);
            }
        }
    }
}
