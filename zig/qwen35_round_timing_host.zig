//! The actual round core reports accepted counts once, and disabled timing allocates and prints nothing.
const std = @import("std");
const lanes = @import("lanes");
const timing = @import("qwen35").native.round_timing;
const t = std.testing;

test "round timing: environment gates actual complete, rejected and shared rounds without changing tokens" {
    for ([_][]const u8{ "full", "partial", "shared", "split" }) |case| {
        var previous: ?[64]u8 = null;
        for ([_]bool{ false, true }) |enabled| {
            var env = std.process.Environ.Map.init(t.allocator);
            defer env.deinit();
            if (enabled) try env.put("TF_TIMING", "1");
            const result = try std.process.run(t.allocator, t.io, .{ .argv = &.{ @import("round_timing_options").probe, case }, .environ_map = &env, .stdout_limit = .limited(4096), .stderr_limit = .limited(65536) });
            defer t.allocator.free(result.stdout);
            defer t.allocator.free(result.stderr);
            if (result.term != .exited or result.term.exited != 0) {
                std.debug.print("timing child: {s}\n", .{result.stderr});
                return error.ChildFailed;
            }
            const Stats = struct { rounds: usize, offered: usize, kept: usize, sha: []const u8 };
            const stats = try std.json.parseFromSlice(Stats, t.allocator, result.stdout, .{});
            defer stats.deinit();
            try t.expect(stats.value.rounds > 0);
            if (previous) |want| try t.expectEqualSlices(u8, &want, stats.value.sha) else previous = stats.value.sha[0..64].*;
            var count: usize = 0;
            var offered: usize = 0;
            var kept: usize = 0;
            var full = false;
            var partial = false;
            var lines = std.mem.tokenizeScalar(u8, result.stderr, '\n');
            while (lines.next()) |line| {
                try t.expect(std.mem.startsWith(u8, line, "[timing] verify_round offered="));
                var fields = std.mem.tokenizeScalar(u8, line, ' ');
                _ = fields.next();
                _ = fields.next();
                const rows = try std.fmt.parseInt(usize, fields.next().?[8..], 10);
                const saved = try std.fmt.parseInt(usize, fields.next().?[5..], 10);
                const bucket = try std.fmt.parseInt(usize, fields.next().?[7..], 10);
                const ms = try std.fmt.parseFloat(f64, fields.next().?[3..]);
                try t.expect(ms >= 0 and std.math.isFinite(ms));
                try t.expectEqual(@as(usize, timing.Shape.init(rows, 1, true, false, 1).total), bucket);
                offered += rows;
                kept += saved;
                full = full or (rows > 1 and rows == saved);
                partial = partial or saved < rows;
                count += 1;
            }
            try t.expectEqual(if (enabled) stats.value.rounds else 0, count);
            if (enabled) {
                try t.expectEqual(stats.value.offered, offered);
                try t.expectEqual(stats.value.kept, kept);
                if (std.mem.eql(u8, case, "partial")) try t.expect(partial) else try t.expect(full);
            } else try t.expectEqual(@as(usize, 0), result.stderr.len);
        }
    }
}

test "round timing: disabled trace allocation succeeds with no allocator capacity" {
    var fail = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    try t.expect((try timing.Trace.create(fail.allocator(), false)) == null);
    try t.expectEqual(@as(usize, 0), fail.alloc_index);
}

test "round timing: drafter and recorder preserve the original observer context" {
    var target: lanes.fake.Fake = .{ .gpa = t.allocator };
    defer target.deinit();
    const Spy = struct {
        fn done(ptr: *anyopaque, paths: []const []const u32) void {
            const n: *usize = @ptrCast(@alignCast(ptr));
            n.* += paths.len;
        }
    };
    var seen: usize = 0;
    var backend = target.backend();
    backend.round_observer = .{ .ptr = &seen, .done = Spy.done };
    var drafter: lanes.drafted.Drafted = .{ .gpa = t.allocator, .target = backend, .drafter = undefined };
    defer drafter.deinit();
    var recorded: lanes.recorder.Recorded = .{ .inner = drafter.backend(), .recorder = undefined };
    const wrapped = recorded.backend();
    try t.expectEqual(@as(*anyopaque, &seen), wrapped.round_observer.?.ptr);
    wrapped.observed(&.{ &.{ 0, 1 }, &.{0} });
    try t.expectEqual(@as(usize, 2), seen);
}
