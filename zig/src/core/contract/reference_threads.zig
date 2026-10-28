//! Bounded scheduling of independent contract-reference jobs; each job retains its own arithmetic order.
const std = @import("std");
const builtin = @import("builtin");

pub const maximum = 8;
pub const Mode = enum { normal, partial_failure, all_failure };

pub fn count(requested: usize, items: usize) usize {
    return @min(if (builtin.single_threaded) 1 else @min(@max(requested, 1), maximum), items);
}

pub fn run(comptime Job: type, jobs: []Job, mode: Mode) void {
    if (jobs.len == 0) return;
    if (builtin.single_threaded or jobs.len > maximum) {
        for (jobs) |*job| job.run();
    } else {
        var threads: [maximum]?std.Thread = @splat(null);
        for (1..jobs.len) |worker| {
            threads[worker] = (if (mode == .all_failure or (mode == .partial_failure and worker % 2 == 0))
                error.ReferenceThreadUnavailable
            else
                std.Thread.spawn(.{}, Job.run, .{&jobs[worker]})) catch null;
            if (threads[worker] == null) jobs[worker].run();
        }
        jobs[0].run();
        for (threads[0..jobs.len]) |thread| if (thread) |t| t.join();
    }
}

test "reference threads complete every partition for normal and failed spawns" {
    const Job = struct {
        completed: usize = 0,
        pub fn run(job: *@This()) void {
            job.completed += 1;
        }
    };
    for ([_]usize{ 0, 1, 3, 8, 11 }) |size| for ([_]Mode{ .normal, .partial_failure, .all_failure }) |mode| {
        var jobs: [11]Job = @splat(.{});
        run(Job, jobs[0..size], mode);
        for (jobs[0..size]) |job| try std.testing.expectEqual(@as(usize, 1), job.completed);
        for (jobs[size..]) |job| try std.testing.expectEqual(@as(usize, 0), job.completed);
    };
    try std.testing.expect(count(99, 99) <= maximum);
}
