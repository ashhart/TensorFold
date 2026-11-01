//! Independent matcher walks share a bounded CPU pool; engine state and round scratch stay on the lane owner.
const std = @import("std");
const grammar = @import("grammar.zig");
const timing = @import("grammar_time.zig");
const window = @import("grammar_window.zig");
const Plan = @import("windows.zig").Plan;
const Allocator = std.mem.Allocator;

const Job = struct {
    arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator),
    prepared: window.Prepared = undefined,
    profile: timing.Profile = undefined,
    result: anyerror!grammar.Window = error.NotRun,

    fn run(j: *Job) void {
        j.result = window.walk(j.prepared, j.arena.allocator(), &j.profile);
    }
};

const Pool = struct {
    threaded: std.Io.Threaded,
    jobs: std.ArrayList(Job) = .empty,

    fn create(a: Allocator) !*Pool {
        const p = try a.create(Pool);
        const cpus = std.Thread.getCpuCount() catch 1;
        p.* = .{ .threaded = .init(std.heap.page_allocator, .{ .async_limit = .limited(@min(7, cpus -| 1)) }) };
        return p;
    }

    fn reserve(p: *Pool, a: Allocator, n: usize) !void {
        try p.jobs.ensureTotalCapacity(a, n);
        while (p.jobs.items.len < n) p.jobs.appendAssumeCapacity(.{});
        for (p.jobs.items[0..n]) |*j| {
            _ = j.arena.reset(.retain_capacity);
            j.result = error.NotRun;
        }
    }

    fn deinit(p: *Pool, a: Allocator) void {
        p.threaded.deinit();
        for (p.jobs.items) |*j| j.arena.deinit();
        p.jobs.deinit(a);
        a.destroy(p);
    }
};

pub const State = struct {
    pool: ?*Pool = null,

    pub fn deinit(s: *State, a: Allocator) void {
        if (s.pool) |p| p.deinit(a);
        s.* = .{};
    }
};

/// A matcher can be shared by a custom caller; it must never be walked by two workers at once.
pub fn independent(plans: []const Plan) bool {
    for (plans, 0..) |p, i| {
        const c = p.stream.grammar orelse continue;
        if (!c.rules.parallel_walk) return false;
        for (plans[0..i]) |before| if (before.stream.grammar) |other| {
            if (c == other or c.rules.ptr == other.rules.ptr) return false;
        };
    }
    return true;
}

pub fn constrain(e: anytype, plans: []Plan) !void {
    var count: usize = 0;
    var runnable: usize = 0;
    for (plans) |p| if (p.stream.grammar) |c| {
        count += 1;
        const reaches = c.active or p.held > 0 or std.mem.indexOfScalar(u32, p.tokens, c.after.?) != null;
        runnable += @intFromBool(!c.finished() and reaches);
    };
    const clock_safe = e.clock.vtable.host_ns == null or e.clock.vtable.parallel_host_ns;
    if (runnable < 4 or !clock_safe or !independent(plans)) {
        for (plans) |*p| try window.constrain(e, p);
        return;
    }
    if (e.grammar_workers.pool == null) e.grammar_workers.pool = try Pool.create(e.gpa);
    const pool = e.grammar_workers.pool.?;
    try pool.reserve(e.gpa, count);
    var at: usize = 0;
    // Held GPU drafts, row allocation and every engine/backend access run on the owner before dispatch.
    for (plans) |*p| if (p.stream.grammar != null) {
        const j = &pool.jobs.items[at];
        j.prepared = try window.prepare(e, p);
        j.profile = .{ .clock = e.clock };
        at += 1;
    };
    var group: std.Io.Group = .init;
    const io = pool.threaded.io();
    defer group.cancel(io);
    for (pool.jobs.items[1..count]) |*j| group.async(io, Job.run, .{j});
    pool.jobs.items[0].run();
    try group.await(io);
    at = 0;
    // Matchers and job arenas stay live through the join; results and errors are consumed in stream order.
    for (plans) |*p| if (p.stream.grammar != null) {
        const j = &pool.jobs.items[at];
        timing.record(e, p.stream, j.profile);
        const w = try j.result;
        try window.apply(e, p, j.prepared, w);
        at += 1;
    };
}
