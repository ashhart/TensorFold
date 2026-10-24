//! Detailed invariants: docs/NOTES-CODE-COMMENTS.md, section 14.
const std = @import("std");
const index = @import("index.zig");
const core_build = @import("core_build.zig");
const Allocator = std.mem.Allocator;

/// Tests only: while set, a merge waits before it builds (so writes come in during it).
pub var hold: std.atomic.Value(bool) = .init(false);

pub const Notify = struct { io: std.Io, event: *std.Io.Event };

/// One merge in flight: the version it reads, and the core it made once `done`.
pub const Job = struct {
    gpa: Allocator,
    from: *index.Index, // the version's index, retained until finish()
    mark: usize, // how many retired places the store had listed then: the ones after it are the new core's
    obsolete: bool = false, // writer-owned: a later full core invalidated this job's retirement mark
    thread: std.Thread = undefined,
    done: std.atomic.Value(bool) = .init(false),
    cancelled: std.atomic.Value(bool) = .init(false),
    result: ?*index.Index = null,
    notify: ?Notify = null,

    /// Starts a merge of version `v` (a layered one), its index retained here. O(1) for the caller.
    pub fn start(gpa: Allocator, v: anytype, mark: usize) !*Job {
        return startNotified(gpa, v, mark, null);
    }

    pub fn startNotified(gpa: Allocator, v: anytype, mark: usize, notify: ?Notify) !*Job {
        const j = try gpa.create(Job);
        errdefer gpa.destroy(j);
        j.* = .{ .gpa = gpa, .from = v.idx.retain(), .mark = mark, .notify = notify };
        errdefer j.from.deinit();
        j.thread = try std.Thread.spawn(.{}, run, .{j});
        return j;
    }

    fn run(j: *Job) void {
        while (hold.load(.acquire) and !j.cancelled.load(.acquire)) std.Thread.yield() catch {};
        if (!j.cancelled.load(.acquire)) j.result = core_build.fromVersionCancellable(j.gpa, j.from, &j.cancelled) catch null;
        j.done.store(true, .release);
        if (j.notify) |n| n.event.set(n.io);
    }

    pub fn ready(j: *const Job) bool {
        return j.done.load(.acquire);
    }

    pub fn cancel(j: *Job) void {
        j.cancelled.store(true, .release);
    }

    /// Waits for the merge, lets its version go and frees the job; the core it made (the caller's reference), or null.
    pub fn finish(j: *Job) ?*index.Index {
        j.thread.join();
        const r = j.result;
        j.from.deinit();
        j.gpa.destroy(j);
        return r;
    }
};
