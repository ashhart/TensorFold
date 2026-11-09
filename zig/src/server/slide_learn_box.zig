//! Own copies of asynchronous learner events, including whether a request failed before its first unit.
const std = @import("std");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;

pub const Box = struct {
    io: std.Io,
    gpa: Allocator,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    events: std.ArrayList(Copy) = .empty,
    done: bool = false,

    pub const Copy = struct { kind: std.meta.Tag(api.LearnEvent), text: []const u8 = "", recalled: bool = false, steps: u32 = 0, before_begin: bool = false, candidate: bool = false, progress: api.LearnProgress = .{ .phase = .reading }, neurons: []const api.Neuron = &.{} };

    pub fn sink(b: *Box) api.LearnSink {
        return .{ .ctx = b, .event = onEvent };
    }

    fn onEvent(ctx: *anyopaque, e: *const api.LearnEvent) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        var c: Copy = .{ .kind = e.* };
        switch (e.*) {
            .learned => |l| {
                c.recalled = l.recalled;
                c.candidate = l.candidate;
                c.steps = l.steps;
                c.neurons = b.gpa.dupe(api.Neuron, l.neurons) catch &.{};
            },
            .progress => |p| {
                c.progress = p;
                c.progress.neurons = &.{};
                c.neurons = b.gpa.dupe(api.Neuron, p.neurons) catch &.{};
            },
            .done => |d| {
                c.text = b.gpa.dupe(u8, d.message) catch "";
                c.before_begin = d.before_begin;
            },
        }
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        b.events.append(b.gpa, c) catch {
            b.gpa.free(c.text);
            b.gpa.free(c.neurons);
        };
        if (e.* == .done) b.done = true;
        b.cond.signal(b.io);
    }

    /// Waits for events and moves them to `out`; true once `done` has arrived.
    pub fn take(b: *Box, out: *std.ArrayList(Copy)) bool {
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        while (b.events.items.len == 0 and !b.done) b.cond.waitUncancelable(b.io, &b.mutex);
        out.appendSlice(b.gpa, b.events.items) catch {};
        b.events.clearRetainingCapacity();
        return b.done;
    }

    pub fn deinit(b: *Box) void {
        for (b.events.items) |c| {
            b.gpa.free(c.text);
            b.gpa.free(c.neurons);
        }
        b.events.deinit(b.gpa);
    }
};
