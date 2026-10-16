//! TF_GLM_PROF=1: where a forward's GPU time goes (events between its sections) against the call's wall time.
const std = @import("std");
const cuda = @import("cuda");

pub const Section = enum { embed, kda, mla, gather_a, dense, router, routed, shared, gather_b, head, mtp };
const names = [_][]const u8{ "embed", "kda", "mla", "gather_a", "dense", "router", "routed", "shared", "gather_b", "head", "mtp" };
const max_marks = 512;

pub const Prof = struct {
    d: *const cuda.Driver,
    events: [max_marks + 1]cuda.Event,
    ends: [max_marks]Section = undefined,
    n: usize = 0, // marks this call
    open: bool = false,
    t0: i96 = 0,
    io: std.Io,
    sections: [names.len]f64 = @splat(0), // ms each section took on the GPU, summed over calls
    gpu_ms: f64 = 0,
    wall_ms: f64 = 0,
    calls: u64 = 0,
    label: []const u8 = "",

    pub fn init(d: *const cuda.Driver, io: std.Io) !*Prof {
        const p = try std.heap.page_allocator.create(Prof);
        p.* = .{ .d = d, .events = undefined, .io = io };
        for (&p.events) |*e| e.* = try cuda.Event.init(d, true);
        return p;
    }

    /// TF_GLM_PROF=1 asks for it.
    pub fn wanted() bool {
        const v = std.c.getenv("TF_GLM_PROF") orelse return false;
        return std.mem.eql(u8, std.mem.span(v), "1");
    }

    /// A call begins on `s` (its wall clock and its first event).
    pub fn begin(p: *Prof, s: cuda.Stream, label: []const u8) !void {
        p.n = 0;
        p.open = true;
        p.label = label;
        p.t0 = std.Io.Clock.awake.now(p.io).toNanoseconds();
        try p.events[0].record(s);
    }

    /// The work queued on `s` since the last mark belongs to `section`.
    pub fn mark(p: *Prof, s: cuda.Stream, section: Section) void {
        if (!p.open or p.n >= max_marks) return;
        p.events[p.n + 1].record(s) catch return;
        p.ends[p.n] = section;
        p.n += 1;
    }

    /// The call ended (its results read: the stream is idle): sections summed, a report every 64 calls.
    pub fn end(p: *Prof) void {
        if (!p.open) return;
        p.open = false;
        const wall: f64 = @as(f64, @floatFromInt(std.Io.Clock.awake.now(p.io).toNanoseconds() - p.t0)) / 1e6;
        if (p.n == 0) return;
        p.events[p.n].synchronize() catch return;
        for (0..p.n) |i| p.sections[@backingInt(p.ends[i])] += cuda.Event.elapsedMs(p.events[i], p.events[i + 1]) catch 0;
        p.gpu_ms += cuda.Event.elapsedMs(p.events[0], p.events[p.n]) catch 0;
        p.wall_ms += wall;
        p.calls += 1;
        if (p.calls % 64 != 0) return;
        const k: f64 = @floatFromInt(p.calls);
        var buf: [512]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        w.print("glm prof {s} ({d} calls): wall {d:.2} ms, gpu {d:.2} ms a call;", .{ p.label, p.calls, p.wall_ms / k, p.gpu_ms / k }) catch {};
        for (names, p.sections) |nm, ms| if (ms > 0) w.print(" {s} {d:.2}", .{ nm, ms / k }) catch {};
        std.log.info("{s}", .{w.buffered()});
        p.sections = @splat(0);
        p.gpu_ms = 0;
        p.wall_ms = 0;
        p.calls = 0;
    }
};
