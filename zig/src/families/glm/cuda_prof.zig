//! TF_GLM_PROF=1: where a forward's GPU time goes (events between its sections) against the call's wall time.
const std = @import("std");
const cuda = @import("cuda");

pub const Section = enum { gate, embed, kda, mla, gather_a, dense, router, routed, shared, gather_b, head, draw, mtp, commit };
const names = [_][]const u8{ "gate", "embed", "kda", "mla", "gather_a", "dense", "router", "routed", "shared", "gather_b", "head", "draw", "mtp", "commit" };
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
    every: u64 = 64, // calls a report sums

    pub fn init(d: *const cuda.Driver, io: std.Io) !*Prof {
        return initWith(d, io, cuda.Event.init);
    }

    pub fn initWith(d: *const cuda.Driver, io: std.Io, comptime create: anytype) !*Prof {
        const p = try std.heap.page_allocator.create(Prof);
        errdefer std.heap.page_allocator.destroy(p);
        p.* = .{ .d = d, .events = undefined, .io = io };
        var made: usize = 0;
        errdefer for (p.events[0..made]) |*event| event.deinit();
        for (&p.events) |*e| {
            e.* = try create(d, true);
            made += 1;
        }
        return p;
    }

    pub fn deinit(p: *Prof) void {
        for (&p.events) |*e| e.deinit();
        std.heap.page_allocator.destroy(p);
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
        if (p.calls % p.every != 0) return;
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

const Counted = struct {
    var calls: usize = 0;
    var live: usize = 0;
    var fail: usize = 0;
    fn event(d: *const cuda.Driver, timing: bool) !cuda.Event {
        const call = calls;
        calls += 1;
        if (call == fail) return error.OutOfDeviceMemory;
        return cuda.Event.init(d, timing);
    }
    fn create(out: *cuda.abi.Event, _: c_uint) callconv(.c) c_int {
        live += 1;
        out.* = @ptrFromInt(live);
        return 0;
    }
    fn name(_: c_int, out: *?[*:0]const u8) callconv(.c) c_int {
        out.* = "mock";
        return 0;
    }
    fn destroy(_: cuda.abi.Event) callconv(.c) c_int {
        live -= 1;
        return 0;
    }
};
test "a profiler whose first, middle or last event fails destroys the events it made" {
    var d: cuda.Driver = undefined;
    d.api.cuGetErrorName = Counted.name;
    d.api.cuGetErrorString = Counted.name;
    d.api.cuEventCreate = Counted.create;
    d.api.cuEventDestroy_v2 = Counted.destroy;
    for ([_]usize{ 0, 256, 512 }) |at| {
        Counted.calls = 0;
        Counted.live = 0;
        Counted.fail = at;
        try std.testing.expectError(error.OutOfDeviceMemory, Prof.initWith(&d, std.testing.io, Counted.event));
        try std.testing.expectEqual(@as(usize, 0), Counted.live);
    }
}

test "a profiler destroys all 513 of its events" {
    var d: cuda.Driver = undefined;
    d.api.cuEventCreate = Counted.create;
    d.api.cuEventDestroy_v2 = Counted.destroy;
    Counted.live = 0;
    const p = try Prof.init(&d, std.testing.io);
    try std.testing.expectEqual(@as(usize, 513), Counted.live);
    p.deinit();
    try std.testing.expectEqual(@as(usize, 0), Counted.live);
}
