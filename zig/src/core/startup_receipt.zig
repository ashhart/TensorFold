//! Optional startup spans share the serving clock; nested phases are inclusive, never additive.
const std = @import("std");
pub const Phase = enum { process_setup, discovery, tokenizer, model_config, checkpoint_headers, read_pack, weight_pipelines, initial_allocation, runtime_setup, runtime_pipelines, runtime_allocation, residency, optional_ane, model_ready, server_setup, server_ready, metal_source_build, metal_pipeline_build, first_completion };
const count = @typeInfo(Phase).@"enum".field_names.len;
pub const Range = struct { phase: Phase, start_ns: ?u64 = null, end_ns: ?u64 = null, total_ns: u64 = 0, calls: u64 = 0 };
pub const Receipt = struct {
    schema: u32 = 1,
    event: enum { ready, first_completion },
    deployment: []const u8 = "qwen27-metal",
    clock: []const u8 = "awake_monotonic",
    origin: []const u8 = "native_main_entry",
    units: []const u8 = "ns",
    inclusive: bool = true,
    at_ns: u64,
    phases: [count]Range,
};

pub const State = struct {
    phases: [count]Range = initRanges(),
    ready_sent: bool = false,
    completion_sent: bool = false,
    target: bool = false,

    pub fn start(s: *State, p: Phase, at: u64) void {
        const r = &s.phases[@backingInt(p)];
        if (r.start_ns == null) r.start_ns = at;
    }

    pub fn end(s: *State, p: Phase, at: u64) void {
        const r = &s.phases[@backingInt(p)];
        s.start(p, at);
        s.finish(p, r.start_ns.?, at);
    }

    pub fn finish(s: *State, p: Phase, began: u64, at: u64) void {
        const r = &s.phases[@backingInt(p)];
        s.start(p, began);
        r.end_ns = at;
        r.total_ns += at - began;
        r.calls += 1;
    }

    pub fn ready(s: *State, at: u64) ?Receipt {
        if (!s.target or s.ready_sent or s.phases[@backingInt(Phase.model_ready)].end_ns == null) return null;
        s.end(.server_setup, at);
        s.end(.server_ready, at);
        s.ready_sent = true;
        return .{ .event = .ready, .at_ns = at, .phases = s.phases };
    }

    pub fn complete(s: *State, start_ns: u64, at: u64) ?Receipt {
        if (!s.ready_sent or s.completion_sent or start_ns < s.phases[@backingInt(Phase.server_ready)].end_ns.?) return null;
        s.start(.first_completion, start_ns);
        s.end(.first_completion, at);
        s.completion_sent = true;
        return .{ .event = .first_completion, .at_ns = at, .phases = s.phases };
    }
};

fn initRanges() [count]Range {
    var out: [count]Range = undefined;
    for (&out, 0..) |*r, i| r.* = .{ .phase = @fromBackingInt(@intCast(i)) };
    return out;
}

var on: std.atomic.Value(bool) = .init(false);
var io: std.Io = undefined;
var origin: i96 = 0;
var mutex: std.Io.Mutex = .init;
var state: State = .{};

/// Enabled once by native main, after environment validation; tools and default serving stay off.
pub fn enable(clock: std.Io, started: i96, wanted: bool) void {
    if (!wanted) return;
    io = clock;
    origin = started;
    state = .{};
    state.start(.process_setup, 0);
    state.end(.process_setup, now());
    on.store(true, .release);
}

pub fn disable() void {
    on.store(false, .release);
}

fn now() u64 {
    return @intCast(@max(0, std.Io.Clock.awake.now(io).toNanoseconds() - origin));
}

pub fn offset() ?u64 {
    return if (on.load(.acquire)) now() else null;
}

pub const Span = struct {
    phase: ?Phase = null,
    began: u64 = 0,
    pub fn end(s: Span) void {
        const p = s.phase orelse return;
        mutex.lockUncancelable(io);
        defer mutex.unlock(io);
        state.finish(p, s.began, now());
    }
};

pub fn span(p: Phase) Span {
    if (!on.load(.acquire)) return .{};
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    const at = now();
    state.start(p, at);
    return .{ .phase = p, .began = at };
}

pub fn target() void {
    if (!on.load(.acquire)) return;
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    state.target = true;
}

pub fn ready() ?Receipt {
    if (!on.load(.acquire)) return null;
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    return state.ready(now());
}

pub fn complete(start: ?u64) ?Receipt {
    const at = start orelse return null;
    if (!on.load(.acquire)) return null;
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    const r = state.complete(at, now()) orelse return null;
    on.store(false, .release);
    return r;
}

test "startup receipt: failure and interruption never produce ready, then completion is once only" {
    var s: State = .{};
    s.target = true;
    s.start(.model_config, 10);
    try std.testing.expect(s.ready(20) == null);
    try std.testing.expect(s.complete(0, 30) == null);
    s.end(.model_config, 40);
    s.end(.model_ready, 50);
    const r = s.ready(60).?;
    try std.testing.expectEqual(@as(u64, 60), r.at_ns);
    try std.testing.expect(s.ready(70) == null);
    try std.testing.expect(s.complete(59, 80) == null);
    const c = s.complete(70, 90).?;
    try std.testing.expectEqual(@as(?u64, 70), c.phases[@backingInt(Phase.first_completion)].start_ns);
    try std.testing.expect(s.complete(80, 100) == null);
    var other: State = .{};
    other.end(.model_ready, 10);
    try std.testing.expect(other.ready(20) == null);
}

test "startup receipt: repeated builds count work rather than gaps between calls" {
    var s: State = .{};
    s.finish(.metal_source_build, 10, 20);
    s.finish(.metal_source_build, 100, 130);
    const r = s.phases[@backingInt(Phase.metal_source_build)];
    try std.testing.expectEqual(@as(?u64, 10), r.start_ns);
    try std.testing.expectEqual(@as(?u64, 130), r.end_ns);
    try std.testing.expectEqual(@as(u64, 40), r.total_ns);
    try std.testing.expectEqual(@as(u64, 2), r.calls);
}

test "startup receipt: disabled hooks do not need a clock or emit a record" {
    disable();
    try std.testing.expect(offset() == null);
    span(.model_config).end();
    target();
    try std.testing.expect(ready() == null);
    try std.testing.expect(complete(null) == null);
}
