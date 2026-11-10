//! Opt-in diagnostic event spans for prompt stages; no events or waits when disabled.
const std = @import("std");
const cuda = @import("cuda");
const Class = enum { embed, norm, delta, attention, post_norm, mlp, finish };
const count = @typeInfo(Class).@"enum".field_names.len;

const State = struct {
    allocator: std.mem.Allocator,
    events: [400]cuda.Event = undefined,
    classes: [400]Class = undefined,
    allocated: usize = 0,
    used: usize = 0,
};

pub const Trace = struct {
    state: ?*State = null,

    pub fn init(allocator: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream, layers: usize, prompt: bool) !Trace {
        if (!prompt) return .{};
        if (!try @import("sys").environment.read(bool, "TF_QWEN_PREFILL_PROFILE", false)) return .{};
        const needed = layers * 4 + 3;
        if (needed > 400) return error.TooManyPromptProfileStages;
        const t = try allocator.create(State);
        errdefer allocator.destroy(t);
        t.* = .{ .allocator = allocator };
        errdefer for (t.events[0..t.allocated]) |*event| event.deinit();
        for (t.events[0..needed]) |*event| {
            event.* = try cuda.Event.init(d, true);
            t.allocated += 1;
        }
        try t.events[0].record(stream);
        t.used = 1;
        return .{ .state = t };
    }

    pub fn deinit(trace: *Trace) void {
        const t = trace.state orelse return;
        for (t.events[0..t.allocated]) |*event| event.deinit();
        t.allocator.destroy(t);
        trace.state = null;
    }

    pub fn mark(trace: *Trace, stream: cuda.Stream, class: Class) !void {
        const t = trace.state orelse return;
        if (t.used == t.allocated) return error.TooManyPromptProfileStages;
        t.classes[t.used - 1] = class;
        try t.events[t.used].record(stream);
        t.used += 1;
    }

    pub fn report(trace: *Trace, rows: usize) !void {
        const t = trace.state orelse return;
        try t.events[t.used - 1].synchronize();
        var milliseconds: [count]f64 = @splat(0);
        for (t.classes[0 .. t.used - 1], 0..) |class, i|
            milliseconds[@backingInt(class)] += try cuda.Event.elapsedMs(t.events[i], t.events[i + 1]);
        std.log.info("prompt profile rows={d} embed={d:.3} norm={d:.3} delta={d:.3} attention={d:.3} post_norm={d:.3} mlp={d:.3} finish={d:.3} ms", .{ rows, milliseconds[0], milliseconds[1], milliseconds[2], milliseconds[3], milliseconds[4], milliseconds[5], milliseconds[6] });
    }
};
