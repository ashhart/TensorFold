//! Read existing prompt pocket products, including cached prefixes; never run a probe or change a kernel.
const builtin = @import("builtin");
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const usage = lanes.pocket_usage;
const state = @import("cuda_state.zig");
const Engine = @import("cuda_engine.zig").Engine;
const Cuda = @import("cuda_lanes.zig").Cuda;
const fill = @import("cuda_lanes_fill.zig");

pub const Read = struct {
    trace: usage.Capture,
    stage: ?cuda.HostBuffer,
    test_ctx: if (builtin.is_test) *anyopaque else void = if (builtin.is_test) undefined else {},
    test_read: if (builtin.is_test) ?*const fn (*anyopaque, *Read, usize, u64, usize) anyerror!void else void = if (builtin.is_test) null else {},

    pub fn init(e: *Engine) !Read {
        var trace = try usage.Capture.init(e.gpa, e.c.layers, state.pass_rows);
        errdefer trace.deinit();
        return .{ .trace = trace, .stage = try cuda.HostBuffer.alloc(e.ctx.d, state.pass_rows * 64 * 4) };
    }
    pub fn initTest(gpa: std.mem.Allocator, layers: usize, capacity: usize, ctx: *anyopaque, read_layer: *const fn (*anyopaque, *Read, usize, u64, usize) anyerror!void) !Read {
        if (!builtin.is_test) unreachable;
        return .{ .trace = try usage.Capture.init(gpa, layers, capacity), .stage = null, .test_ctx = ctx, .test_read = read_layer };
    }
    pub fn deinit(r: *Read) void {
        if (r.stage) |*stage| stage.free();
        r.trace.deinit();
    }
    pub fn read(r: *Read, o: anytype, layer: usize, ptr: u64, width: usize) !void {
        if (builtin.is_test) if (r.test_read) |callback| return callback(r.test_ctx, r, layer, ptr, width);
        const values = r.stage.?.slice(f32)[0 .. r.trace.rows * 64];
        errdefer cuda.completion.drain(o.k.d, o.s, "pocket usage read did not drain");
        try o.download(std.mem.sliceAsBytes(values), ptr);
        try o.s.synchronize();
        try r.trace.layer(layer, values, width, 64);
    }
};

pub fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
    const b: *Cuda = @ptrCast(@alignCast(ptr));
    if (!b.usage_enabled or s.untaught) return fill.whole(b, s);
    var r = try Read.init(b.e);
    defer r.deinit();
    b.e.usage = &r;
    defer b.e.usage = null;
    try fill.whole(b, s);
    s.pocket_usage = try r.trace.through(s.prompt().len);
}

/// Warm passes keep their end after prefill returns; interior marks use the live chunk before it returns.
pub fn saved(e: *Engine, owner: ?*anyopaque, at: usize) !?usage.Masks {
    if (e.usage) |u| return try u.trace.through(at);
    const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return null));
    return if (at == s.prompt_len) s.pocket_usage else null;
}

pub const StepHook = struct {
    ctx: *anyopaque,
    run: *const fn (*anyopaque, *Cuda, *lanes.Stream) anyerror!bool,
    read: *const fn (*anyopaque, *Read, usize, u64, usize) anyerror!void,
};

fn initRead(b: *Cuda) !Read {
    if (builtin.is_test) if (b.test_step) |hook| return Read.initTest(b.gpa, 4, state.pass_rows, hook.ctx, hook.read);
    return Read.init(b.e);
}

fn runStep(b: *Cuda, s: *lanes.Stream, max_rows: u32) anyerror!bool {
    if (builtin.is_test) if (b.test_step) |hook| return hook.run(hook.ctx, b, s);
    return fill.step(b, s, max_rows);
}

pub fn prefillStep(ptr: *anyopaque, s: *lanes.Stream, max_rows: u32) anyerror!bool {
    const b: *Cuda = @ptrCast(@alignCast(ptr));
    if (!b.usage_enabled or s.untaught) return runStep(b, s, max_rows);
    if (b.usage_reads.getPtr(s)) |r| {
        b.e.usage = r;
    } else {
        var r = try initRead(b);
        errdefer r.deinit();
        try b.usage_reads.putNoClobber(b.gpa, s, r);
        b.e.usage = b.usage_reads.getPtr(s).?;
    }
    const done = runStep(b, s, max_rows) catch |e| {
        b.e.usage = null;
        release(b, s);
        return e;
    };
    b.e.usage = null;
    if (done) {
        const r = b.usage_reads.getPtr(s).?;
        s.pocket_usage = r.trace.through(s.prompt_len) catch |e| {
            release(b, s);
            return e;
        };
        release(b, s);
    }
    return done;
}

pub fn release(b: *Cuda, s: *const lanes.Stream) void {
    if (b.usage_reads.fetchRemove(s)) |kv| {
        var r = kv.value;
        r.deinit();
    }
}

pub fn deinit(b: *Cuda) void {
    var it = b.usage_reads.valueIterator();
    while (it.next()) |r| r.deinit();
    b.usage_reads.deinit(b.gpa);
}
