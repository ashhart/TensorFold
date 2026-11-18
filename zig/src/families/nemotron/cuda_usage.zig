//! Read existing prompt pocket products, including cached prefixes; never run a probe or change a kernel.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const usage = lanes.pocket_usage;
const state = @import("cuda_state.zig");
const Engine = @import("cuda_engine.zig").Engine;
const Cuda = @import("cuda_lanes.zig").Cuda;

pub const Read = struct {
    trace: usage.Capture,
    stage: cuda.HostBuffer,

    pub fn init(e: *Engine) !Read {
        var trace = try usage.Capture.init(e.gpa, e.c.layers, state.pass_rows);
        errdefer trace.deinit();
        return .{ .trace = trace, .stage = try cuda.HostBuffer.alloc(e.ctx.d, state.pass_rows * 64 * 4) };
    }
    pub fn deinit(r: *Read) void {
        r.stage.free();
        r.trace.deinit();
    }
    pub fn read(r: *Read, o: anytype, layer: usize, ptr: u64, width: usize) !void {
        const values = r.stage.slice(f32)[0 .. r.trace.rows * 64];
        errdefer cuda.completion.drain(o.k.d, o.s, "pocket usage read did not drain");
        try o.download(std.mem.sliceAsBytes(values), ptr);
        try o.s.synchronize();
        try r.trace.layer(layer, values, width, 64);
    }
};

pub fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
    const b: *Cuda = @ptrCast(@alignCast(ptr));
    if (!b.usage_enabled or s.untaught) return b.prefillPlain(s);
    var r = try Read.init(b.e);
    defer r.deinit();
    b.e.usage = &r;
    defer b.e.usage = null;
    try b.prefillPlain(s);
    s.pocket_usage = try r.trace.through(s.prompt().len);
}

/// Warm passes keep their end after prefill returns; interior marks use the live chunk before it returns.
pub fn saved(e: *Engine, owner: ?*anyopaque, at: usize) !?usage.Masks {
    if (e.usage) |u| return try u.trace.through(at);
    const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return null));
    return if (at == s.prompt_len) s.pocket_usage else null;
}
