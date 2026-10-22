//! Staged references from the CUDA engine: each point's rows copied to the host once the engine's stream drains.
const std = @import("std");
const cuda = @import("cuda");
const stages = @import("stages.zig");

pub const Capture = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    round: stages.Round,
    host: []u8,

    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream, recorder: *stages.Recorder, capacity: u32, logit_capacity: u32) !Capture {
        const round = try stages.Round.init(recorder, capacity, logit_capacity);
        return .{ .d = d, .stream = stream, .round = round, .host = try a.alloc(u8, round.bytes()) };
    }

    pub fn deinit(c: Capture, a: std.mem.Allocator) void {
        a.free(c.host);
    }

    pub fn hook(c: *Capture) stages.DeviceHook {
        return .{ .ctx = c, .f = copy };
    }

    fn copy(ctx: *anyopaque, at: stages.Point, ptr: u64, rows: usize) !void {
        const c: *Capture = @ptrCast(@alignCast(ctx));
        try c.round.mark(at, rows);
        try c.d.check(c.d.api.cuStreamSynchronize(c.stream.handle), "cuStreamSynchronize");
        const bytes = rows * c.round.recorder.width(at) * 2;
        try c.d.check(c.d.api.cuMemcpyDtoH_v2(c.host[c.round.offset(at)..].ptr, ptr, bytes), "cuMemcpyDtoH stages");
    }

    pub fn begin(c: *Capture) void {
        c.round.begin();
    }

    /// Hands a finished forward, whose rows start at cache position `first`, to the recorder.
    pub fn end(c: *Capture, first: u32) !void {
        try c.round.end(c.host, first);
    }
};
