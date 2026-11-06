//! Staged references on Metal: a round copies each stage's rows into its own plane, read once the round has run.
const std = @import("std");
const mtl = @import("metal");
const Ref = @import("projection.zig").Ref;
const Glue = @import("glue_gpu.zig").Kernels;
const stages = @import("core").qwen35_stages;

pub const Planes = struct {
    buffer: mtl.Buffer,
    round: stages.Round,

    pub fn init(device: mtl.Device, recorder: *stages.Recorder, capacity: u32, logit_capacity: u32) !Planes {
        const round = try stages.Round.init(recorder, capacity, logit_capacity);
        return .{ .buffer = try device.buffer(round.bytes(), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked), .round = round };
    }

    pub fn deinit(p: Planes) void {
        p.buffer.deinit();
    }

    pub fn begin(p: *Planes) void {
        p.round.begin();
    }

    pub fn record(p: *Planes, glue: Glue, e: mtl.ComputeEncoder, at: stages.Point, x: Ref, rows: u32) !void {
        try p.round.mark(at, rows);
        const width: u32 = @intCast(p.round.recorder.width(at));
        try glue.unstack(e, x, .{ .buffer = p.buffer, .offset = p.round.offset(at) }, .{ .rows = rows, .width = width, .stride = width, .offset = 0 });
        e.barrier();
    }

    /// Hands a finished round, whose rows start at cache position `first`, to the recorder.
    pub fn end(p: *Planes, first: u32) !void {
        try p.round.end(p.buffer.slice(u8, p.buffer.length()), first);
    }
};
