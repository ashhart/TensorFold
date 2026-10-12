//! EXLR8 trellis jobs for prompt rows on the tensor units: each job's tiles decoded once for up to 32 rows (bf16 rows).
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("frags.zig");
const op = @import("exlr8_projection.zig");
const job = @import("exlr8_job.zig");

/// The rows one job takes at most (its member list's length).
pub const job_rows = 32;

pub const Pipelines = struct {
    prompt: mtl.Pipeline,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const joined = try std.mem.concat(gpa, u8, &.{ sources.core_exlr8_projection, sources.core_exlr8_prompt });
        defer gpa.free(joined);
        const text = try frags.source(device, gpa, joined);
        defer gpa.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const p = try mtl.Pipeline.init(device, lib, "tf_exlr8_prompt", false);
        if (p.simdWidth() != 32 or p.maxThreads() < 256) return error.UnsupportedExlr8Pipeline;
        return .{ .prompt = p };
    }

    pub fn deinit(p: *Pipelines) void {
        p.prompt.deinit();
    }
};

fn bind(p: *const Pipelines, e: mtl.ComputeEncoder, x: op.Ref, t: op.Tables, y: op.Ref, x_stride: usize, y_stride: usize, y_f32: bool, members: ?op.Ref) void {
    e.setPipeline(p.prompt);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(t.weights, 0, 1);
    e.setBuffer(t.bases, 0, 2);
    e.setBuffer(t.halves, 0, 3);
    e.setBuffer(t.signs, 0, 4);
    e.setBuffer(t.jobs, t.jobs_off, 5);
    e.setValue(job.Rows{ .rows = job_rows, .x_stride = @intCast(x_stride), .y_stride = @intCast(y_stride), .y_f32 = @intFromBool(y_f32) }, 6);
    e.setBuffer(y.buf, y.off, 7);
    const m = members orelse op.Ref{ .buf = t.jobs };
    e.setBuffer(m.buf, m.off, 8);
}

/// `t.count` jobs, a threadgroup each, over their member rows (or rows 0-31 without members) of x into y.
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, x: op.Ref, t: op.Tables, y: op.Ref, x_stride: usize, y_stride: usize, y_f32: bool, members: ?op.Ref) !void {
    if (t.count == 0 or t.jobs.length() < t.jobs_off + t.count * @sizeOf(job.Job)) return error.InvalidExlr8Jobs;
    bind(p, e, x, t, y, x_stride, y_stride, y_f32, members);
    e.dispatchGroups(.{ .width = t.count }, .{ .width = 256 });
}

/// `encode` with the job count read on the GPU (three u32 at `args`).
pub fn encodeIndirect(p: *const Pipelines, e: mtl.ComputeEncoder, x: op.Ref, t: op.Tables, y: op.Ref, x_stride: usize, y_stride: usize, y_f32: bool, members: op.Ref, args: op.Ref) void {
    bind(p, e, x, t, y, x_stride, y_stride, y_f32, members);
    e.dispatchIndirect(args.buf, args.off, .{ .width = 256 });
}
