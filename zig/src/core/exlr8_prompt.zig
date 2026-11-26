//! EXLR8 prompt rows on the tensor units: trellis jobs (tiles decoded once for 32 rows) and dense matrices, bf16 rows.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("frags.zig");
const op = @import("exlr8_projection.zig");
const job = @import("exlr8_job.zig");
const dg = @import("exlr8_dense_gpu.zig");

/// The rows one job takes at most (its member list's length).
pub const job_rows = 32;

pub const Pipelines = struct {
    prompt: mtl.Pipeline,
    dense: mtl.Pipeline,
    dense_fast: ?mtl.Pipeline = null,
    dense_fast64: ?mtl.Pipeline = null,
    dense_bf16_32: ?mtl.Pipeline = null,

    fn library(gpa: std.mem.Allocator, device: mtl.Device, parts: []const []const u8) !mtl.Library {
        const joined = try std.mem.concat(gpa, u8, parts);
        defer gpa.free(joined);
        const raw = try frags.source(device, gpa, joined);
        defer gpa.free(raw);
        const text = if (!device.tensorUnits())
            try std.mem.concat(gpa, u8, &.{ "#define TF_GLM_M3_FAST 1\n", raw })
        else
            try gpa.dupe(u8, raw);
        defer gpa.free(text);
        return mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    }

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lt = try library(gpa, device, &.{ sources.core_exlr8_projection, sources.core_exlr8_prompt });
        defer lt.deinit();
        const ld = try library(gpa, device, &.{ sources.core_exlr8_dense, sources.core_exlr8_dense_prompt });
        defer ld.deinit();
        const p = try mtl.Pipeline.init(device, lt, "tf_exlr8_prompt", false);
        const d = try mtl.Pipeline.init(device, ld, "tf_exlr8_dense_prompt", false);
        const df: ?mtl.Pipeline = if (!device.tensorUnits())
            try mtl.Pipeline.init(device, ld, "tf_exlr8_dense_prompt_fast", false)
        else
            null;
        const df64: ?mtl.Pipeline = if (!device.tensorUnits())
            try mtl.Pipeline.init(device, ld, "tf_exlr8_dense_prompt_64x64", false)
        else
            null;
        const dbf32: ?mtl.Pipeline = if (!device.tensorUnits())
            try mtl.Pipeline.init(device, ld, "tf_exlr8_dense_prompt_bf16_32x64", false)
        else
            null;
        if (p.simdWidth() != 32 or p.maxThreads() < 256 or d.maxThreads() < 64) return error.UnsupportedExlr8Pipeline;
        return .{ .prompt = p, .dense = d, .dense_fast = df, .dense_fast64 = df64, .dense_bf16_32 = dbf32 };
    }

    pub fn deinit(p: *Pipelines) void {
        p.prompt.deinit();
        p.dense.deinit();
        if (p.dense_fast) |df| df.deinit();
        if (p.dense_fast64) |df64| df64.deinit();
        if (p.dense_bf16_32) |dbf| dbf.deinit();
    }
};

/// Row tiles a window of `rows` takes.
pub fn tiles(rows: usize) usize {
    return (rows + job_rows - 1) / job_rows;
}

fn bind(p: *const Pipelines, e: mtl.ComputeEncoder, x: op.Ref, t: op.Tables, y: op.Ref, rows: usize, x_stride: usize, y_stride: usize, y_f32: bool, members: ?op.Ref) void {
    e.setPipeline(p.prompt);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(t.weights, 0, 1);
    e.setBuffer(t.bases, 0, 2);
    e.setBuffer(t.halves, 0, 3);
    e.setBuffer(t.signs, 0, 4);
    e.setBuffer(t.jobs, t.jobs_off, 5);
    e.setValue(job.Rows{ .rows = @intCast(rows), .x_stride = @intCast(x_stride), .y_stride = @intCast(y_stride), .y_f32 = @intFromBool(y_f32) }, 6);
    e.setBuffer(y.buf, y.off, 7);
    const m = members orelse op.Ref{ .buf = t.jobs };
    e.setBuffer(m.buf, m.off, 8);
}

/// `t.count` jobs over member rows (32 a job), or over the window's `rows` 32 at a time without members, x into y.
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, x: op.Ref, t: op.Tables, y: op.Ref, rows: usize, x_stride: usize, y_stride: usize, y_f32: bool, members: ?op.Ref) !void {
    if (t.count == 0 or rows == 0 or t.jobs.length() < t.jobs_off + t.count * @sizeOf(job.Job)) return error.InvalidExlr8Jobs;
    bind(p, e, x, t, y, rows, x_stride, y_stride, y_f32, members);
    e.dispatchGroups(.{ .width = t.count, .height = if (members == null) tiles(rows) else 1 }, .{ .width = 256 });
}

/// `encode` over member jobs with the job count read on the GPU (three u32 at `args`).
pub fn encodeIndirect(p: *const Pipelines, e: mtl.ComputeEncoder, x: op.Ref, t: op.Tables, y: op.Ref, x_stride: usize, y_stride: usize, y_f32: bool, members: op.Ref, args: op.Ref) void {
    bind(p, e, x, t, y, job_rows, x_stride, y_stride, y_f32, members);
    e.dispatchIndirect(args.buf, args.off, .{ .width = 256 });
}

/// Row-stored matrices (any format but fp32) over the window's `rows`, 32 outputs and 32 rows a threadgroup.
pub fn encodeDense(p: *const Pipelines, e: mtl.ComputeEncoder, x: op.Ref, t: dg.Tables, y: op.Ref, rows: usize, x_stride: usize, y_stride: usize, y_f32: bool) !void {
    if (rows == 0 or t.count == 0) return error.InvalidExlr8Rows;
    e.setPipeline(p.dense);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(t.weights, 0, 1);
    e.setBuffer(t.jobs, 0, 2);
    e.setBuffer(t.groups, t.groups_off, 3);
    e.setValue(job.Rows{ .rows = @intCast(rows), .x_stride = @intCast(x_stride), .y_stride = @intCast(y_stride), .y_f32 = @intFromBool(y_f32) }, 4);
    e.setBuffer(y.buf, y.off, 5);
    e.dispatchGroups(.{ .width = t.count, .height = tiles(rows) }, .{ .width = 64 });
}

/// Fast 4-simdgroup (64x64 half MMA for rows>=64, 64x32 bfloat MMA otherwise) dense prompt dispatch on M3/M4.
pub fn encodeDenseFast(p: *const Pipelines, e: mtl.ComputeEncoder, x: op.Ref, t64: dg.Tables, t32: dg.Tables, y: op.Ref, rows: usize, x_stride: usize, y_stride: usize, y_f32: bool) !void {
    if (rows >= 64) {
        if (p.dense_fast64) |df64| {
            if (t64.count == 0) return error.InvalidExlr8Rows;
            e.setPipeline(df64);
            e.setBuffer(x.buf, x.off, 0);
            e.setBuffer(t64.weights, 0, 1);
            e.setBuffer(t64.jobs, 0, 2);
            e.setBuffer(t64.groups, t64.groups_off, 3);
            e.setValue(job.Rows{ .rows = @intCast(rows), .x_stride = @intCast(x_stride), .y_stride = @intCast(y_stride), .y_f32 = @intFromBool(y_f32) }, 4);
            e.setBuffer(y.buf, y.off, 5);
            e.dispatchGroups(.{ .width = t64.count, .height = (rows + 63) / 64 }, .{ .width = 128 });
            return;
        }
    }
    if (p.dense_fast) |df| {
        if (rows == 0 or t64.count == 0) return error.InvalidExlr8Rows;
        e.setPipeline(df);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(t64.weights, 0, 1);
        e.setBuffer(t64.jobs, 0, 2);
        e.setBuffer(t64.groups, t64.groups_off, 3);
        e.setValue(job.Rows{ .rows = @intCast(rows), .x_stride = @intCast(x_stride), .y_stride = @intCast(y_stride), .y_f32 = @intFromBool(y_f32) }, 4);
        e.setBuffer(y.buf, y.off, 5);
        e.dispatchGroups(.{ .width = t64.count, .height = tiles(rows) }, .{ .width = 128 });
        return;
    }
    return encodeDense(p, e, x, t32, y, rows, x_stride, y_stride, y_f32);
}

/// Native BF16 MMA 32x64 prompt dispatch (ideal for 256-expert router GEMM on M3/M4, -21% vs 64x64 half).
pub fn encodeDenseBf16Fast(p: *const Pipelines, e: mtl.ComputeEncoder, x: op.Ref, t64: dg.Tables, t32: dg.Tables, y: op.Ref, rows: usize, x_stride: usize, y_stride: usize, y_f32: bool) !void {
    if (rows >= 64) {
        if (p.dense_bf16_32) |dbf| {
            if (t32.count == 0) return error.InvalidExlr8Rows;
            e.setPipeline(dbf);
            e.setBuffer(x.buf, x.off, 0);
            e.setBuffer(t32.weights, 0, 1);
            e.setBuffer(t32.jobs, 0, 2);
            e.setBuffer(t32.groups, t32.groups_off, 3);
            e.setValue(job.Rows{ .rows = @intCast(rows), .x_stride = @intCast(x_stride), .y_stride = @intCast(y_stride), .y_f32 = @intFromBool(y_f32) }, 4);
            e.setBuffer(y.buf, y.off, 5);
            e.dispatchGroups(.{ .width = t32.count, .height = (rows + 63) / 64 }, .{ .width = 128 });
            return;
        }
    }
    return encodeDenseFast(p, e, x, t64, t32, y, rows, x_stride, y_stride, y_f32);
}
