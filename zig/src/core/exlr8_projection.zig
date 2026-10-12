//! EXLR8 on Metal: activations rotated once a token, a fused trellis-decode projection of 1-16 rows, a down fan-in.

const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

const job = @import("exlr8").job;
pub const max_rows = job.max_rows;
pub const none = job.none;
pub const capacities = job.capacities;
pub const Job = job.Job;
pub const Rotate = job.Rotate;
pub const Combine = job.Combine;
const Rows = job.Rows;

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

/// What the projection reads besides its rows: trellis bytes, tile-run offsets, fp16 scales, i8 signs and jobs.
pub const Tables = struct {
    weights: mtl.Buffer,
    bases: mtl.Buffer,
    halves: mtl.Buffer,
    signs: mtl.Buffer,
    jobs: mtl.Buffer,
    count: usize,
};

pub const Pipelines = struct {
    rotate: mtl.Pipeline,
    project: [capacities.len]mtl.Pipeline,
    combine: mtl.Pipeline,
    weights: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = try mtl.Library.fromSource(device, sources.core_exlr8_projection, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const rotate = try mtl.Pipeline.init(device, lib, "tf_exlr8_rotate", false);
        errdefer rotate.deinit();
        var project: [capacities.len]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (project[0..made]) |q| q.deinit();
        for (capacities, &project) |r, *q| {
            var name: [32]u8 = undefined;
            q.* = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&name, "tf_exlr8_project_{d}", .{r}), false);
            made += 1;
            if (q.simdWidth() != 32 or q.maxThreads() < 256) return error.UnsupportedExlr8Pipeline;
        }
        const combine = try mtl.Pipeline.init(device, lib, "tf_exlr8_combine", false);
        errdefer combine.deinit();
        const weights = try mtl.Pipeline.init(device, lib, "tf_exlr8_weights", false);
        errdefer weights.deinit();
        return .{ .rotate = rotate, .project = project, .combine = combine, .weights = weights };
    }

    pub fn deinit(p: *Pipelines) void {
        p.rotate.deinit();
        for (p.project) |q| q.deinit();
        p.combine.deinit();
        p.weights.deinit();
    }
};

fn fits(r: Ref, bytes: usize) !void {
    if (r.off > r.buf.length() or bytes > r.buf.length() - r.off) return error.Exlr8ViewTooSmall;
}

/// One simdgroup per (row, 128-block, slot); slot_scales holds each slot's scale offset when a.scales is set.
pub fn encodeRotate(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, signs: mtl.Buffer, halves: mtl.Buffer, r: Ref, a: Rotate, blocks: usize, slots: usize, slot_scales: ?Ref) !void {
    if (a.rows == 0 or blocks == 0 or slots == 0 or (a.scales != 0) != (slot_scales != null)) return error.InvalidExlr8Rows;
    const elem: usize = if (a.x_f32 != 0) 4 else 2;
    const span = (slots - 1) * @as(usize, a.slot_cols) + blocks * 128;
    try fits(x, ((a.rows - 1) * @as(usize, a.x_stride) + a.x_col + span) * elem);
    try fits(r, ((a.rows - 1) * @as(usize, a.r_stride) + a.r_col + span) * 4);
    if (slot_scales) |s| try fits(s, slots * 4);
    e.setPipeline(p.rotate);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(signs, 0, 1);
    e.setBuffer(halves, 0, 2);
    e.setValue(a, 3);
    e.setBuffer(r.buf, r.off, 4);
    const s = slot_scales orelse r;
    e.setBuffer(s.buf, s.off, 5);
    e.dispatchGroups(.{ .width = blocks, .height = a.rows, .depth = slots }, .{ .width = 32 });
}

/// One threadgroup per job over 1-16 rows of rotated fp32 input; y holds rows of y_stride elements (f32 or bf16).
pub fn encodeProject(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, t: Tables, y: Ref, rows: usize, x_stride: usize, y_stride: usize, y_f32: bool) !void {
    if (rows == 0 or rows > max_rows) return error.InvalidExlr8Rows;
    if (t.count == 0 or t.jobs.length() < t.count * @sizeOf(Job)) return error.InvalidExlr8Jobs;
    try fits(x, ((rows - 1) * x_stride + 128) * 4);
    try fits(y, ((rows - 1) * y_stride + 128) * @as(usize, if (y_f32) 4 else 2));
    e.setPipeline(p.project[job.capacity(rows)]);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(t.weights, 0, 1);
    e.setBuffer(t.bases, 0, 2);
    e.setBuffer(t.halves, 0, 3);
    e.setBuffer(t.signs, 0, 4);
    e.setBuffer(t.jobs, 0, 5);
    e.setValue(Rows{ .rows = @intCast(rows), .x_stride = @intCast(x_stride), .y_stride = @intCast(y_stride), .y_f32 = @intFromBool(y_f32) }, 6);
    e.setBuffer(y.buf, y.off, 7);
    e.dispatchGroups(.{ .width = t.count }, .{ .width = 256 });
}

/// One simdgroup per (row, 128-block) over `blocks` output blocks; sv[e] indexes slot e's block scales.
pub fn encodeCombine(p: *const Pipelines, e: mtl.ComputeEncoder, z: Ref, weights: Ref, sv: Ref, halves: mtl.Buffer, signs: mtl.Buffer, y: Ref, a: Combine, blocks: usize) !void {
    if (a.rows == 0 or a.slots == 0 or blocks == 0) return error.InvalidExlr8Rows;
    try fits(z, ((a.rows - 1) * @as(usize, a.z_stride) + (a.slots - 1) * @as(usize, a.z_slot) + blocks * 128) * 4);
    try fits(weights, @as(usize, a.rows) * a.slots * 4);
    try fits(sv, @as(usize, a.slots) * 4);
    try fits(y, ((a.rows - 1) * @as(usize, a.y_stride) + blocks * 128) * @as(usize, if (a.y_f32 != 0) 4 else 2));
    e.setPipeline(p.combine);
    e.setBuffer(z.buf, z.off, 0);
    e.setBuffer(weights.buf, weights.off, 1);
    e.setBuffer(sv.buf, sv.off, 2);
    e.setBuffer(halves, 0, 3);
    e.setBuffer(signs, 0, 4);
    e.setValue(a, 5);
    e.setBuffer(y.buf, y.off, 6);
    e.dispatchGroups(.{ .width = blocks, .height = a.rows }, .{ .width = 32 });
}

test {
    _ = job;
}
