//! EXLR8 dense matrices stored by row on Metal: 1-16 rows a weight read, each row's sums in one fixed order.

const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

const job = @import("exlr8_job.zig");
pub const DenseJob = job.DenseJob;
pub const capacities = job.capacities;
pub const max_rows = job.max_rows;
const Rows = job.Rows;

/// A buffer view, shared with the trellis kernels so one value binds to either.
pub const Ref = @import("exlr8_projection.zig").Ref;

/// One dispatch's matrices: the weight buffer the jobs' offsets index, the jobs, and their (job, block) groups.
pub const Tables = struct {
    weights: mtl.Buffer,
    jobs: mtl.Buffer,
    groups: mtl.Buffer,
    count: usize,
    /// The first group's byte offset in `groups` (groups index jobs from the start of `jobs`).
    groups_off: usize = 0,
    /// Groups of 8 rows a threadgroup (a row a simdgroup) for 1-4 rows: four times the simdgroups, the same sums.
    narrow: bool = false,
    /// Groups of 16 rows a threadgroup (2 rows a simdgroup) for 1 row: higher memory bandwidth on wide matrices.
    q2: bool = false,
};

/// The rows a narrow (8 or 16 outs/TG, TG-cached) dispatch takes at most (covers depth-5 MTP r=6!).
pub const narrow_rows = 6;

pub const Pipelines = struct {
    project: [capacities.len]mtl.Pipeline,
    q2_1: mtl.Pipeline,
    tg2_2: mtl.Pipeline,
    tg2_3: mtl.Pipeline,
    tg2_4: mtl.Pipeline,
    tg2_5: mtl.Pipeline,
    tg2_6: mtl.Pipeline,
    tg2_q2_4: mtl.Pipeline,
    tg2_q2_5: mtl.Pipeline,
    tg2_q2_6: mtl.Pipeline,
    pack_bf16_1: mtl.Pipeline,
    pack_bf16_2: mtl.Pipeline,
    pack_bf16_3: mtl.Pipeline,
    pack_bf16_4: mtl.Pipeline,
    pack_bf16_5: mtl.Pipeline,
    pack_bf16_6: mtl.Pipeline,
    decode: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = try mtl.Library.fromSource(device, sources.core_exlr8_dense, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var project: [capacities.len]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (project[0..made]) |q| q.deinit();
        for (capacities, &project) |r, *q| {
            var name: [32]u8 = undefined;
            q.* = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&name, "tf_exlr8_dense_{d}", .{r}), false);
            made += 1;
            if (q.simdWidth() != 32 or q.maxThreads() < 256) return error.UnsupportedExlr8Pipeline;
        }
        return .{
            .project = project,
            .q2_1 = try mtl.Pipeline.init(device, lib, "tf_exlr8_dense_q2_1", false),
            .tg2_2 = try mtl.Pipeline.init(device, lib, "tf_exlr8_dense_tg2_2", false),
            .tg2_3 = try mtl.Pipeline.init(device, lib, "tf_exlr8_dense_tg2_3", false),
            .tg2_4 = try mtl.Pipeline.init(device, lib, "tf_exlr8_dense_tg2_4", false),
            .tg2_5 = try mtl.Pipeline.init(device, lib, "tf_exlr8_dense_tg2_5", false),
            .tg2_6 = try mtl.Pipeline.init(device, lib, "tf_exlr8_dense_tg2_6", false),
            .tg2_q2_4 = try mtl.Pipeline.init(device, lib, "tf_exlr8_dense_tg2_q2_4", false),
            .tg2_q2_5 = try mtl.Pipeline.init(device, lib, "tf_exlr8_dense_tg2_q2_5", false),
            .tg2_q2_6 = try mtl.Pipeline.init(device, lib, "tf_exlr8_dense_tg2_q2_6", false),
            .pack_bf16_1 = try mtl.Pipeline.init(device, lib, "tf_exlr8_dense_1_pack_bf16", false),
            .pack_bf16_2 = try mtl.Pipeline.init(device, lib, "tf_exlr8_dense_pack_bf16_2", false),
            .pack_bf16_3 = try mtl.Pipeline.init(device, lib, "tf_exlr8_dense_pack_bf16_3", false),
            .pack_bf16_4 = try mtl.Pipeline.init(device, lib, "tf_exlr8_dense_pack_bf16_4", false),
            .pack_bf16_5 = try mtl.Pipeline.init(device, lib, "tf_exlr8_dense_pack_bf16_5", false),
            .pack_bf16_6 = try mtl.Pipeline.init(device, lib, "tf_exlr8_dense_pack_bf16_6", false),
            .decode = try mtl.Pipeline.init(device, lib, "tf_exlr8_dense_decode", false),
        };
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.project) |q| q.deinit();
        inline for (.{
            p.q2_1,        p.tg2_2,       p.tg2_3,       p.tg2_4,       p.tg2_5,       p.tg2_6,
            p.tg2_q2_4,    p.tg2_q2_5,    p.tg2_q2_6,    p.pack_bf16_1, p.pack_bf16_2, p.pack_bf16_3,
            p.pack_bf16_4, p.pack_bf16_5, p.pack_bf16_6, p.decode,
        }) |q| q.deinit();
    }
};

fn fits(r: Ref, bytes: usize) !void {
    if (r.off > r.buf.length() or bytes > r.buf.length() - r.off) return error.Exlr8ViewTooSmall;
}

/// Every job's rows of x (fp32, x_stride apart) through its matrix into y (y_stride apart, f32 or bf16).
pub fn encodeProject(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, t: Tables, y: Ref, rows: usize, x_stride: usize, y_stride: usize, y_f32: bool) !void {
    if (rows == 0 or rows > max_rows) return error.InvalidExlr8Rows;
    if (t.count == 0 or t.groups.length() < t.groups_off + t.count * 8) return error.InvalidExlr8Jobs;
    try fits(x, ((rows - 1) * x_stride + 128) * 4);
    try fits(y, ((rows - 1) * y_stride + 32) * @as(usize, if (y_f32) 4 else 2));
    if (t.narrow and rows > narrow_rows) return error.InvalidExlr8Rows;
    const use_fast = t.narrow and rows <= narrow_rows;
    const pipe = if (use_fast) switch (rows) {
        1 => p.q2_1,
        2 => p.tg2_2,
        3 => p.tg2_3,
        4 => if (t.q2) p.tg2_q2_4 else p.tg2_4,
        5 => if (t.q2) p.tg2_q2_5 else p.tg2_5,
        else => if (t.q2) p.tg2_q2_6 else p.tg2_6,
    } else p.project[job.capacity(rows)];
    const grid_w: usize = if (use_fast and rows == 1) t.count * 2 else t.count;
    const th: usize = if (use_fast and rows == 1) 64 else 256;
    e.setPipeline(pipe);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(t.weights, 0, 1);
    e.setBuffer(t.jobs, 0, 2);
    e.setBuffer(t.groups, t.groups_off, 3);
    e.setValue(Rows{ .rows = @intCast(rows), .x_stride = @intCast(x_stride), .y_stride = @intCast(y_stride), .y_f32 = @intFromBool(y_f32) }, 4);
    e.setBuffer(y.buf, y.off, 5);
    e.dispatchGroups(.{ .width = grid_w }, .{ .width = th });
}

/// R=1..6 FP8_BLOCK projection with direct peer BF16 pack into `send_base` (skips mesh_pack_bf16 + barrier).
pub fn encodeProjectPackBf16(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, t: Tables, y: Ref, rows: usize, x_stride: usize, y_stride: usize, pack_info: [4]u32, send_base: Ref) !void {
    if (rows == 0 or rows > narrow_rows) return error.InvalidExlr8Rows;
    if (t.count == 0 or t.groups.length() < t.groups_off + t.count * 8) return error.InvalidExlr8Jobs;
    try fits(x, ((rows - 1) * x_stride + 128) * 4);
    try fits(y, ((rows - 1) * y_stride + 32) * 4);
    const pipe = switch (rows) {
        1 => p.pack_bf16_1,
        2 => p.pack_bf16_2,
        3 => p.pack_bf16_3,
        4 => p.pack_bf16_4,
        5 => p.pack_bf16_5,
        else => p.pack_bf16_6,
    };
    const grid_w: usize = t.count;
    const th: usize = if (rows == 1) 128 else 256;
    e.setPipeline(pipe);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(t.weights, 0, 1);
    e.setBuffer(t.jobs, 0, 2);
    e.setBuffer(t.groups, t.groups_off, 3);
    e.setValue(Rows{ .rows = @intCast(rows), .x_stride = @intCast(x_stride), .y_stride = @intCast(y_stride), .y_f32 = 1 }, 4);
    e.setBuffer(y.buf, y.off, 5);
    e.setValue(pack_info, 6);
    e.setBuffer(send_base.buf, send_base.off, 7);
    e.dispatchGroups(.{ .width = grid_w }, .{ .width = th });
}

/// The first job's weights as fp32 rows (outs x ins), as the kernels decode them, for checks against a reference.
pub fn encodeDecode(p: *const Pipelines, e: mtl.ComputeEncoder, weights: mtl.Buffer, jobs: Ref, j: DenseJob, out: Ref) !void {
    if (j.ins % 128 != 0 or j.outs == 0) return error.InvalidExlr8Jobs;
    try fits(out, @as(usize, j.outs) * j.ins * 4);
    e.setPipeline(p.decode);
    e.setBuffer(weights, 0, 0);
    e.setBuffer(jobs.buf, jobs.off, 1);
    e.setBuffer(out.buf, out.off, 2);
    e.dispatchGroups(.{ .width = j.ins / 128, .height = j.outs }, .{ .width = 32 });
}

test {
    _ = job;
}
