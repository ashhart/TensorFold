//! Hy4-flash 5-layer DFly2 drafter glue CUDA launchers (`hy4_dfly2_glue.cu`).
const std = @import("std");
const cuda = @import("cuda");

const Stream = cuda.Stream;
const Function = cuda.Function;

pub const hidden: usize = 3072;
pub const qw: usize = 6;
pub const target_topk: usize = 2048;
pub const topk_pad: usize = 2176;

pub const DflyRows = extern struct {
    base: u64,
    ids: u64 = 0,
    ld: i64,
    ids_stride: i32 = 1,
    nrows: i32 = 0,

    comptime {
        std.debug.assert(@sizeOf(DflyRows) == 32);
    }
};

pub const TopkArgs = extern struct {
    mode: i32 = 0,
    src: u64 = 0,
    src_slot_stride: i64 = 0,
    src_row_stride: i64 = 0,
    src_rows_cap: i32 = 0,
    dense_below: i32 = 2048,
    layer_slot: u64 = 0,
    num_layers: i32 = 1,
    active: u64 = 0,
    source_rows: u64 = 0,
    anchors: u64 = 0,
    b: i32 = 1,
    qw: i32 = qw,
    target_topk: i32 = target_topk,
    local_window: i32 = 0,
    k_pad: i32 = topk_pad,
    out: u64 = 0,
    out_layer_stride: i64 = 0,
    slot_out: u64 = 0,
    row_pos_out: u64 = 0,
    block_table: u64 = 0,
    bt_stride: i64 = 0,
    blocks_per_req: i32 = 0,
    block_size: i32 = 0,
    max_model_len: i32 = 0,
    req_stream: u64 = 0,
    ctx: i64 = 0,
    pad_slot: i64 = -1,

    comptime {
        std.debug.assert(@sizeOf(TopkArgs) == 192);
    }
};

pub const Kernels = struct {
    mod: cuda.Module,
    dual: Function,
    fusion: Function,
    boundary: Function,
    extend_topk: Function,
    pick_argmax: Function,
    gather_hidden: Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_dfly2_glue);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .dual = try mod.function("hy4_dfly_dual_rmsnorm_cat_s1"),
            .fusion = try mod.function("hy4_dfly_fusion_gated_add_s1"),
            .boundary = try mod.function("hy4_dfly_sublayer_boundary_s1"),
            .extend_topk = try mod.function("hy4_dfly_extend_topk_k"),
            .pick_argmax = try mod.function("hy4_dfly_pick_argmax"),
            .gather_hidden = try mod.function("hy4_dfly_gather_hidden"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }

    pub fn runDual(k: *const Kernels, s: Stream, a: DflyRows, wa: u64, b: DflyRows, wb: u64, eps: f32, out: u64, ld_out: usize, rows: usize) !void {
        if (rows == 0) return;
        var args: cuda.Args = .{};
        args.add(a);
        args.add(wa);
        args.add(b);
        args.add(wb);
        args.add(eps);
        args.add(out);
        args.add(@as(i64, @intCast(ld_out)));
        try cuda.launch.launch(k.dual, .{ .grid = .{ .x = 1, .y = 2, .z = @intCast(rows) }, .block = .{ .x = 384 } }, s, &args);
    }

    pub fn runFusion(k: *const Kernels, s: Stream, e: DflyRows, eh: u64, gate: u64, w_norm: u64, eps: f32, res_out: u64, norm_out: u64, n: usize) !void {
        if (n == 0) return;
        var args: cuda.Args = .{};
        args.add(e);
        args.add(eh);
        args.add(@as(i64, hidden));
        args.add(@as(i64, 0));
        args.add(@as(i32, 1));
        args.add(gate);
        args.add(w_norm);
        args.add(eps);
        args.add(@as(i32, qw));
        args.add(res_out);
        args.add(@as(i64, hidden));
        args.add(norm_out);
        args.add(@as(i64, hidden));
        try cuda.launch.launch(k.fusion, .{ .grid = .{ .x = 1, .y = @intCast(n), .z = 1 }, .block = .{ .x = 384 } }, s, &args);
    }

    pub fn runBoundary(k: *const Kernels, s: Stream, part: u64, part_ld: usize, world_stride: usize, world: usize, sandwich_w: u64, residual: u64, next_w: u64, res_out: u64, next_out: u64, eps: f32, rows: usize) !void {
        if (rows == 0) return;
        var args: cuda.Args = .{};
        args.add(part);
        args.add(@as(i64, @intCast(part_ld)));
        args.add(@as(i64, @intCast(world_stride)));
        args.add(@as(i32, @intCast(world)));
        args.add(sandwich_w);
        args.add(residual);
        args.add(@as(i64, hidden));
        args.add(next_w);
        args.add(res_out);
        args.add(@as(i64, hidden));
        args.add(next_out);
        args.add(@as(i64, hidden));
        args.add(eps);
        try cuda.launch.launch(k.boundary, .{ .grid = .{ .x = 1, .y = @intCast(rows), .z = 1 }, .block = .{ .x = 384 } }, s, &args);
    }

    pub fn runExtendTopk(k: *const Kernels, s: Stream, ta: TopkArgs) !void {
        const blocks_x: u32 = @intCast(ta.b * ta.num_layers);
        const blocks_y: u32 = @intCast(@divTrunc(ta.k_pad + 511, 512));
        if (blocks_x == 0 or blocks_y == 0) return;
        var args: cuda.Args = .{};
        args.add(ta);
        try cuda.launch.launch(k.extend_topk, .{ .grid = .{ .x = blocks_x, .y = blocks_y, .z = 1 }, .block = .{ .x = 128 } }, s, &args);
    }

    pub fn runPickArgmax(k: *const Kernels, s: Stream, argmax_full: u64, world: usize, batch: usize, step: usize, stride: usize, prev_ids: u64, draft_out: u64) !void {
        if (batch == 0) return;
        var args: cuda.Args = .{};
        args.add(argmax_full);
        args.add(@as(i32, @intCast(world)));
        args.add(@as(i32, @intCast(batch)));
        args.add(@as(i32, @intCast(step)));
        args.add(@as(i32, @intCast(stride)));
        args.add(prev_ids);
        args.add(draft_out);
        try cuda.launch.launch(k.pick_argmax, .{ .grid = .{ .x = @intCast((batch + 63) / 64), .y = 1, .z = 1 }, .block = .{ .x = 64 } }, s, &args);
    }

    pub fn runGatherHidden(k: *const Kernels, s: Stream, src_ptrs: u64, dst: u64, batch: usize) !void {
        if (batch == 0) return;
        var args: cuda.Args = .{};
        args.add(src_ptrs);
        args.add(dst);
        args.add(@as(i32, @intCast(batch)));
        try cuda.launch.launch(k.gather_hidden, .{ .grid = .{ .x = @intCast(batch), .y = 1, .z = 1 }, .block = .{ .x = 384 } }, s, &args);
    }
};

test "dfly2_glue_cuda: extern struct ABI sizes match CUDA declarations" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(DflyRows));
    try std.testing.expectEqual(@as(usize, 192), @sizeOf(TopkArgs));
}
