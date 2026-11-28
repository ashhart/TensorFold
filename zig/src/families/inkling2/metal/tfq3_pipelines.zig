//! Inkling 2 Mini TFQ3/TFQ4 Metal compute pipelines and dispatch wrappers.
const mtl = @import("metal");
const sources = @import("kernel_sources");
const state = @import("state.zig");
const tfq3 = @import("tfq3.zig");

const Ref = tfq3.Ref;
const Layer = tfq3.Layer;
const Args = tfq3.Args;
const Dense4Args = tfq3.Dense4Args;

pub const Pipelines = struct {
    fwht_x: mtl.Pipeline,
    moe_gu: mtl.Pipeline,
    moe_down: mtl.Pipeline,
    moe4_gu: mtl.Pipeline,
    moe4_down: mtl.Pipeline,
    dense4: mtl.Pipeline,
    xrot: mtl.Buffer,

    pub fn load(device: mtl.Device) !Pipelines {
        const lib = try mtl.Library.fromSource(device, sources.inkling2_tfq3, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const fwht_x = try mtl.Pipeline.init(device, lib, "tf_ink2_tfq3_fwht_x", false);
        errdefer fwht_x.deinit();
        if (fwht_x.maxThreads() < 32) return error.Inkling2Threads;
        const moe_gu = try mtl.Pipeline.init(device, lib, "tf_ink2_tfq3_moe_gu_softcap", false);
        errdefer moe_gu.deinit();
        if (moe_gu.maxThreads() < 256) return error.Inkling2Threads;
        const moe_down = try mtl.Pipeline.init(device, lib, "tf_ink2_tfq3_moe_down", false);
        errdefer moe_down.deinit();
        if (moe_down.maxThreads() < 256) return error.Inkling2Threads;
        const moe4_gu = try mtl.Pipeline.init(device, lib, "tf_ink2_tfq4_moe_gu_softcap", false);
        errdefer moe4_gu.deinit();
        if (moe4_gu.maxThreads() < 256) return error.Inkling2Threads;
        const moe4_down = try mtl.Pipeline.init(device, lib, "tf_ink2_tfq4_moe_down", false);
        errdefer moe4_down.deinit();
        if (moe4_down.maxThreads() < 256) return error.Inkling2Threads;
        const dense4 = try mtl.Pipeline.init(device, lib, "tf_ink2_tfq4_dense", false);
        errdefer dense4.deinit();
        if (dense4.maxThreads() < 256) return error.Inkling2Threads;
        const xrot = try device.buffer(tfq3.xrot_bytes, mtl.ResourceOptions.private);
        return .{
            .fwht_x = fwht_x,
            .moe_gu = moe_gu,
            .moe_down = moe_down,
            .moe4_gu = moe4_gu,
            .moe4_down = moe4_down,
            .dense4 = dense4,
            .xrot = xrot,
        };
    }

    pub fn deinit(p: *Pipelines) void {
        p.xrot.deinit();
        p.dense4.deinit();
        p.moe4_down.deinit();
        p.moe4_gu.deinit();
        p.moe_down.deinit();
        p.moe_gu.deinit();
        p.fwht_x.deinit();
        p.* = undefined;
    }

    /// Dispatches forward FWHT-128, softcapped SwiGLU gate+up, and down into BF16 `down_out[rows * 8, 2304]`.
    pub fn dispatch(
        p: *const Pipelines,
        e: mtl.ComputeEncoder,
        layer: Layer,
        mlp_normed: Ref,
        route_ids: Ref,
        act_rot_gu: Ref,
        down_out: Ref,
        rows: usize,
    ) !void {
        if (layer.hidden != state.hidden or layer.inter != state.intermediate) return error.Tfq3Geometry;
        try p.dispatchWith(e, layer, mlp_normed, route_ids, act_rot_gu, down_out, rows, state.experts_per_token, 4.0, 25.0);
    }

    /// Dispatches TFQ3 or hybrid TFQ4 MoE with caller-specified `topk`, `gate_cap`, and `up_cap`.
    pub fn dispatchWith(
        p: *const Pipelines,
        e: mtl.ComputeEncoder,
        layer: Layer,
        mlp_normed: Ref,
        route_ids: Ref,
        act_rot_gu: Ref,
        down_out: Ref,
        rows: usize,
        topk: u32,
        gate_cap: f32,
        up_cap: f32,
    ) !void {
        if (rows == 0 or rows * layer.hidden * 2 > tfq3.xrot_bytes) return error.Inkling2RowsExceeded;
        if (layer.hidden == 0 or layer.inter == 0 or (layer.hidden % 128) != 0 or (layer.inter % 128) != 0) return error.Tfq3Geometry;
        const a: Args = .{
            .rows = @intCast(rows),
            .hidden = layer.hidden,
            .inter = layer.inter,
            .experts = layer.num_experts,
            .topk = topk,
            .gate_cap = gate_cap,
            .up_cap = up_cap,
            .table4_kind = layer.table4_kind,
        };
        const slots: usize = rows * topk;

        e.setPipeline(p.fwht_x);
        e.setBuffer(mlp_normed.buf, mlp_normed.off, 0);
        e.setBuffer(p.xrot, 0, 1);
        e.setValue(a, 2);
        e.dispatchGroups(mtl.Size.of(layer.hidden / 128, rows, 1), mtl.Size.of(32, 1, 1));

        if (layer.has_tfq4) {
            const t4 = layer.tables4 orelse return error.MissingTfq4Tables;
            const t3 = layer.tables orelse t4;
            e.setPipeline(p.moe4_gu);
            e.setBuffer(layer.weights, 0, 0);
            e.setBuffer(layer.scales, 0, 1);
            e.setBuffer(layer.meta, 0, 2);
            e.setBuffer(t3, 0, 3);
            e.setBuffer(t4, 64, 4);
            e.setBuffer(p.xrot, 0, 5);
            e.setBuffer(route_ids.buf, route_ids.off, 6);
            e.setBuffer(act_rot_gu.buf, act_rot_gu.off, 7);
            e.setValue(a, 8);
            e.dispatchGroups(mtl.Size.of(layer.inter / 128, slots, 1), mtl.Size.of(256, 1, 1));

            e.setPipeline(p.moe4_down);
            e.setBuffer(layer.weights, 0, 0);
            e.setBuffer(layer.scales, 0, 1);
            e.setBuffer(layer.meta, 0, 2);
            e.setBuffer(t3, 0, 3);
            e.setBuffer(t4, 64, 4);
            e.setBuffer(act_rot_gu.buf, act_rot_gu.off, 5);
            e.setBuffer(route_ids.buf, route_ids.off, 6);
            e.setBuffer(down_out.buf, down_out.off, 7);
            e.setValue(a, 8);
            e.dispatchGroups(mtl.Size.of(layer.hidden / 128, slots, 1), mtl.Size.of(256, 1, 1));
            return;
        }

        const t3 = layer.tables orelse return error.MissingTfq3Tables;
        e.setPipeline(p.moe_gu);
        e.setBuffer(layer.weights, 0, 0);
        e.setBuffer(layer.scales, 0, 1);
        e.setBuffer(layer.meta, 0, 2);
        e.setBuffer(t3, 0, 3);
        e.setBuffer(p.xrot, 0, 4);
        e.setBuffer(route_ids.buf, route_ids.off, 5);
        e.setBuffer(act_rot_gu.buf, act_rot_gu.off, 6);
        e.setValue(a, 7);
        e.dispatchGroups(mtl.Size.of(layer.inter / 128, slots, 1), mtl.Size.of(256, 1, 1));

        e.setPipeline(p.moe_down);
        e.setBuffer(layer.weights, 0, 0);
        e.setBuffer(layer.scales, 0, 1);
        e.setBuffer(layer.meta, 0, 2);
        e.setBuffer(t3, 0, 3);
        e.setBuffer(act_rot_gu.buf, act_rot_gu.off, 4);
        e.setBuffer(route_ids.buf, route_ids.off, 5);
        e.setBuffer(down_out.buf, down_out.off, 6);
        e.setValue(a, 7);
        e.dispatchGroups(mtl.Size.of(layer.hidden / 128, slots, 1), mtl.Size.of(256, 1, 1));
    }

    /// Dispatches standalone TFQ4 dense (`b_bits` 2 or 225, `table_kind` 0..3) into BF16 or FP32 `out`.
    pub fn dispatchDense4(
        p: *const Pipelines,
        e: mtl.ComputeEncoder,
        weights: mtl.Buffer,
        scales: mtl.Buffer,
        tables4_file: mtl.Buffer,
        x_bf16: Ref,
        out: Ref,
        args: Dense4Args,
    ) !void {
        if (args.rows == 0 or args.rows * args.k * 2 > tfq3.xrot_bytes) return error.Inkling2RowsExceeded;
        if (args.n == 0 or args.k == 0 or (args.n % 128) != 0 or (args.k % 128) != 0) return error.Tfq3Geometry;
        const fa: Args = .{
            .rows = args.rows,
            .hidden = args.k,
            .inter = 0,
            .experts = 0,
            .topk = 1,
            .gate_cap = 0,
            .up_cap = 0,
            .table4_kind = args.table_kind,
        };
        e.setPipeline(p.fwht_x);
        e.setBuffer(x_bf16.buf, x_bf16.off, 0);
        e.setBuffer(p.xrot, 0, 1);
        e.setValue(fa, 2);
        e.dispatchGroups(mtl.Size.of(args.k / 128, args.rows, 1), mtl.Size.of(32, 1, 1));

        e.setPipeline(p.dense4);
        e.setBuffer(weights, 0, 0);
        e.setBuffer(scales, 0, 1);
        e.setBuffer(tables4_file, 64, 2);
        e.setBuffer(p.xrot, 0, 3);
        e.setBuffer(out.buf, out.off, 4);
        e.setValue(args, 5);
        e.dispatchGroups(mtl.Size.of(args.n / 128, args.rows, 1), mtl.Size.of(256, 1, 1));
    }
};
