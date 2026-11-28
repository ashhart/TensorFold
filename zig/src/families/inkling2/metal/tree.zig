//! Inkling 2 Mini Metal tree-convolution and surviving-path compaction pipelines.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("inkling2_metal");
pub const plan = @import("inkling2_tree_plan");

pub const Ref = @import("inkling2_metal_ref").Ref;
const unit = mtl.Size.of(32, 1, 1);

pub const Pipelines = struct {
    conv_static: mtl.Pipeline,
    conv_dynamic: mtl.Pipeline,
    keep_kv: mtl.Pipeline,
    keep_conv: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.tree);
    }

    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [4]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |pipe| pipe.deinit();
        for ([_][]const u8{
            "tf_ink2_conv_static_tree",
            "tf_ink2_conv_dynamic_tree",
            "tf_ink2_keep_kv",
            "tf_ink2_keep_conv",
        }, [_]usize{ 32, 256, 128, 256 }, &pipes) |name, req, *pipe| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
            if (pipe.maxThreads() < req) return error.TreeThreads;
        }
        return .{ .conv_static = pipes[0], .conv_dynamic = pipes[1], .keep_kv = pipes[2], .keep_conv = pipes[3] };
    }

    pub fn deinit(p: *Pipelines) void {
        p.keep_conv.deinit();
        p.keep_kv.deinit();
        p.conv_dynamic.deinit();
        p.conv_static.deinit();
        p.* = undefined;
    }
};

pub fn convStatic(p: *const Pipelines, e: mtl.ComputeEncoder, history: Ref, raw: Ref, weight: Ref, src: Ref, out: Ref, rows: u32, channels: u32) void {
    std.debug.assert(rows > 0 and channels > 0);
    e.setPipeline(p.conv_static);
    inline for (.{ history, raw, weight, src, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(plan.ConvTreeArgs{ .rows = rows, .channels = channels }, 5);
    e.dispatchThreads(mtl.Size.of(channels, rows, 1), unit);
}

pub fn convDynamic(p: *const Pipelines, e: mtl.ComputeEncoder, history: Ref, raw: Ref, input: Ref, static_w: Ref, projection: Ref, src: Ref, out: Ref, rows: u32, channels: u32) void {
    std.debug.assert(rows > 0 and channels > 0 and channels <= 1024);
    e.setPipeline(p.conv_dynamic);
    inline for (.{ history, raw, input, static_w, projection, src, out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(plan.ConvTreeArgs{ .rows = rows, .channels = channels }, 7);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

pub fn keepKv(p: *const Pipelines, e: mtl.ComputeEncoder, kv: Ref, args: plan.KeepKvArgs) void {
    std.debug.assert(args.path_len > 0 and args.path_len <= plan.window_rows);
    e.setPipeline(p.keep_kv);
    e.setBuffer(kv.buf, kv.off, 0);
    e.setValue(args, 1);
    e.dispatchGroups(mtl.Size.of(plan.visit_slots, plan.kv_heads, 1), mtl.Size.of(plan.head_dim, 1, 1));
}

pub fn keepConv(p: *const Pipelines, e: mtl.ComputeEncoder, src_conv: Ref, taps: Ref, conv: Ref, args: plan.KeepConvArgs) void {
    e.setPipeline(p.keep_conv);
    e.setBuffer(src_conv.buf, src_conv.off, 0);
    e.setBuffer(taps.buf, taps.off, 1);
    e.setBuffer(conv.buf, conv.off, 2);
    e.setValue(args, 3);
    e.dispatchThreads(mtl.Size.of(plan.hidden, plan.visit_slots * 4, 1), mtl.Size.of(256, 1, 1));
}
