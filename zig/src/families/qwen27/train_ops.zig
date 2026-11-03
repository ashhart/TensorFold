//! Metal learner primitives over the native engine's saved rows and original affine words.
const std = @import("std");
const mtl = @import("metal");
pub const Ref = @import("projection.zig").Ref;
const sources = @import("qwen_runtime_sources");

pub const Shape = extern struct {
    u: [4]u32 align(16) = @splat(0),
    f: [4]f32 align(16) = @splat(0),
};
pub const GdnShape = extern struct { rows: i32, nk: i32, nv: i32, conv: i32, proj: i32 };
pub const Heads = extern struct { rows: i32, heads: i32, kv_heads: i32, dim: i32, qkv: i32, scale: f32, prefix: i32 = 0 };
pub const Q4 = struct { words: Ref, scales: Ref, biases: Ref, n: usize, k: usize };

const gdn_names = [_][]const u8{
    "tf_gdn_norm_back", "tf_gdn_scan_back",
    "tf_gdn_qk_back",   "tf_gdn_gates_back",
    "tf_gdn_conv_back",
};
const attn_names = [_][]const u8{
    "tf_attn_q", "tf_attn_gate_back", "tf_attn_prep_back",
};
const row_names = [_][]const u8{
    "tf_learn_copy",     "tf_learn_widen",       "tf_learn_gdn_act",     "tf_learn_zero", "tf_learn_nudge", "tf_learn_loss",
    "tf_learn_rms_back", "tf_learn_swiglu_back", "tf_learn_pocket_back",
};
const linear_names = [_][]const u8{ "tf_learn_q4_back", "tf_learn_attn_kv_back", "tf_learn_attn_gate_grad" };

pub const Kernels = struct {
    allocator: std.mem.Allocator,
    pipelines: std.StringHashMapUnmanaged(mtl.Pipeline) = .empty,

    pub fn init(a: std.mem.Allocator, device: mtl.Device) !Kernels {
        var k: Kernels = .{ .allocator = a };
        errdefer k.deinit();
        const units = [_]struct { text: []const u8, names: []const []const u8 }{
            .{ .text = sources.gdn_train, .names = &gdn_names },
            .{ .text = sources.attn_train, .names = &attn_names },
            .{ .text = sources.learn_rows, .names = &row_names },
            .{ .text = sources.pocket, .names = &.{"tf_q27_pocket_scaled"} },
            .{ .text = sources.learn_linear, .names = &linear_names },
        };
        for (units) |unit| {
            const library = try mtl.Library.fromSource(device, unit.text, mtl.CompileOptions.mlx());
            defer library.deinit();
            for (unit.names) |name| {
                const pipeline = try mtl.Pipeline.init(device, library, name, false);
                k.pipelines.put(a, name, pipeline) catch |err| {
                    pipeline.deinit();
                    return err;
                };
            }
        }
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        var values = k.pipelines.valueIterator();
        while (values.next()) |p| p.deinit();
        k.pipelines.deinit(k.allocator);
        k.* = undefined;
    }
};

pub const Work = struct { decodes: usize = 0, fmas: usize = 0 };

pub const Ops = struct {
    kernels: *const Kernels,
    e: mtl.ComputeEncoder,
    work: ?*Work = null,

    pub fn go(o: Ops, name: []const u8, buffers: []const Ref, params: anytype, groups: mtl.Size, threads: usize) !void {
        const pipeline = o.kernels.pipelines.get(name) orelse return error.NoLearningKernel;
        if (threads == 0 or threads > pipeline.maxThreads()) return error.UnsupportedLearningGroup;
        o.e.setPipeline(pipeline);
        for (buffers, 0..) |buffer, slot| o.e.setBuffer(buffer.buffer, buffer.offset, slot);
        o.e.setValue(params, buffers.len);
        o.e.dispatchGroups(groups, .{ .width = threads });
        o.e.barrier();
    }

    pub fn linear(o: Ops, name: []const u8, buffers: []const Ref, shape: Shape, count: usize) !void {
        if (count == 0) return;
        try o.go(name, buffers, shape, .{ .width = (count + 255) / 256 }, 256);
    }

    pub fn copy(o: Ops, source: Ref, destination: Ref, bytes: usize) !void {
        const aligned = source.offset % 16 == 0 and destination.offset % 16 == 0;
        try o.linear("tf_learn_copy", &.{ source, destination }, .{ .u = .{ @intCast(bytes), @intFromBool(aligned), 0, 0 } }, (bytes + 15) / 16);
    }

    pub fn widen(o: Ops, source: Ref, destination: Ref, count: usize) !void {
        try o.linear("tf_learn_widen", &.{ source, destination }, .{ .u = .{ @intCast(count), 0, 0, 0 } }, count);
    }

    pub fn zero(o: Ops, destination: Ref, count: usize) !void {
        try o.linear("tf_learn_zero", &.{destination}, .{ .u = .{ @intCast(count), 0, 0, 0 } }, count);
    }

    pub fn nudge(o: Ops, destination: Ref, delta: Ref, count: usize) !void {
        try o.linear("tf_learn_nudge", &.{ destination, delta }, .{ .u = .{ @intCast(count), 0, 0, 0 } }, count);
    }

    pub fn q4Back(o: Ops, gradient: Ref, w: Q4, destination: Ref, rows: usize, add: bool) !void {
        if (o.work) |work| {
            work.decodes += @import("train_schedule.zig").q4Threads(rows, w.k) * w.n;
            work.fmas += rows * w.k * w.n;
        }
        try o.linear("tf_learn_q4_back", &.{ gradient, w.words, w.scales, w.biases, destination }, .{
            .u = .{ @intCast(rows), @intCast(w.n), @intCast(w.k), @intFromBool(add) },
        }, @import("train_schedule.zig").q4Threads(rows, w.k));
    }

    pub fn rmsBack(o: Ops, x: Ref, gain: Ref, dy: Ref, residual_gradient: Ref, rows: usize, width: usize, eps: f32) !void {
        try o.go("tf_learn_rms_back", &.{ x, gain, dy, residual_gradient }, Shape{
            .u = .{ @intCast(rows), @intCast(width), 0, 0 },
            .f = .{ eps, 0, 0, 0 },
        }, .{ .width = rows }, 256);
    }

    pub fn loss(o: Ops, logits: Ref, targets: Ref, weights: Ref, gradient: Ref, stats: Ref, rows: usize, vocab: usize) !void {
        try o.go("tf_learn_loss", &.{ logits, targets, weights, gradient, stats }, Shape{
            .u = .{ @intCast(rows), @intCast(vocab), 0, 0 },
        }, .{ .width = rows }, 256);
    }

    pub fn swigluBack(o: Ops, pair: Ref, gradient: Ref, output: Ref, rows: usize, width: usize) !void {
        try o.linear("tf_learn_swiglu_back", &.{ pair, gradient, output }, .{
            .u = .{ @intCast(rows), @intCast(width), 0, 0 },
        }, rows * width);
    }
};

pub fn at(buffer: mtl.Buffer, offset: usize) Ref {
    return .{ .buffer = buffer, .offset = offset };
}

pub fn plus(ref: Ref, offset: usize) Ref {
    return .{ .buffer = ref.buffer, .offset = ref.offset + offset };
}
