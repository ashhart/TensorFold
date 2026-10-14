//! A native Gemma 4 dense model: admitted config, packed weights, the glue pipelines and the lane projections.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const ckpt = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const weights = @import("weights.zig");
const kernels = @import("kernels.zig");
const affine_mm = @import("../../core/affine_mm.zig");

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ckpt.Checkpoint,
    weights: weights.Weights,
    kernels: kernels.Kernels,
    projections: weights.Projections = .{},
    /// core/affine_mm's pipelines for prompt chunks: 4-bit groups of 64 from the checkpoint's own tensors.
    mm: affine_mm.Pipes,
    /// RoPE inverse frequencies, float32 [dim / 2], for sliding and for global layers.
    inv_sliding: mtl.Buffer,
    inv_global: mtl.Buffer,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Model {
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = undefined;
        m.gpa = gpa;
        m.projections = .{};
        m.config = try cfg.Config.read(gpa, io, dir);
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        if (!m.device.tensorUnits()) return error.GemmaNeedsTensorUnits;
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.checkpoint = ckpt.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        const files = try shards.shardFiles(gpa, io, dir);
        defer shards.freeShardFiles(gpa, files);
        for (files) |path| try m.checkpoint.addFileSelected(m.device, path, "", "language_model.");
        m.weights = try weights.load(gpa, m.device, &m.checkpoint);
        errdefer m.weights.deinit(gpa);
        m.kernels = try kernels.Kernels.init(gpa, m.device);
        errdefer m.kernels.deinit();
        {
            const text = try affine_mm.source(m.device, gpa, .{});
            defer gpa.free(text);
            const lib = try mtl.Library.fromSource(m.device, text, mtl.CompileOptions.mlx());
            defer lib.deinit();
            for (&m.mm, affine_mm.names) |*p, name| p.* = try mtl.Pipeline.init(m.device, lib, name, false);
        }
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        m.inv_sliding = try m.device.buffer(cfg.sliding_dim / 2 * 4, opts);
        m.inv_global = try m.device.buffer(cfg.global_dim / 2 * 4, opts);
        try kernels.inverseFrequencies(&m.kernels, m.queue, m.device, false, m.inv_sliding.slice(f32, cfg.sliding_dim / 2));
        try kernels.inverseFrequencies(&m.kernels, m.queue, m.device, true, m.inv_global.slice(f32, cfg.global_dim / 2));
        return m;
    }

    pub fn deinit(m: *Model) void {
        m.inv_sliding.deinit();
        m.inv_global.deinit();
        m.projections.deinit(m.gpa);
        for (m.mm) |p| p.deinit();
        m.kernels.deinit();
        m.weights.deinit(m.gpa);
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};
