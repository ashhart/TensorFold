//! A native Gemma 4 dense model (31B or 12B): admitted config, packed weights, the glue pipelines, lane projections.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const ckpt = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const weights = @import("weights.zig");
const kernels = @import("kernels.zig");
const affine_mm = @import("affine_mm.zig");
const round = @import("tree_round_gpu");
const vision = @import("vision.zig");
const vision_unified = @import("vision_unified.zig");
const image = @import("image.zig");
const lane = @import("../../core/lane_projection.zig");

/// The image input: the 31B's tower (27 encoder layers, pooling) or the 12B's embedder (48-px patches, no layers).
pub const Tower = union(enum) {
    encoder: *vision.Vision,
    embedder: *vision_unified.Embedder,

    /// An image's patches (image.prepare's, in `layout`'s order) to `out` [gh * gw / 9, the model's hidden].
    pub fn encode(t: Tower, e: mtl.ComputeEncoder, pixels: []const f32, gh: u32, gw: u32, out: lane.Ref) !void {
        switch (t) {
            inline else => |v| try v.encode(e, pixels, gh, gw, out, null),
        }
    }

    pub fn inject(t: Tower, e: mtl.ComputeEncoder, src: lane.Ref, dst: lane.Ref, n: u32) void {
        switch (t) {
            inline else => |v| v.inject(e, src, dst, n),
        }
    }

    /// The patches' order the processor writes for this input.
    pub fn layout(t: Tower) image.Layout {
        return if (t == .encoder) .patches else .merged;
    }

    /// The working buffers a residency set holds.
    pub fn buffers(t: Tower) []const mtl.Buffer {
        return switch (t) {
            .encoder => |v| &.{ v.scratch, v.patches, v.cos, v.sin },
            .embedder => |v| &.{ v.scratch, v.patches },
        };
    }

    pub fn deinit(t: Tower) void {
        switch (t) {
            inline else => |v| v.deinit(),
        }
    }
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ckpt.Checkpoint,
    weights: weights.Weights,
    kernels: kernels.Kernels,
    projections: weights.Projections,
    /// core/affine_mm's pipelines for prompt chunks: 4-bit groups of 64 from the checkpoint's own tensors.
    mm: affine_mm.Pipes,
    /// RoPE inverse frequencies, float32 [dim / 2], for sliding and for global layers.
    inv_sliding: mtl.Buffer,
    inv_global: mtl.Buffer,
    /// The core's bf16 argmax: a window's greedy tokens on the GPU (the lowest index among ties, as mx.argmax).
    picks: round.Ops,
    /// A drafter's tapped layers: each layer's index among the taps (-1: untapped), and how many there are.
    tap_of: [cfg.max_layers]i8 = @splat(-1),
    taps: u32 = 0,
    /// One-row windows take the fused pipelines (the glue as the projections' prologues); off for the bench's check.
    fuse: bool = true,
    /// The image input, once loadVision read it (serve --vision).
    vision: ?Tower = null,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Model {
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = undefined;
        m.gpa = gpa;
        m.tap_of = @splat(-1);
        m.taps = 0;
        m.fuse = true;
        m.vision = null;
        m.config = try cfg.Config.read(gpa, io, dir);
        m.projections = .{ .geo = m.config.geo };
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
        m.weights = try weights.load(gpa, m.device, &m.checkpoint, m.config.geo);
        errdefer m.weights.deinit(gpa);
        m.kernels = try kernels.Kernels.init(gpa, m.device, m.config.geo);
        errdefer m.kernels.deinit();
        m.picks = try round.Ops.init(m.device);
        errdefer m.picks.deinit();
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

    /// The checkpoint's image input (the 12B's embedder, the 31B's tower); error.NoGemmaVision when it has none.
    pub fn loadVision(m: *Model, io: std.Io, dir: []const u8) !void {
        if (m.vision != null) return;
        const files = try shards.shardFiles(m.gpa, io, dir);
        defer shards.freeShardFiles(m.gpa, files);
        if (m.config.geo.unified()) {
            try vision_unified.addTensors(&m.checkpoint, m.device, files);
            if (!vision_unified.present(&m.checkpoint)) return error.NoGemmaVision;
            m.vision = .{ .embedder = try vision_unified.Embedder.load(m.gpa, m.device, &m.checkpoint, m.config.geo.hidden) };
            return;
        }
        try vision.addTensors(&m.checkpoint, m.device, files);
        if (!vision.present(&m.checkpoint)) return error.NoGemmaVision;
        m.vision = .{ .encoder = try vision.Vision.load(m.gpa, m.device, &m.checkpoint, m.mm, m.config.geo.hidden) };
    }

    /// The drafter's taps, in its feature order: each a layer whose output (layer scalar applied) it reads.
    pub fn setTaps(m: *Model, layers: []const u32) !void {
        if (layers.len == 0 or layers.len > 8) return error.BadGemmaTaps;
        m.tap_of = @splat(-1);
        for (layers, 0..) |l, t| {
            if (l >= m.config.geo.layers or m.tap_of[l] >= 0) return error.BadGemmaTaps;
            m.tap_of[l] = @intCast(t);
        }
        m.taps = @intCast(layers.len);
    }

    pub fn deinit(m: *Model) void {
        if (m.vision) |v| v.deinit();
        m.inv_sliding.deinit();
        m.inv_global.deinit();
        m.projections.deinit(m.gpa);
        m.picks.deinit();
        for (m.mm) |p| p.deinit();
        m.kernels.deinit();
        m.weights.deinit(m.gpa);
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};
