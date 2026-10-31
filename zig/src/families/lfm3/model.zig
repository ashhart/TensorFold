//! An owned LFM3 checkpoint and its native Metal text pipelines; VL tensors remain intact without vision execution.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const ckpt = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const weights = @import("metal_weights.zig");
const kernels = @import("kernels.zig");
const binned = @import("../../core/binned_projection.zig");
const prompt = @import("prompt_projection.zig");
const counters = @import("../../core/gpu_counters.zig");

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ckpt.Checkpoint,
    weights: weights.Weights,
    kernels: kernels.Kernels,
    resident: ?mtl.ResidencySet = null,
    profile: ?*counters.Recorder = null,
    tiled_attention: ?mtl.Pipeline = null,
    residual_ffn: ?@import("residual_ffn.zig").Candidate = null,
    projection_candidate: ?prompt.Candidate = null,
    binned_candidate: ?binned.Pipelines = null,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Model {
        return loadWithResidency(gpa, io, dir, true);
    }

    pub fn loadWithResidency(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, hold_weights: bool) !*Model {
        var config = try cfg.Config.read(gpa, io, dir);
        errdefer config.deinit();
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.profile = null;
        m.tiled_attention = null;
        m.residual_ffn = null;
        m.projection_candidate = null;
        m.binned_candidate = null;
        m.gpa = gpa;
        m.config = config;
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.checkpoint = ckpt.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        const files = try shards.shardFiles(gpa, io, dir);
        defer shards.freeShardFiles(gpa, files);
        for (files) |path| try m.checkpoint.addFile(m.device, path, "");
        m.weights = try weights.load(&m.checkpoint, &m.config);
        m.kernels = try kernels.load(m.device);
        m.resident = null;
        if (hold_weights) {
            if (m.device.residencySet(m.checkpoint.shards.items.len)) |set| {
                for (m.checkpoint.shards.items) |shard| set.add(shard.buffer);
                set.commit();
                set.requestResidency();
                m.queue.addResidencySet(set);
                m.resident = set;
            } else |_| {}
        }
        return m;
    }

    /// Explicit developer qualification path; ordinary model loading retains the established row arithmetic.
    pub fn enableProjectionCandidate(m: *Model) !void {
        if (m.projection_candidate != null or m.binned_candidate != null) return error.ProjectionCandidateAlreadyEnabled;
        var list: std.ArrayList(weights.Linear) = .empty;
        defer list.deinit(m.gpa);
        try list.append(m.gpa, m.weights.embedding);
        for (m.weights.blocks) |b| {
            switch (b.mixer) {
                .conv => |cv| try list.appendSlice(m.gpa, &.{ cv.in, cv.out }),
                .attention => |at| try list.appendSlice(m.gpa, &.{ at.q, at.k, at.v, at.gate, at.out }),
            }
            try list.appendSlice(m.gpa, &.{ b.ffn.gate, b.ffn.up, b.ffn.down });
        }
        m.projection_candidate = try prompt.Candidate.init(m.device, list.items);
    }

    pub fn enableBinnedProjectionCandidate(m: *Model) !void {
        if (m.projection_candidate != null or m.binned_candidate != null) return error.ProjectionCandidateAlreadyEnabled;
        m.binned_candidate = try binned.Pipelines.load(m.device);
    }

    pub fn dropResident(m: *Model) void {
        if (m.resident) |set| {
            m.queue.removeResidencySet(set);
            set.deinit();
        }
        m.resident = null;
    }

    pub fn enableResidualFFN(m: *Model) !void {
        if (m.residual_ffn != null or !m.device.tensorUnits()) return;
        m.residual_ffn = try @import("residual_ffn.zig").Candidate.init(m.device);
    }

    pub fn enableTiledAttention(m: *Model) !void {
        if (m.tiled_attention != null) return;
        m.tiled_attention = try @import("attention_tiled.zig").load(m.device);
    }

    pub fn deinit(m: *Model) void {
        m.dropResident();
        if (m.tiled_attention) |pipeline| pipeline.deinit();
        if (m.residual_ffn) |*candidate| candidate.deinit();
        if (m.projection_candidate) |*candidate| candidate.deinit();
        if (m.binned_candidate) |*candidate| candidate.deinit();
        m.kernels.deinit();
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.config.deinit();
        m.gpa.destroy(m);
    }
};
