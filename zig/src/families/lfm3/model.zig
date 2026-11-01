//! An owned LFM3 checkpoint and its native Metal text pipelines; a VL checkpoint holds its language model only.
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
    split_attention: ?@import("attention_split.zig").Pipelines = null,
    prompt_nax: ?@import("prompt_nax.zig").Candidate = null,
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
        m.split_attention = null;
        m.prompt_nax = null;
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
        for (files) |path| try addShard(&m.checkpoint, m.device, path, config.kind);
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

    pub fn enableNaxPrompt(m: *Model) !void {
        if (m.prompt_nax != null) return;
        const w = m.weights.blocks[0].ffn.gate;
        m.prompt_nax = try @import("prompt_nax.zig").Candidate.init(m.device, w.bits, @intCast(w.group));
    }

    pub fn enableTiledAttention(m: *Model) !void {
        if (m.tiled_attention != null) return;
        m.tiled_attention = try @import("attention_tiled.zig").load(m.device);
        m.split_attention = try @import("attention_split.zig").Pipelines.load(m.device);
    }

    pub fn deinit(m: *Model) void {
        m.dropResident();
        if (m.tiled_attention) |pipeline| pipeline.deinit();
        if (m.split_attention) |pipelines| pipelines.deinit();
        if (m.prompt_nax) |*candidate| candidate.deinit();
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

/// A VL checkpoint's text tensors, wrapped (`language_model.`) or bare (`model.`), pack without tower and projector.
pub fn addShard(c: *ckpt.Checkpoint, device: mtl.Device, path: [:0]const u8, kind: cfg.Kind) !void {
    if (kind == .text) return c.addFile(device, path, "");
    // both namespaces metal_weights.load accepts; a shard holding both is still refused there (MixedNamespaces)
    for ([_][]const u8{ "language_model.", "model." }) |text| try c.addFileSplit(device, path, "", text, never);
}

fn never(_: []const u8, _: usize) bool {
    return false;
}

test "an LFM3-VL shard, wrapped or bare, holds its language model only; a text shard holds everything as before" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    inline for (.{ "language_model.model.embed_tokens.weight", "model.embed_tokens.weight" }) |name|
        try textOnly(device, name);
}

/// A shard in LFM3-VL's order (text tensor `name`, multi_modal_projector., vision_tower.) read both ways.
fn textOnly(device: mtl.Device, comptime name: []const u8) !void {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header = "{\"" ++ name ++ "\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}," ++
        "\"multi_modal_projector.p\":{\"dtype\":\"U8\",\"shape\":[1024],\"data_offsets\":[4,1028]}," ++
        "\"vision_tower.x\":{\"dtype\":\"U8\",\"shape\":[4096],\"data_offsets\":[1028,5124]}}";
    var data: [8 + header.len + 5124]u8 = undefined;
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    for (data[8 + header.len ..], 0..) |*b, i| b.* = @truncate(i);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = &data });
    const path = try std.fmt.allocPrintSentinel(gpa, ".zig-cache/tmp/{s}/model.safetensors", .{tmp.sub_path}, 0);
    defer gpa.free(path);
    var all = ckpt.Checkpoint.init(gpa);
    defer all.deinit();
    try addShard(&all, device, path, .text);
    try std.testing.expectEqual(@as(usize, 5124), all.residentBytes());
    var text = ckpt.Checkpoint.init(gpa);
    defer text.deinit();
    try addShard(&text, device, path, .vision_language);
    try std.testing.expectEqual(@as(usize, 4), text.residentBytes());
    try std.testing.expect(!text.has("vision_tower.x") and !text.has("multi_modal_projector.p"));
    try std.testing.expectEqualSlices(u16, (try all.get(name)).host(u16), (try text.get(name)).host(u16));
}
