//! A native Qwen3.5-2B model: validated config, shared affine checkpoint views and Metal pipelines.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const ckpt = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const weights = @import("weights.zig");
const kernels = @import("kernels.zig");

/// The language model's tensors pack into their own buffer, not the file span between them that holds the vision tower.
fn never(_: []const u8, _: usize) bool {
    return false;
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ckpt.Checkpoint,
    weights: weights.Weights,
    kernels: kernels.Kernels,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Model {
        const config = try cfg.Config.read(gpa, io, dir);
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
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
        for (files) |path| try m.checkpoint.addFileSplit(m.device, path, "", "language_model.", never);
        m.weights = try weights.load(&m.checkpoint, config.tied);
        m.kernels = try kernels.load(m.device);
        return m;
    }

    pub fn deinit(m: *Model) void {
        m.kernels.deinit();
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};

test "the language model's tensors are held without the vision tower stored between them" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // the order a Qwen3.5 checkpoint stores: a language-model tensor, the vision tower, then the next one
    const header =
        \\{"language_model.a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},
        \\ "vision_tower.x":{"dtype":"U8","shape":[4096],"data_offsets":[4,4100]},
        \\ "language_model.b":{"dtype":"BF16","shape":[3],"data_offsets":[4100,4106]}}
    ;
    var data: [8 + header.len + 4106]u8 = undefined;
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    for (data[8 + header.len ..], 0..) |*b, i| b.* = @truncate(i);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = &data });
    const path = try std.fmt.allocPrintSentinel(gpa, ".zig-cache/tmp/{s}/model.safetensors", .{tmp.sub_path}, 0);
    defer gpa.free(path);
    var span = ckpt.Checkpoint.init(gpa);
    defer span.deinit();
    try span.addFileSelected(device, path, "", "language_model.");
    try std.testing.expectEqual(@as(usize, 4106), span.residentBytes());
    var packed_ = ckpt.Checkpoint.init(gpa);
    defer packed_.deinit();
    try packed_.addFileSplit(device, path, "", "language_model.", never);
    try std.testing.expectEqual(@as(usize, ckpt.alignment + 6), packed_.residentBytes());
    try std.testing.expect(!packed_.has("vision_tower.x"));
    inline for (.{ "language_model.a", "language_model.b" }) |name|
        try std.testing.expectEqualSlices(u16, (try span.get(name)).host(u16), (try packed_.get(name)).host(u16));
}
