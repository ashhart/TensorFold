//! Read the real bf16 checkpoint through the native mapped source, including dtype and shape rejection.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("ling_checkpoint");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ModelDirectoryRequired;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    var model = try ck.Checkpoint.open(init.gpa, init.io, device, args[1]);
    defer model.deinit();
    const embed = try model.get("model.word_embeddings.weight", .bf16, &.{ 157184, 1536 });
    _ = try model.get("model.layers.0.attention.q_proj.weight", .bf16, &.{ 2048, 1536 });
    _ = try model.get("model.norm.weight", .bf16, &.{1536});
    try std.testing.expectError(error.TensorShape, model.get("model.norm.weight", .f32, &.{1536}));
    try std.testing.expectError(error.TensorShape, model.get("model.norm.weight", .bf16, &.{1535}));
    try std.testing.expectError(error.MissingTensor, model.get("not.a.tensor", .bf16, &.{1}));
    std.debug.print("Ling checkpoint: files={d} tensors={d} embedding_bytes={d}; 3 reads, 3 rejection checks passed\n", .{ model.files.items.len, model.where.count(), embed.entry.end - embed.entry.begin });
}
