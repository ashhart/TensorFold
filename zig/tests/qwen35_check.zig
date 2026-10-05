//! Qwen3.5 checkpoint and native-operation checks at the model's real dimensions.
const std = @import("std");
const mtl = @import("metal");
const qwen = @import("tensorfold").qwen35;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2 or args.len > 3) return error.ExpectedModelDirectoryAndOptionalFixtures;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try qwen.Model.load(init.gpa, init.io, args[1]);
    defer m.deinit();
    const head = m.weights.head();
    try std.testing.expect(head.weight.buffer.id == m.weights.embedding.weight.buffer.id);
    try std.testing.expectEqual(head.weight.offset, m.weights.embedding.weight.offset);
    std.debug.print("Qwen3.5-2B: checkpoint loaded, {d} native pipelines, shared tied head\n", .{qwen.kernels.total});
    if (args.len == 3) {
        var check = @import("qwen35_fixtures.zig").Checker{ .gpa = init.gpa, .io = init.io, .model = m, .dir = args[2] };
        try check.check();
    }
}
