//! Actual Q8 replacement and planted-control registration, with host buffers and no shader execution.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const suite = @import("tests/contract/metal_minimax_nax.zig");
const t = std.testing;

test "MiniMax Q8 contract: indexed oracle and wrong-expert control use independent outputs and unchanged ids" {
    var record: mtl.Record = .{ .gpa = t.allocator };
    defer record.deinit();
    var kernels: std.ArrayList(contract.gates.Kernel) = .empty;
    defer kernels.deinit(t.allocator);
    var s: suite.Suite = .{ .a = t.allocator, .device = record.device(), .queue = record.queue(), .kernels = &kernels };
    defer s.deinit();
    try s.collect("metal minimax_m3 mm_experts_grouped_q8");
    try t.expectEqual(@as(usize, 2), kernels.items.len);
    for (kernels.items, 0..) |kernel, i| {
        try t.expectEqualStrings("tf_row_projection_indexed_q8_f32", kernel.spec.replaces.?);
        try t.expectEqual(if (i == 0) null else @as(?contract.spec.Gate, .shapes), kernel.control);
    }
    for (s.grouped.items) |p| {
        try t.expect(try p.prepare(.{ .rows = 2, .n = 8, .k = 64 }, .random, 73));
        try t.expect(p.out.buf.id != p.baseline_out.buf.id);
        try t.expect(p.baseline_pipe != null);
        const ids = p.rig.bytes(p.baseline_ids, p.ids.len * 4);
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(p.ids), ids);
        const group_bytes = p.rig.bytes(p.refs[5], 8);
        const first_expert = std.mem.readInt(u32, group_bytes[4..8], .little);
        var actual_first: u32 = std.math.maxInt(u32);
        for (p.ids) |id| actual_first = @min(actual_first, @as(u32, @intCast(id)));
        try t.expectEqual(if (p.control) (actual_first + 1) % 6 else actual_first, first_expert);
        try t.expect(try p.rig.intact());
    }
}
