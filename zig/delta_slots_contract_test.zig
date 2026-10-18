//! CPU controls for the production slot planner and independent FP64 recurrence.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const ref = contract.delta_slots_ref;
fn bf(value: f64) u16 {
    return c.toBf16(@floatCast(value));
}
test "DeltaNet FP64 corruption and slot-parent controls" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const slots = @import("src/core/tree_slots.zig");
    const f = try ref.make(slots, arena.allocator(), 4, 7, false, 173);
    const bytes = try arena.allocator().alloc(u8, 2 * f.output.len);
    for (f.output, 0..) |want, i| std.mem.writeInt(u16, bytes[2 * i ..][0..2], bf(want), .little);
    try std.testing.expect(f.accepts(bytes));
    @memset(bytes[0..2], 0x40);
    try std.testing.expect(!f.accepts(bytes));
    var bad = try arena.allocator().dupe(ref.Step, f.steps[0..7]);
    bad[1].read_slot = 3;
    try std.testing.expectError(error.BadSlotPlan, slots.validate(f.parents[0..7], @as([]slots.Step, @ptrCast(bad)), 0, 4));
}

test "zero keys leave a decayed root and zero queries give exactly zero, including advance" {
    const planner = @import("src/core/tree_slots.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f = try ref.make(planner, arena.allocator(), 1, 1, true, 19);
    @memset(f.key, 0);
    @memset(f.query, 0);
    try ref.reference(arena.allocator(), &f, 1);
    for (f.output) |want| try std.testing.expectEqual(@as(f64, 0), want);
    const stride = 48 * 128 * 128;
    for (f.advanced, f.state, 0..) |want, initial, i| {
        const slot = i / stride;
        const head = (i % stride) / (128 * 128);
        const decay: f64 = if (slot == 1) 1 else f.decay[(if (slot == 0) @as(usize, 1) else 0) * 48 + head];
        try std.testing.expectEqual(@as(f64, initial) * decay, want);
    }
}
