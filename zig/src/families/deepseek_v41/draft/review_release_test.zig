const std = @import("std");
const lanes = @import("lanes");
const twin = @import("twin.zig");
const iface = @import("iface.zig");
const t = std.testing;
const Fault = struct {
    var failing = true;
    fn checked(p: *anyopaque, slot: u32) !void {
        if (failing) return error.RollbackFailed;
        const target: *twin.Twin = @ptrCast(@alignCast(p));
        target.target().vtable.release(p, slot);
    }
    fn release(p: *anyopaque, slot: u32) void {
        checked(p, slot) catch {};
    }
    fn collect(_: *anyopaque, _: []iface.Proposal) !void {
        return error.CollectFailed;
    }
};
test "Completion ownership: failed target release quarantines the slot until a later successful release" {
    const target = try twin.Twin.init(t.allocator, .{}, 1);
    defer target.deinit();
    const pass = try twin.Drafter.init(t.allocator, target, 1);
    defer pass.deinit();
    var vt = target.target().vtable.*;
    vt.release = Fault.release;
    vt.release_checked = Fault.checked;
    var prices = try @import("costs.zig").defaults(t.allocator, 16);
    defer prices.deinit(t.allocator);
    const x = try @import("lanes.zig").Lanes.init(t.allocator, .{ .ptr = target, .vtable = &vt }, pass.pass(), prices, .{ .shape = pass.shape, .slots = 1 });
    defer x.deinit();
    var stream = try lanes.Stream.init(t.allocator, .{ .id = "a", .prompt = &.{ 1, 2, 3 }, .max_new = 4, .drafts = false });
    defer stream.deinit(t.allocator);
    Fault.failing = true;
    try x.backend().prefill(&stream);
    x.backend().release(&stream);
    try t.expectEqual(@as(usize, 0), x.free.items.len);
    try t.expectError(error.NoSlot, x.backend().prefill(&stream));
    Fault.failing = false;
    try x.backend().prefill(&stream);
    x.backend().release(&stream);
    try t.expectEqual(@as(usize, 1), x.free.items.len);
}
test "Completion ownership: unsuccessful speculative collection retains its pending ownership for retry" {
    var spec = try @import("spec.zig").Spec.init(t.allocator, .{ .on = true }, 1, 1, 1);
    defer spec.deinit();
    spec.begun = true;
    spec.n = 0;
    var vt: iface.Pass.VTable = undefined;
    vt.collect = Fault.collect;
    try t.expectError(error.CollectFailed, spec.settle(.{ .ptr = &spec, .vtable = &vt }));
    try t.expect(spec.begun);
}

test "Completion ownership: failed stream-map publication quarantines acquired target state on cleanup failure" {
    const target = try twin.Twin.init(t.allocator, .{}, 1);
    defer target.deinit();
    const pass = try twin.Drafter.init(t.allocator, target, 1);
    defer pass.deinit();
    var vt = target.target().vtable.*;
    vt.release = Fault.release;
    vt.release_checked = Fault.checked;
    var prices = try @import("costs.zig").defaults(t.allocator, 16);
    defer prices.deinit(t.allocator);
    const x = try @import("lanes.zig").Lanes.init(t.allocator, .{ .ptr = target, .vtable = &vt }, pass.pass(), prices, .{ .shape = pass.shape, .slots = 1 });
    defer x.deinit();
    var stream = try lanes.Stream.init(t.allocator, .{ .id = "a", .prompt = &.{ 1, 2, 3 }, .max_new = 4, .drafts = false });
    defer stream.deinit(t.allocator);
    var fail = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 1 });
    x.gpa = fail.allocator();
    Fault.failing = true;
    try t.expectError(error.OutOfMemory, x.backend().prefill(&stream));
    try t.expect(target.slots[0].live);
    try t.expectEqual(@as(usize, 0), x.free.items.len);
    x.gpa = t.allocator;
    Fault.failing = false;
    try x.backend().prefill(&stream);
    x.backend().release(&stream);
    try t.expectEqual(@as(usize, 1), x.free.items.len);
}
