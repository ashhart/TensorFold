//! Test the actual read-slot handoff; no model, file reader or CUDA library runs.
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("cuda_kernels.zig");
const Source = @import("cuda_source.zig").Source;
const Fake = @import("upload_fake").Fake;
const t = std.testing;
fn kernels(d: *cuda.Driver, discrete: bool) kern.Kernels {
    var k: kern.Kernels = undefined;
    k.d = d;
    k.discrete = discrete;
    return k;
}
test "weight upload: integrated Nemotron DMA reads the producer slot and settles before reuse" {
    var d = Fake.driver();
    var k = kernels(&d, false);
    cuda.copy_source.setCurrent(@ptrFromInt(63));
    var s: Source = .{ .gpa = t.allocator, .ops = .{ .k = &k, .s = .{ .d = &d, .handle = null } } };
    s.slots[0] = @alignCast(try s.stages[0].take(&d, 4096));
    defer Source.Testing.freeSlots(&s, 1);
    @memset(s.slots[0], 0x6b);
    try Source.Testing.finish(&s, 0, .{ .dst = 0x1000, .file = undefined, .offset = 3, .len = 17, .at = 3 }, s.slots[0][3..20]);
    try t.expectEqual(@intFromPtr(s.slots[0].ptr) + 3, Fake.last_source);
    try t.expectEqual(@as(usize, 1), Fake.allocs);
    try t.expect(Fake.pending != null);
    try Source.Testing.settle(&s, 0);
    try t.expect(Fake.pending == null);
    @memset(s.slots[0], 0xef);
    try t.expectEqualSlices(u8, &@as([17]u8, @splat(0x6b)), Fake.gpu[0..17]);
    try t.expectEqual(@as(usize, 0), Fake.early);
}
test "weight upload: Nemotron failed event record retains its slot through deinit" {
    var d = Fake.driver();
    var k = kernels(&d, false);
    cuda.copy_source.setCurrent(@ptrFromInt(64));
    var s: Source = .{ .gpa = t.allocator, .ops = .{ .k = &k, .s = .{ .d = &d, .handle = null } } };
    s.slots[0] = @alignCast(try s.stages[0].take(&d, 4096));
    @memset(s.slots[0], 0x43);
    const before = cuda.copy_source.retained();
    Fake.fail_record = true;
    Fake.fail = true;
    try t.expectError(error.NotReady, Source.Testing.finish(&s, 0, .{ .dst = 0x1000, .file = undefined, .offset = 0, .len = 9, .at = 0 }, s.slots[0][0..9]));
    Source.Testing.freeSlots(&s, 1);
    try t.expectEqual(@as(usize, 0), Fake.frees);
    try t.expectEqual(before + 1, cuda.copy_source.retained());
    Fake.fail = false;
    Fake.fail_record = false;
    try cuda.copy_source.synchronizeContext(&d);
    try t.expectEqual(@as(usize, 1), Fake.frees);
    try t.expectEqual(@as(usize, 0), Fake.early);
}
test "weight upload: discrete Nemotron retains the original private copy and event path" {
    var d = Fake.driver();
    var k = kernels(&d, true);
    cuda.copy_source.setCurrent(@ptrFromInt(65));
    var s: Source = .{ .gpa = t.allocator, .ops = .{ .k = &k, .s = .{ .d = &d, .handle = null } } };
    s.pinned[0] = try cuda.HostBuffer.alloc(&d, 4096);
    s.slots[0] = @alignCast(s.pinned[0].?.bytes);
    s.copied[0] = try cuda.Event.init(&d, false);
    defer Source.Testing.freeSlots(&s, 1);
    @memset(s.slots[0], 0x7a);
    try Source.Testing.finish(&s, 0, .{ .dst = 0x1000, .file = undefined, .offset = 0, .len = 9, .at = 0 }, s.slots[0][0..9]);
    try t.expect(Fake.last_source != @intFromPtr(s.slots[0].ptr));
    try t.expectEqual(@as(usize, 2), Fake.allocs);
    try Source.Testing.settle(&s, 0);
    try cuda.copy_source.synchronizeContext(&d);
    try t.expectEqualSlices(u8, &@as([9]u8, @splat(0x7a)), Fake.gpu[0..9]);
    try t.expectEqual(@as(usize, 0), Fake.early);
}
test "weight upload: Nemotron partial pinned-slot construction frees prior slots" {
    var d = Fake.driver();
    var k = kernels(&d, false);
    Fake.fail_alloc_at = 2;
    try t.expectError(error.NotReady, Source.init(t.allocator, .{ .k = &k, .s = .{ .d = &d, .handle = null } }));
    try t.expectEqual(@as(usize, 1), Fake.frees);
    for (Fake.pins) |p| try t.expect(p == null);
}
