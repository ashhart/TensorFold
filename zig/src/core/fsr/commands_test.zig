const std = @import("std");
const commands = @import("commands.zig");
const Fake = struct {
    pub const DispatchType = enum { serial, concurrent };
    const Record = struct { opened: bool = false, ended: bool = false, committed: bool = false, waited: bool = false, failed: bool = false, refs: usize = 0, fence_update: bool = false, fence_wait: bool = false };
    var records: [12]Record = @splat(.{});
    var used: usize = 0;
    pub const ComputeEncoder = struct {
        id: usize,
        pub fn end(e: @This()) void {
            const r = &records[e.id];
            std.debug.assert(r.opened and !r.ended);
            r.ended = true;
        }
    };
    pub const CommandBuffer = struct {
        id: usize,
        pub fn compute(c: @This(), _: DispatchType) ComputeEncoder {
            std.debug.assert(!records[c.id].opened);
            records[c.id].opened = true;
            return .{ .id = c.id };
        }
        pub fn commit(c: @This()) void {
            std.debug.assert(records[c.id].ended and !records[c.id].committed);
            records[c.id].committed = true;
        }
        pub fn wait(c: @This()) void {
            std.debug.assert(records[c.id].committed);
            records[c.id].waited = true;
        }
        pub fn failure(c: @This()) ?u8 {
            return if (records[c.id].failed) 1 else null;
        }
        pub fn gpuSeconds(_: @This()) f64 {
            return 1;
        }
        pub fn retain(c: @This()) @This() {
            records[c.id].refs += 1;
            return c;
        }
        pub fn release(c: @This()) void {
            records[c.id].refs -= 1;
        }
    };
    pub const Fence = struct {
        pub fn update(_: @This(), e: ComputeEncoder) void {
            std.debug.assert(records[e.id].opened and !records[e.id].ended);
            records[e.id].fence_update = true;
        }
        pub fn wait(_: @This(), e: ComputeEncoder) void {
            std.debug.assert(records[e.id].opened and !records[e.id].ended);
            records[e.id].fence_wait = true;
        }
    };
    pub const Queue = struct {
        pub fn commandBuffer(_: @This()) CommandBuffer {
            const id = used;
            used += 1;
            return .{ .id = id };
        }
    };
};

test "FSR cuts end the actual buffer, preserve order and surface an earlier asynchronous failure" {
    Fake.records = @splat(.{});
    Fake.used = 0;
    const queue: Fake.Queue = .{};
    var c: commands.Commands(Fake) = .{};
    var enc = c.begin(queue.commandBuffer(), .serial);
    try c.pause(queue, &enc);
    try std.testing.expect(Fake.records[0].waited);
    try c.overlap(queue, &enc, Fake.Fence{});
    try std.testing.expect(Fake.records[1].fence_update);
    try std.testing.expect(Fake.records[2].fence_wait);
    try std.testing.expect(Fake.records[1].committed and !Fake.records[1].waited);
    Fake.records[1].failed = true;
    try std.testing.expectError(error.GpuFailed, c.pause(queue, &enc));
    try std.testing.expect(Fake.records[1].waited and Fake.records[2].waited);
    try std.testing.expectEqual(@as(usize, 0), Fake.records[1].refs);
    c.abort(enc);
    enc = c.begin(queue.commandBuffer(), .concurrent);
    try c.pause(queue, &enc);
    const last = try c.end(enc);
    try std.testing.expectEqual(@as(usize, 4), last.id);
    last.commit();
    last.wait();
    try std.testing.expectEqual(@as(f64, 4), try c.completed());
    enc = c.begin(queue.commandBuffer(), .serial);
    c.abort(enc);
    try std.testing.expect(Fake.records[5].ended and !Fake.records[5].committed);
}
