//! The actual session owner runs on retained recording command buffers; no Metal call or device is opened.
const std = @import("std");
const a = std.testing.allocator;
const T = std.testing;
const Fake = struct {
    const Event = enum { compute, end_compute, blit, copy, fill, end_blit, commit };
    var events: [64]Event = undefined;
    var used: usize = 0;
    var made: usize = 0;
    var pools: usize = 0;
    var waits: usize = 0;
    var refs: [16]usize = @splat(0);
    var autoreleased: [16]bool = @splat(false);
    var done: [16]bool = @splat(false);
    var bad: [16]bool = @splat(false);
    var compute_open = false;
    var blit_open = false;
    var can_complete = true;
    fn log(e: Event) void {
        events[used] = e;
        used += 1;
    }
    fn reset() void {
        used = 0;
        made = 0;
        pools = 0;
        waits = 0;
        refs = @splat(0);
        autoreleased = @splat(false);
        done = @splat(false);
        bad = @splat(false);
        compute_open = false;
        blit_open = false;
        can_complete = true;
    }
    pub const Buffer = struct {};
    pub const objc = struct {
        pub const Pool = struct {
            first: usize,
            pub fn push() Pool {
                pools += 1;
                return .{ .first = made };
            }
            pub fn pop(p: Pool) void {
                for (p.first..made) |i| if (autoreleased[i]) {
                    refs[i] -= 1;
                    autoreleased[i] = false;
                };
                pools -= 1;
            }
        };
    };
    pub const Queue = struct {
        pub fn commandBuffer(_: Queue) CommandBuffer {
            const i = made;
            made += 1;
            refs[i] = 1;
            autoreleased[i] = true;
            return .{ .index = i };
        }
    };
    pub const CommandBuffer = struct {
        index: usize,
        pub fn retain(c: CommandBuffer) CommandBuffer {
            refs[c.index] += 1;
            return c;
        }
        pub fn release(c: CommandBuffer) void {
            std.debug.assert(refs[c.index] > 0);
            refs[c.index] -= 1;
        }
        pub fn compute(_: CommandBuffer, _: enum { serial, concurrent }) ComputeEncoder {
            std.debug.assert(!compute_open and !blit_open);
            compute_open = true;
            log(.compute);
            return .{};
        }
        pub fn blit(_: CommandBuffer) Blit {
            std.debug.assert(!compute_open and !blit_open);
            blit_open = true;
            log(.blit);
            return .{};
        }
        pub fn commit(_: CommandBuffer) void {
            std.debug.assert(!compute_open and !blit_open);
            log(.commit);
        }
        pub fn wait(c: CommandBuffer) void {
            waits += 1;
            if (can_complete) done[c.index] = true;
        }
        pub fn status(c: CommandBuffer) enum { committed, completed, @"error" } {
            return if (!done[c.index]) .committed else if (bad[c.index]) .@"error" else .completed;
        }
        pub fn failure(c: CommandBuffer) ?[*:0]const u8 {
            return if (bad[c.index]) "planted" else null;
        }
    };
    pub const ComputeEncoder = struct {
        pub fn end(_: ComputeEncoder) void {
            std.debug.assert(compute_open);
            compute_open = false;
            log(.end_compute);
        }
    };
    pub const Blit = struct {
        pub fn copy(_: Blit, _: Buffer, _: usize, _: Buffer, _: usize, _: usize) void {
            std.debug.assert(blit_open);
            log(.copy);
        }
        pub fn fill(_: Blit, _: Buffer, _: usize, _: usize, _: u8) void {
            std.debug.assert(blit_open);
            log(.fill);
        }
        pub fn end(_: Blit) void {
            std.debug.assert(blit_open);
            blit_open = false;
            log(.end_blit);
        }
    };
};
const Session = @import("metal").CommandSessionFor(Fake);
test "Metal session: compute and blit stay in one ordered buffer without host waits" {
    Fake.reset();
    var s = Session.init(a, .{});
    defer s.deinit() catch unreachable;
    try s.begin();
    _ = try s.encoder();
    _ = try s.encoder();
    try s.copy(.{}, 0, .{}, 0, 64);
    _ = try s.encoder();
    try s.zero(.{}, 0, 64);
    _ = try s.encoder();
    try T.expectError(error.MetalCommandUnsubmitted, s.drain());
    try s.submit();
    try T.expectEqual(@as(usize, 1), Fake.made);
    try T.expectEqual(@as(usize, 0), Fake.waits);
    try T.expectEqualSlices(Fake.Event, &.{ .compute, .end_compute, .blit, .copy, .end_blit, .compute, .end_compute, .blit, .fill, .end_blit, .compute, .end_compute, .commit }, Fake.events[0..Fake.used]);
    try T.expectEqual(@as(usize, 1), Fake.refs[0]);
    try T.expectEqual(@as(usize, 0), Fake.pools);
    try s.poll();
    try T.expectEqual(@as(usize, 1), s.pending.items.len);
    try T.expectEqual(@as(usize, 0), Fake.waits);
    Fake.done[0] = true;
    try s.poll();
    try T.expectEqual(@as(usize, 0), s.pending.items.len);
    try T.expectEqual(@as(usize, 0), Fake.refs[0]);
    try s.begin();
    _ = try s.encoder();
    s.discard();
    try T.expectEqual(@as(usize, 0), Fake.refs[1]);
    try T.expectEqual(@as(usize, 0), Fake.pools);
}
test "Metal session: a later successful empty buffer cannot hide an earlier device fault" {
    Fake.reset();
    var s = Session.init(a, .{});
    try s.copy(.{}, 0, .{}, 0, 64);
    try s.begin();
    try s.submit();
    Fake.done[0] = true;
    Fake.bad[0] = true;
    Fake.done[1] = true;
    try T.expectError(error.MetalDeviceFailed, s.poll());
    try T.expect(s.failed);
    try T.expectEqual(@as(usize, 0), s.pending.items.len);
    try T.expectEqual(@as(usize, 0), Fake.refs[0] + Fake.refs[1]);
    try T.expectError(error.MetalDeviceFailed, s.begin());
    try s.deinit();
}
test "Metal session: incomplete drains retain buffers; a retry releases only after completion" {
    Fake.reset();
    var s = Session.init(a, .{});
    try s.zero(.{}, 0, 64);
    Fake.can_complete = false;
    try T.expectError(error.MetalCommandIncomplete, s.deinit());
    try T.expectEqual(@as(usize, 1), Fake.refs[0]);
    try T.expectEqual(@as(usize, 1), s.pending.items.len);
    Fake.can_complete = true;
    try s.deinit();
    try T.expectEqual(@as(usize, 0), Fake.refs[0]);
}
test "Metal session: allocation failure before commit retains the active buffer until explicit discard" {
    Fake.reset();
    var failure = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    var s = Session.init(failure.allocator(), .{});
    try s.begin();
    _ = try s.encoder();
    try T.expectError(error.OutOfMemory, s.submit());
    try T.expect(s.active != null);
    try T.expectEqual(@as(usize, 2), Fake.refs[0]);
    try T.expectEqual(@as(usize, 1), Fake.pools);
    s.discard();
    try T.expectEqual(@as(usize, 0), Fake.refs[0]);
    try T.expectEqual(@as(usize, 0), Fake.pools);
    try s.deinit();
}
