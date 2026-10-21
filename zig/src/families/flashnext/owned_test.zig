//! Resource handles stand in for owned Objective-C objects; no device or shader is opened.
const std = @import("std");
const prefix = @import("load_prefix.zig");
const testing = std.testing;

const Token = struct { released: u32 = 0, record: *Record };
const Record = struct { waits: u32 = 0, queue_releases: u32 = 0, early_release: bool = false };
const Fake = struct {
    const Self = @This();
    pub const objc = struct {
        pub const Id = *Token;
        pub fn release(t: Id) void {
            if (t.record.waits == 0) t.record.early_release = true;
            t.released += 1;
        }
    };
    pub const Queue = struct {
        record: *Record,
        pub fn commandBuffer(q: Queue) Self.CommandBuffer {
            return .{ .record = q.record };
        }
        pub fn deinit(q: Queue) void {
            q.record.queue_releases += 1;
        }
    };
    pub const CommandBuffer = struct {
        record: *Record,
        pub fn commit(_: @This()) void {}
        pub fn wait(c: @This()) void {
            c.record.waits += 1;
        }
    };
    pub const Object = struct {
        id: *Token,
        pub fn deinit(o: @This()) void {
            // An unpublished object has no submitted use and may be released immediately.
            o.id.released += 1;
        }
    };
};
const Owner = @import("owned.zig").Owner(Fake);

const Build = struct {
    owner: Owner = .{},
    record: *Record,
    tokens: *[3]Token,
    pub fn deinit(b: *Build) void {
        b.owner.deinit();
    }
    fn construct(_: void, b: *Build) !void {
        _ = try b.owner.keep(testing.allocator, Fake.Queue{ .record = b.record });
        for (b.tokens) |*t| _ = try b.owner.keep(testing.allocator, Fake.Object{ .id = t });
        return error.LaterConstructorFailure;
    }
};

test "Flash Next cleanup: owned prefix drains queues before releasing every successful object once" {
    var record: Record = .{};
    var tokens = [_]Token{ .{ .record = &record }, .{ .record = &record }, .{ .record = &record } };
    var build: Build = .{ .record = &record, .tokens = &tokens };
    defer build.deinit();
    try testing.expectError(error.LaterConstructorFailure, prefix.build(void, &build, {}, Build.construct));
    for (tokens) |t| try testing.expectEqual(@as(u32, 1), t.released);
    try testing.expect(!record.early_release);
    try testing.expectEqual(@as(u32, 1), record.waits);
    try testing.expectEqual(@as(u32, 1), record.queue_releases);
    build.deinit();
    try testing.expectEqual(@as(u32, 1), record.queue_releases);
}

test "Flash Next cleanup: failed object publication returns the unowned acquisition once" {
    var record: Record = .{};
    var token: Token = .{ .record = &record };
    var owner: Owner = .{};
    defer owner.deinit();
    var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    try testing.expectError(error.OutOfMemory, owner.keep(failing.allocator(), Fake.Object{ .id = &token }));
    try testing.expectEqual(@as(u32, 1), token.released);
    try testing.expectEqual(@as(usize, 0), owner.entries.items.len);
}

const Pair = struct {
    wbuf: Fake.Object,
    one: Fake.Object,
    pick_tmp: Fake.Object,
    record: *Record,
    stopped: bool = false,
    closed: bool = false,
    pub fn stopService(p: *@This()) void {
        p.stopped = true;
    }
    pub fn deinit(p: *@This()) void {
        std.debug.assert(p.stopped and p.record.waits > 0);
        Fake.objc.release(p.pick_tmp.id);
        Fake.objc.release(p.one.id);
        Fake.objc.release(p.wbuf.id);
        p.closed = true;
    }
};

test "Flash Next cleanup: the pair owns borrowed aliases until all queue work is drained, then closes its endpoint" {
    const Metal = @import("metal");
    const Run = struct {
        closed: bool = false,
        tp: ?*Pair,
        objects: Owner = .{},
        queue: Fake.Queue,
        held: std.ArrayList(Fake.Object) = .empty,
        follow: ?struct { ctx: *anyopaque, leave: *const fn (*anyopaque, Fake.Object) void } = null,
        rows_cleanup: ?struct { ctx: *anyopaque, call: *const fn (*anyopaque) void } = null,
        wide_pipes: std.AutoHashMapUnmanaged(u8, Fake.Object) = .empty,
        maps: std.ArrayList(Metal.MappedFile) = .empty,
        files: std.ArrayList(c_int) = .empty,
    };
    var record: Record = .{};
    var tokens = [_]Token{ .{ .record = &record }, .{ .record = &record }, .{ .record = &record }, .{ .record = &record } };
    var pair: Pair = .{ .wbuf = .{ .id = &tokens[0] }, .one = .{ .id = &tokens[1] }, .pick_tmp = .{ .id = &tokens[2] }, .record = &record };
    var run: Run = .{ .tp = &pair, .queue = .{ .record = &record } };
    defer run.held.deinit(testing.allocator);
    try run.held.appendSlice(testing.allocator, &.{ pair.wbuf, pair.one, pair.pick_tmp, .{ .id = &tokens[3] } });
    @import("run_cleanup.zig").drop(&run);
    for (tokens) |t| try testing.expectEqual(@as(u32, 1), t.released);
    try testing.expect(pair.stopped and pair.closed and !record.early_release);
    try testing.expect(run.tp == null);
    try testing.expectEqual(@as(u32, 1), record.queue_releases);
}
