//! Lane admission through the real lane opener with virtual device allocations, without loading CUDA.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const fixtures = @import("cuda_lanes_restore_test.zig");

/// A drafting backend over a fake driver; `init` in place, since the backend points into it.
fn Rig(comptime Cuda: type, comptime drop: anytype) type {
    return struct {
        driver: cuda.Driver,
        stream: cuda.Stream,
        engine: prompt.Engine,
        ids: [1]u32,
        head: mtp.Head,
        self: Cuda,

        fn init(r: *@This(), a: std.mem.Allocator) void {
            r.driver = fixtures.fakeDriver();
            r.stream = .{ .d = &r.driver, .handle = null };
            r.engine = undefined;
            r.engine.gpa = a;
            r.engine.driver = &r.driver;
            r.engine.stream = &r.stream;
            r.engine.eos = 7;
            r.engine.chunk = 16;
            r.engine.round_graph = .{};
            r.engine.heads = .full;
            r.ids = .{1};
            r.head = undefined;
            r.head.ids = &r.ids;
            r.self = undefined;
            r.self.gpa = a;
            r.self.context = 1024;
            r.self.engine = &r.engine;
            r.self.stream = r.stream;
            r.self.ranks = null;
            r.self.head = &r.head;
            r.self.lanes = .empty;
            r.self.parts = .{};
            r.self.kv = .{};
        }

        fn deinit(r: *@This()) void {
            var it = r.self.lanes.valueIterator();
            while (it.next()) |lane| drop(&r.self, lane);
            r.self.lanes.deinit(r.self.gpa);
        }
    };
}

fn run(comptime Cuda: type, comptime open: anytype, comptime drop: anytype, count: usize) !void {
    const a = std.testing.allocator;
    var streams: [8]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    var rig: Rig(Cuda, drop) = undefined;
    rig.init(a);
    defer rig.deinit();
    for (streams[0..count]) |*s| {
        s.* = try lanes.Stream.init(a, .{
            .id = "drafted",
            .prompt = &.{ 1, 2, 3 },
            .max_new = 2,
            .drafts = true,
        });
        made += 1;
        const lane = try open(&rig.self, s);
        try std.testing.expect(s.drafts);
        try std.testing.expect(lane.draft != null);
        try std.testing.expect(lane.seq.cap >= s.prompt().len + s.max_new + mtp.max_rows);
    }
    try std.testing.expectEqual(count, rig.self.lanes.count());
    for (streams[0..count]) |*s| {
        try std.testing.expect(s.drafts);
        try std.testing.expect(rig.self.lanes.getPtr(s).?.draft != null);
    }
}

/// A warm pass's yield hook, real work waiting from the start, and the keeps its pass offered the cache.
const Waiting = struct {
    asked: usize = 0,
    kept: usize = 0,
    fn yields(ptr: *anyopaque) bool {
        const w: *Waiting = @ptrCast(@alignCast(ptr));
        w.asked += 1;
        return true;
    }
    fn keep(ptr: *anyopaque, _: *lanes.Stream, _: u32) void {
        const w: *Waiting = @ptrCast(@alignCast(ptr));
        w.kept += 1;
    }
};

pub fn Cases(comptime Cuda: type, comptime open: anytype, comptime drop: anytype) type {
    return struct {
        test "four concurrent streams all retain MTP draft state" {
            try run(Cuda, open, drop, 4);
        }

        test "eight concurrent streams all retain MTP draft state" {
            try run(Cuda, open, drop, 8);
        }

        test "a reply's warm pass that real work waits on steps aside before its first chunk, keeping nothing" {
            const a = std.testing.allocator;
            var waiting: Waiting = .{};
            var s = try lanes.Stream.init(a, .{ .id = "warm", .prompt = &.{ 1, 2, 3 }, .max_new = 0, .drafts = true, .yield_check = .{ .ptr = &waiting, .check = Waiting.yields }, .reuse = .{ .hook = .{ .ptr = &waiting, .at = Waiting.keep } } });
            defer s.deinit(a);
            var rig: Rig(Cuda, drop) = undefined;
            rig.init(a);
            defer rig.deinit();
            const be = rig.self.backend();
            try std.testing.expectError(error.Yielded, be.vtable.prefill(be.ptr, &s));
            try std.testing.expect(waiting.asked == 1 and waiting.kept == 0); // it stands where it opened: nothing new
            try std.testing.expectEqual(@as(?usize, 0), rig.self.lanes.getPtr(&s).?.filled); // and no chunk ran
        }
    };
}
