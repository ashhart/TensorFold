//! Draft admission exercises the real lane opener with virtual device allocations, without loading CUDA.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const fixtures = @import("cuda_lanes_restore_test.zig");

fn run(comptime Cuda: type, comptime open: anytype, comptime drop: anytype, count: usize) !void {
    const a = std.testing.allocator;
    var driver = fixtures.fakeDriver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var engine: prompt.Engine = undefined;
    engine.gpa = a;
    engine.driver = &driver;
    engine.stream = &stream;
    engine.eos = 7;
    engine.chunk = 16;
    engine.round_graph = .{};
    var ids = [_]u32{1};
    var head: mtp.Head = undefined;
    head.ids = &ids;
    var streams: [8]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    var self: Cuda = undefined;
    self.gpa = a;
    self.context = 1024;
    self.engine = &engine;
    self.stream = stream;
    self.ranks = null;
    self.head = &head;
    self.lanes = .empty;
    defer {
        var it = self.lanes.valueIterator();
        while (it.next()) |lane| drop(&self, lane);
        self.lanes.deinit(a);
    }
    for (streams[0..count]) |*s| {
        s.* = try lanes.Stream.init(a, .{
            .id = "drafted",
            .prompt = &.{ 1, 2, 3 },
            .max_new = 2,
            .drafts = true,
        });
        made += 1;
        const lane = try open(&self, s);
        try std.testing.expect(s.drafts);
        try std.testing.expect(lane.draft != null);
        try std.testing.expect(lane.seq.cap >= s.prompt().len + s.max_new + mtp.max_rows);
    }
    try std.testing.expectEqual(count, self.lanes.count());
    for (streams[0..count]) |*s| {
        try std.testing.expect(s.drafts);
        try std.testing.expect(self.lanes.getPtr(s).?.draft != null);
    }
}

pub fn Cases(comptime Cuda: type, comptime open: anytype, comptime drop: anytype) type {
    return struct {
        test "four concurrent streams all retain MTP draft state" {
            try run(Cuda, open, drop, 4);
        }

        test "eight concurrent streams all retain MTP draft state" {
            try run(Cuda, open, drop, 8);
        }
    };
}
