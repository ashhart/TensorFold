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
    self.parts = .{};
    self.kv = .{};
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

pub fn Cases(comptime Cuda: type, comptime open: anytype, comptime drop: anytype, comptime load: anytype) type {
    return struct {
        test "four concurrent streams all retain MTP draft state" {
            try run(Cuda, open, drop, 4);
        }

        test "eight concurrent streams all retain MTP draft state" {
            try run(Cuda, open, drop, 8);
        }

        test "a bad TF_FLASHNEXT_CHUNK, _CHUNK_SOLO or TF_FLASHNEXT_LOAD_THREADS is refused at load before the loader takes a stream, a module or a buffer" {
            const env = @import("sys").env;
            // every driver entry point undefined: the refusal must come before the first call
            var driver: cuda.Driver = undefined;
            const ctx: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
            for ([_]struct { [*:0]const u8, [*:0]const u8, anyerror }{ .{ "TF_FLASHNEXT_CHUNK", "0", error.BadChunk }, .{ "TF_FLASHNEXT_CHUNK", "16385", error.BadChunk }, .{ "TF_FLASHNEXT_CHUNK", "4k", error.BadChunk }, .{ "TF_FLASHNEXT_CHUNK_SOLO", "1024", error.BadChunk }, .{ "TF_FLASHNEXT_LOAD_THREADS", "0", error.BadLoadThreads }, .{ "TF_FLASHNEXT_LOAD_THREADS", "33", error.BadLoadThreads }, .{ "TF_FLASHNEXT_LOAD_THREADS", "8x", error.BadLoadThreads } }) |bad| {
                _ = env.setenv(bad[0], bad[1], 1);
                defer _ = env.unsetenv(bad[0]);
                try std.testing.expectError(bad[2], load(std.testing.allocator, std.testing.io, &ctx, "no-such-checkpoint", null, .{ .context = 4096, .drafts = true }));
            }
        }
    };
}
