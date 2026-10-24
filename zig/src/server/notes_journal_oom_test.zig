test "review: first journal allocation failure releases file and writer lock" {
    const std = @import("std");
    const testing = std.testing;
    const journal = @import("notes/journal.zig");
    const write = @import("notes/write.zig");
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "oom.journal" });
    const sha: [64]u8 = @splat('a');
    var recs: std.ArrayList(write.Record) = .empty;
    var problem: []const u8 = "";
    var j = try journal.Journal.open(testing.allocator, a, path, sha, &recs, &problem);
    var alive = true;
    defer if (alive) j.deinit();
    var empty: [0]u8 = .{};
    var fba = std.heap.FixedBufferAllocator.init(&empty);
    try testing.expectError(error.OutOfMemory, j.append(fba.allocator(), .{
        .op = .add,
        .source = "w1",
        .told = 739900,
        .subject = "me",
        .attribute = "note:w1",
        .text = "My colour is green.",
    }));
    j.deinit();
    alive = false;
    var reopened = try journal.Journal.open(testing.allocator, a, path, sha, &recs, &problem);
    defer reopened.deinit();
}
