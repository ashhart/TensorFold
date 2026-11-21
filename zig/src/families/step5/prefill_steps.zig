//! Step-5 Metal prompt chunks between lane rounds, preserving the whole pass's cuts and native width observations.
const std = @import("std");
const lanes = @import("lanes");

pub const Fill = struct { at: usize };

pub fn chunkEnd(len: usize, at: usize, rows: usize, chunks: []const u32, marks: []const u32) usize {
    var end = @min(len, at + rows);
    for ([_][]const u32{ chunks, marks }) |cuts| for (cuts) |x| if (x > at and x < end) {
        end = x;
    };
    return end;
}

pub fn step(r: anytype, stream: *lanes.Stream, fill: *Fill, rows: usize) !bool {
    if (stream.isCancelled()) return error.Cancelled;
    const len = stream.prompt().len;
    const end = chunkEnd(len, fill.at, rows, stream.chunks, stream.reuse.marks);
    try r.run(stream, fill.at, end);
    lanes.prefill_widths.completed(r.m.gpa, stream, end - fill.at);
    stream.fill_at.store(@intCast(end), .monotonic);
    fill.at = end;
    if (std.mem.indexOfScalar(u32, stream.reuse.marks, @intCast(end)) != null) r.keep(stream, @intCast(end));
    return end == len;
}

const Owner = struct { gpa: std.mem.Allocator };
const Event = union(enum) { chunk: [2]usize, keep: u32 };

const Rec = struct {
    m: *const Owner,
    log: *std.ArrayList(Event),

    pub fn run(r: Rec, _: *lanes.Stream, at: usize, end: usize) !void {
        try r.log.append(r.m.gpa, .{ .chunk = .{ at, end } });
    }

    pub fn keep(r: Rec, _: *lanes.Stream, at: u32) void {
        r.log.append(r.m.gpa, .{ .keep = at }) catch @panic("out of memory");
    }
};

fn checkPass(ids: []const u32, from: usize, want: []const Event, widths: []const u32) !void {
    const gpa = std.testing.allocator;
    const owner: Owner = .{ .gpa = gpa };
    var stream = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = ids, .max_new = 4, .chunks = &.{6}, .reuse = .{ .marks = &.{9} } });
    defer stream.deinit(gpa);
    stream.cached = @intCast(from);
    var log: std.ArrayList(Event) = .empty;
    defer log.deinit(gpa);
    var fill: Fill = .{ .at = from };
    while (fill.at < ids.len) _ = try step(Rec{ .m = &owner, .log = &log }, &stream, &fill, 4);
    try std.testing.expectEqual(@as(usize, ids.len), fill.at);
    try std.testing.expectEqual(@as(u32, @intCast(ids.len)), stream.fill_at.load(.monotonic));
    try std.testing.expectEqual(want.len, log.items.len);
    for (want, log.items) |w, got| try std.testing.expectEqual(w, got);
    try std.testing.expectEqualSlices(u32, widths, stream.prefill_widths.items);
}

test "step5 host: stepped prefill keeps fresh and resumed cuts and native widths" {
    var ids: [15]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i + 1);
    try checkPass(&ids, 0, &.{
        .{ .chunk = .{ 0, 4 } },
        .{ .chunk = .{ 4, 6 } },
        .{ .chunk = .{ 6, 9 } },
        .{ .keep = 9 },
        .{ .chunk = .{ 9, 13 } },
        .{ .chunk = .{ 13, 15 } },
    }, &.{ 4, 2, 3, 4, 2 });
    try checkPass(&ids, 6, &.{
        .{ .chunk = .{ 6, 9 } },
        .{ .keep = 9 },
        .{ .chunk = .{ 9, 13 } },
        .{ .chunk = .{ 13, 15 } },
    }, &.{ 3, 4, 2 });
}
