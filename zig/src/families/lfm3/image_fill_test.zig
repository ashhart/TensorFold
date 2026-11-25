//! The chunk path opens a pass's images once, splices them per chunk and ends them once.
const std = @import("std");
const lanes = @import("lanes");
const steps = @import("prefill_steps.zig");
const pass = @import("image_pass.zig");
const fill = @import("image_fill.zig");
const gpa = std.testing.allocator;

const Image = lanes.stream.Image;

const Tower = struct {
    calls: usize = 0,

    pub fn project(t: *Tower, im: anytype, row: u32) !void {
        _ = row;
        _ = im;
        t.calls += 1;
    }
};

/// Counts the provider's begin/end and projects through `image_pass.encode`.
const Fake = struct {
    tower: Tower = .{},
    begins: usize = 0,
    ends: usize = 0,
    from: usize = 0,
    spans: [8]pass.Span = undefined,

    const Images = struct { spans: []const pass.Span };

    fn begin(ptr: *anyopaque, stream: *const lanes.Stream, from: usize) !?Images {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        f.begins += 1;
        f.from = from;
        var list = std.ArrayList(pass.Span).initBuffer(&f.spans);
        try pass.encode(gpa, stream.images, from, &list, &f.tower);
        return .{ .spans = list.items };
    }

    fn end(ptr: *anyopaque, _: *const lanes.Stream) void {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        f.ends += 1;
    }

    const Provider = struct {
        ctx: *anyopaque,
        begin: *const fn (*anyopaque, *const lanes.Stream, usize) anyerror!?Images,
        end: *const fn (*anyopaque, *const lanes.Stream) void,
    };
    const Fill = fill.Fill(Provider, Images);

    fn provider(f: *Fake) Provider {
        return .{ .ctx = f, .begin = begin, .end = end };
    }
};

/// Records each chunk and asks the pass for its images, as the backend's Run.run does.
const Rec = struct {
    fill: *Fake.Fill,
    chunks: usize = 0,

    pub fn run(r: *Rec, s: *lanes.Stream, at: usize, end: usize) !void {
        _ = s;
        _ = try r.fill.chunk(at, end);
        r.chunks += 1;
    }

    pub fn keep(_: Rec, _: *lanes.Stream, _: u32) void {}
};

var images = [_]Image{
    .{ .pixels = &.{}, .gh = 2, .gw = 20, .hash = 1, .at = 100, .tokens = 70 },
    .{ .pixels = &.{}, .gh = 2, .gw = 20, .hash = 2, .at = 250, .tokens = 40 },
    .{ .pixels = &.{}, .gh = 2, .gw = 20, .hash = 1, .at = 300, .tokens = 70 }, // the same bytes and grid
};

fn make(ids: []const u32) !lanes.Stream {
    var s = try lanes.Stream.init(gpa, .{ .id = "images", .prompt = ids, .max_new = 8 });
    s.images = &images;
    return s;
}

/// Runs a pass from `from` at `rows` a chunk through the real step loop.
fn run(s: *lanes.Stream, from: usize, rows: usize, f: *Fake) !Rec {
    var state = Fake.Fill{ .provider = f.provider(), .stream = s };
    try state.begin(from);
    var r: Rec = .{ .fill = &state };
    var at = from;
    while (!try steps.step(&r, s, &at, rows, false)) {}
    state.close();
    return r;
}

test "LFM3 prefill images: 128-row chunks open the tower once and splice every chunk" {
    var ids: [400]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i + 3);
    var s = try make(&ids);
    defer s.deinit(gpa);
    var f: Fake = .{};
    const r = try run(&s, 0, fill.rows(128, 0), &f);
    try std.testing.expectEqual(@as(usize, 1), f.begins);
    try std.testing.expectEqual(@as(usize, 1), f.ends);
    try std.testing.expectEqual(@as(usize, 4), r.chunks);
    try std.testing.expectEqual(@as(usize, 2), f.tower.calls); // the repeated image reuses the first one's rows
}

test "LFM3 prefill images: a host cap narrows chunks without another projection" {
    var ids: [40]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i + 3);
    var s = try make(&ids);
    defer s.deinit(gpa);
    var f: Fake = .{};
    const narrow = fill.rows(128, 1);
    try std.testing.expectEqual(@as(usize, 1), narrow);
    const r = try run(&s, 0, narrow, &f);
    try std.testing.expectEqual(@as(usize, 1), f.begins);
    try std.testing.expectEqual(@as(usize, 1), f.ends);
    try std.testing.expectEqual(@as(usize, 40), r.chunks);
    try std.testing.expectEqual(@as(usize, 2), f.tower.calls);
}

test "LFM3 prefill images: a warm pass from inside an image projects only what it reads" {
    var ids: [400]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i + 3);
    var s = try make(&ids);
    defer s.deinit(gpa);
    var f: Fake = .{};
    const r = try run(&s, 130, fill.rows(128, 0), &f); // inside the first image's rows
    try std.testing.expectEqual(@as(usize, 1), f.begins);
    try std.testing.expectEqual(@as(usize, 1), f.ends);
    try std.testing.expectEqual(@as(usize, 130), f.from);
    try std.testing.expectEqual(@as(usize, 3), r.chunks);
    // the first image again (its span ends past 130), the second, then the repeat reuses the first's rows
    try std.testing.expectEqual(@as(usize, 2), f.tower.calls);
}

test "LFM3 prefill images: a stream released mid-pass ends the pass once" {
    var ids: [400]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i + 3);
    var s = try make(&ids);
    defer s.deinit(gpa);
    var f: Fake = .{};
    var state = Fake.Fill{ .provider = f.provider(), .stream = &s };
    try state.begin(0);
    try std.testing.expect((try state.chunk(0, 128)) != null);
    state.close(); // releaseFn
    state.close(); // a second close is not another end
    try std.testing.expectEqual(@as(usize, 1), f.begins);
    try std.testing.expectEqual(@as(usize, 1), f.ends);
}
