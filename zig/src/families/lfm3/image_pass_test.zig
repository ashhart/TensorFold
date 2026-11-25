//! The tower runs once per distinct pending image, under every chunking a pass takes.
const std = @import("std");
const pass = @import("image_pass.zig");

const Image = struct { hash: u64, at: u32, tokens: u32 };

/// Counts tower calls and the rows they wrote.
const Tower = struct {
    calls: usize = 0,
    rows: usize = 0,

    pub fn project(t: *Tower, im: anytype, row: u32) !void {
        t.calls += 1;
        t.rows = @max(t.rows, row + im.tokens);
    }
};

const images = [_]Image{
    .{ .hash = 1, .at = 100, .tokens = 70 },
    .{ .hash = 2, .at = 600, .tokens = 70 },
    .{ .hash = 1, .at = 900, .tokens = 70 },
    .{ .hash = 0, .at = 1000, .tokens = 40 },
};

test "LFM3 image pass: one tower call per distinct pending image, ordinary, narrowed and warm chunks" {
    const gpa = std.testing.allocator;
    // ordinary 128-row chunks, host-narrowed one-row chunks, and a kept prefix inside an image
    const cases = [_]struct { chunk: usize, from: usize, calls: usize, rows: usize, spans: usize, spliced: usize }{
        .{ .chunk = 128, .from = 0, .calls = 3, .rows = 180, .spans = 4, .spliced = 250 },
        .{ .chunk = 1, .from = 0, .calls = 3, .rows = 180, .spans = 4, .spliced = 250 },
        .{ .chunk = 17, .from = 630, .calls = 3, .rows = 180, .spans = 3, .spliced = 150 },
        .{ .chunk = 128, .from = 100, .calls = 3, .rows = 180, .spans = 4, .spliced = 250 },
        .{ .chunk = 128, .from = 1070, .calls = 0, .rows = 0, .spans = 0, .spliced = 0 },
    };
    for (cases) |case| {
        var spans: std.ArrayList(pass.Span) = .empty;
        defer spans.deinit(gpa);
        var tower: Tower = .{};
        try pass.encode(gpa, &images, case.from, &spans, &tower);
        try std.testing.expectEqual(case.calls, tower.calls);
        try std.testing.expectEqual(case.rows, tower.rows);
        try std.testing.expectEqual(case.spans, spans.items.len);
        try std.testing.expectEqual(case.rows, pass.rows(&images, case.from));
        // the chunks that follow at this cut read the same spans: no chunk reaches the tower again
        var spliced: usize = 0;
        var at = case.from;
        while (at < 1040) : (at = @min(at + case.chunk, 1040)) {
            const end = @min(at + case.chunk, 1040);
            for (spans.items) |span| {
                const first = @max(at, span.at);
                const last = @min(end, span.at + span.rows);
                if (first < last) spliced += last - first;
            }
        }
        try std.testing.expectEqual(case.spliced, spliced);
        try std.testing.expectEqual(case.calls, tower.calls);
    }
}

test "LFM3 image pass: a warm prefix before the image keeps it out of the pass" {
    const gpa = std.testing.allocator;
    var spans: std.ArrayList(pass.Span) = .empty;
    defer spans.deinit(gpa);
    var tower: Tower = .{};
    try pass.encode(gpa, &images, 170, &spans, &tower); // image 1's span ends at 170
    try std.testing.expectEqual(@as(usize, 3), tower.calls);
    try std.testing.expectEqual(@as(usize, 3), spans.items.len);
    try std.testing.expectEqual(pass.Span{ .at = 600, .rows = 70, .source = 0 }, spans.items[0]);
    try std.testing.expectEqual(pass.Span{ .at = 900, .rows = 70, .source = 70 }, spans.items[1]);
}
