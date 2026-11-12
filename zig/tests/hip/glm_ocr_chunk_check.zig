//! Compare prompt cuts with a complete prefill, including cuts inside image rows.
const std = @import("std");
const hip = @import("hip");
const family = @import("glm_ocr");
const c = family.config;
const a = std.testing.allocator;

pub fn check(graph: *const family.decoder.Graph, embeddings: u64, positions: u64, rows: usize, full: u64, fixed: usize) !void {
    const stream = graph.stream;
    const expected = try a.alloc(u16, rows * c.hidden);
    defer a.free(expected);
    const view: hip.DeviceBuffer = .{ .r = stream.r, .ptr = @ptrFromInt(full), .len = expected.len * 2 };
    try view.download(0, std.mem.sliceAsBytes(expected));
    var ctx = try hip.Context.init(stream.r, 0);
    defer ctx.deinit();
    var cache = try family.decoder.Cache.init(stream, rows + 256, (try ctx.memInfo()).free -| (256 << 20));
    defer cache.deinit();
    var at: usize = 0;
    var piece: usize = 0;
    const widths = [_]usize{ 1, 7, 31, 64 };
    while (at < rows) {
        const count = @min(if (fixed != 0) fixed else widths[@min(piece, widths.len - 1)], rows - at);
        const result = try graph.forwardMany(&.{.{ .cache = &cache, .embeddings = embeddings + at * c.hidden * 2, .positions = positions + at * 12, .rows = count, .phase = .prompt }}, null);
        const out = try a.alloc(u16, count * c.hidden);
        defer a.free(out);
        const actual: hip.DeviceBuffer = .{ .r = stream.r, .ptr = @ptrFromInt(result), .len = out.len * 2 };
        try actual.download(0, std.mem.sliceAsBytes(out));
        const want = expected[at * c.hidden ..][0..out.len];
        var differences: usize = 0;
        for (out, want) |x, y| if (x != y) {
            differences += 1;
        };
        std.debug.print("prompt chunk {d}..{d}: {d}/{d} BF16 differences from complete prefill\n", .{ at, at + count, differences, out.len });
        try std.testing.expectEqual(@as(usize, 0), differences);
        at += count;
        piece += 1;
    }
}
