//! Host proof for the CUDA DFlash round trace's exact, opt-in JSONL format.
const std = @import("std");
const lanes = @import("lanes");
const trace = @import("cuda_draft_trace.zig");

fn bits(x: f64) u64 {
    return @bitCast(x);
}

test "N15 trace: selector inputs and the accepted path round-trip exactly" {
    const a = std.testing.allocator;
    var words: [trace.lp_words * 2]u32 = undefined;
    for (&words, 0..) |*x, i| x.* = @intCast(1000 + i * 17);
    var hp: [8]u16 = .{ 0x0000, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc0, 0x0001 };
    const nodes = [_]u32{ 41, 42, 43 };
    const parents = [_]i32{ -1, 0, 0 };
    const own = [_]f64{ -0.0, -std.math.inf(f64), 0.25 };
    const accepted_rows = [_]u32{ 1, 2, 5 };
    const accepted_tokens = [_]u32{ 41, 42, 99 };
    const candidates: lanes.draft_tree.Candidates = .{
        .words = std.mem.sliceAsBytes(&words),
        .projected = std.mem.sliceAsBytes(&hp),
        .topk = 3,
        .predecessor = &.{},
        .successor = &.{},
        .rank = 4,
    };
    var text: std.Io.Writer.Allocating = .init(a);
    defer text.deinit();
    try trace.writeRecord(&text.writer, .{
        .stream = "request\\\"one",
        .position = 123,
        .pending = 40,
        .offered = 3,
        .nodes = &nodes,
        .parents = &parents,
        .own = &own,
        .accepted_rows = &accepted_rows,
        .accepted_tokens = &accepted_tokens,
        .candidates = candidates,
    });
    const got = try trace.parse(a, std.mem.trimEnd(u8, text.written(), "\n"));
    defer got.deinit();
    const r = got.value;
    try std.testing.expectEqual(@as(u8, 1), r.v);
    try std.testing.expectEqualStrings("request\\\"one", r.stream);
    try std.testing.expectEqual(@as(u64, 123), r.position);
    try std.testing.expectEqual(@as(u32, 40), r.pending);
    try std.testing.expectEqual(@as(u32, 3), r.offered);
    try std.testing.expectEqualSlices(u32, &nodes, r.nodes);
    try std.testing.expectEqualSlices(i32, &parents, r.parents);
    for (own, r.own) |want, actual| try std.testing.expectEqual(bits(want), actual);
    try std.testing.expectEqualSlices(u32, &accepted_rows, r.accepted_rows);
    try std.testing.expectEqualSlices(u32, &accepted_tokens, r.accepted_tokens);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, -1 }, r.accepted_nodes);
    try std.testing.expectEqual(@as(usize, 2), r.candidates.len);
    for (r.candidates, 0..) |row, i| {
        const w = words[i * trace.lp_words ..][0..trace.lp_words];
        const ids = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(w)[trace.ids_offset .. trace.ids_offset + 3 * 4]);
        const unary = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(w)[trace.unary_offset .. trace.unary_offset + 3 * 4]);
        try std.testing.expectEqualSlices(u32, ids, row.ids);
        try std.testing.expectEqualSlices(u32, unary, row.unary);
        const encoded = try std.fmt.allocPrint(a, "{x:0>4}{x:0>4}{x:0>4}{x:0>4}", .{ hp[i * 4], hp[i * 4 + 1], hp[i * 4 + 2], hp[i * 4 + 3] });
        defer a.free(encoded);
        try std.testing.expectEqualStrings(encoded, row.hp);
    }
    try std.testing.expectError(error.UnsupportedTraceVersion, trace.parse(a, "{\"v\":2}"));
}

test "N15 trace: all fifteen rows and 256 selector inputs survive" {
    const a = std.testing.allocator;
    var words: [15 * trace.lp_words]u32 = undefined;
    for (&words, 0..) |*x, i| x.* = @bitCast(@as(f32, @floatFromInt(i)));
    for (0..15) |row| for (0..16) |j| {
        words[row * trace.lp_words + 2 + j] = @intCast(row * 16 + j);
        words[row * trace.lp_words + 2 + 20 + j] = @bitCast(@as(f32, @floatFromInt(row * 16 + j)));
    };
    var hp: [15 * 256]u16 = undefined;
    for (&hp, 0..) |*x, i| x.* = @truncate(i *% 65521);
    const candidates: lanes.draft_tree.Candidates = .{
        .words = std.mem.sliceAsBytes(&words),
        .projected = std.mem.sliceAsBytes(&hp),
        .topk = 16,
        .rank = 256,
        .predecessor = &.{},
        .successor = &.{},
    };
    var text: std.Io.Writer.Allocating = .init(a);
    defer text.deinit();
    try trace.writeRecord(&text.writer, .{
        .stream = "n15",
        .position = 1,
        .pending = 0,
        .offered = 0,
        .nodes = &.{},
        .parents = &.{},
        .own = &.{},
        .accepted_rows = &.{},
        .accepted_tokens = &.{},
        .candidates = candidates,
    });
    const got = try trace.parse(a, std.mem.trimEnd(u8, text.written(), "\n"));
    defer got.deinit();
    try std.testing.expectEqual(@as(usize, 15), got.value.candidates.len);
    for (got.value.candidates, 0..) |row, i| {
        const w = words[i * trace.lp_words ..][0..trace.lp_words];
        try std.testing.expectEqualSlices(u32, w[2..18], row.ids);
        try std.testing.expectEqualSlices(u32, w[22..38], row.unary);
        try std.testing.expectEqual(@as(usize, 4 * 256), row.hp.len);
        for (hp[i * 256 ..][0..256], 0..) |value, j| {
            var encoded: [4]u8 = undefined;
            const want = try std.fmt.bufPrint(&encoded, "{x:0>4}", .{value});
            try std.testing.expectEqualStrings(want, row.hp[j * 4 ..][0..4]);
        }
    }
}

test "N15 trace: no environment path means no file and no logger" {
    try std.testing.expectEqual(@as(?*trace.Trace, null), trace.openPath(std.testing.allocator, std.testing.io, null));
    try std.testing.expectEqual(@as(?*trace.Trace, null), trace.openPath(std.testing.allocator, std.testing.io, ""));
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}/rounds.jsonl", .{tmp.sub_path});
    defer std.testing.allocator.free(path);
    var slot = try trace.openPath(std.testing.allocator, std.testing.io, path);
    try std.testing.expect(slot != null);
    trace.drop(&slot);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, std.testing.allocator, .limited(1));
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqual(@as(usize, 0), bytes.len);
}
