//! A restored prefix must retain complete-prompt arithmetic, even when its old embedding rows are zero.
const std = @import("std");
const hip = @import("hip");
const family = @import("glm_ocr");
const c = family.config;
const a = std.testing.allocator;

fn download(stream: hip.Stream, address: u64, count: usize) ![]u16 {
    const out = try a.alloc(u16, count);
    errdefer a.free(out);
    const view: hip.DeviceBuffer = .{ .r = stream.r, .ptr = @ptrFromInt(address), .len = count * 2 };
    try stream.synchronize();
    try view.download(0, std.mem.sliceAsBytes(out));
    return out;
}
fn packedKv(cache: *const family.decoder.Cache) ![]u16 {
    const out = try a.alloc(u16, c.layers * cache.len * c.kv * 2);
    errdefer a.free(out);
    for (0..c.layers * 2) |plane| {
        try cache.buffer.download(plane * cache.capacity * c.kv * 2, std.mem.sliceAsBytes(out[plane * cache.len * c.kv ..][0 .. cache.len * c.kv]));
    }
    return out;
}

pub fn check(graph: *const family.decoder.Graph, source: *family.decoder.Cache, embeddings: u64, positions: u64, rows: usize, full: u64) !void {
    const expected = try download(graph.stream, full, rows * c.hidden);
    defer a.free(expected);
    const expected_kv = try packedKv(source);
    defer a.free(expected_kv);
    var ctx = try hip.Context.init(graph.stream.r, 0);
    defer ctx.deinit();
    for ([_]usize{ 1, 7, rows / 2, rows - 1, rows }) |prefix| {
        // Later iterations reuse the graph scratch: save from the original retained host hidden rows.
        var original = try hip.DeviceBuffer.fromHost(graph.stream.r, std.mem.sliceAsBytes(expected));
        defer original.free();
        var snap = try family.snapshot.Snapshot.take(source, original.base(), @intCast(prefix - 1), rows, (try ctx.memInfo()).free -| (512 << 20));
        defer snap.deinit();
        var restored = try family.decoder.Cache.init(graph.stream, rows + 256, (try ctx.memInfo()).free -| (512 << 20));
        defer restored.deinit();
        try snap.restore(&restored, @intCast(prefix - 1), rows);
        try std.testing.expectEqual(prefix, restored.len);
        if (prefix == rows) {
            const prepared = try download(graph.stream, snap.hidden.base(), c.hidden);
            defer a.free(prepared);
            try std.testing.expectEqualSlices(u16, expected[(rows - 1) * c.hidden ..], prepared);
        } else {
            var input = try hip.DeviceBuffer.alloc(graph.stream.r, rows * c.hidden * 2);
            defer input.free();
            try input.copyFromAsync(0, embeddings, input.len, graph.stream);
            const skipped: hip.DeviceBuffer = .{ .r = graph.stream.r, .ptr = input.ptr, .len = prefix * c.hidden * 2 };
            try skipped.fill8Async(0, graph.stream);
            const address = try graph.forwardMany(&.{.{ .cache = &restored, .embeddings = input.base(), .positions = positions, .rows = rows, .prefix_rows = prefix }}, null);
            const actual = try download(graph.stream, address + prefix * c.hidden * 2, (rows - prefix) * c.hidden);
            defer a.free(actual);
            try std.testing.expectEqualSlices(u16, expected[prefix * c.hidden ..], actual);
        }
        try std.testing.expectEqual(rows, restored.len);
        const actual_kv = try packedKv(&restored);
        defer a.free(actual_kv);
        try std.testing.expectEqualSlices(u16, expected_kv, actual_kv);
        std.debug.print("full-shape resume {d}/{d}: 0 hidden or KV differences\n", .{ prefix, rows });
    }
    // A different final text token must also match a fresh pass; the original cached prefix is unchanged.
    const prefix = rows / 2;
    var alternate = try hip.DeviceBuffer.alloc(graph.stream.r, rows * c.hidden * 2);
    defer alternate.free();
    try alternate.copyFromAsync(0, embeddings, alternate.len, graph.stream);
    const table = try graph.weights.get(family.schema.text ++ "embed_tokens.weight");
    try alternate.copyFromAsync((rows - 1) * c.hidden * 2, table + 100 * c.hidden * 2, c.hidden * 2, graph.stream);
    var fresh = try family.decoder.Cache.init(graph.stream, rows + 256, (try ctx.memInfo()).free -| (512 << 20));
    defer fresh.deinit();
    const branch = try graph.forward(&fresh, alternate.base(), positions, rows, null);
    const want = try download(graph.stream, branch + prefix * c.hidden * 2, (rows - prefix) * c.hidden);
    defer a.free(want);
    const want_kv = try packedKv(&fresh);
    defer a.free(want_kv);
    var original = try hip.DeviceBuffer.fromHost(graph.stream.r, std.mem.sliceAsBytes(expected));
    defer original.free();
    var snap = try family.snapshot.Snapshot.take(source, original.base(), @intCast(prefix - 1), rows, (try ctx.memInfo()).free -| (512 << 20));
    defer snap.deinit();
    try snap.restore(&fresh, @intCast(prefix - 1), rows);
    const skipped: hip.DeviceBuffer = .{ .r = graph.stream.r, .ptr = alternate.ptr, .len = prefix * c.hidden * 2 };
    try skipped.fill8Async(0, graph.stream);
    const resumed = try graph.forwardMany(&.{.{ .cache = &fresh, .embeddings = alternate.base(), .positions = positions, .rows = rows, .prefix_rows = prefix }}, null);
    const actual = try download(graph.stream, resumed + prefix * c.hidden * 2, (rows - prefix) * c.hidden);
    defer a.free(actual);
    const actual_kv = try packedKv(&fresh);
    defer a.free(actual_kv);
    try std.testing.expectEqualSlices(u16, want, actual);
    try std.testing.expectEqualSlices(u16, want_kv, actual_kv);
    std.debug.print("full-shape changed tail {d}/{d}: 0 hidden or KV differences\n", .{ prefix, rows });
}
