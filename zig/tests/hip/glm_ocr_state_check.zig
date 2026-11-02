//! Cache restore and shared decoder comparisons on the same image prompt and generated token.
const std = @import("std");
const hip = @import("hip");
const family = @import("glm_ocr");
const c = family.config;
const a = std.testing.allocator;

const Stages = struct {
    stream: hip.Stream,
    saved: std.StringHashMap([]u16) = .init(a),
    shared_rows: usize = 1,
    fn deinit(s: *Stages) void {
        var entries = s.saved.iterator();
        while (entries.next()) |e| {
            a.free(e.key_ptr.*);
            a.free(e.value_ptr.*);
        }
        s.saved.deinit();
    }
    fn tap(context: *anyopaque, name: []const u8, address: u64, values: usize) !void {
        const s: *Stages = @ptrCast(@alignCast(context));
        try s.stream.synchronize();
        const out = try a.alloc(u16, values);
        var retained = false;
        defer if (!retained) a.free(out);
        const view: hip.DeviceBuffer = .{ .r = s.stream.r, .ptr = @ptrFromInt(address), .len = values * 2 };
        try view.download(0, std.mem.sliceAsBytes(out));
        if (s.shared_rows == 1) {
            const key = try a.dupe(u8, name);
            errdefer a.free(key);
            try s.saved.put(key, out);
            retained = true;
            return;
        }
        const expected = s.saved.get(name) orelse return error.MissingSoloStage;
        try std.testing.expectEqual(expected.len * s.shared_rows, out.len);
        var differences: usize = 0;
        for (out, 0..) |v, i| if (v != expected[i % expected.len]) {
            differences += 1;
        };
        std.debug.print("shared {s}: {d}/{d} BF16 differences\n", .{ name, differences, out.len });
    }
};

pub fn shared(graph: *const family.decoder.Graph, source: *family.decoder.Cache, tokens: []const u32, position: i32) !void {
    const stream = graph.stream;
    var ctx = try hip.Context.init(stream.r, 0);
    defer ctx.deinit();
    var copies: [4]family.decoder.Cache = undefined;
    var initialized: usize = 0;
    defer for (copies[0..initialized]) |*cache| cache.deinit();
    for (&copies, 0..) |*cache, i| {
        cache.* = try .init(stream, source.capacity + i, (try ctx.memInfo()).free -| (256 << 20));
        initialized += 1;
        try cache.copyPrefix(source, source.len);
    }
    const coordinates: [3]i32 = @splat(position);
    var positions = try hip.DeviceBuffer.fromHost(stream.r, std.mem.asBytes(&coordinates));
    defer positions.free();
    const table = try graph.weights.get(family.schema.text ++ "embed_tokens.weight");
    const embedding = table + tokens[0] * c.hidden * 2;
    var stages: Stages = .{ .stream = stream };
    defer stages.deinit();
    const solo = try graph.forward(&copies[0], embedding, positions.base(), 1, .{ .context = &stages, .capture = Stages.tap });
    const expected = try a.alloc(u16, c.hidden);
    defer a.free(expected);
    const solo_view: hip.DeviceBuffer = .{ .r = stream.r, .ptr = @ptrFromInt(solo), .len = c.hidden * 2 };
    try solo_view.download(0, std.mem.sliceAsBytes(expected));
    try copies[0].truncate(source.len);
    var segments: [4]family.decoder.Segment = undefined;
    for (&copies, &segments) |*cache, *segment| segment.* = .{ .cache = cache, .embeddings = embedding, .positions = positions.base(), .rows = 1, .phase = .decode };
    stages.shared_rows = copies.len;
    const result = try graph.forwardMany(&segments, .{ .context = &stages, .capture = Stages.tap });
    const actual = try a.alloc(u16, c.hidden * copies.len);
    defer a.free(actual);
    const shared_view: hip.DeviceBuffer = .{ .r = stream.r, .ptr = @ptrFromInt(result), .len = actual.len * 2 };
    try shared_view.download(0, std.mem.sliceAsBytes(actual));
    for (&copies, 0..) |*cache, i| {
        try std.testing.expectEqual(source.len + 1, cache.len);
        const row = actual[i * c.hidden ..][0..c.hidden];
        var different: usize = 0;
        for (row, expected) |x, y| {
            if (x != y) different += 1;
        }
        std.debug.print("shared decoder row {d}: {d}/{d} BF16 differences from solo\n", .{ i, different, c.hidden });
        try std.testing.expectEqualSlices(u16, expected, row);
    }
    for ([_]usize{ 1, 2, 3, 4, 8, 16, 32 }) |count| {
        const sequential = try a.alloc(u16, count * c.hidden);
        defer a.free(sequential);
        var window_input = try hip.DeviceBuffer.alloc(stream.r, count * c.hidden * 2);
        defer window_input.free();
        const coordinates_window = try a.alloc([3]i32, count);
        defer a.free(coordinates_window);
        for (coordinates_window, 0..) |*row, i| row.* = @splat(position + @as(i32, @intCast(i)));
        var window_positions = try hip.DeviceBuffer.fromHost(stream.r, std.mem.sliceAsBytes(coordinates_window));
        defer window_positions.free();
        try copies[0].truncate(source.len);
        try copies[1].truncate(source.len);
        for (0..count) |i| {
            try window_input.copyFromAsync(i * c.hidden * 2, table + tokens[i % tokens.len] * c.hidden * 2, c.hidden * 2, stream);
            const row = try graph.forward(&copies[1], window_input.base() + i * c.hidden * 2, window_positions.base() + i * 12, 1, null);
            const view: hip.DeviceBuffer = .{ .r = stream.r, .ptr = @ptrFromInt(row), .len = c.hidden * 2 };
            try view.download(0, std.mem.sliceAsBytes(sequential[i * c.hidden ..][0..c.hidden]));
        }
        const window = try graph.forwardMany(&.{.{ .cache = &copies[0], .embeddings = window_input.base(), .positions = window_positions.base(), .rows = count, .phase = .decode }}, null);
        const window_values = try a.alloc(u16, count * c.hidden);
        defer a.free(window_values);
        const window_view: hip.DeviceBuffer = .{ .r = stream.r, .ptr = @ptrFromInt(window), .len = count * c.hidden * 2 };
        try window_view.download(0, std.mem.sliceAsBytes(window_values));
        var differences: usize = 0;
        for (window_values, sequential) |x, y| if (x != y) {
            differences += 1;
        };
        std.debug.print("verification window {d}: {d}/{d} BF16 differences from plain\n", .{ count, differences, window_values.len });
        try std.testing.expectEqualSlices(u16, sequential, window_values);
    }
}
