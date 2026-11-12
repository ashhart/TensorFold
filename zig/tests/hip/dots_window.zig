//! Real-checkpoint window and partial-keep checks, including a key-tile boundary and poisoned unaccepted cache rows.
const std = @import("std");
const hip = @import("hip");
const family = @import("dots_mocr");
const Graph = family.graph.Graph;
const Cache = family.text.Cache;
const Tensor = hip.ops.Tensor;
const width: usize = 8;

pub fn check(a: std.mem.Allocator, stream: hip.Stream, graph: Graph, source: *Cache, prompt_rows: usize, tokens: []const u32) !void {
    const cross: usize = (64 - prompt_rows % 64 + 64 - 4) % 64;
    if (tokens.len < cross + width or source.total < prompt_rows + cross + width) return error.ShortWindowFixture;
    var other = try Cache.init(source.data.r, graph.c, source.total);
    defer other.deinit();
    var ids = try hip.DeviceBuffer.alloc(source.data.r, width * 4);
    defer ids.free();
    var positions = try hip.DeviceBuffer.alloc(source.data.r, width * 4);
    defer positions.free();
    const expected = try a.alloc(u8, width * graph.c.vocab * 2);
    defer a.free(expected);
    const actual = try a.alloc(u8, graph.c.vocab * 2);
    defer a.free(actual);
    defer stream.synchronize() catch {};
    var g = graph;
    g.ops.window = true;
    for ([_]usize{ 0, cross }) |offset| {
        // Each case starts with the same prompt; forcing these rows overwrites the earlier case's unaccepted suffix.
        for (0..offset) |row| _ = try run(g, source, stream, &ids, &positions, tokens[row..][0..1], prompt_rows + row);
        const start = prompt_rows + offset;
        try source.copyPrefix(&other, start, stream);
        const out = try run(g, source, stream, &ids, &positions, tokens[offset..][0..width], start);
        try download(source.data.r, out, expected);
        for (0..width) |row| {
            const single = try run(g, &other, stream, &ids, &positions, tokens[offset + row ..][0..1], start + row);
            try download(source.data.r, single, actual);
            try same("window logits", expected[row * actual.len ..][0..actual.len], actual);
        }
        try samePrefix(a, source, &other, start + width);
        // The accepted two-row prefix survives, while the restored cache has NaN in every unaccepted future slot.
        try stream.synchronize();
        try other.data.fill32(0x7fc07fc0);
        try source.copyPrefix(&other, start + 2, stream);
        const alternate = [1]u32{(tokens[offset + 2] + 1) % graph.c.vocab};
        const kept = try run(g, source, stream, &ids, &positions, &alternate, start + 2);
        try download(source.data.r, kept, expected[0..actual.len]);
        const restored = try run(g, &other, stream, &ids, &positions, &alternate, start + 2);
        try download(source.data.r, restored, actual);
        try same("partial-keep logits", expected[0..actual.len], actual);
        try samePrefix(a, source, &other, start + 3);
        std.debug.print("dots native window: prefix {d}, width 8; all logits and live KV bytes equal serial; partial keep 2 passes\n", .{start});
    }
}

fn run(g: Graph, cache: *Cache, stream: hip.Stream, ids: *hip.DeviceBuffer, pos: *hip.DeviceBuffer, tokens: []const u32, start: usize) !Tensor {
    if (tokens.len == 0 or tokens.len > width) return error.BadWindowFixture;
    try stream.synchronize();
    g.ops.arena.reset();
    try ids.upload(0, std.mem.sliceAsBytes(tokens));
    var positions: [width]i32 = undefined;
    for (positions[0..tokens.len], 0..) |*p, row| p.* = @intCast(start + row);
    try pos.upload(0, std.mem.sliceAsBytes(positions[0..tokens.len]));
    const input = try family.text.embed(g, ids.base(), tokens.len, &.{});
    const output = try family.text.forward(g, input, pos.base(), tokens.len, start, cache, true);
    try stream.synchronize();
    return output.logits;
}

fn download(r: *const hip.Runtime, src: Tensor, out: []u8) !void {
    try hip.runtime.check(r.api.hipMemcpyDtoH(out.ptr, @ptrFromInt(src.ptr), out.len));
}

fn same(what: []const u8, expected: []const u8, actual: []const u8) !void {
    if (std.mem.indexOfDiff(u8, expected, actual)) |at| {
        std.debug.print("dots {s}: first different byte {d}\n", .{ what, at });
        return error.WindowMismatch;
    }
}

fn samePrefix(a: std.mem.Allocator, x: *Cache, y: *Cache, rows: usize) !void {
    const step: usize = @as(usize, x.c.headDim()) * 2;
    const bytes = rows * step;
    const from = try a.alloc(u8, bytes);
    defer a.free(from);
    const to = try a.alloc(u8, bytes);
    defer a.free(to);
    for (0..x.c.layers * 2 * x.c.kv_heads) |group| {
        try x.data.download(group * x.total * step, from);
        try y.data.download(group * y.total * step, to);
        try same("window KV prefix", from, to);
    }
}
