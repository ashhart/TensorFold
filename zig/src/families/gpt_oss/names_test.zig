//! All 459 real tensor headers, without model payloads or a second safetensors parser.
const std = @import("std");
const st = @import("safetensors");
const names = @import("names.zig");
const config = @import("config.zig");
const t = std.testing;

pub fn headers(a: std.mem.Allocator) !st.Header {
    var all: st.Header = .empty;
    const jsons = [_][]const u8{
        @embedFile("fixtures/model-00000-of-00002.safetensors.header.json"),
        @embedFile("fixtures/model-00001-of-00002.safetensors.header.json"),
        @embedFile("fixtures/model-00002-of-00002.safetensors.header.json"),
    };
    for (jsons, [_]usize{ 4792250240, 4798679872, 4170334656 }) |json, size| {
        const h = try st.parseHeader(a, json, size);
        for (h.keys(), h.values()) |key, value| {
            try t.expect(!all.contains(key));
            try all.put(a, key, value);
        }
    }
    return all;
}

test "gpt-oss tensor map accounts for every real name dtype and dimension" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var h = try headers(arena.allocator());
    const c = try @import("config_test.zig").load();
    var why: config.Why = .{};
    try names.check(&c, &h, &why);
    try t.expectEqual(@as(usize, 459), h.count());
    const q = names.find(&c, "model.layers.0.self_attn.q_proj.weight").?;
    try t.expectEqual(@as(usize, 4096), q.shape[0]);
    try t.expectEqual(@as(usize, 2880), q.shape[1]);
    const blocks = names.find(&c, "model.layers.23.mlp.experts.gate_up_proj_blocks").?;
    try t.expectEqualSlices(usize, &.{ 32, 5760, 90, 16 }, blocks.shape[0..blocks.rank]);
    try t.expectEqual(st.DType.u8, blocks.dtype);
    try t.expect(names.find(&c, "model.layers.24.self_attn.sinks") == null);
    try t.expect(names.find(&c, "model.layers.00.self_attn.sinks") == null);
    try t.expect(names.find(&c, "model.layers.-1.self_attn.sinks") == null);
}

test "gpt-oss missing sink and malformed mixed projections are refused by tensor name" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var h = try headers(arena.allocator());
    const c = try @import("config_test.zig").load();
    var why: config.Why = .{};
    const sink = "model.layers.0.self_attn.sinks";
    const saved = h.get(sink).?;
    _ = h.swapRemove(sink);
    try t.expectError(error.BadGptOssTensor, names.check(&c, &h, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), sink) != null);
    try h.put(arena.allocator(), sink, saved);
    const key = "model.layers.0.mlp.experts.gate_up_proj_blocks";
    const original = h.get(key).?;
    h.getPtr(key).?.shape[2] = 89;
    try t.expectError(error.BadGptOssTensor, names.check(&c, &h, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), key) != null);
    h.getPtr(key).?.* = original;
    h.getPtr(key).?.dtype = .bf16;
    try t.expectError(error.BadGptOssTensor, names.check(&c, &h, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), key) != null);
    h.getPtr(key).?.* = original;
    try h.put(arena.allocator(), "model.layers.0.mlp.experts.unexpected", saved);
    try t.expectError(error.BadGptOssTensor, names.check(&c, &h, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), "unexpected") != null);
}
