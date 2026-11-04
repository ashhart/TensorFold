//! The XPU runner's fixed Qwen3.8-27B shape and layer pattern, checked against the checkpoint's config.json at load.
const std = @import("std");
const layer_pattern = @import("qwen35_layers");

/// The widths and depth the runner's kernels and buffers are built for.
pub const hidden = 5120;
pub const intermediate = 17408;
pub const layers = 64;
pub const vocab = 248320;

/// Whether layer `i` is full attention in the runner's one schedule: every fourth.
pub fn isAttention(i: usize) bool {
    return layer_pattern.every(4, i);
}

/// config.json's text tower (`text_config`, else the root): the runner's shape, and its pattern by the shared rule.
pub fn check(gpa: std.mem.Allocator, json: []const u8) !void {
    const doc = std.json.parseFromSlice(std.json.Value, gpa, json, .{}) catch return error.BadQwenConfig;
    defer doc.deinit();
    if (doc.value != .object) return error.BadQwenConfig;
    const nested = doc.value.object.get("text_config");
    const tower = if (nested != null and nested.? == .object) nested.?.object else doc.value.object;
    const n = tower.get("num_hidden_layers") orelse return error.BadQwenConfig;
    if (n != .integer or n.integer < 1 or n.integer > 128) return error.BadQwenConfig;
    var full: [128]bool = undefined;
    try layer_pattern.read(full[0..@intCast(n.integer)], tower);
    for ([_]struct { []const u8, usize }{ .{ "hidden_size", hidden }, .{ "intermediate_size", intermediate }, .{ "num_hidden_layers", layers }, .{ "vocab_size", vocab } }) |want| {
        const v = tower.get(want[0]) orelse return error.BadQwenConfig;
        if (v != .integer or v.integer != want[1]) return error.UnsupportedQwenShape;
    }
    for (full[0..layers], 0..) |f, i| if (f != isAttention(i)) return error.UnsupportedLayerPattern;
}

/// `check` of `dir`/config.json, read whole (at most 16 MiB).
pub fn admit(gpa: std.mem.Allocator, dir: []const u8) !void {
    const path = try std.fmt.allocPrintSentinel(gpa, "{s}/config.json", .{dir}, 0);
    defer gpa.free(path);
    const fd = std.c.open(path.ptr, .{});
    if (fd < 0) return error.MissingQwenConfig;
    defer _ = std.c.close(fd);
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    var chunk: [1 << 16]u8 = undefined;
    while (true) {
        const got = std.c.read(fd, &chunk, chunk.len);
        if (got < 0) return error.ReadFailed;
        if (got == 0) break;
        if (bytes.items.len + @as(usize, @intCast(got)) > 16 << 20) return error.BadQwenConfig;
        try bytes.appendSlice(gpa, chunk[0..@intCast(got)]);
    }
    return check(gpa, bytes.items);
}

const t = std.testing;

/// A 27B config.json with `layer_keys` (JSON members, or empty) in its text tower.
fn config(comptime layer_keys: []const u8) []const u8 {
    return "{\"model_type\":\"qwen3_5\",\"text_config\":{\"hidden_size\":5120,\"intermediate_size\":17408," ++
        "\"num_hidden_layers\":64,\"vocab_size\":248320" ++ layer_keys ++ "}}";
}

fn types(comptime every: usize) []const u8 {
    comptime var out: []const u8 = ",\"layer_types\":[";
    inline for (0..layers) |i| out = out ++ (if (i > 0) "," else "") ++ (if ((i + 1) % every == 0) "\"full_attention\"" else "\"linear_attention\"");
    return out ++ "]";
}

test "the 27B's config: the every-fourth pattern, from layer_types or the interval, is admitted" {
    for ([_][]const u8{ config(""), config(types(4)), config(types(4) ++ ",\"full_attention_interval\":4"), config(",\"layer_types\":null"), config(types(4) ++ ",\"full_attention_interval\":null") }) |json| try check(t.allocator, json);
    try check(t.allocator, "{\"hidden_size\":5120,\"intermediate_size\":17408,\"num_hidden_layers\":64,\"vocab_size\":248320}");
}

test "another shape or schedule is refused, and a contradiction by the shared reader's name" {
    try t.expectError(error.UnsupportedLayerPattern, check(t.allocator, config(types(2))));
    try t.expectError(error.UnsupportedLayerPattern, check(t.allocator, config(",\"full_attention_interval\":8")));
    try t.expectError(error.LayerTypesContradictFullAttentionInterval, check(t.allocator, config(types(4) ++ ",\"full_attention_interval\":2")));
    try t.expectError(error.BadFullAttentionInterval, check(t.allocator, config(",\"full_attention_interval\":0")));
    const nine = "{\"text_config\":{\"hidden_size\":4096,\"intermediate_size\":12288,\"num_hidden_layers\":32,\"vocab_size\":248320}}";
    try t.expectError(error.UnsupportedQwenShape, check(t.allocator, nine));
    try t.expectError(error.BadQwenConfig, check(t.allocator, "{\"text_config\":{\"hidden_size\":5120}}"));
    try t.expectError(error.BadQwenConfig, check(t.allocator, "[]"));
}

test "admit reads the folder's config.json; a folder without one is refused" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(t.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer t.allocator.free(dir);
    try t.expectError(error.MissingQwenConfig, admit(t.allocator, dir));
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = config(types(4)) });
    try admit(t.allocator, dir);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = config(types(2)) });
    try t.expectError(error.UnsupportedLayerPattern, admit(t.allocator, dir));
}
