//! The engine's directly read tensors take their declared widths: experts by layer, the embedding, the n-gram tables.
const std = @import("std");
const config = @import("config.zig");
const widths = @import("widths.zig");
const ew = @import("engine_widths.zig");
const a = std.testing.allocator;
const checked_config = @embedFile("fixtures/config.json");

/// The checked config at global `bits`/`group` (both quantization blocks) with `entries` as overrides.
fn with(bits: []const u8, group: []const u8, entries: []const u8) ![]u8 {
    const declaration = try std.fmt.allocPrint(a, "\"mode\": \"affine\", {s}", .{entries});
    defer a.free(declaration);
    const once = try std.mem.replaceOwned(u8, a, checked_config, "\"mode\": \"affine\"", declaration);
    defer a.free(once);
    const twice = try std.mem.replaceOwned(u8, a, once, "\"bits\": 6,", bits);
    defer a.free(twice);
    return std.mem.replaceOwned(u8, a, twice, "\"group_size\": 32,", group);
}

fn readSet(bytes: []const u8) !ew.Set {
    var c = try config.parse(a, bytes);
    defer c.deinit();
    var w = try widths.Widths.init(a, &c);
    defer w.deinit();
    return ew.read(&c, &w, "shard_");
}

test "oQ2's experts, embedding and n-gram tables take their own widths, layer by layer" {
    const bytes = try with("\"bits\": 2,", "\"group_size\": 32,",
        \\"language_model.model.embed_tokens": {"bits": 8, "group_size": 64},
        \\"language_model.model.layers.3.mlp.shared_expert.gate_proj": {"bits": 8, "group_size": 128},
        \\"language_model.model.layers.3.mlp.shared_expert.up_proj": {"bits": 8, "group_size": 128},
        \\"language_model.model.layers.3.mlp.shared_expert.down_proj": {"bits": 8, "group_size": 128},
        \\"language_model.mtp.layers.0.mlp.switch_mlp.gate_proj": {"bits": 4, "group_size": 32},
        \\"language_model.mtp.layers.0.mlp.switch_mlp.up_proj": {"bits": 4, "group_size": 32}
    );
    defer a.free(bytes);
    const s = try readSet(bytes);
    try std.testing.expect(!s.uniform);
    try std.testing.expectEqual([4]usize{ 2, 32, 8, 128 }, s.layers[3].?.gu);
    try std.testing.expectEqual([4]usize{ 2, 32, 8, 128 }, s.layers[3].?.down);
    try std.testing.expectEqual([4]usize{ 2, 32, 2, 32 }, s.layers[0].?.gu);
    try std.testing.expectEqual([4]usize{ 4, 32, 2, 32 }, s.mtp.?.gu);
    try std.testing.expectEqual([4]usize{ 2, 32, 2, 32 }, s.mtp.?.down);
    try std.testing.expectEqual([2]usize{ 8, 64 }, s.embed.?);
    try std.testing.expectEqual([2]usize{ 2, 32 }, s.ple.?);
}

test "the uniform 6/32 checkpoint keeps every recorded kernel" {
    const s = try readSet(checked_config);
    try std.testing.expect(s.uniform);
    try std.testing.expect(s.embed == null and s.ple == null and s.mtp == null);
    for (s.layers) |l| try std.testing.expect(l == null);
}

test "an expert whose gate and up differ in width is refused: one kernel reads both" {
    const bytes = try with("\"bits\": 6,", "\"group_size\": 32,",
        \\"language_model.model.layers.0.mlp.switch_mlp.up_proj": {"bits": 4, "group_size": 64}
    );
    defer a.free(bytes);
    try std.testing.expectError(error.ExpertGateUpWidths, readSet(bytes));
}

test "n-gram shards of different widths are refused: one lookup kernel reads every shard" {
    const bytes = try with("\"bits\": 6,", "\"group_size\": 32,",
        \\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_1": {"bits": 4, "group_size": 32}
    );
    defer a.free(bytes);
    try std.testing.expectError(error.NgramTableWidths, readSet(bytes));
}

test "a check program that reads only 6/32 refuses a 4/32 checkpoint and one 6/32 checkpoint with an override" {
    const io = std.testing.io;
    const cases = [_]struct { bits: []const u8, entries: []const u8, six: bool }{
        .{ .bits = "\"bits\": 6,", .entries = "", .six = true },
        .{ .bits = "\"bits\": 4,", .entries = "", .six = false },
        .{ .bits = "\"bits\": 6,", .entries = "\"language_model.model.embed_tokens\": {\"bits\": 8, \"group_size\": 64}", .six = false },
    };
    for (cases) |case| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const bytes = if (case.entries.len == 0)
            try std.mem.replaceOwned(u8, a, checked_config, "\"bits\": 6,", case.bits)
        else
            try with(case.bits, "\"group_size\": 32,", case.entries);
        defer a.free(bytes);
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = bytes });
        const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
        defer a.free(dir);
        try std.testing.expectEqual(case.six, try ew.uniformDir(a, io, dir));
    }
}
