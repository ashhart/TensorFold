//! The Metal/HIP reader refuses a config that asks for scaled RoPE; plain configs keep theta and rotary dims.
const std = @import("std");
const config = @import("config.zig");

test "scaled RoPE is refused in either form; plain and null rope_scaling keep theta and rotary dims" {
    const a = std.testing.allocator;
    const bytes = @embedFile("fixtures/config.json");
    const plain = "\"rope_parameters\": {";
    const keep = try std.mem.replaceOwned(u8, a, bytes, plain, "\"rope_scaling\": null, " ++ plain);
    defer a.free(keep);
    for ([_][]const u8{ bytes, keep }) |accepted| {
        var c = try config.parse(a, accepted);
        defer c.deinit();
        try std.testing.expectEqual(@as(f64, 10_000_000), c.rope_theta);
        try std.testing.expectEqual(@as(usize, 64), c.rotary_dim);
    }
    const yarn = "{\"rope_type\": \"yarn\", \"factor\": 4.0, \"original_max_position_embeddings\": 262144}";
    for ([_][2][]const u8{
        .{ "\"type\": \"default\"", "\"rope_type\": \"yarn\", \"factor\": 4.0, \"original_max_position_embeddings\": 262144" },
        .{ plain, "\"rope_scaling\": " ++ yarn ++ ", " ++ plain },
    }) |change| {
        const changed = try std.mem.replaceOwned(u8, a, bytes, change[0], change[1]);
        defer a.free(changed);
        try std.testing.expectError(error.UnsupportedFlashRope, config.parse(a, changed));
    }
}
