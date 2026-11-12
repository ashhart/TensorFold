//! Host-only Ling family package: config, names, fixtures, synth JSON, the float64 oracle text. No GPU.
const std = @import("std");

test {
    _ = @import("config.zig");
    _ = @import("config_test.zig");
    _ = @import("names.zig");
    _ = @import("names_test.zig");
    _ = @import("tiny_test.zig");
    _ = @import("synth.zig");
    _ = @import("route.zig");
    _ = @import("kda.zig");
}

test "the float64 oracle handles tiny's null SwiGLU lists" {
    const src = @embedFile("oracle/ling_forward.py");
    try std.testing.expect(std.mem.indexOf(u8, src, "null (Ling-3.0-tiny)") != null);
}

test "kda_inner.py is a stdlib copy; ling_forward.py is left untouched" {
    const src = @embedFile("oracle/ling_forward.py");
    const inner = @embedFile("oracle/kda_inner.py");
    try std.testing.expect(std.mem.indexOf(u8, src, "kda_inner") == null);
    try std.testing.expect(std.mem.indexOf(u8, inner, "def kda_inner") != null);
    try std.testing.expect(std.mem.indexOf(u8, inner, "does not import Model.kda") != null);
    try std.testing.expect(std.mem.indexOf(u8, inner, "has no mixer-only entry point") != null);
    const vs = @embedFile("oracle/kda_vs_model.py");
    try std.testing.expect(std.mem.indexOf(u8, vs, "No edit to ling_forward.py") != null);
    try std.testing.expect(std.mem.indexOf(u8, vs, "class CaptureModel") != null);
}
