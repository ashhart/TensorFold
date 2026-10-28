//! Host-only Ling family package: config, names, fixtures, synth JSON, the float64 oracle text. No GPU.
const std = @import("std");

test {
    _ = @import("config.zig");
    _ = @import("config_test.zig");
    _ = @import("names.zig");
    _ = @import("names_test.zig");
    _ = @import("tiny_test.zig");
    _ = @import("synth.zig");
}

test "the float64 oracle handles tiny's null SwiGLU lists" {
    const src = @embedFile("oracle/ling_forward.py");
    try std.testing.expect(std.mem.indexOf(u8, src, "null (Ling-3.0-tiny)") != null);
}
