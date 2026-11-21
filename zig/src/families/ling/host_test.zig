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
    _ = @import("mla.zig");
    _ = @import("mla_rope_test.zig");
}

test "the float64 oracle handles tiny's null SwiGLU lists" {
    const src = @embedFile("oracle/ling_forward.py");
    try std.testing.expect(std.mem.indexOf(u8, src, "null (Ling-3.0-tiny)") != null);
}

test "the float64 oracle's mla takes a pos0 so a chunk is not always arange(T) from 0" {
    const src = @embedFile("oracle/ling_forward.py");
    try std.testing.expect(std.mem.indexOf(u8, src, "def mla(self, x, p, pos0=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "def one_mla(") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "pos0 + torch.arange(T, dtype=torch.float64)") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "attends only over the rows it is given") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "kv_a_proj_with_mqa") != null);
}

test "mla dump runs ling_forward.py --mla-layer for TFLINGA1" {
    const src = @embedFile("oracle/mla_dump.py");
    try std.testing.expect(std.mem.indexOf(u8, src, "ling_forward.py") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "--mla-layer") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "TFLINGA1") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "Random values in Ling-3.0-tiny tensor shapes") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "sys.argv") != null);
}
