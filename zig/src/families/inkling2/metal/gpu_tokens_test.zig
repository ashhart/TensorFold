//! Host-only invariants for Inkling 2 Mini Metal's GPU sampling dispatch, scratch layout, and flight metadata.
const std = @import("std");
const builtin = @import("builtin");
const lanes = @import("lanes");
const fwd = @import("forward.zig");
const st = @import("state.zig");
const flight = @import("flight.zig");
const backend = @import("backend.zig");

test {
    if (builtin.os.tag.isDarwin()) _ = @import("gpu_tokens_gpu_test.zig");
}

test "drawMode splits greedy, two-pass top-k (1..64), rule_1024 (65..1024), and full-vocabulary draws" {
    const dummy: fwd.Ref = .{ .buf = .{ .id = @ptrFromInt(16) }, .off = 0 };
    const vocab: u32 = st.vocabulary;
    try std.testing.expectEqual(fwd.DrawMode.greedy, fwd.drawMode(.{ .how = .greedy, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.full, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 0 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.topk, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 1 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.topk, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 0.7, .top_k = 20 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.topk, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 64 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.rule_1024, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 65 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.rule_1024, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 128 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.rule_1024, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 1024 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.full, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 1025 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.full, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 2048 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.rule_1024, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 20 }, .position = 1 } }, .out = dummy }, 16));
}

test "flight metadata stride and offsets respect Metal alignment rules" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(flight.FlightMeta));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(flight.FlightMeta, "pos") % 4);
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(flight.FlightMeta, "seg_of") % 4);
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(flight.FlightMeta, "seg") % 8);
}

test "scratch reserves picks and two-pass top-k buffers and Metal defaults gpu_tokens off for TTFT safety" {
    try std.testing.expect(st.scratchBytes(st.batch_rows) > st.scratchBytes(1));
    var m: backend.Metal = undefined;
    m.options = .{ .capacity = 16, .budget = 0 };
    m.timed = 0;
    m.shared_timed = 0;
    try std.testing.expect(!m.facts().gpu_tokens);
    m.options.gpu_tokens = true;
    try std.testing.expect(m.facts().gpu_tokens);
}
