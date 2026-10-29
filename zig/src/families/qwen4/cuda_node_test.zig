//! A source-backed CUDA profile refuses the old normalized-BF16 probability boundary, without a GPU.
const std = @import("std");
const n = @import("../../../tests/qwen4/numeric_gpu.zig");
const f = @import("device_fixture.zig");
const t = std.testing;
fn read(_: *n.B, buffer: n.B.Buffer, out: []u8) !void {
    @memcpy(out, buffer[0..out.len]);
}
test "Qwen4 CUDA nodes: every two-row storage node matches and queued state has no first divergence" {
    var b: n.B = .{};
    const summary = try @import("node_gate.zig").run(n.G, read, t.allocator, std.testing.io, &b, n.Loader{}, false);
    try t.expectEqual(@as(usize, 0), summary.different);
    try t.expectEqual(@as(usize, 142), summary.checked);
    try t.expect(summary.first[0] == null and summary.first[1] == null);
    try t.expectEqual(@as(usize, 0), b.held);
}
test "Qwen4 CUDA nodes: the real probability-store and merge bodies preserve the reported counterexample" {
    const a = t.allocator;
    const old = try std.json.parseFromSlice(f.Oracle, a, f.legacy_reference, .{});
    defer old.deinit();
    const corrected = try std.json.parseFromSlice(f.Oracle, a, f.cuda_reference, .{});
    defer corrected.deinit();
    const columns = [_]usize{ 1, 2, 4, 5, 6, 7, 8, 10 };
    const words = [_]u16{ 0x3f7f, 0xc03d, 0xbfb6, 0x3e4a, 0xc018, 0xbe02, 0x3f6d, 0xbee2 };
    const bf = @import("core").quant.convert.bf16;
    for (columns, words) |col, want| {
        try t.expectEqual(want, bf(@floatCast(corrected.value.logits[1][col])));
        try t.expect(want != bf(@floatCast(old.value.logits[1][col])));
    }
    var different: usize = 0;
    for (old.value.logits, corrected.value.logits) |before, after| for (before, after) |x, y| {
        different += @intFromBool(bf(@floatCast(x)) != bf(@floatCast(y)));
    };
    try t.expectEqual(@as(usize, 219), different);
}

extern fn tf_q4_cuda_node_attention([*]const u16, [*]const u16, [*]const u16, [*]const i32, c_int, [*]u16, c_int) void;
test "Qwen4 CUDA nodes: normalized-before-BF16 control changes the actual source-backed result" {
    const quant = @import("core").quant.convert;
    const bf = quant.bf16;
    var q: [256]u16 = @splat(0);
    var k: [512]u16 = @splat(0);
    var v: [512]u16 = undefined;
    q[0] = bf(1);
    k[256] = bf(8);
    @memset(v[0..256], bf(-3));
    @memset(v[256..], bf(6));
    const before = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(&v));
    const ids = [_]i32{ 0, 1 };
    var output: [258]u16 = @splat(0x5aa5);
    var wrong: [256]u16 = undefined;
    const e: f64 = @exp(@as(f64, -0.5));
    const stored: f64 = @as(f32, @bitCast(@as(u32, bf(@floatCast(e))) << 16));
    const expected = bf(@floatCast((-3 * stored + 6) / (e + 1)));
    for (0..5) |_| {
        tf_q4_cuda_node_attention(&q, &k, &v, &ids, 2, output[1..257].ptr, 0);
        try t.expectEqual(@as(u16, 0x5aa5), output[0]);
        try t.expectEqual(@as(u16, 0x5aa5), output[257]);
        for (output[1..257]) |word| try t.expectEqual(expected, word);
    }
    tf_q4_cuda_node_attention(&q, &k, &v, &ids, 2, &wrong, 1);
    try t.expect(!std.mem.eql(u16, output[1..257], &wrong));
    try t.expectEqual(before, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(&v)));
}
