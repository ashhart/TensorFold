//! Host-only source and lifecycle checks for the real chain contract; no Metal operation is called.
const std = @import("std");
const tf = @import("tensorfold");
const g = @import("contract").gates;
const chain = @import("tests/contract/metal_q27_chain.zig");
const t = std.testing;

test "q27 contract: both chain libraries contain the served pre-M5 prompt shader" {
    for ([_]bool{ false, true }) |baseline| for ([_]bool{ false, true }) |units| {
        const text = try chain.source(t.allocator, units, baseline, false, .q27b);
        defer t.allocator.free(text);
        const symbol = if (units) "Q27_PROMPT(8, false," else "Q27_PROMPT_SG(8, false,";
        try t.expect(std.mem.indexOf(u8, text, symbol) != null);
        try t.expect(std.mem.indexOf(u8, text, "q27_attn_prompt_merge_single") != null);
    };
    const text = try tf.qwen27.simd_attention.library(t.allocator, false, true);
    defer t.allocator.free(text);
    try t.expect(std.mem.indexOf(u8, text, "Q27_PROMPT_SG(8, false,") != null);
}

noinline fn fromAnotherFrame(x: *chain.Case) g.Kernel {
    return x.kernel();
}

test "q27 contract: returned hooks survive the registering stack frame" {
    var x: chain.Case = undefined;
    x.reg = .{ .heads = .q27b };
    x.shape = .{ .rows = 3, .n = 24, .k = 0 };
    const k = fromAnotherFrame(&x);
    const expected = comptime g.hooks(chain.Case);
    try t.expectEqualDeep(expected, k.vt.*);
    try t.expectEqual(@as(usize, 3 * 24 * 256), k.vt.count(k.ctx));
}

test "q27 contract: missing prompt pipelines become named gate failures without Metal calls" {
    var x: chain.Case = undefined;
    x.rig = .{ .a = t.allocator, .device = undefined, .queue = undefined };
    defer x.rig.deinit();
    x.reg = .{ .heads = .q27b };
    x.op.prompt = null;
    x.old.prompt = null;
    const shape = @import("contract").spec.Shape{ .rows = 16, .n = 24, .k = 0 };
    try t.expectError(error.Qwen27ChainPromptUnavailable, x.prepare(shape, .random, 1));
    var tally: g.Tally = .{};
    try g.runAll(t.allocator, &.{x.kernel()}, 1, &tally);
    try t.expectEqual(@as(usize, 1), tally.kernels);
    try t.expectEqual(@as(usize, 1), tally.failed);
    try t.expectEqual(@as(usize, 0), tally.passed);
}
