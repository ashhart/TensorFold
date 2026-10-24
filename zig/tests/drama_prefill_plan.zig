//! Drama 3's CUDA prompt prefill plan: the default single row and the opt-in batched plan.
const std = @import("std");
const d3 = @import("drama3");
const host = @import("drama_host");
const tool = @import("drama_generate");

test "the default plan is the single row the stock path used" {
    const t = std.testing;
    // the serving host's default: one row a chunk, the arithmetic the single-row path used
    try t.expectEqual(@as(u32, 1), (host.Options{}).prefill_rows);
    const serve: d3.common.Options = .{
        .max_rows = @intCast((host.Options{}).prefill_rows),
        .attn_chunk = @intCast((host.Options{}).prefill_rows),
    };
    try serve.validatePrefill();
    for ([_]usize{ 1, 2, 127, 4096 }) |n| try t.expectEqual(@as(usize, 1), serve.promptRows(n));
    // the generation tool's default: the same single row
    try t.expectEqual(@as(usize, 1), (tool.Args{}).opts.max_rows);
    try t.expectEqual(@as(usize, 1), (tool.Args{}).opts.attn_chunk);
    for ([_]usize{ 1, 2, 127, 4096 }) |n| try t.expectEqual(@as(usize, 1), (tool.Args{}).opts.promptRows(n));
}

test "an opt-in capacity plans N-row chunks and a partial tail" {
    const t = std.testing;
    const total: usize = 4096;
    for ([_]usize{ 2, 64, 128, 1024 }) |n| {
        const o: d3.common.Options = .{ .max_rows = n, .attn_chunk = @min(n, 1024) };
        try o.validatePrefill();
        var at: usize = 0;
        var rounds: usize = 0;
        while (at < total) {
            const count = o.promptRows(total - at);
            try t.expect(count > 0 and count <= n and count <= total - at);
            at += count;
            rounds += 1;
        }
        try t.expectEqual(total, at);
        try t.expectEqual((total + n - 1) / n, rounds);
        try t.expectEqual(@as(usize, 0), o.promptRows(total - at));
    }
}
