const std = @import("std");
const weights = @import("cuda_prompt_wt.zig");

pub fn failedFill(comptime makeBuffers: anytype) !void {
    const Fake = struct {
        live: *usize,
        pub fn free(buffer: *@This()) void {
            buffer.live.* -= 1;
        }
    };
    const Src = struct {
        live: *usize,
        fills: *usize,
        fail_at: usize,
        pub fn make(source: @This()) !Fake {
            source.live.* += 1;
            return .{ .live = source.live };
        }
        pub fn fill(source: @This(), _: *Fake) !void {
            source.fills.* += 1;
            if (source.fills.* == source.fail_at) return error.FillFailed;
        }
    };
    for ([_]usize{ 1, 2, 4 }) |fail_at| {
        var live: usize = 0;
        var fills: usize = 0;
        try std.testing.expectError(error.FillFailed, makeBuffers(Fake, std.testing.allocator, 4, Src{ .live = &live, .fills = &fills, .fail_at = fail_at }));
        try std.testing.expectEqual(@as(usize, 0), live);
    }
    var live: usize = 0;
    var fills: usize = 0;
    const all = try makeBuffers(Fake, std.testing.allocator, 3, Src{ .live = &live, .fills = &fills, .fail_at = 0 });
    defer std.testing.allocator.free(all);
    try std.testing.expectEqual(@as(usize, 3), live);
}

pub fn layerIndices() !void {
    try std.testing.expect(weights.linearLayer(0));
    try std.testing.expect(!weights.linearLayer(3));
    try std.testing.expect(!weights.linearLayer(47));
    try std.testing.expectEqual(@as(usize, 0), weights.linearIndex(0));
    try std.testing.expectEqual(@as(usize, 2), weights.linearIndex(2));
    try std.testing.expectEqual(@as(usize, 35), weights.linearIndex(46));
    try std.testing.expectEqual(@as(usize, 0), weights.attnIndex(3));
    try std.testing.expectEqual(@as(usize, 11), weights.attnIndex(47));
}

pub fn sequenceBytes(comptime sequenceSize: anytype) !void {
    const per_token = weights.attn_n * (2 * weights.kv_heads * weights.head_dim * 2 + weights.index_dim * 2);
    try std.testing.expectEqual(per_token, sequenceSize(6) - sequenceSize(5));
    try std.testing.expectEqual(per_token + weights.attn_n * weights.index_dim * 2, sequenceSize(5) - sequenceSize(4));
    try std.testing.expect(sequenceSize(weights.window) > weights.window * per_token);
}
