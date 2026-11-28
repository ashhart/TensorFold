//! Finite BF16 bounded top-k, following Qwen's integer ordering and canonical FP64 sampler.
const std = @import("std");
const lanes = @import("lanes");
const capacity = 128;
const Entry = struct { key: u16, word: u16, id: u64 };

fn key(word: u16) u16 {
    if (word & 0x7fff == 0) return 0x8000;
    return word ^ (if (word & 0x8000 == 0) @as(u16, 0x8000) else @as(u16, 0xffff));
}
fn value(word: u16) f64 {
    return @as(f32, @bitCast(@as(u32, word) << 16));
}

/// Null retains the general draw for greedy, unbounded or larger top-k requests.
pub fn draw(logits: []const u16, sampling: lanes.Sampling, position: u64, mask: []const u32) !?u32 {
    if (sampling.temperature <= 0 or sampling.top_k == 0 or sampling.top_k > capacity) return null;
    const k: usize = sampling.top_k;
    var selected: [capacity]Entry = undefined;
    var filled: usize = 0;
    for (logits, 0..) |word, id| {
        if (word & 0x7f80 == 0x7f80) return error.NonfiniteLfmLogits;
        if (mask.len > 0 and !lanes.grammar.allows(mask, id)) continue;
        const rank = key(word);
        if (filled == k and rank <= selected[k - 1].key) continue;
        var at = @min(filled, k - 1);
        while (at > 0 and rank > selected[at - 1].key) : (at -= 1) selected[at] = selected[at - 1];
        selected[at] = .{ .key = rank, .word = word, .id = id };
        filled = @min(filled + 1, k);
    }
    if (filled == 0) return error.NoAllowedLfmToken;
    var candidates: [capacity]lanes.sampling.Candidate = undefined;
    for (candidates[0..filled], selected[0..filled]) |*out, entry| out.* = .{ .value = value(entry.word), .id = entry.id };
    var words: [2 * capacity]f64 = undefined;
    var scratch = std.heap.FixedBufferAllocator.init(std.mem.sliceAsBytes(&words));
    return @as(u32, @intCast(try lanes.sampling.chooseOrdered(scratch.allocator(), candidates[0..filled], position, sampling)));
}

test "bounded draws keep the lowest id across signed zeros and reject hidden nonfinite logits" {
    const gpa = std.testing.allocator;
    var logits = [_]u16{ 0x8000, 0x0000, 0xbf80, 0xc000 };
    const values = [_]f64{ -0.0, 0.0, -1.0, -2.0 };
    const ids = [_]u64{ 0, 1, 2, 3 };
    for ([_]u32{ 1, 2 }) |k| {
        const sampling: lanes.Sampling = .{ .seed = 1234, .temperature = 1, .top_k = k };
        const want = try lanes.sampling.choose(gpa, &values, &ids, 128, sampling);
        try std.testing.expectEqual(want, (try draw(&logits, sampling, 128, &.{})).?);
    }
    logits[3] = 0x7f80;
    try std.testing.expectError(error.NonfiniteLfmLogits, draw(&logits, .{ .seed = 1234, .temperature = 1, .top_k = 1 }, 0, &.{1}));
}
