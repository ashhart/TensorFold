//! A finite BF16 row's bounded top-k, ordered by value then id, with scratch owned by the draw.
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

/// Null retains the general draw for unbounded or larger top-k and greedy requests.
pub fn draw(logits: []const u16, sampling: lanes.Sampling, position: u64, mask: []const u32) !?u32 {
    if (sampling.temperature <= 0 or sampling.top_k == 0 or sampling.top_k > capacity) return null;
    const k: usize = sampling.top_k;
    var selected: [capacity]Entry = undefined;
    var filled: usize = 0;
    for (logits, 0..) |word, id| {
        // Validate the whole row, including masked tokens and candidates below the cut.
        if (word & 0x7f80 == 0x7f80) return error.NonfiniteQwenLogits;
        if (mask.len > 0 and !lanes.grammar.allows(mask, id)) continue;
        const rank = key(word);
        if (filled == k and rank <= selected[k - 1].key) continue;
        var at = @min(filled, k - 1);
        while (at > 0 and rank > selected[at - 1].key) : (at -= 1) selected[at] = selected[at - 1];
        selected[at] = .{ .key = rank, .word = word, .id = id };
        filled = @min(filled + 1, k);
    }
    if (filled == 0) return error.NoAllowedQwenToken;
    var candidates: [capacity]lanes.sampling.Candidate = undefined;
    for (candidates[0..filled], selected[0..filled]) |*out, entry| out.* = .{ .value = value(entry.word), .id = entry.id };
    // The canonical FP64 sampler needs only its scaled values and optional probabilities for this ordered cut.
    var words: [2 * capacity]f64 = undefined;
    var scratch = std.heap.FixedBufferAllocator.init(std.mem.sliceAsBytes(&words));
    return @as(u32, @intCast(try lanes.sampling.chooseOrdered(scratch.allocator(), candidates[0..filled], position, sampling)));
}

test "bounded BF16 keys preserve numeric order over every finite word and both zero encodings" {
    for (0..65536) |bits| {
        const a: u16 = @intCast(bits);
        if (a & 0x7f80 == 0x7f80) continue;
        for ([_]u16{ a ^ 0x8000, a +% 1, 0, 0x8000 }) |b| {
            if (b & 0x7f80 == 0x7f80) continue;
            try std.testing.expectEqual(value(a) < value(b), key(a) < key(b));
            try std.testing.expectEqual(value(a) == value(b), key(a) == key(b));
        }
    }
}
