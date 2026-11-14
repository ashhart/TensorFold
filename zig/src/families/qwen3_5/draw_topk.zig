//! A finite BF16 row's bounded top-k, ordered by value then id, with scratch owned by the draw.
const std = @import("std");
const lanes = @import("lanes");
const capacity = 128;
const Entry = struct { key: u16, word: u16, id: u64 };

const row_draw = lanes.sampling_row;
const key = row_draw.key;
/// One mask word's logits: a vector compare skips 32 that are not allowed or cannot enter a full selection.
const block = 32;
const Block = @Vector(block, u16);

/// Inserts a finite allowed word that beats the k-th key, after equal keys so ties keep id order.
inline fn consider(selected: *[capacity]Entry, filled: *usize, k: usize, word: u16, id: usize, mask: []const u32) void {
    if (mask.len > 0 and !lanes.grammar.allows(mask, id)) return;
    const rank = key(word);
    if (filled.* == k and rank <= selected[k - 1].key) return;
    var at = @min(filled.*, k - 1);
    while (at > 0 and rank > selected[at - 1].key) : (at -= 1) selected[at] = selected[at - 1];
    selected[at] = .{ .key = rank, .word = word, .id = id };
    filled.* = @min(filled.* + 1, k);
}

fn value(word: u16) f64 {
    return @as(f32, @bitCast(@as(u32, word) << 16));
}

/// The k best allowed words of a finite row, ordered by key then id; skips blocks that allow nothing or can't win.
fn select(logits: []const u16, k: usize, mask: []const u32, selected: *[capacity]Entry) error{NonfiniteQwenLogits}!usize {
    var filled: usize = 0;
    var at: usize = 0;
    while (at + block <= logits.len) : (at += block) {
        const row: Block = logits[at..][0..block].*;
        if (@reduce(.Or, row & @as(Block, @splat(0x7f80)) == @as(Block, @splat(0x7f80)))) return error.NonfiniteQwenLogits;
        const bits = row_draw.maskWord(mask, at / block);
        if (bits == 0) continue;
        var ranks = row_draw.keys(block, row);
        if (~bits != 0) ranks = @select(u16, row_draw.spread(block, bits), ranks, @as(Block, @splat(0)));
        if (filled == k and @reduce(.Max, ranks) <= selected[k - 1].key) continue;
        var left = bits;
        while (left != 0) : (left &= left - 1) consider(selected, &filled, k, logits[at + @ctz(left)], at + @ctz(left), mask);
    }
    for (logits[at..], at..) |word, id| {
        if (word & 0x7f80 == 0x7f80) return error.NonfiniteQwenLogits;
        consider(selected, &filled, k, word, id, mask);
    }
    return filled;
}

/// Null retains the general draw for unbounded or larger top-k and greedy requests.
pub fn draw(logits: []const u16, sampling: lanes.Sampling, position: u64, mask: []const u32) !?u32 {
    if (sampling.temperature <= 0 or sampling.top_k == 0 or sampling.top_k > capacity) return null;
    var selected: [capacity]Entry = undefined;
    // Validate the whole row, including masked tokens and candidates below the cut.
    const filled = try select(logits, sampling.top_k, mask, &selected);
    if (filled == 0) return error.NoAllowedQwenToken;
    var candidates: [capacity]lanes.sampling.Candidate = undefined;
    for (candidates[0..filled], selected[0..filled]) |*out, entry| out.* = .{ .value = value(entry.word), .id = entry.id };
    // The canonical FP64 sampler needs only its scaled values and optional probabilities for this ordered cut.
    var words: [2 * capacity]f64 = undefined;
    var scratch = std.heap.FixedBufferAllocator.init(std.mem.sliceAsBytes(&words));
    return @as(u32, @intCast(try lanes.sampling.chooseOrdered(scratch.allocator(), candidates[0..filled], position, sampling)));
}

/// The scalar walk `select` replaces: every word validated, then inserted after its equal keys.
fn selectScalar(logits: []const u16, k: usize, mask: []const u32, selected: *[capacity]Entry) !usize {
    var filled: usize = 0;
    for (logits, 0..) |word, id| {
        if (word & 0x7f80 == 0x7f80) return error.NonfiniteQwenLogits;
        consider(selected, &filled, k, word, id, mask);
    }
    return filled;
}

test "the blocked selection equals the scalar walk over ties, zeros, masks, tails and non-finite words" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const r = prng.random();
    const words = try gpa.alloc(u16, 4099);
    defer gpa.free(words);
    var mask: [(4099 + 31) / 32]u32 = undefined;
    for (0..400) |case| {
        const n = 1 + r.uintLessThan(usize, words.len);
        const narrow = case % 2 == 1; // eight adjacent words: k-th keys and block maxima often differ by one
        for (words[0..n]) |*w| w.* = if (narrow) 0x3f80 + r.uintLessThan(u16, 8) else switch (r.uintLessThan(u8, 8)) {
            0 => 0, // +0 and -0 share a key
            1 => 0x8000,
            2 => 0x4000 | r.uintLessThan(u16, 4), // a few tied values
            else => r.int(u16) & ~@as(u16, 0x0080), // never all-ones exponents here
        };
        if (case % 3 == 0) for (0..n / 64 + 1) |_| { // big values late in the row, so early blocks are full and later skipped
            words[n - 1 - r.uintLessThan(usize, @min(n, 64))] = 0x4700;
        };
        const shape = (case / 2) % 8; // no mask, dense, mostly zero words, one id, none allowed, short, dense, no mask
        const zero = shape == 3 or shape == 4;
        for (&mask) |*m| m.* = if (zero or shape == 2 and r.uintLessThan(u8, 8) != 0) 0 else r.int(u32);
        if (shape == 3) mask[r.uintLessThan(usize, n) / 32] = @as(u32, 1) << @intCast(r.uintLessThan(usize, 32));
        const width = if (shape == 5) 1 + r.uintLessThan(usize, (n + 31) / 32) else (n + 31) / 32;
        const m: []const u32 = if (shape == 0 or shape == 7) &.{} else mask[0..width];
        if (case % 5 == 2) words[r.uintLessThan(usize, n)] = if (case % 2 == 0) 0x7fc0 else 0xff80; // NaN or -inf
        const k = 1 + r.uintLessThan(usize, capacity);
        var a: [capacity]Entry = undefined;
        var b: [capacity]Entry = undefined;
        const want = selectScalar(words[0..n], k, m, &a);
        const got = select(words[0..n], k, m, &b);
        if (want) |filled| {
            try std.testing.expectEqual(filled, try got);
            for (a[0..filled], b[0..filled]) |x, y| {
                try std.testing.expectEqual(x.id, y.id);
                try std.testing.expectEqual(x.word, y.word);
            }
        } else |err| try std.testing.expectError(err, got);
    }
}

test "a non-finite word inside a block the selection skips is still refused" {
    var words: [96]u16 = @splat(0x3f80); // 1.0
    for (words[0..32]) |*w| w.* = 0x4000; // 2.0 fills k = 4 in the first block
    words[70] = 0xff80; // -inf in the third block, whose largest key is below the selection's
    var selected: [capacity]Entry = undefined;
    try std.testing.expectError(error.NonfiniteQwenLogits, select(&words, 4, &.{}, &selected));
}
