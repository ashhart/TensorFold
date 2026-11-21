//! A finite BF16 row's bounded top-k, ordered by value then id, with scratch owned by the draw.
const std = @import("std");
const sampling = @import("sampling.zig");
const grammar = @import("grammar.zig");
const capacity = 128;
const Entry = struct { key: u16, word: u16, id: u64 };

fn key(word: u16) u16 {
    if (word & 0x7fff == 0) return 0x8000;
    return word ^ (if (word & 0x8000 == 0) @as(u16, 0x8000) else @as(u16, 0xffff));
}
/// Words a block: one vector compare skips 32 logits that cannot enter a full selection.
const block = 32;
const Block = @Vector(block, u16);

/// `key` over a block of words.
fn keys(words: Block) Block {
    const negative = words & @as(Block, @splat(0x8000)) != @as(Block, @splat(0));
    const flipped = words ^ @select(u16, negative, @as(Block, @splat(0xffff)), @as(Block, @splat(0x8000)));
    return @select(u16, words & @as(Block, @splat(0x7fff)) == @as(Block, @splat(0)), @as(Block, @splat(0x8000)), flipped);
}

/// Inserts a finite allowed word that beats the k-th key, after equal keys so ties keep id order.
inline fn consider(selected: *[capacity]Entry, filled: *usize, k: usize, word: u16, id: usize, mask: []const u32) void {
    if (mask.len > 0 and !grammar.allows(mask, id)) return;
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

/// The k best allowed words of a finite row, ordered by key then id; a block that can't beat the k-th key is skipped.
fn select(logits: []const u16, k: usize, mask: []const u32, selected: *[capacity]Entry) error{NonfiniteLogits}!usize {
    var filled: usize = 0;
    var at: usize = 0;
    while (at + block <= logits.len) : (at += block) {
        const row: Block = logits[at..][0..block].*;
        if (@reduce(.Or, row & @as(Block, @splat(0x7f80)) == @as(Block, @splat(0x7f80)))) return error.NonfiniteLogits;
        if (filled == k and @reduce(.Max, keys(row)) <= selected[k - 1].key) continue;
        for (logits[at..][0..block], at..) |word, id| consider(selected, &filled, k, word, id, mask);
    }
    for (logits[at..], at..) |word, id| {
        if (word & 0x7f80 == 0x7f80) return error.NonfiniteLogits;
        consider(selected, &filled, k, word, id, mask);
    }
    return filled;
}

/// Null retains the general draw for unbounded or larger top-k and greedy requests.
pub fn draw(logits: []const u16, settings: sampling.Sampling, position: u64, mask: []const u32) !?u32 {
    if (settings.temperature <= 0 or settings.top_k == 0 or settings.top_k > capacity) return null;
    var selected: [capacity]Entry = undefined;
    // Validate the whole row, including masked tokens and candidates below the cut.
    const filled = try select(logits, settings.top_k, mask, &selected);
    if (filled == 0) return error.NoAllowedToken;
    var candidates: [capacity]sampling.Candidate = undefined;
    for (candidates[0..filled], selected[0..filled]) |*out, entry| out.* = .{ .value = value(entry.word), .id = entry.id };
    // The canonical FP64 sampler needs only its scaled values and optional probabilities for this ordered cut.
    var words: [2 * capacity]f64 = undefined;
    var scratch = std.heap.FixedBufferAllocator.init(std.mem.sliceAsBytes(&words));
    const chosen = try sampling.chooseOrdered(scratch.allocator(), candidates[0..filled], position, settings);
    return @as(u32, @intCast(chosen));
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

/// The scalar walk `select` replaces: every word validated, then inserted after its equal keys.
fn selectScalar(logits: []const u16, k: usize, mask: []const u32, selected: *[capacity]Entry) !usize {
    var filled: usize = 0;
    for (logits, 0..) |word, id| {
        if (word & 0x7f80 == 0x7f80) return error.NonfiniteLogits;
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
        const masked = case % 4 == 1;
        for (&mask) |*m| m.* = r.int(u32);
        const m: []const u32 = if (masked) mask[0 .. (n + 31) / 32] else &.{};
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
    try std.testing.expectError(error.NonfiniteLogits, select(&words, 4, &.{}, &selected));
}
