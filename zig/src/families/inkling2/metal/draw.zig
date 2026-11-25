//! Host logit draws, grammar masks and logprobs for Inkling 2 Mini, bit-identical to the Metal reference.
const std = @import("std");
const lanes = @import("lanes");

const topk_capacity = 128;
const topk_block = 16;

pub fn minMaskWords(vocab_len: usize) usize {
    return (vocab_len + 31) / 32;
}

pub fn maskStride(masks_len: usize, rows: usize, unpadded_vocab: usize) !usize {
    if (masks_len == 0) return 0;
    if (rows == 0 or masks_len % rows != 0) return error.InvalidInklingMask;
    const words = masks_len / rows;
    if (words < minMaskWords(unpadded_vocab)) return error.InvalidInklingMask;
    return words;
}

/// Row `r` of a window's masks, or none where the window has none.
pub fn maskRow(masks: []const u32, words: usize, r: usize) []const u32 {
    return if (masks.len == 0) &.{} else masks[r * words ..][0..words];
}

/// First index of the greatest logit, matching the scalar scan: a NaN is chosen only at index 0.
pub fn greedyArgmax(l: []const f32) u32 {
    std.debug.assert(l.len > 0);
    if (l[0] != l[0]) return 0;
    const V = @Vector(16, f32);
    const least: V = @splat(-std.math.inf(f32));
    var top = least;
    var i: usize = 0;
    while (i + 16 <= l.len) : (i += 16) {
        const v: V = l[i..][0..16].*;
        top = @max(top, @select(f32, v != v, least, v));
    }
    var best = @reduce(.Max, top);
    for (l[i..]) |w| if (w > best) {
        best = w;
    };
    const want: V = @splat(best);
    i = 0;
    while (i + 16 <= l.len) : (i += 16) {
        const v: V = l[i..][0..16].*;
        if (@reduce(.Or, v == want)) break;
    }
    for (l[i..], i..) |w, id| if (w == best) return @intCast(id);
    return 0;
}

pub fn logprobRow(l: []const f32, pick: u32, count: u8) lanes.LogprobRow {
    const top = lanes.logprob.max_top;
    var most: f64 = -std.math.inf(f64);
    for (l) |w| most = @max(most, w);
    var sum: f64 = 0;
    for (l) |w| sum += @exp(@as(f64, w) - most);
    const lse: f32 = @floatCast(most + @log(sum));
    var r: lanes.LogprobRow = .{ .token = pick, .logprob = l[pick] - lse, .count = 0 };
    var filled: usize = 0;
    for (l, 0..) |w, id| {
        if (filled == top and w <= r.logprobs[top - 1] + lse) continue;
        var at = @min(filled, top - 1);
        while (at > 0 and w > r.logprobs[at - 1] + lse) : (at -= 1) {
            r.ids[at] = r.ids[at - 1];
            r.logprobs[at] = r.logprobs[at - 1];
        }
        r.ids[at] = @intCast(id);
        r.logprobs[at] = w - lse;
        filled = @min(filled + 1, top);
    }
    r.count = @min(count, top);
    return r;
}

pub fn firstLogprobRow(l: []const f32, tokens: []const u32, next: u64, count: u8) lanes.LogprobRow {
    std.debug.assert(tokens.len > 0 and next > 0);
    return logprobRow(l, tokens[@intCast((next - 1) % tokens.len)], count);
}

const Entry = struct { key: u32, v: f32, i: u32 };

inline fn consider(selected: *[topk_capacity]Entry, filled: *usize, k: usize, rank: u32, v: f32, id: u32) void {
    if (filled.* == k and rank <= selected[k - 1].key) return;
    var at = @min(filled.*, k - 1);
    while (at > 0 and rank > selected[at - 1].key) : (at -= 1) selected[at] = selected[at - 1];
    selected[at] = .{ .key = rank, .v = v, .i = id };
    filled.* = @min(filled.* + 1, k);
}

/// Heap-free SIMD top-k selection matching lanes.gpu_rule.sample for finite rows with 1 <= top_k <= 128.
fn fastTopKDraw(logits: []const f32, s: lanes.Sampling, position: u32) ?u32 {
    if (s.temperature <= 0 or s.top_k == 0 or s.top_k > topk_capacity or logits.len == 0) return null;
    const FVec = @Vector(topk_block, f32);
    const UVec = @Vector(topk_block, u32);
    const r = lanes.gpu_rule.rule(s);
    const k: usize = @min(logits.len, r.top_k);
    const inv_vec: FVec = @splat(r.inv_t);
    const sign: UVec = @splat(0x8000_0000);
    const ones: UVec = @splat(0xFFFF_FFFF);
    const abs_mask: UVec = @splat(0x7FFF_FFFF);
    const inf_bits: UVec = @splat(0x7F80_0000);
    const zero: UVec = @splat(0);
    var selected: [topk_capacity]Entry = undefined;
    var filled: usize = 0;
    var at: usize = 0;
    while (at + topk_block <= logits.len) : (at += topk_block) {
        const raw: FVec = logits[at..][0..topk_block].*;
        const scaled = raw * inv_vec;
        const bits: UVec = @bitCast(scaled);
        if (@reduce(.Or, (bits & abs_mask) > inf_bits)) return null;
        const neg = (bits & sign) != zero;
        const ks = bits ^ @select(u32, neg, ones, sign);
        if (filled == k and @reduce(.Max, ks) <= selected[k - 1].key) continue;
        const vals: [topk_block]f32 = scaled;
        const ranks: [topk_block]u32 = ks;
        for (vals, ranks, 0..) |v, rank, lane| consider(&selected, &filled, k, rank, v, @intCast(at + lane));
    }
    for (logits[at..], at..) |l, id| {
        const v = l * r.inv_t;
        if (std.math.isNan(v)) return null;
        consider(&selected, &filled, k, lanes.gpu_rule.key(v), v, @intCast(id));
    }
    const m = selected[0].v;
    var norm: f32 = 0.0;
    for (selected[0..filled]) |c| norm += @exp(c.v - m);
    var keep: usize = filled;
    if (r.top_p > 0.0 and r.top_p < 1.0) {
        var cum: f32 = 0.0;
        for (selected[0..filled], 0..) |c, j| {
            cum += @exp(c.v - m) / norm;
            if (cum >= r.top_p) {
                keep = j + 1;
                break;
            }
        }
    }
    if (r.min_log > -std.math.inf(f32)) {
        const floor = m + r.min_log;
        var j: usize = 0;
        while (j < keep and selected[j].v >= floor) j += 1;
        keep = j;
    }
    var best: usize = 0;
    var best_score = -std.math.inf(f32);
    for (selected[0..keep], 0..) |c, t| {
        const score = c.v - @log(-@log(lanes.gpu_rule.uniform24(s.seed, position, c.i)));
        if (t == 0 or score > best_score) {
            best = t;
            best_score = score;
        }
    }
    return selected[best].i;
}

/// Host draw matching greedyArgmax or lanes.gpu_full.draw over finite, masked, and nonfinite rows.
pub fn draw(gpa: std.mem.Allocator, l: []const f32, scratch: []f32, sampling: ?lanes.Sampling, position: u64, mask: []const u32) !u32 {
    const t = if (sampling) |s| (if (s.temperature > 0) s else null) else null;
    if (t == null) {
        if (mask.len == 0) return greedyArgmax(l);
        var best: ?usize = null;
        for (l, 0..) |w, id| {
            if (!lanes.grammar.allows(mask, id)) continue;
            if (best == null or w > l[best.?]) best = id;
        }
        return @intCast(best orelse return error.InklingEmptyTokenMask);
    }
    const values = if (mask.len == 0) l else blk: {
        for (l, scratch, 0..) |w, *v, id| v.* = if (lanes.grammar.allows(mask, id)) w else -std.math.inf(f32);
        break :blk scratch;
    };
    if (fastTopKDraw(values, t.?, @intCast(position))) |token| return token;
    return lanes.gpu_full.draw(gpa, values, t.?, @intCast(position), null);
}

test "a greedy pick is the scan's: the first of equal maxima, and a NaN only at index 0" {
    const scan = struct {
        fn pick(l: []const f32) u32 {
            var best: usize = 0;
            for (l, 0..) |w, id| if (w > l[best]) {
                best = id;
            };
            return @intCast(best);
        }
    }.pick;
    var l: [203]f32 = undefined;
    for (0..64) |round| {
        for (&l, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 37 + round * 11) % 101)) - 50);
        l[(round * 7) % l.len] = 60;
        l[(round * 13 + 5) % l.len] = 60;
        if (round % 3 == 0) l[(round * 5 + 1) % l.len] = std.math.nan(f32);
        if (round % 4 == 0) l[(round * 17 + 2) % l.len] = std.math.inf(f32);
        if (round % 5 == 0) l[(round * 3) % l.len] = -std.math.inf(f32);
        if (round % 7 == 0) l[0] = std.math.nan(f32);
        try std.testing.expectEqual(scan(&l), greedyArgmax(&l));
        try std.testing.expectEqual(scan(l[0..17]), greedyArgmax(l[0..17]));
    }
    var z = [_]f32{ -0.0, 0.0, -std.math.inf(f32) };
    try std.testing.expectEqual(@as(u32, 0), greedyArgmax(&z));
    z = .{ -std.math.inf(f32), std.math.nan(f32), -std.math.inf(f32) };
    try std.testing.expectEqual(@as(u32, 0), greedyArgmax(&z));
}

/// A grammar mask's words a row: a bit a token of the tokenizer's vocabulary, padding included.
pub fn maskWords(vocab: u32) usize {
    return minMaskWords(vocab);
}

/// Row `r` of a window's masks, or none where the window has none.
pub fn maskRow(masks: []const u32, words: usize, r: usize) []const u32 {
    return if (masks.len == 0) &.{} else masks[r * words ..][0..words];
}

/// The token behind `handle`: only the last `ring` handles drawn are still held.
pub fn ringToken(tokens: []const u32, next: u64, handle: u64, ring_size: usize) !u32 {
    if (handle >= next or next - handle > ring_size) return error.InvalidInklingTokenHandle;
    return tokens[handle % ring_size];
}

pub fn firstLogprobRow(l: []const f32, tokens: []const u32, next: usize, count: u8, ring_size: usize) lanes.LogprobRow {
    std.debug.assert(tokens.len == ring_size and next > 0);
    return logprobRow(l, tokens[(next - 1) % ring_size], count);
}

test "a window's masks step by the padded vocabulary's words" {
    const words = maskWords(201024);
    try std.testing.expectEqual(@as(usize, 6282), words);
    var masks: [3 * 6282]u32 = @splat(0);
    masks[2 * 6282] = 1;
    try std.testing.expectEqual(@as(usize, 0), maskRow(&.{}, words, 2).len);
    try std.testing.expectEqual(@as(u32, 0), maskRow(&masks, words, 1)[0]);
    try std.testing.expect(lanes.grammar.allows(maskRow(&masks, words, 2), 0));
    try std.testing.expect(!lanes.grammar.allows(maskRow(&masks, words, 1), 0));
}

test "sampled first-token logprob uses the last written handle, not argmax" {
    const logits = [_]f32{ 0, -1, -2, -3 };
    var tokens: [4096]u32 = @splat(0);
    tokens[2] = 3;
    const row = firstLogprobRow(&logits, &tokens, 3, 0, 4096);
    try std.testing.expectEqual(@as(u32, 3), row.token);
    try std.testing.expect(std.math.isFinite(row.logprob));
    const lse = @log(@as(f64, 1) + @exp(-@as(f64, 1)) + @exp(-@as(f64, 2)) + @exp(-@as(f64, 3)));
    try std.testing.expectApproxEqAbs(@as(f32, -3) - @as(f32, @floatCast(lse)), row.logprob, 1e-6);
}

test "a token handle expires once the ring has wrapped past it" {
    var tokens: [4096]u32 = @splat(0);
    tokens[0] = 7;
    try std.testing.expectEqual(@as(u32, 7), try ringToken(&tokens, 4096, 0, 4096));
    try std.testing.expectError(error.InvalidInklingTokenHandle, ringToken(&tokens, 4097, 0, 4096));
    try std.testing.expectEqual(@as(u32, 7), try ringToken(&tokens, 4097, 4096, 4096));
    try std.testing.expectError(error.InvalidInklingTokenHandle, ringToken(&tokens, 4096, 4096, 4096));
}

test "fastTopKDraw matches lanes.gpu_full.draw bit-for-bit without heap allocation" {
    const gpa = std.testing.allocator;
    const n = 4099;
    const logits = try gpa.alloc(f32, n);
    defer gpa.free(logits);
    const scratch = try gpa.alloc(f32, n);
    defer gpa.free(scratch);
    var mask: [(n + 31) / 32]u32 = @splat(0);
    for (logits, 0..) |*l, i| {
        l.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 37) % 199)) - 99)) / 16.0;
        if (i % 7 == 3 or i % 251 == 0) mask[i / 32] |= @as(u32, 1) << @intCast(i % 32);
    }
    logits[0] = -0.0;
    logits[1] = 0.0;
    logits[2] = 0.0;
    logits[3] = -0.0;
    logits[n - 3] = 5.0;
    logits[n - 2] = 5.0;
    logits[n - 1] = -std.math.inf(f32);
    var failing = std.testing.FailingAllocator.init(gpa, .{ .fail_index = 0 });
    for ([_]u32{ 1, 2, 20, 64, 128, 129, 0 }) |k| {
        for ([_]f64{ 0.0, 0.5, 0.95, 1.0 }) |top_p| for ([_]f64{ 0.0, 0.05 }) |min_p| {
            const s: lanes.Sampling = .{ .seed = 0x1234_5678_9abc_def0, .temperature = 0.85, .top_k = k, .top_p = top_p, .min_p = min_p };
            const want = try lanes.gpu_full.draw(gpa, logits, s, 42, null);
            const alloc = if (k >= 1 and k <= 128) failing.allocator() else gpa;
            try std.testing.expectEqual(want, try draw(alloc, logits, scratch, s, 42, &.{}));
            for (logits, scratch, 0..) |w, *v, id| v.* = if (lanes.grammar.allows(&mask, id)) w else -std.math.inf(f32);
            const want_m = try lanes.gpu_full.draw(gpa, scratch, s, 99, null);
            try std.testing.expectEqual(want_m, try draw(gpa, logits, scratch, s, 99, &mask));
        };
    }
    try std.testing.expectEqual(@as(usize, 0), failing.alloc_index);
    var bad = try gpa.dupe(f32, logits);
    defer gpa.free(bad);
    bad[500] = std.math.inf(f32);
    const s_top20: lanes.Sampling = .{ .seed = 7, .temperature = 1.0, .top_k = 20, .top_p = 0.95 };
    try std.testing.expectEqual(try lanes.gpu_full.draw(gpa, bad, s_top20, 11, null), try draw(failing.allocator(), bad, scratch, s_top20, 11, &.{}));
    bad[2000] = std.math.nan(f32);
    try std.testing.expectEqual(try lanes.gpu_full.draw(gpa, bad, s_top20, 11, null), try draw(gpa, bad, scratch, s_top20, 11, &.{}));
}

test "a window's masks step by the padded vocabulary's words" {
    const words = minMaskWords(201024);
    try std.testing.expectEqual(@as(usize, 6282), words);
    var masks: [3 * 6282]u32 = @splat(0);
    masks[2 * 6282] = 1;
    try std.testing.expectEqual(@as(usize, 0), maskRow(&.{}, words, 2).len);
    try std.testing.expectEqual(@as(u32, 0), maskRow(&masks, words, 1)[0]);
    try std.testing.expect(lanes.grammar.allows(maskRow(&masks, words, 2), 0));
    try std.testing.expect(!lanes.grammar.allows(maskRow(&masks, words, 1), 0));
}

test "sampled first-token logprob uses the last written handle, not argmax" {
    const logits = [_]f32{ 0, -1, -2, -3 };
    var tokens: [16]u32 = @splat(0);
    tokens[2] = 3;
    const row = firstLogprobRow(&logits, &tokens, 3, 0);
    try std.testing.expectEqual(@as(u32, 3), row.token);
    try std.testing.expect(std.math.isFinite(row.logprob));
    const lse = @log(@as(f64, 1) + @exp(-@as(f64, 1)) + @exp(-@as(f64, 2)) + @exp(-@as(f64, 3)));
    try std.testing.expectApproxEqAbs(@as(f32, -3) - @as(f32, @floatCast(lse)), row.logprob, 1e-6);
}

test "draw preserves candidate-order norm and unclamped uniform24 boundaries against lanes.gpu_full.draw" {
    const gpa = std.testing.allocator;
    const n = 4096;
    const logits = try gpa.alloc(f32, n);
    defer gpa.free(logits);
    const scratch = try gpa.alloc(f32, n);
    defer gpa.free(scratch);
    @memset(logits, -50.0);
    logits[9] = 0.0;
    logits[1035] = -16.7;
    @memset(logits[1056..1088], -16.7);
    const s40: lanes.Sampling = .{ .seed = 18508, .temperature = 1.0, .top_k = 40, .top_p = 0.9999995 };
    const s0: lanes.Sampling = .{ .seed = 18508, .temperature = 1.0, .top_k = 0, .top_p = 0.9999995 };
    try std.testing.expectEqual(@as(u32, 9), try lanes.gpu_full.draw(gpa, logits, s40, 42, null));
    try std.testing.expectEqual(@as(u32, 9), try draw(gpa, logits, scratch, s40, 42, &.{}));
    try std.testing.expectEqual(@as(u32, 1035), try lanes.gpu_full.draw(gpa, logits, s0, 42, null));
    try std.testing.expectEqual(@as(u32, 1035), try draw(gpa, logits, scratch, s0, 42, &.{}));

    @memset(logits, -50.0);
    logits[3122] = 0.0;
    logits[1035] = -4.5;
    @memset(logits[1040..1058], -4.5);
    const s20: lanes.Sampling = .{ .seed = 18508, .temperature = 1.0, .top_k = 20, .top_p = 0.9 };
    const s0_p9: lanes.Sampling = .{ .seed = 18508, .temperature = 1.0, .top_k = 0, .top_p = 0.9 };
    try std.testing.expectEqual(@as(u32, 1035), try lanes.gpu_full.draw(gpa, logits, s20, 42, null));
    try std.testing.expectEqual(@as(u32, 1035), try draw(gpa, logits, scratch, s20, 42, &.{}));
    try std.testing.expectEqual(@as(u32, 3122), try lanes.gpu_full.draw(gpa, logits, s0_p9, 42, null));
    try std.testing.expectEqual(@as(u32, 3122), try draw(gpa, logits, scratch, s0_p9, 42, &.{}));
}
