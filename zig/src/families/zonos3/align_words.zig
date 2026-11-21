//! The forced aligner's host side: Dia's byte ids, the standardized waveform, RoPE tables, the words assembled.
const std = @import("std");
const tables = @import("unicode_words.zig");

const A = std.mem.Allocator;

pub const rate = 16000;
/// One forward's limits: 30 s of audio, 1022 UTF-8 bytes of transcript.
pub const max_samples = 30 * rate;
pub const max_bytes = 1022;
pub const presence_threshold: f32 = 0.1;

/// A token of the transcript in the clip; `placed` false: the uniform fallback, or punctuation alone.
pub const Word = struct { start: f64, end: f64, presence: f32, placed: bool };

/// _is_wordy: the token has a letter or a digit.
pub fn isWordy(token: []const u8) bool {
    var it = (std.unicode.Utf8View.init(token) catch return false).iterator();
    while (it.nextCodepoint()) |cp| {
        var lo: usize = 0;
        var hi: usize = tables.wordy.len / 2;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (cp > tables.wordy[2 * mid + 1]) lo = mid + 1 else hi = mid;
        }
        if (lo < tables.wordy.len / 2 and cp >= tables.wordy[2 * lo]) return true;
    }
    return false;
}

/// The words as Dia reads them: ids [1, 32, bytes joined by spaces], and each word's byte span after the prefix.
pub const Transcript = struct { ids: []u32, spans: [][2]u32 };

pub fn transcript(a: A, words: []const []const u8) !Transcript {
    var ids: std.ArrayList(u32) = .empty;
    try ids.appendSlice(a, &.{ 1, 32 });
    const spans = try a.alloc([2]u32, words.len);
    for (words, spans, 0..) |w, *span, i| {
        if (i != 0) try ids.append(a, 32);
        span[0] = @intCast(ids.items.len - 2);
        for (w) |byte| try ids.append(a, byte);
        span[1] = @intCast(ids.items.len - 2);
    }
    if (ids.items.len - 2 > max_bytes) return error.TranscriptTooLong;
    return .{ .ids = try ids.toOwnedSlice(a), .spans = spans };
}

/// FuseAlign._standardize: zero mean, unit deviation (plus 1e-5), the sums in double as torch's reductions round.
pub fn standardize(a: A, pcm: []const f32) ![]f32 {
    const n: f32 = @floatFromInt(pcm.len);
    var sum: f64 = 0;
    for (pcm) |x| sum += x;
    const mean = @as(f32, @floatCast(sum)) / n;
    const out = try a.alloc(f32, pcm.len);
    var sq: f64 = 0;
    for (out, pcm) |*o, x| {
        o.* = x - mean;
        sq += @as(f64, o.*) * o.*;
    }
    const denom = @sqrt(@as(f32, @floatCast(sq)) / n) + 1e-5;
    for (out) |*o| o.* /= denom;
    return out;
}

pub fn bf16(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    // round to nearest, ties to even, on the 16 bits dropped
    const rounded = (bits +% 0x7fff +% ((bits >> 16) & 1)) & 0xffff0000;
    return @bitCast(rounded);
}

/// cos, sin [positions][hd / 2] of fp32 angles; `dia`: transformers' frequencies and bf16 tables, else FuseAlign's.
pub fn ropeTables(a: A, positions: []const f32, hd: usize, theta: f32, dia: bool) ![2][]f32 {
    const half = hd / 2;
    const cos = try a.alloc(f32, positions.len * half);
    const sin = try a.alloc(f32, positions.len * half);
    for (0..half) |i| {
        const e = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(hd));
        const inv: f32 = if (dia) 1.0 / @as(f32, @floatCast(std.math.pow(f64, theta, e))) else @floatCast(std.math.pow(f64, theta, -e));
        for (positions, 0..) |p, row| {
            const angle: f64 = p * inv;
            const c: f32 = @floatCast(@cos(angle));
            const s: f32 = @floatCast(@sin(angle));
            cos[row * half + i] = if (dia) bf16(c) else c;
            sin[row * half + i] = if (dia) bf16(s) else s;
        }
    }
    return .{ cos, sin };
}

/// The joint sequence's positions: patch a at a, text byte j at j * patches / max(bytes - 1, 1).
pub fn jointPositions(a: A, patches: usize, bytes: usize) ![]f32 {
    const out = try a.alloc(f32, patches + bytes);
    const scale = @as(f32, @floatFromInt(patches)) / @as(f32, @floatFromInt(@max(bytes, 2) - 1));
    for (out[0..patches], 0..) |*p, i| p.* = @floatFromInt(i);
    for (out[patches..], 0..) |*p, j| p.* = @as(f32, @floatFromInt(j)) * scale;
    return out;
}

/// rotate_half in place on `n` rows' heads of `hd`: (x1, x2) -> (x1 c - x2 s, x2 c + x1 s), a table row a row.
pub fn rope(rows: []f32, stride: usize, off: usize, heads: usize, hd: usize, cos: []const f32, sin: []const f32) void {
    const half = hd / 2;
    for (0..rows.len / stride) |r| for (0..heads) |h| {
        const x = rows[r * stride + off + h * hd ..][0..hd];
        for (x[0..half], x[half..], cos[r * half ..][0..half], sin[r * half ..][0..half]) |*x1, *x2, c, s| {
            const p = x1.*;
            const q = x2.*;
            x1.* = p * c - q * s;
            x2.* = q * c + p * s;
        }
    };
}

/// fill_unplaced, align_batch: an unplaced run shares its gap; punctuation is zero-width at the previous end.
pub fn assemble(a: A, tokens: []const []const u8, placed: []const ?[2]f64, presence: []const f32, duration: f64) ![]Word {
    const filled = try a.alloc([2]f64, placed.len);
    var index: usize = 0;
    while (index < placed.len) {
        if (placed[index]) |iv| {
            filled[index] = iv;
            index += 1;
            continue;
        }
        const lo = index;
        while (index < placed.len and placed[index] == null) index += 1;
        const left = if (lo != 0) filled[lo - 1][1] else 0;
        const right = if (index < placed.len) placed[index].?[0] else duration;
        const width = (right - left) / @as(f64, @floatFromInt(index - lo));
        for (0..index - lo) |o| {
            const of: f64 = @floatFromInt(o);
            filled[lo + o] = .{ left + of * width, if (lo + o + 1 == index) right else left + (of + 1) * width };
        }
    }
    const out = try a.alloc(Word, tokens.len);
    const first: f64 = if (filled.len != 0) filled[0][0] else 0;
    var previous: ?f64 = null;
    var j: usize = 0;
    for (out, tokens) |*w, token| {
        if (isWordy(token)) {
            w.* = .{ .start = filled[j][0], .end = filled[j][1], .presence = presence[j], .placed = placed[j] != null };
            previous = filled[j][1];
            j += 1;
        } else {
            const t = previous orelse first;
            w.* = .{ .start = t, .end = t, .presence = 0, .placed = false };
            previous = t;
        }
    }
    return out;
}

test "transcript: the prefix, the words' bytes joined by spaces, a span a word" {
    const t = std.testing;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try transcript(a, &.{ "hi", "caf\u{e9}" });
    try t.expectEqualSlices(u32, &.{ 1, 32, 'h', 'i', 32, 'c', 'a', 'f', 0xc3, 0xa9 }, got.ids);
    try t.expectEqual([2]u32{ 0, 2 }, got.spans[0]);
    try t.expectEqual([2]u32{ 3, 8 }, got.spans[1]);
    try t.expect(isWordy("dog,") and isWordy("\u{6771}") and !isWordy("--") and !isWordy("\u{2014}"));
    try t.expectEqual(@as(f32, 1.0), bf16(1.0));
    // a tie rounds to the even neighbour: down from 0x3f80 8000, up from 0x3f81 8000
    try t.expectEqual(@as(u32, 0x3f800000), @as(u32, @bitCast(bf16(@bitCast(@as(u32, 0x3f808000))))));
    try t.expectEqual(@as(u32, 0x3f820000), @as(u32, @bitCast(bf16(@bitCast(@as(u32, 0x3f818000))))));
}

test "assemble: an unplaced run shares its gap, punctuation sits at the previous end" {
    const t = std.testing;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try assemble(a, &.{ "a", "--", "b", "c", "d" }, &.{ .{ 0.5, 1.0 }, null, null, .{ 3.0, 3.5 } }, &.{ 0.9, 0.01, 0.02, 0.8 }, 4.0);
    try t.expectEqual(@as(f64, 1.0), got[1].start);
    try t.expect(!got[1].placed and got[1].start == got[1].end);
    try t.expectEqual(@as(f64, 1.0), got[2].start);
    try t.expectEqual(@as(f64, 2.0), got[2].end);
    try t.expectEqual(@as(f64, 3.0), got[3].end);
    try t.expect(!got[2].placed and got[4].placed);
}
