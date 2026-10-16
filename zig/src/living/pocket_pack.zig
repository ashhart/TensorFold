//! The pocket's passes with their sequences side by side: as many as one pass of the trainer holds, run together.
const std = @import("std");
const train_mode = @import("train_mode.zig");
const lesson_route = @import("lesson_route.zig");
const Example = @import("lesson_examples.zig").Example;

pub const Seq = train_mode.Seq;
pub const Result = train_mode.Result;
const Mode = train_mode.Mode;

/// How many of `seqs`, from the first, one pass of trainer t holds: as many as fit, at least one.
pub fn fits(t: anytype, seqs: []const Seq) usize {
    if (!t.packs()) return @min(seqs.len, 1);
    var rows: usize = 0;
    var n: usize = 0;
    while (n < seqs.len and n < t.spans.len) : (n += 1) {
        rows += seqs[n].ids.len - 1 - seqs[n].skip;
        if (rows > t.pack and n > 0) break;
    }
    return n;
}

/// The sequence an example makes, its pocket unscaled and nothing nudged.
pub fn seq(ex: Example) Seq {
    return .{ .ids = ex.ids, .start = ex.start, .decide = ex.decide };
}

/// The turn every pocket pass takes: lessons of the model itself, no router deciding.
pub const turn: lesson_route.Turn = .{ .decide = 0, .bits = lesson_route.Acting.empty, .whole = true };

/// Each of `xs` read in `mode` in as few passes as hold them, each one's result into `out`.
pub fn read(gpa: std.mem.Allocator, t: anytype, xs: []const Example, mode: Mode, out: []Result) !void {
    const seqs = try gpa.alloc(Seq, xs.len);
    defer gpa.free(seqs);
    for (seqs, xs) |*s, ex| s.* = seq(ex);
    var at: usize = 0;
    while (at < seqs.len) {
        const n = fits(t, seqs[at..]);
        try t.pass(seqs[at..][0..n], mode, turn, out[at..][0..n]);
        at += n;
    }
}
