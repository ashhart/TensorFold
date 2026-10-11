//! What a training step does, for both backends' trainers.

/// A step's loss, its gradient (and Adam: `learn`), inputs only (`reads`), or the gradient at a nudged row (`target`).
pub const Mode = enum { loss, grad, learn, avoid, seek, project, keys, target, prime };

/// Whether a step only reads the site inputs (sketches, projections or keys), with no loss.
pub fn reads(mode: Mode) bool {
    return switch (mode) {
        .avoid, .seek, .project, .keys, .prime => true,
        else => false,
    };
}

/// Where a `target` step nudges and reads the gradient: three rows, each after its mask's layers, row i's in third i.
pub const Nudge = struct { layers: [3]u64 = @splat(0), rows: [3]usize = @splat(0) };

/// The answer's mean loss, and whether each of its tokens was the likeliest at its row.
pub const Result = struct { loss: f32, recalled: bool };

/// A pass's sequence: tokens, answer start, decision row, pocket scale, nudges, and `skip` rows from prefix `slot`.
pub const Seq = struct { ids: []const u32, start: usize, decide: usize, scale: f32 = 1, nudge: Nudge = .{}, skip: usize = 0, slot: u32 = 0 };

/// Where each sequence of a pass sits: its first row, its rows (ids less the last and the skipped), its prefix.
pub const Span = struct {
    at: usize,
    rows: usize,
    skip: usize = 0,
    slot: u32 = 0,

    /// The pass's row holding the sequence's row r (r at least `skip`).
    pub fn row(sp: Span, r: usize) usize {
        return sp.at + r - sp.skip;
    }
};

/// A sequence's rows that predict its answer: the first in the pass, where they sit among the head's rows, how many.
pub const Answer = struct { first: usize, at: usize, rows: usize };

/// Each sequence's answer rows, packed for the head (no other row's loss or gradient there is anything but zero).
pub fn answers(seqs: []const Seq, spans: []const Span, out: []Answer) usize {
    var at: usize = 0;
    for (seqs, spans, out) |sq, sp, *a| {
        a.* = .{ .first = sp.row(sq.start - 1), .at = at, .rows = sq.ids.len - sq.start };
        at += a.rows;
    }
    return at;
}
