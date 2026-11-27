//! The batched MTP head's plan: the rows a pass takes and the lanes' groups, from the admitted capacity.

/// The head's rows a pass takes at most (the MTP's own rows).
pub const piece_rows: usize = 256;

/// Prompt-only rows a piece takes under TF_FLASHNEXT_PREFILL_MTP_LEAN; proposal paths keep `piece_rows`.
pub const prefill_piece_rows: usize = 3072;

/// The batched head's requests: the pass's rows, the score rows and the rows a pass takes.
pub const Plan = struct {
    /// The pass's rows to reserve: the admitted capacity bounds them, so the head never grows the shared scratch.
    pass_rows: usize,
    /// The score rows to reserve: the piece, as the single-stream head prices them.
    score_rows: usize,
    /// The rows a pass takes: the capacity's half, as the single-stream head takes them.
    piece: usize,
};

/// Two scratch rows per token, with the bounded count widened before multiplication.
pub fn reserveRows(total: usize) usize {
    const rows: usize = @min(total, piece_rows);
    return @max(@as(usize, 32), 2 * rows);
}

/// Two scratch rows per prompt-only token, bounded at the wider opt-in piece.
pub fn reservePrefillRows(total: usize) usize {
    const rows: usize = @min(total, prefill_piece_rows);
    return @max(@as(usize, 32), 2 * rows);
}

/// A lean piece's two four-stream fc_h slices fit `partRows(cap)` without growing scratch.
pub fn prefillPiece(cap: usize) usize {
    return @max(@as(usize, 1), @min(prefill_piece_rows, cap / 4));
}

/// The head's plan from the admitted capacity and the rows to draft.
pub fn plan(admitted: usize, total: usize) Plan {
    // the pass's rows as they stand: the head asks for at least 32, as the single-stream head does
    const cap = @max(admitted, 32);
    const want = reserveRows(total);
    return .{
        .pass_rows = @min(want, cap),
        .score_rows = @max(@as(usize, 1), @min(piece_rows, cap / 2)),
        .piece = @max(@as(usize, 1), @min(piece_rows, cap / 2)),
    };
}

/// The lanes' groups: each group's rows fit the piece, the first lane always in.
pub fn groups(lanes: []const usize, piece: usize, out: []usize) usize {
    var n: usize = 0;
    var at: usize = 0;
    while (at < lanes.len) {
        var g: usize = 0;
        var rows: usize = 0;
        while (at + g < lanes.len and (g == 0 or rows + lanes[at + g] <= piece)) : (g += 1) rows += lanes[at + g];
        out[n] = g;
        n += 1;
        at += g;
    }
    return n;
}
