//! The card top-k of a sampled row: the candidates a forward keeps beside its greedy tokens, and the host's view.
const std = @import("std");
const q = @import("qwen27.zig");
const shape = @import("qwen27_config.zig");

pub const TOPK_MAX = 64; // candidates a row keeps on the card (a sampled row with a larger top_k draws on the host)
pub const CAND_WORDS = 132; // a row's words: TOPK_MAX values (fp32 bits), TOPK_MAX ids, the not-finite flag, padding
pub const TOPK_PARTS = (shape.vocab + 1023) / 1024; // topk_part's groups a row, one per 1024 ids
comptime {
    if (TOPK_PARTS > 256) @compileError("the card top-k merges at most 262144 ids a row");
}

/// The card top-k to keep: `k` candidates a row, copied to `out` ([16][CAND_WORDS] words) before the forward's sync.
pub const Mode = struct { k: u32, out: *[16 * CAND_WORDS]u32, rows: *u32 };

/// A row's card top-k: `vals` and `ids` by value, then lower id; `bad`: the row holds a value that is not finite.
pub const Cand = struct { vals: []const f32, ids: []const u32, bad: bool };

/// What the host sets and the last forward left: `k` wanted next (0: none), the rows and candidates held, their words.
pub const State = struct { k: u32 = 0, rows: u32 = 0, have: u32 = 0, words: [16 * CAND_WORDS]u32 = @splat(0) };

/// Row `row`'s `k` best logits and their ids from the last forward's card top-k; null when it did not keep that many.
pub fn candidates(r: *const q.Runner, row: u32, k: u32) ?Cand {
    const c = &r.cand;
    if (row >= c.rows or k == 0 or k > c.have) return null;
    const w = c.words[row * CAND_WORDS ..][0..CAND_WORDS];
    return .{ .vals = std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(w[0..k])), .ids = w[TOPK_MAX..][0..k], .bad = w[2 * TOPK_MAX] != 0 };
}

/// The card top-k a forward keeps (null when none is wanted); the forward notes its rows after its sync.
pub fn mode(r: *q.Runner) ?Mode {
    const c = &r.cand;
    c.rows = 0;
    if (c.k == 0) return null;
    c.have = @min(c.k, TOPK_MAX);
    return .{ .k = c.have, .out = &c.words, .rows = &c.rows };
}
