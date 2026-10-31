//! Planted controls for LFM2's shared-round kernels: lfm2.metal's text with one fault, which one gate must report.
const std = @import("std");
const contract = @import("contract");
const sources = @import("kernel_sources");

pub const Mutation = enum {
    none,
    argmax_last_tie,
    argmax_no_flag,
    conv_first_state,
    attention_first_cache,
    tap_packed_rows,
    tap_clobbers_column,
    markov_nan_dropped,
    markov_first_prev,
    draft_ids_unzeroed,
    block_attention_causal,
    markov_bf16_nan_dropped,
    markov_bf16_first_prev,
    bf16_rows_short_k,
    bf16_rows_last_row,
    bf16_tile_untransposed,
    bf16_tile_last_column,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .argmax_last_tie => "control: metal lfm2 argmax, a tie goes to the highest id",
            .argmax_no_flag => "control: metal lfm2 argmax, a nonfinite logit is not flagged",
            .conv_first_state => "control: metal lfm2 stream conv, every stream reads the first stream's state",
            .attention_first_cache => "control: metal lfm2 stream attention, every stream writes the first stream's cache",
            .tap_packed_rows => "control: metal lfm2 tap copy, rows packed at the hidden size, not the taps' row",
            .tap_clobbers_column => "control: metal lfm2 tap copy, a column of another tap is overwritten",
            .markov_nan_dropped => "control: metal lfm2 markov step, a NaN score is not flagged",
            .markov_first_prev => "control: metal lfm2 markov step, every stream reads the first stream's previous token",
            .draft_ids_unzeroed => "control: metal lfm2 draft ids, a pick past the vocabulary keeps its token",
            .block_attention_causal => "control: metal lfm2 block attention, a row misses the last key",
            .markov_bf16_nan_dropped => "control: metal lfm2 markov step bf16, a NaN score is not flagged",
            .markov_bf16_first_prev => "control: metal lfm2 markov step bf16, every stream reads the first stream's previous token",
            .bf16_rows_short_k => "control: metal lfm2 bf16 rows, a lane skips every second step of k",
            .bf16_rows_last_row => "control: metal lfm2 bf16 rows, a group's last row is not computed",
            .bf16_tile_untransposed => "control: metal lfm2 bf16 tile, W's tile is read untransposed",
            .bf16_tile_last_column => "control: metal lfm2 bf16 tile, the last column is not written",
        };
    }

    /// The gate that must go red.
    pub fn gate(m: Mutation) ?contract.gates.Gate {
        return switch (m) {
            .none => null,
            .argmax_no_flag, .markov_nan_dropped, .markov_bf16_nan_dropped => .nonfinite,
            else => .shapes,
        };
    }

    const Edit = struct { from: []const u8, to: []const u8 };

    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .argmax_last_tie => .{ .from = "(ov == bv && oi < bi)", .to = "(ov == bv && oi > bi)" },
            .argmax_no_flag => .{ .from = "if (!isfinite(f)) bad = 1;", .to = "if (false) bad = 1;" },
            .conv_first_state => .{ .from = "lfm2_conv_row(bcx + size_t(s.base) * 3 * D, s.conv,", .to = "lfm2_conv_row(bcx + size_t(s.base) * 3 * D, segs[0].conv," },
            .attention_first_cache => .{ .from = "queries + b * QH * HD, s.keys, s.values,", .to = "queries + b * QH * HD, segs[0].keys, segs[0].values," },
            .tap_packed_rows => .{ .from = "(taps + size_t(i.y) * at.x + at.y))[i.x]", .to = "(taps + size_t(i.y) * D + at.y))[i.x]" },
            .markov_nan_dropped => .{ .from = "nan = nan || isnan(sc);", .to = "nan = false;" },
            .markov_first_prev => .{ .from = "id = prev[size_t(s) * p.prev_stride];", .to = "id = prev[0];" },
            .draft_ids_unzeroed => .{ .from = "ids[i] = ok ? token : 0u;", .to = "ids[i] = token;" },
            .block_attention_causal => .{ .from = "count[b] = FULL ? p.len + 1 :", .to = "count[b] = FULL ? p.len :" },
            .markov_bf16_nan_dropped => .{ .from = "nan = nan || isnan(sc);", .to = "nan = false;" },
            .markov_bf16_first_prev => .{ .from = "id = prev[size_t(s) * p.prev_stride];", .to = "id = prev[0];" },
            .bf16_rows_short_k => .{ .from = "for (uint i = 8 * lane; i < p.k; i += 256) {", .to = "for (uint i = 8 * lane; i < p.k; i += 512) {" },
            .bf16_rows_last_row => .{ .from = "if (r >= rows) break;", .to = "if (r + 1 >= rows) break;" },
            .bf16_tile_untransposed => .{ .from = "32, ulong2(0, 0), true);", .to = "32, ulong2(0, 0), false);" },
            .bf16_tile_last_column => .{ .from = "if (row < p.rows && col < p.n) y[", .to = "if (row < p.rows && col + 1 < p.n) y[" },
            .tap_clobbers_column => .{ .from = "[i.x] = from[i.x];\n}\n", .to = "[i.x] = from[i.x];\n  *(device bfloat*)(taps + size_t(i.y) * at.x) = 0;\n}\n" },
        };
    }
};

/// lfm2.metal's text as built for the kernel under test; a control's one edit must change it, or it proves nothing.
pub fn source(a: std.mem.Allocator, m: Mutation) ![]const u8 {
    if (m == .none) return sources.lfm2;
    const e = m.edit();
    if (std.mem.indexOf(u8, sources.lfm2, e.from) == null) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, sources.lfm2, e.from, e.to);
}
