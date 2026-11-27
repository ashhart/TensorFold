//! The q35 attention kernels' parameter lists, as their .cu sources declare them.
const std = @import("std");

/// A parameter's kind: geometry struct, device pointer, 32-bit count, float scale or Boolean mode.
pub const Kind = enum { geo, ptr, i32, f32, boolean };

/// One kernel's parameter list, in the .cu source's order.
pub const Sig = struct {
    /// The name a launch site names this list by.
    name: []const u8,
    /// The engine's field path that holds the kernel's handle.
    handle: []const u8,
    /// The symbol the .cu source declares.
    symbol: []const u8,
    /// The parameters, in order.
    params: []const Kind,
};

/// The bf16 decode partial: a block a (row, chunk, key head) of the query heads' rows.
pub const attn_partial = Sig{ .name = "attn_partial", .handle = "e.fns.attn_partial", .symbol = "tf_q35_attn_partial", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .f32, .ptr, .ptr, .ptr } };

/// The e4m3 decode partial: the bf16 one's tiles in the e4m3 scales.
pub const attn_partial8 = Sig{ .name = "attn_partial8", .handle = "e.fp8.attn_partial", .symbol = "tf_q35_attn_partial8", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .f32, .ptr, .ptr, .ptr } };

/// The bf16 decode attention over a long stream's rows: a block a group of rows.
pub const attn_rows = Sig{ .name = "attn_rows", .handle = "e.fns.attn_rows", .symbol = "tf_q35_attn_rows", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .f32, .i32, .ptr, .ptr, .ptr, .boolean } };

/// The e4m3 decode attention over a long stream's rows.
pub const attn_rows8 = Sig{ .name = "attn_rows8", .handle = "e.fp8.attn_rows", .symbol = "tf_q35_attn_rows8", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .f32, .i32, .ptr, .ptr, .ptr } };

/// The rows kernels for a forward whose streams share pages (Seg.share), the same ABI.
pub const attn_rows_join = Sig{ .name = "attn_rows_join", .handle = "e.fns.attn_rows_join", .symbol = "tf_q35_attn_rows_join", .params = attn_rows.params };
pub const attn_rows8_join = Sig{ .name = "attn_rows8_join", .handle = "e.fp8.attn_rows_join", .symbol = "tf_q35_attn_rows8_join", .params = attn_rows8.params };

/// Whole-window instances use the same ABI without consulting Seg.pt.
pub const attn_partial_whole = Sig{ .name = "attn_partial_whole", .handle = "e.fns.attn_partial_whole", .symbol = "tf_q35_attn_partial_whole", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .f32, .ptr, .ptr, .ptr } };
pub const attn_partial8_whole = Sig{ .name = "attn_partial8_whole", .handle = "e.fp8.attn_partial8_whole", .symbol = "tf_q35_attn_partial8_whole", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .f32, .ptr, .ptr, .ptr } };
pub const attn_rows_whole = Sig{ .name = "attn_rows_whole", .handle = "e.fns.attn_rows_whole", .symbol = "tf_q35_attn_rows_whole", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .f32, .i32, .ptr, .ptr, .ptr, .boolean } };
pub const attn_rows8_whole = Sig{ .name = "attn_rows8_whole", .handle = "e.fp8.attn_rows8_whole", .symbol = "tf_q35_attn_rows8_whole", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .f32, .i32, .ptr, .ptr, .ptr } };

/// The merge: a row's chunks' partials into each (row, query head)'s output.
pub const attn_merge = Sig{ .name = "attn_merge", .handle = "e.fns.attn_merge", .symbol = "tf_q35_attn_merge", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .ptr, .ptr, .ptr, .ptr } };

/// The gate: the attention's outputs through the second half of each query head's q_proj rows.
pub const attn_gate = Sig{ .name = "attn_gate", .handle = "e.fns.attn_gate", .symbol = "tf_q35_attn_gate", .params = &.{ .geo, .ptr, .ptr, .ptr, .i32, .ptr, .ptr } };

/// The rope: the queries rotated in place, the keys rotated and the values copied into the cache.
pub const rope = Sig{ .name = "rope", .handle = "e.fns.rope", .symbol = "tf_q35_rope", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .ptr, .ptr } };

/// The e4m3 rope: the bf16 one's tiles in the e4m3 scales.
pub const rope8 = Sig{ .name = "rope8", .handle = "e.fp8.rope", .symbol = "tf_q35_rope8", .params = &.{ .geo, .ptr, .ptr, .ptr, .ptr, .ptr, .ptr, .i32, .ptr, .ptr, .ptr } };

/// The draft frame's partial: the draft rows' own cache over the ring.
pub const draft_partial = Sig{ .name = "draft_partial", .handle = "d.fns.attn_partial", .symbol = "tf_q35d_attn_partial", .params = &.{ .ptr, .ptr, .ptr, .ptr, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .ptr, .ptr, .ptr } };

/// The draft frame's merge: the draft rows' chunks' partials into each row's output.
pub const draft_merge = Sig{ .name = "draft_merge", .handle = "d.fns.attn_merge", .symbol = "tf_q35d_attn_merge", .params = &.{ .ptr, .ptr, .ptr, .i32, .i32, .ptr, .ptr } };

/// The DFlash frame's dynamic convolution: one branch, either the incumbent bf16 rounding or f32 accumulate.
pub const draft_conv = Sig{ .name = "draft_conv", .handle = "d.fns.dyn_conv_batch", .symbol = "tf_q35d_dyn_conv_batch", .params = &.{ .ptr, .ptr, .ptr, .ptr, .ptr, .i32, .i32, .i32, .i32, .i32 } };

/// The DFlash frame's batched norm-rope: the frame's q/k rows, either rounding.
pub const draft_rope = Sig{ .name = "draft_rope", .handle = "d.fns.norm_rope_batch", .symbol = "tf_q35d_norm_rope_batch", .params = &.{ .ptr, .ptr, .ptr, .i32, .i32, .ptr, .i32, .f32, .f32, .i32 } };

/// Every q35 attention kernel's list: the decode partials, the rows, the merge, the gate, the rope and the draft pair.
pub const attn_rows8_mt2 = Sig{ .name = "attn_rows8_mt2", .handle = "e.mt2.paged", .symbol = "tf_q35_attn_rows8_mt2", .params = attn_rows8.params };
pub const attn_rows8_mt2_whole = Sig{ .name = "attn_rows8_mt2_whole", .handle = "e.mt2.whole", .symbol = "tf_q35_attn_rows8_mt2_whole", .params = attn_rows8.params };

/// The lilicorr fade's f32 RMS: X through the W-weighted norm.
pub const lilicorr_rms_f32 = Sig{ .name = "lilicorr_rms_f32", .handle = "d.fns.lilicorr_rms_f32", .symbol = "tf_lilicorr_rms_f32", .params = &.{ .ptr, .ptr, .ptr, .i32, .f32 } };

/// The lilicorr widen: bf16 rows to f32.
pub const lilicorr_widen = Sig{ .name = "lilicorr_widen", .handle = "d.fns.lilicorr_widen", .symbol = "tf_lilicorr_widen", .params = &.{ .ptr, .ptr, .i32 } };

/// The lilicorr narrow: f32 rows to bf16.
pub const lilicorr_narrow = Sig{ .name = "lilicorr_narrow", .handle = "d.fns.lilicorr_narrow", .symbol = "tf_lilicorr_narrow", .params = &.{ .ptr, .ptr, .i32 } };

/// The lilicorr factor: the A_IN/ANCHOR pairing.
pub const lilicorr_factor = Sig{ .name = "lilicorr_factor", .handle = "d.fns.lilicorr_factor", .symbol = "tf_lilicorr_factor", .params = &.{ .ptr, .ptr, .ptr, .ptr, .ptr, .i32, .i32 } };

/// The token projection, extended.
pub const token_proj_extended = Sig{ .name = "token_proj_extended", .handle = "d.fns.token_proj_extended", .symbol = "tf_token_proj_extended", .params = &.{ .ptr, .ptr, .ptr, .ptr, .i32, .i32 } };

/// The l2 norm over the tail width.
pub const lilicorr_l2norm = Sig{ .name = "lilicorr_l2norm", .handle = "d.fns.lilicorr_l2norm", .symbol = "tf_lilicorr_l2norm", .params = &.{ .ptr, .i32, .i32, .i32, .f32 } };

/// The dot products.
pub const lilicorr_dots = Sig{ .name = "lilicorr_dots", .handle = "d.fns.lilicorr_dots", .symbol = "tf_lilicorr_dots", .params = &.{ .ptr, .ptr, .ptr, .ptr, .ptr, .i32 } };

/// The lattice attention.
pub const lilicorr_lat_attn = Sig{ .name = "lilicorr_lat_attn", .handle = "d.fns.lilicorr_lat_attn", .symbol = "tf_lilicorr_lat_attn", .params = &.{ .ptr, .ptr, .ptr, .i32, .i32, .i32 } };

/// The lilicorr RMS (bf16 in).
pub const lilicorr_rms = Sig{ .name = "lilicorr_rms", .handle = "d.fns.lilicorr_rms", .symbol = "tf_lilicorr_rms", .params = &.{ .ptr, .ptr, .ptr, .i32, .f32 } };

/// The f32 linear.
pub const lilicorr_lin_f32 = Sig{ .name = "lilicorr_lin_f32", .handle = "d.fns.lilicorr_lin_f32", .symbol = "tf_lilicorr_lin_f32", .params = &.{ .ptr, .ptr, .ptr, .ptr, .i32, .i32 } };

/// The SiLU downcast.
pub const lilicorr_silu_bf16 = Sig{ .name = "lilicorr_silu_bf16", .handle = "d.fns.lilicorr_silu_bf16", .symbol = "tf_lilicorr_silu_bf16", .params = &.{ .ptr, .ptr, .i32 } };

/// The bf16 add.
pub const lilicorr_add_bf16 = Sig{ .name = "lilicorr_add_bf16", .handle = "d.fns.lilicorr_add_bf16", .symbol = "tf_lilicorr_add_bf16", .params = &.{ .ptr, .ptr, .ptr, .i32 } };

/// The lattice walk.
pub const lilicorr_lattice = Sig{ .name = "lilicorr_lattice", .handle = "d.fns.lilicorr_lattice", .symbol = "tf_lilicorr_lattice", .params = &.{ .ptr, .ptr, .ptr, .ptr, .ptr, .ptr } };

/// The bf16 GEMV row set.
pub const gemv_bf16_rows = Sig{ .name = "gemv_bf16_rows", .handle = "d.fns.gemv_bf16_rows", .symbol = "tf_gemv_bf16_rows", .params = &.{ .ptr, .ptr, .ptr, .ptr, .i32, .i32, .i32 } };

/// The f32 GEMV row set.
pub const gemv_f32_rows = Sig{ .name = "gemv_f32_rows", .handle = "d.fns.gemv_f32_rows", .symbol = "tf_gemv_f32_rows", .params = &.{ .ptr, .ptr, .ptr, .ptr, .i32, .i32, .i32 } };

/// The factor rows variant.
pub const lilicorr_factor_rows = Sig{ .name = "lilicorr_factor_rows", .handle = "d.fns.lilicorr_factor_rows", .symbol = "tf_lilicorr_factor_rows", .params = &.{ .ptr, .ptr, .ptr, .ptr, .ptr, .i32, .i32 } };

pub const all = [_]Sig{ attn_rows8_mt2, attn_rows8_mt2_whole, attn_partial, attn_partial8, attn_rows, attn_rows8, attn_merge, attn_gate, rope, rope8, draft_partial, draft_merge, draft_conv, draft_rope, attn_partial_whole, attn_partial8_whole, attn_rows_whole, attn_rows8_whole, attn_rows_join, attn_rows8_join, lilicorr_rms_f32, lilicorr_widen, lilicorr_narrow, lilicorr_factor, token_proj_extended, lilicorr_l2norm, lilicorr_dots, lilicorr_lat_attn, lilicorr_rms, lilicorr_lin_f32, lilicorr_silu_bf16, lilicorr_add_bf16, lilicorr_lattice, gemv_bf16_rows, gemv_f32_rows, lilicorr_factor_rows };

/// The list a launch site's handle names, if the table holds one.
pub fn byHandle(handle: []const u8) ?Sig {
    for (all) |s| if (std.mem.eql(u8, s.handle, handle)) return s;
    return null;
}

/// The list a launch site names, if the table holds one.
pub fn byName(name: []const u8) ?Sig {
    for (all) |s| if (std.mem.eql(u8, s.name, name)) return s;
    return null;
}

/// The kind a host value carries: the geometry struct, a device pointer, a count or a scale.
pub fn kindOf(comptime T: type) ?Kind {
    if (T == u64) return .ptr;
    if (T == c_int) return .i32;
    if (T == f32) return .f32;
    if (T == bool) return .boolean;
    const info = @typeInfo(T);
    if (info == .@"struct" and info.@"struct".layout == .@"extern") return .geo;
    return null;
}

/// A launch site's arguments against a kernel's list: a wrong count or kind does not build.
pub fn check(comptime s: Sig, comptime Args: type) void {
    comptime {
        const info = @typeInfo(Args);
        if (info != .@"struct") @compileError("q35 launch arguments are not a tuple: " ++ s.symbol);
        const types = info.@"struct".field_types;
        if (types.len != s.params.len) @compileError("q35 launch " ++ s.symbol ++ " takes " ++ std.fmt.comptimePrint("{d}", .{s.params.len}) ++ " arguments, the launch passes " ++ std.fmt.comptimePrint("{d}", .{types.len}));
        for (types, 0..) |T, i| {
            const got = kindOf(T) orelse @compileError("q35 launch argument kind is unknown: " ++ s.symbol);
            if (got != s.params[i]) @compileError("q35 launch argument " ++ std.fmt.comptimePrint("{d}", .{i}) ++ " of " ++ s.symbol ++ " has the wrong kind");
        }
    }
}
