//! Planted controls for Llama's kernels: a source as the loader builds it with one fault, which one gate must report.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const tf = @import("tensorfold");
const family = tf.llama.contracts;

pub const Mutation = enum {
    none,
    embed_own_row,
    norm_mean_over_threads,
    add_wrong_sign,
    add_norm_no_residual,
    silu_swapped,
    prep_first_position,
    argmax_nan_loses,
    decode_no_rescale,
    merge_unscaled_mass,
    matvec_drops_tail,
    gemm_untransposed,
    prompt_sees_next_key,
    q4_matvec_first_zero,
    q4_dequant_reversed_nibbles,
    prep_table_first_position,
    prep_table_sin_for_cos,
    matvec_bias_dropped,
    gemm_f32_untransposed,
    bias_round_no_bias,
    bias_round_twice,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .embed_own_row => "control: metal llama embed, row r reads table row r, not row ids[r]",
            .norm_mean_over_threads => "control: metal llama norm, the mean of squares divides by the threads, not the width",
            .add_wrong_sign => "control: metal llama add, the rows are subtracted",
            .add_norm_no_residual => "control: metal llama add_norm, the sum is not written back to the residual",
            .silu_swapped => "control: metal llama silu, the gate and the up are read swapped",
            .prep_first_position => "control: metal llama prep, every row is rotated at the window's first position",
            .argmax_nan_loses => "control: metal llama argmax, a NaN logit never wins",
            .decode_no_rescale => "control: metal llama decode, the running sums are not rescaled when the max grows",
            .merge_unscaled_mass => "control: metal llama merge, each block's mass is summed without its max's scale",
            .matvec_drops_tail => "control: metal llama matvec, the columns past the last whole 256 are dropped",
            .gemm_untransposed => "control: metal llama gemm_00, the weights are read untransposed, as [k, n]",
            .prompt_sees_next_key => "control: metal llama prompt attention, each row also sees the next key",
            .q4_matvec_first_zero => "control: metal llama q4 matvec, every group takes its row's first zero point",
            .q4_dequant_reversed_nibbles => "control: metal llama q4 dequant, a word's eight nibbles are read in reverse",
            .prep_table_first_position => "control: metal llama prep_table, every row reads the window's first position's cos and sin",
            .prep_table_sin_for_cos => "control: metal llama prep_table, the cosine and sine are read from each other's half of the table",
            .matvec_bias_dropped => "control: metal llama matvec_bias, the bias is never added",
            .gemm_f32_untransposed => "control: metal llama gemm_f32_00, the weights are read untransposed, as [k, n]",
            .bias_round_no_bias => "control: metal llama bias_round, the bias is left out",
            .bias_round_twice => "control: metal llama bias_round, the sum is rounded to bf16 before the bias joins it",
        };
    }

    /// The gate that must go red.
    pub fn gate(m: Mutation) ?contract.gates.Gate {
        return switch (m) {
            .none => null,
            .argmax_nan_loses => .nonfinite,
            else => .shapes,
        };
    }

    const Edit = struct { from: []const u8, to: []const u8 };

    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .embed_own_row => .{ .from = "E[(size_t)ids[p.y] * a.width + p.x]", .to = "E[(size_t)p.y * a.width + p.x]" },
            .norm_mean_over_threads => .{ .from = "total / float(a.width) + a.eps);\n  for (uint i = t;", .to = "total / float(n) + a.eps);\n  for (uint i = t;" },
            .add_wrong_sign => .{ .from = "R[i] = bfloat(float(R[i]) + float(X[i]));", .to = "R[i] = bfloat(float(R[i]) - float(X[i]));" },
            .add_norm_no_residual => .{ .from = "        x[t + p * n] = v[p];\n", .to = "" },
            .silu_swapped => .{ .from = "const float g = float(GU[at]), u = float(GU[at + a.width]);", .to = "const float u = float(GU[at]), g = float(GU[at + a.width]);" },
            .prep_first_position => .{ .from = "const float angle = position * inv[i];", .to = "const float angle = float(a.pos) * inv[i];" },
            .argmax_nan_loses => .{ .from = "if (isnan(a)) return !isnan(b) || ia < ib;", .to = "if (isnan(a)) return false;" },
            .decode_no_rescale => .{ .from = "const float rescale = fast::exp(top - peak);", .to = "const float rescale = 1.0f;" },
            .merge_unscaled_mass => .{ .from = "ls[b] * fast::exp(ms[b] - top) : 0.0f;", .to = "ls[b] : 0.0f;" },
            .matvec_drops_tail => .{ .from = "_Pragma(\"clang loop unroll(full)\")\n  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;\n  for (uint i = 8 * lane; i < a.k; i += 256) {", .to = "_Pragma(\"clang loop unroll(full)\")\n  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;\n  for (uint i = 8 * lane; i + 256 - 8 * lane <= a.k; i += 256) {" },
            .gemm_untransposed => .{ .from = "TF_GEMM_NAX_KERNEL(ll_gemm_00, bfloat16_t, false, true, false, false)", .to = "TF_GEMM_NAX_KERNEL(ll_gemm_00, bfloat16_t, false, false, false, false)" },
            .prompt_sees_next_key => .{ .from = "const bool out = col > row ||", .to = "const bool out = col > row + 1 ||" },
            .q4_matvec_first_zero => .{ .from = "z = ll_q4_zero(Z, j, g, groups);", .to = "z = ll_q4_zero(Z, j, 0, groups);" },
            .q4_dequant_reversed_nibbles => .{ .from = "out[e] = bfloat(ll_q4(word, e, z, s));", .to = "out[e] = bfloat(ll_q4(word, 7 - e, z, s));" },
            .prep_table_first_position => .{ .from = "const device bfloat* tab = T + slot * DH;", .to = "const device bfloat* tab = T + (size_t)a.pos * DH;" },
            .prep_table_sin_for_cos => .{ .from = "const float c = float(tab[i]), s = float(tab[HALF + i]);", .to = "const float c = float(tab[HALF + i]), s = float(tab[i]);" },
            .matvec_bias_dropped => .{ .from = "    s += float(Bv[j]);\n", .to = "" },
            .gemm_f32_untransposed => .{ .from = "TF_GEMM_NAX_KERNEL_TO(ll_gemm_f32_00, bfloat16_t, float, false, true,", .to = "TF_GEMM_NAX_KERNEL_TO(ll_gemm_f32_00, bfloat16_t, float, false, false," },
            .bias_round_no_bias => .{ .from = "bfloat(C[(size_t)p.y * a.n + p.x] + float(Bv[p.x]))", .to = "bfloat(C[(size_t)p.y * a.n + p.x])" },
            .bias_round_twice => .{ .from = "bfloat(C[(size_t)p.y * a.n + p.x] + float(Bv[p.x]))", .to = "bfloat(float(bfloat(C[(size_t)p.y * a.n + p.x])) + float(Bv[p.x]))" },
        };
    }
};

/// `text` with the control's one edit (a copy either way); an edit that finds nothing proves nothing, so it errors.
pub fn apply(a: std.mem.Allocator, text: []const u8, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, text);
    const e = m.edit();
    if (std.mem.count(u8, text, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, text, e.from, e.to);
}

/// The glue library as the loader compiles it (precise math functions), with the control's edit.
pub fn glueLibrary(device: mtl.Device, a: std.mem.Allocator, m: Mutation) !mtl.Library {
    if (m == .none) return family.glueLibrary(device, a);
    const plain = try family.glueSource(a);
    defer a.free(plain);
    const src = try apply(a, plain, m);
    defer a.free(src);
    return mtl.Library.fromSource(device, src, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
}

/// A spec's name and control for a probe built with `m`.
pub fn mark(k: contract.gates.Kernel, m: Mutation) contract.gates.Kernel {
    var out = k;
    if (m != .none) {
        out.spec.name = m.name();
        out.control = m.gate();
    }
    return out;
}

test "llama source control: GEMM edits only the weight layout of ll_gemm_00" {
    const a = std.testing.allocator;
    const plain = @import("kernel_sources").llama_gemm;
    const from = "TF_GEMM_NAX_KERNEL(ll_gemm_00, bfloat16_t, false, true, false, false)";
    const to = "TF_GEMM_NAX_KERNEL(ll_gemm_00, bfloat16_t, false, false, false, false)";
    const at = std.mem.indexOf(u8, plain, from).?;
    const edited = try apply(a, plain, .gemm_untransposed);
    defer a.free(edited);
    try std.testing.expectEqualStrings(plain[0..at], edited[0..at]);
    try std.testing.expectEqualStrings(to, edited[at..][0..to.len]);
    try std.testing.expectEqualStrings(plain[at + from.len ..], edited[at + to.len ..]);
    try std.testing.expectEqual(contract.gates.Gate.shapes, Mutation.gemm_untransposed.gate().?);
    const unchanged = try apply(a, plain, .none);
    defer a.free(unchanged);
    try std.testing.expectEqualStrings(plain, unchanged);
}

test "llama source control: a missing or repeated GEMM anchor refuses the edit" {
    const a = std.testing.allocator;
    const plain = @import("kernel_sources").llama_gemm;
    try std.testing.expectError(error.ControlEditMissing, apply(a, "", .gemm_untransposed));
    const repeated = try std.mem.concat(a, u8, &.{ plain, plain });
    defer a.free(repeated);
    try std.testing.expectError(error.ControlEditMissing, apply(a, repeated, .gemm_untransposed));
    const edited = try apply(a, plain, .gemm_untransposed);
    defer a.free(edited);
    try std.testing.expectError(error.ControlEditMissing, apply(a, edited, .gemm_untransposed));
}
