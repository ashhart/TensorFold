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
            .matvec_drops_tail => .{ .from = "for (uint i = 8 * lane; i < a.k; i += 256) {", .to = "for (uint i = 8 * lane; i + 256 - 8 * lane <= a.k; i += 256) {" },
            .gemm_untransposed => .{ .from = "gemm_nax<bfloat16_t, false, true, AM, AN", .to = "gemm_nax<bfloat16_t, false, false, AM, AN" },
            .prompt_sees_next_key => .{ .from = "col > row ? kMasked", .to = "col > row + 1 ? kMasked" },
            .q4_matvec_first_zero => .{ .from = "z = ll_q4_zero(Z, j, g, groups);", .to = "z = ll_q4_zero(Z, j, 0, groups);" },
            .q4_dequant_reversed_nibbles => .{ .from = "out[e] = bfloat(ll_q4(word, e, z, s));", .to = "out[e] = bfloat(ll_q4(word, 7 - e, z, s));" },
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
