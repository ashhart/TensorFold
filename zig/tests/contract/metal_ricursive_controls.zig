//! Planted controls for Ripple's kernels: a source as the loader builds it with one fault, which one gate must report.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const tf = @import("tensorfold");
const ks = @import("kernel_sources");
const family = tf.ricursive.contracts;

pub const Mutation = enum {
    none,
    embed_own_row,
    add_norm_no_residual,
    lerp_no_blend,
    prep_ropes_global,
    prep_next_slot,
    gate_early_round,
    clamp_no_lower,
    decode_past_window,
    prompt_past_window,
    prompt_short_ring,
    fp4_high_first,
    fp4_scale_bias,
    fp4_global_late,
    fp4_dequant_unscaled,
    fp4_gemm_no_global,
    df_tap_next_slot,
    df_norm_half_mean,
    df_norm16_no_weight,
    df_embed_own_row,
    df_prep_ropes_nope,
    df_conv_three_taps,
    df_silu_no_negative,
    df_vec_no_floor,
    df_topk_ascending,
    df_walk_from_pending,

    /// A control of the DFlash2 drafter's kernels (dflash2.metal), planted by metal_ricursive_dflash2.zig.
    pub fn dflash2(m: Mutation) bool {
        return @backingInt(m) >= @backingInt(Mutation.df_tap_next_slot);
    }

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .embed_own_row => "control: metal ricursive embed, row r reads table row r, not row ids[r]",
            .add_norm_no_residual => "control: metal ricursive add_norm, the sum is not written back to the residual",
            .lerp_no_blend => "control: metal ricursive add_norm, v's input takes the normed row whole, not its blend",
            .prep_ropes_global => "control: metal ricursive prep, a global layer's q and k are rotated too",
            .prep_next_slot => "control: metal ricursive prep, k and v land one ring slot late",
            .gate_early_round => "control: metal ricursive gate, the sigmoid's exp is rounded to bf16 before the division",
            .clamp_no_lower => "control: metal ricursive silu_clamp, up is clamped above only",
            .decode_past_window => "control: metal ricursive decode, each row also sees the key before its window",
            .prompt_past_window => "control: metal ricursive prompt attention, each row also sees the 64 keys before its window",
            .prompt_short_ring => "control: metal ricursive prompt attention, keys wrap 64 slots before the ring's end",
            .fp4_high_first => "control: metal ricursive fp4 matvec, each byte's nibbles read high first",
            .fp4_scale_bias => "control: metal ricursive fp4 matvec, the E4M3 scale's exponent bias off by one",
            .fp4_global_late => "control: metal ricursive fp4 matvec, the global applied after the sum rounds to bf16",
            .fp4_dequant_unscaled => "control: metal ricursive fp4 dequant, a block's values left without its scale",
            .fp4_gemm_no_global => "control: metal ricursive fp4 gemm, the sums never take the global",
            .df_tap_next_slot => "control: metal ricursive dflash2 tap, the residual lands in the next tap's slot",
            .df_norm_half_mean => "control: metal ricursive dflash2 norm, the mean of squares is halved",
            .df_norm16_no_weight => "control: metal ricursive dflash2 norm16, the weight is never applied",
            .df_embed_own_row => "control: metal ricursive dflash2 embed, row r reads table row r, not row ids[r]",
            .df_prep_ropes_nope => "control: metal ricursive dflash2 prep, the NoPE layer's q and k are rotated too",
            .df_conv_three_taps => "control: metal ricursive dflash2 conv, the fourth tap is dropped",
            .df_silu_no_negative => "control: metal ricursive dflash2 silu, a negative gate takes its magnitude's sigmoid",
            .df_vec_no_floor => "control: metal ricursive dflash2 vec, each row also sees the keys below the floor",
            .df_topk_ascending => "control: metal ricursive dflash2 topk, the sort runs ascending",
            .df_walk_from_pending => "control: metal ricursive dflash2 walk, every slot scores after the pending token",
        };
    }

    /// The gate that must go red.
    pub fn gate(m: Mutation) ?contract.gates.Gate {
        return if (m == .none) null else .shapes;
    }

    const Edit = struct { from: []const u8, to: []const u8 };

    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .embed_own_row => .{ .from = "E[(size_t)ids[p.y] * a.width + p.x]", .to = "E[(size_t)p.y * a.width + p.x]" },
            .add_norm_no_residual => .{ .from = "      R[base + t + p * n] = bfloat(r[p]);\n", .to = "" },
            .lerp_no_blend => .{ .from = "rc_lerp(float(V0[base + i]), float(x), a.blend)", .to = "rc_lerp(float(V0[base + i]), float(x), 1.0f)" },
            .prep_ropes_global => .{ .from = "if (a.rope == 0) {", .to = "if (false) {" },
            .prep_next_slot => .{ .from = "const size_t slot = (a.pos + row) % a.ring;", .to = "const size_t slot = (a.pos + row + 1) % a.ring;" },
            .gate_early_round => .{ .from = "1.0f / (1.0f + precise::exp(-g))", .to = "1.0f / (1.0f + float(bfloat(precise::exp(-g))))" },
            .clamp_no_lower => .{ .from = "clamp(u0, -a.scale, a.scale)", .to = "min(u0, a.scale)" },
            .decode_past_window => .{ .from = "p + 1 > a.window ? p + 1 - a.window : 0;", .to = "p + 1 > a.window ? p - a.window : 0;" },
            .prompt_past_window => .{ .from = "col < row - (window - 1)", .to = "col < row - (window + 63)" },
            .prompt_short_ring => .{ .from = "kc + long(kpos % ring) * LD", .to = "kc + long(kpos % (ring - BKT)) * LD" },
            .fp4_high_first => .{ .from = "rc_e2m1(word >> sh), rc_e2m1(word >> (sh + 4u)), rc_e2m1(word >> (sh + 8u)), rc_e2m1(word >> (sh + 12u))", .to = "rc_e2m1(word >> (sh + 4u)), rc_e2m1(word >> sh), rc_e2m1(word >> (sh + 12u)), rc_e2m1(word >> (sh + 8u))" },
            .fp4_scale_bias => .{ .from = "int(e) - 7", .to = "int(e) - 6" },
            .fp4_global_late => .{ .from = "bfloat(sum * a.global)", .to = "bfloat(float(bfloat(sum)) * a.global)" },
            .fp4_dequant_unscaled => .{ .from = "D[at * 4 + c] = bfloat4(w[c]);", .to = "D[at * 4 + c] = bfloat4(w[c] / rc_e4m3(S[at]));" },
            .fp4_gemm_no_global => .{ .from = "acc[i][j] *= global;", .to = "acc[i][j] *= 1.0f;" },
            .df_tap_next_slot => .{ .from = "F[((size_t)p.y * a.taps + a.at) * a.width + p.x]", .to = "F[((size_t)p.y * a.taps + a.at + 1) * a.width + p.x]" },
            .df_norm_half_mean => .{ .from = "precise::rsqrt(total / float(width) + eps)", .to = "precise::rsqrt(total / float(2 * width) + eps)" },
            .df_norm16_no_weight => .{ .from = "bfloat(float(w[i]) * float(bfloat(float(x[i]) * inv)))", .to = "bfloat(float(bfloat(float(x[i]) * inv)))" },
            .df_embed_own_row => .{ .from = "float(E[(size_t)ids[p.y] * a.width + p.x])", .to = "float(E[(size_t)p.y * a.width + p.x])" },
            .df_prep_ropes_nope => .{ .from = "if (a.rope == 0) {", .to = "if (false) {" },
            .df_conv_three_taps => .{ .from = "for (uint t = 0; t < a.taps; ++t) {", .to = "for (uint t = 0; t + 1 < a.taps; ++t) {" },
            .df_silu_no_negative => .{ .from = "const float s = g < 0.0f ? e * y : y;", .to = "const float s = y;" },
            .df_vec_no_floor => .{ .from = "max(p + 1 > a.window ? p + 1 - a.window : 0, a.floor)", .to = "p + 1 > a.window ? p + 1 - a.window : 0" },
            .df_topk_ascending => .{ .from = "const bool down = (i & k) == 0;", .to = "const bool down = (i & k) != 0;" },
            .df_walk_from_pending => .{ .from = "prev = UI[e * a.top + bi];\n      seq[1 + e] = prev;", .to = "seq[1 + e] = UI[e * a.top + bi];" },
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

/// dflash2.metal as dflash2.zig and model.zig's setTaps compile it, with the control's edit.
pub fn dflash2Library(device: mtl.Device, a: std.mem.Allocator, m: Mutation) !mtl.Library {
    const src = try apply(a, ks.ricursive_dflash2, m);
    defer a.free(src);
    return family.dflash2Library(device, src);
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
