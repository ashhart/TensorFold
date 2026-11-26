//! One host-checkable source-planted fault for every Inkling 2 Mini image-tower kernel family.
const std = @import("std");
const contract = @import("contract");

pub const Kind = enum { pack, embed, qkv, heads, bias_gelu, gelu_rows, residual, layer_norm, softmax, gemm_tail, gemm_rows, dot_tail };

pub const Edit = struct { label: []const u8, from: []const u8, to: []const u8, gate: contract.spec.Gate };

pub fn edit(kind: Kind) Edit {
    const g = contract.spec.Gate;
    return switch (kind) {
        .pack => .{
            .label = "control: metal inkling2 vision ik2v_pack writes past payload",
            .from = "if (i < n) y[i] = bfloat(x[i]);",
            .to = "if (i < n) y[i] = bfloat(x[i]); if (i == 0) y[n] = bfloat(0);",
            .gate = g.repeat,
        },
        .embed => .{
            .label = "control: metal inkling2 vision ik2v_embed reuses the first bilinear weight",
            .from = "* weights[i * 4 + k];",
            .to = "* weights[i * 4];",
            .gate = g.shapes,
        },
        .qkv => .{
            .label = "control: metal inkling2 vision ik2v_qkv swaps rotary pairing",
            .from = "const uint e = d ^ 1u;",
            .to = "const uint e = d ^ 2u;",
            .gate = g.shapes,
        },
        .heads => .{
            .label = "control: metal inkling2 vision ik2v_heads ignores the chunk's first row",
            .from = "ulong(p.r0 + at.y)",
            .to = "ulong(at.y)",
            .gate = g.shapes,
        },
        .bias_gelu => .{
            .label = "control: metal inkling2 vision ik2v_bias_gelu drops the bias",
            .from = "ik2v_gelu(tf_vis_bf(acc[i] + float(b[at.x])))",
            .to = "ik2v_gelu(tf_vis_bf(acc[i]))",
            .gate = g.shapes,
        },
        .gelu_rows => .{
            .label = "control: metal inkling2 vision ik2v_gelu_rows loses the sign",
            .from = "ik2v_gelu(float(y[i]))",
            .to = "ik2v_gelu(fabs(float(y[i])))",
            .gate = g.shapes,
        },
        .residual => .{
            .label = "control: metal inkling2 vision ik2v_residual omits LayerScale",
            .from = "float(lambda[at.x]) * branch",
            .to = "branch",
            .gate = g.shapes,
        },
        .layer_norm => .{
            .label = "control: metal inkling2 vision ik2v_layer_norm repeats channels",
            .from = "#define IK2V_ROW(i) float(xr[i])",
            .to = "#define IK2V_ROW(i) float(xr[(i) % 2])",
            .gate = g.shapes,
        },
        .softmax => .{
            .label = "control: metal inkling2 vision ik2v_softmax ignores NaN exp",
            .from = "const float e = metal::precise::exp(s[i] - m);",
            .to = "const float e = isnan(s[i] - m) ? 0.0f : metal::precise::exp(s[i] - m);",
            .gate = g.nonfinite,
        },
        .gemm_tail => .{
            .label = "control: metal inkling2 vision ik2_gemm_bf16f_nt drops ragged K tail",
            .from = "gemm_block<T, TA, TB, false, 256>(acc, A, B, lda, ldb, P[2], P[9], rows, cols, home);",
            .to = "gemm_block<T, TA, TB, false, 256>(acc, A, B, lda, ldb, P[2] - 1, (P[2] - 1) / 256, rows, cols, home);",
            .gate = g.shapes,
        },
        .gemm_rows => .{
            .label = "control: metal inkling2 vision ik2_gemm_bf16_nt_rows feeds block 1 the weights of block 0",
            .from = "simdgroup_multiply_accumulate(acc1, fa, fb1, acc1);",
            .to = "simdgroup_multiply_accumulate(acc1, fa, fb0, acc1);",
            .gate = g.shapes,
        },
        .dot_tail => .{
            .label = "control: metal inkling2 vision ik2_gemm_f32_nt drops the last k",
            .from = "kc = min(IK2_DOT_KC, K - next);",
            .to = "kc = min(IK2_DOT_KC, K - next - 1);",
            .gate = g.shapes,
        },
    };
}

pub fn source(a: std.mem.Allocator, production: []const u8, kind: Kind) ![]u8 {
    const e = edit(kind);
    if (std.mem.count(u8, production, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, e.from, e.to);
}

test "every vision control applies once and changes only its anchor" {
    const production = @import("inkling2_metal").vision;
    const a = std.testing.allocator;
    for (std.enums.values(Kind)) |kind| {
        const e = edit(kind);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, production, e.from));
        try std.testing.expect(!std.mem.eql(u8, e.from, e.to));
        const changed = try source(a, production, kind);
        defer a.free(changed);
        const at = std.mem.indexOf(u8, production, e.from).?;
        try std.testing.expectEqualStrings(production[0..at], changed[0..at]);
        try std.testing.expectEqualStrings(e.to, changed[at..][0..e.to.len]);
        try std.testing.expectEqualStrings(production[at + e.from.len ..], changed[at + e.to.len ..]);
        try std.testing.expectError(error.ControlEditMissing, source(a, "", kind));
    }
}

test "a repeated vision control anchor is ambiguous and refuses to prove a fault" {
    const a = std.testing.allocator;
    const production = @import("inkling2_metal").vision;
    const twice = try std.mem.concat(a, u8, &.{ production, production });
    defer a.free(twice);
    for (std.enums.values(Kind)) |kind| try std.testing.expectError(error.ControlEditMissing, source(a, twice, kind));
}
