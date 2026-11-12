const std = @import("std");

pub fn source(a: std.mem.Allocator, extra: []const u8) ![]u8 {
    const attention = try attentionSource(a);
    defer a.free(attention);
    const combined = try std.mem.concat(a, u8, &.{
        "#define TF_NAX_STRICT_FP32\n",          @embedFile("prefill/gemm_nax.metal") ++ @embedFile("prefill/gemm_nax_kernel.metal"), @embedFile("core/qwen_vision.metal"),
        @embedFile("tensor_attention_pv.metal"), attention,                                                                           @embedFile("tensor_ops.metal"),
        @embedFile("tensor_attention.metal"),    extra,
    });
    defer a.free(combined);
    return std.mem.replaceOwned(u8, a, combined, "#include \"../nax.h\"", "\n#ifndef TF_TENSOR_NAX_INCLUDED\n#define TF_TENSOR_NAX_INCLUDED\n" ++ @embedFile("nax.h") ++ "\n#endif\n");
}

fn attentionSource(a: std.mem.Allocator) ![]u8 {
    var text = try a.dupe(u8, @embedFile("prefill/attention_nax.metal"));
    errdefer a.free(text);
    const changes = .{
        .{ "attention_pv(acc[d], acc[d + 1], s[k], v0, v1, ushort(lane));", "if constexpr (DIM >= 128) tensor_attention_pv(acc[d], acc[d + 1], s[k], v0, v1, ushort(lane)); else attention_pv(acc[d], acc[d + 1], s[k], v0, v1, ushort(lane));", 1 },
        .{ "template <typename T, bool AQ, bool AK, bool CAUSAL>", "template <typename T, bool AQ, bool AK, bool CAUSAL, int DIM = 128, int WINDOW = 0, bool TILE_MASK = false>", 1 },
        .{ "constexpr int BQ = 64, BK = 32;", "constexpr int BQ = 64, BK = 32, VDIM = DIM > 128 ? 128 : DIM;\n  const int batch_index = int(tg.z) / (DIM / VDIM), value_part = int(tg.z) % (DIM / VDIM);", 1 },
        .{ "long(tg.z)", "long(batch_index)", 4 },
        .{ "V += long(batch_index) * P[18] + long(kv) * P[19];", "V += long(batch_index) * P[18] + long(kv) * P[19] + value_part * VDIM;", 1 },
        .{ "long(tg.x) * BQ * os;", "long(tg.x) * BQ * os + value_part * VDIM;", 1 },
        .{ "uint3 tg, uint sg, uint lane)", "uint3 tg, uint sg, uint lane, const device int* sizes = nullptr)", 1 },
        .{ "float top_new[2]", "if (TILE_MASK) { for (short f = 0; f < 2; f++) { for (short e = 0; e < 8; e++) { const int col = kb * BK + 16 * f + home.x + TF_COL(e); if (col % 64 >= sizes[col / 64]) s[f][e] = kMasked; } } }\n    float top_new[2]", 1 },
        .{ "if (CAUSAL && kb >= first_masked)", "if (CAUSAL && (WINDOW > 0 || kb >= first_masked))", 1 },
        .{ "(r0 + (e >> 2) * 8) < (c0 + 16 * f + TF_COL(e)) ?", "((r0 + (e >> 2) * 8) < (c0 + 16 * f + TF_COL(e)) || (WINDOW > 0 && (r0 + (e >> 2) * 8) >= (c0 + 16 * f + TF_COL(e)) + WINDOW)) ?", 1 },
        .{ "acc[8]", "acc[VDIM / 16]", 1 },
        .{ "short i = 0; i < 8;", "short i = 0; i < VDIM / 16;", 5 },
        .{ "short d = 0; d < 8; d++)", "short d = 0; d < DIM / 16; d++)", 1 },
        .{ "short d = 0; d < 8; d += 2)", "short d = 0; d < VDIM / 16; d += 2)", 1 },
    };
    inline for (changes) |change| {
        if (std.mem.count(u8, text, change[0]) != change[2]) return error.TensorFoldKernelChanged;
        const next = try std.mem.replaceOwned(u8, a, text, change[0], change[1]);
        a.free(text);
        text = next;
    }
    return text;
}
