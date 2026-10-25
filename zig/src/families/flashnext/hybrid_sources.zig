//! Checked specializations reuse maintained Flash Next kernels without changing its existing source or entry points.
const std = @import("std");
fn once(a: std.mem.Allocator, source: []const u8, before: []const u8, after: []const u8) ![]u8 {
    if (std.mem.count(u8, source, before) != 1) return error.HybridSourceChanged;
    return std.mem.replaceOwned(u8, a, source, before, after);
}
pub const Gdn = struct {
    pub fn metal(a: std.mem.Allocator, source: []const u8, recorded: []const u8, keys: usize, values: usize, sigmoid: bool) ![]u8 {
        if (keys == 0 or values == 0 or keys > values or values % keys != 0 or values > std.math.maxInt(i32) / (16 * 514))
            return error.HybridGdnShape;
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        const mem = arena.allocator();
        const start = std.mem.indexOf(u8, recorded, "inline float bsig(") orelse return error.HybridSourceChanged;
        const last = std.mem.indexOfPos(u8, recorded, start, "inline float fsoftplus(") orelse return error.HybridSourceChanged;
        const end = last + (std.mem.indexOfScalar(u8, recorded[last..], '\n') orelse return error.HybridSourceChanged);
        var math = recorded[start..end];
        math = try once(mem, math, "inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }", "inline float bsig(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? e : 1.0f) / (1.0f + e))); }");
        math = try once(mem, math, "inline float bsilu(float x) { return float(bfloat(x / (1.0f + metal::exp(-x)))); }", "inline float bsilu(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e))); }");
        math = try once(mem, math, "inline float fsig(float x) { return 1.0f / (1.0f + metal::exp(-x)); }", "inline float fsig(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? e : 1.0f) / (1.0f + e); }\ninline float fsilu(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? x * e : x) / (1.0f + e); }");
        const geometry = try std.fmt.allocPrint(mem, "constant constexpr int GDN_NK = {d}, GDN_NV = {d}, GDN_DK = 128, GDN_DV = 128, GDN_TAPS = 4, GDN_MAXR = 16;", .{ keys, values });
        var text = try once(mem, source, "constant constexpr int GDN_NK = 16, GDN_NV = 48, GDN_DK = 128, GDN_DV = 128, GDN_TAPS = 4, GDN_MAXR = 16;", geometry);
        text = try once(mem, text, "void fz_gdn(", "void qwen4_gdn_chain(");
        text = try once(mem, text, "const device bfloat* CW [[buffer(3)]]", "const device float* CW [[buffer(3)]]");
        text = try once(mem, text, "const device bfloat* ALOG [[buffer(4)]]", "const device float* ALOG [[buffer(4)]]");
        text = try once(mem, text, "const device bfloat* DT [[buffer(5)]]", "const device float* DT [[buffer(5)]]");
        text = try once(mem, text, "const device bfloat* NW [[buffer(6)]]", "const device float* NW [[buffer(6)]]");
        text = try once(mem, text, "bfloat(y * fsig(z))", if (sigmoid) "bfloat(y * fsig(z))" else "bfloat(y * fsilu(z))");
        text = try once(mem, text, "const bfloat act = bfloat(bsilu(conv));", "const bfloat act = bfloat(bsilu(CONV_ROUND ? float(bfloat(conv)) : conv));");
        text = try once(mem, text, "device float* SO [[buffer(11)]], constant uint& H0 [[buffer(12)]],", "device float* SO [[buffer(11)]], constant uint& H0 [[buffer(12)]], constant ulong& STATE_STRIDE [[buffer(14)]], constant uint& NORM_ROUND [[buffer(15)]], constant uint& CONV_ROUND [[buffer(16)]],");
        text = try once(mem, text, "const float y = float(bfloat(float(NW[d]) * float(bfloat(float(ys[r][d]) * red[r]))));", "const float weighted = float(NW[d]) * float(bfloat(float(ys[r][d]) * red[r]));\n    const float y = NORM_ROUND ? float(bfloat(weighted)) : weighted;");
        text = try once(mem, text, "CSO[(r * (TAPS - 1) + q) * C + c]", "CSO[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 2 : (TAPS - 1) * C) + q * C + c]");
        text = try once(mem, text, "SO4[((size_t(r) * NV + hv) * DV + sg * RPS + j) * DK / 4 + lane]", "SO4[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 16 : NV * DV * DK / 4) + ((size_t(hv) * DV + sg * RPS + j) * DK / 4 + lane)]");
        return std.mem.concat(a, u8, &.{ "#include <metal_stdlib>\nusing namespace metal;\n#define FZ_GDN_REPLAY 0\n#define FZ_GDN_TREE 0\n", math, "\n", text });
    }
    /// Native state and norm arithmetic retain 128-wide heads; head counts are runtime geometry.
    pub fn cuda(a: std.mem.Allocator, source: []const u8) ![]u8 {
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        const mem = arena.allocator();
        var text = try once(mem, source, "constexpr int NK = 16, NV = 48, DK = 128, DV = 128, TAPS = 4;", "constexpr int DK = 128, DV = 128, TAPS = 4;");
        text = try once(mem, text, "flashnext_gdn_chain(", "qwen4_gdn_chain(");
        text = try once(mem, text, "static __device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }", "static __device__ __forceinline__ float sigmoidf_(float x) { const float e = expf(-fabsf(x)); return (x < 0 ? e : 1.0f) / (1.0f + e); }\nstatic __device__ __forceinline__ float siluf_(float x) { const float e = expf(-fabsf(x)); return (x < 0 ? x * e : x) / (1.0f + e); }");
        text = try once(mem, text, "const __nv_bfloat16* __restrict__ cw,", "const void* __restrict__ cw,");
        text = try once(mem, text, "const __nv_bfloat16* __restrict__ norm_w,", "const void* __restrict__ norm_w,");
        text = try once(mem, text, "float* __restrict__ state_out) {", "float* __restrict__ state_out, int NK, int NV, int gate, int weight_flags, unsigned long long STATE_STRIDE) {");
        text = try once(mem, text, "constexpr int C = 2 * NK * DK + NV * DV;", "const int C = 2 * NK * DK + NV * DV;");
        text = try once(mem, text, "constexpr int PW = C + NV * DV + 2 * NV;", "const int PW = C + NV * DV + 2 * NV;");
        text = try once(mem, text, "const float act = bf(acc / (1.0f + expf(-acc)));", "const float conv = (weight_flags & 8) ? bf(acc) : acc;\n            const float act = bf(siluf_(conv));");
        text = try once(mem, text, "__bfloat162float(cw[c * TAPS + tap])", "((weight_flags & 1) ? static_cast<const float*>(cw)[c * TAPS + tap] : __bfloat162float(static_cast<const __nv_bfloat16*>(cw)[c * TAPS + tap]))");
        text = try once(mem, text, "__bfloat162float(norm_w[t])", "((weight_flags & 2) ? static_cast<const float*>(norm_w)[t] : __bfloat162float(static_cast<const __nv_bfloat16*>(norm_w)[t]))");
        text = try once(mem, text, "const float yn = bf(bf(ys[t] * rinv) * ((weight_flags & 2) ? static_cast<const float*>(norm_w)[t] : __bfloat162float(static_cast<const __nv_bfloat16*>(norm_w)[t])));", "const float weighted = bf(ys[t] * rinv) * ((weight_flags & 2) ? static_cast<const float*>(norm_w)[t] : __bfloat162float(static_cast<const __nv_bfloat16*>(norm_w)[t]));\n            const float yn = (weight_flags & 4) ? weighted : bf(weighted);");
        text = try once(mem, text, "const float o = bf(yn * sigmoidf_(z));", "const float o = bf(yn * (gate ? sigmoidf_(z) : siluf_(z)));");
        // Keep every verified row, so partial acceptance restores its own recurrent state without replay.
        text = try once(mem, text, "const float* __restrict__ state_in,", "const float* state_in,");
        text = try once(mem, text, "float* __restrict__ state_out, int NK,", "float* state_out, int NK,");
        text = try once(mem, text,
            \\    }
            \\    if (state_out != nullptr) {
            \\#pragma unroll
            \\        for (int j = 0; j < 4; ++j)
            \\#pragma unroll
            \\            for (int i = 0; i < 4; ++i)
            \\                state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
            \\    }
            \\}
            \\
        ,
            \\        if (state_out != nullptr) {
            \\#pragma unroll
            \\            for (int j = 0; j < 4; ++j)
            \\#pragma unroll
            \\                for (int i = 0; i < 4; ++i)
            \\                    state_out[(size_t)r * STATE_STRIDE + sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
            \\        }
            \\    }
            \\}
            \\
        );
        return a.dupe(u8, text);
    }
};
