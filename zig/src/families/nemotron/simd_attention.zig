//! The decode attention before M5: generated kernels rewritten to read the simdgroup-matrix result layout, checked first.
const std = @import("std");
const mtl = @import("metal");

/// A lane's results of a 16 x N op: rows fm and fm + 8 take the two halves, columns fn + (i & 1) + 8 ((i >> 1) % (N / 8)).
const rewrites = [_][2][]const u8{
    .{ "  const short fn = ((qid & 2) | (lane & 1)) * 4;\n", "  const short fn = ((lane & 8) >> 1) | ((lane & 1) << 1);\n" },
    .{
        "      const int key = kt + (i >> 3) * 16 + fn + (i & 3);\n      s[i] = key < ((i & 4) ? n1 : n0) ? S[i] * scale[0] : -INFINITY;\n",
        "      const int key = kt + fn + (i & 1) + 8 * ((i >> 1) % (TK / 8));\n      s[i] = key < ((i & (TK / 4)) ? n1 : n0) ? S[i] * scale[0] : -INFINITY;\n",
    },
    .{ "{ if (i & 4) x1 = max(x1, s[i]); else x0 = max(x0, s[i]); }", "{ if (i & (TK / 4)) x1 = max(x1, s[i]); else x0 = max(x0, s[i]); }" },
    .{ "fast::exp(s[i] - ((i & 4) ? nm1 : nm0));", "fast::exp(s[i] - ((i & (TK / 4)) ? nm1 : nm0));" },
    .{
        "      y0 += (p[b * 8] + p[b * 8 + 1]) + (p[b * 8 + 2] + p[b * 8 + 3]);\n      y1 += (p[b * 8 + 4] + p[b * 8 + 5]) + (p[b * 8 + 6] + p[b * 8 + 7]);\n",
        "      y0 += (p[b * 4] + p[b * 4 + 1]) + (p[b * 4 + 2] + p[b * 4 + 3]);\n      y1 += (p[TK / 4 + b * 4] + p[TK / 4 + b * 4 + 1]) + (p[TK / 4 + b * 4 + 2] + p[TK / 4 + b * 4 + 3]);\n",
    },
    .{ "{ const float f = (i & 4) ? f1 : f0; Olo[i] *= f; }", "{ const float f = (i & (D / 4)) ? f1 : f0; Olo[i] *= f; }" },
    .{
        "  for (int q = 0; q < 16; q++) {\n    device float* dst = PO + (base + tile * 16 + fm + (q & 1) * 8) * D + (q >> 1) * 16 + fn;\n    *(device float4*)dst = float4(Olo[4 * q], Olo[4 * q + 1], Olo[4 * q + 2], Olo[4 * q + 3]);\n",
        "  for (int q = 0; q < D / 4; q++) {\n    device float* dst = PO + (base + tile * 16 + fm + (q / (D / 8)) * 8) * D + fn + 8 * (q % (D / 8));\n    *(device float2*)dst = float2(Olo[2 * q], Olo[2 * q + 1]);\n",
    },
};

/// A generated attention source with its result indexing in the simdgroup-matrix layout; the caller frees it.
pub fn rewrite(a: std.mem.Allocator, source: []const u8) ![]u8 {
    var text = try a.dupe(u8, source);
    errdefer a.free(text);
    for (rewrites) |r| {
        if (std.mem.count(u8, text, r[0]) != 1) return error.AttentionSource;
        const next = try std.mem.replaceOwned(u8, a, text, r[0], r[1]);
        a.free(text);
        text = next;
    }
    return text;
}

/// The attention's score (16 x 64) and output (16 x 128) results, and the tree tail's (16 x 32, threadgroup operands).
const check_source =
    \\#include <metal_stdlib>
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace metal;
    \\using namespace mpp::tensor_ops;
    \\[[kernel]] void tf_attn_layout(const device bfloat* X [[buffer(0)]], device int* OK [[buffer(1)]],
    \\    uint lane [[thread_index_in_simdgroup]]) {
    \\  constexpr int TK = 64, D = 128;
    \\  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tQ((device bfloat*)X, dextents<int32_t, 2>(D, 16));
    \\  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tK((device bfloat*)X, dextents<int32_t, 2>(D, TK));
    \\  threadgroup half Ps[16 * TK];
    \\  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(Ps, dextents<int32_t, 2>(TK, 16));
    \\  constexpr auto dS = matmul2d_descriptor(16, TK, D, false, true, false, matmul2d_descriptor::mode::multiply);
    \\  constexpr auto dO = matmul2d_descriptor(16, D, TK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    \\  matmul2d<dS, execution_simdgroup> opS;
    \\  matmul2d<dO, execution_simdgroup> opO;
    \\  auto aQ = tQ.slice(0, 0);
    \\  auto bK = tK.slice(0, 0);
    \\  auto S = opS.template get_destination_cooperative_tensor<decltype(aQ), decltype(bK), float>();
    \\  auto O = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(bK), float>();
    \\  const int fm = ((lane & 16) >> 2) | ((lane >> 1) & 3), fn = ((lane & 8) >> 1) | ((lane & 1) << 1);
    \\  bool ok = true;
    \\  for (int i = 0; i < TK / 2; i++) {
    \\    auto at = S.get_multidimensional_index(i);
    \\    ok = ok && at[1] == fm + ((i & (TK / 4)) ? 8 : 0) && at[0] == fn + (i & 1) + 8 * ((i >> 1) % (TK / 8));
    \\  }
    \\  for (int i = 0; i < D / 2; i++) {
    \\    auto at = O.get_multidimensional_index(i);
    \\    ok = ok && at[1] == fm + ((i & (D / 4)) ? 8 : 0) && at[0] == fn + (i & 1) + 8 * ((i >> 1) % (D / 8));
    \\  }
    \\  threadgroup bfloat KV[TK * D];
    \\  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tK32(KV, dextents<int32_t, 2>(D, 32));
    \\  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tVh(KV, dextents<int32_t, 2>(D, TK));
    \\  constexpr auto dS32 = matmul2d_descriptor(16, 32, D, false, true, false, matmul2d_descriptor::mode::multiply);
    \\  matmul2d<dS32, execution_simdgroup> opS32;
    \\  auto S32 = opS32.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK32), float>();
    \\  for (int i = 0; i < 16; i++) {
    \\    auto at = S32.get_multidimensional_index(i);
    \\    ok = ok && at[1] == fm + ((i & 8) ? 8 : 0) && at[0] == fn + (i & 1) + 8 * ((i >> 1) % 4);
    \\  }
    \\  auto Ot = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(tVh), float>();
    \\  for (int i = 0; i < D / 2; i++) {
    \\    auto at = Ot.get_multidimensional_index(i);
    \\    ok = ok && at[1] == fm + ((i & (D / 4)) ? 8 : 0) && at[0] == fn + (i & 1) + 8 * ((i >> 1) % (D / 8));
    \\  }
    \\  OK[lane] = ok ? 1 : 0;
    \\}
;

/// The layout the rewrite reads, checked on this GPU: error.AttentionLayout when its tensor ops lay results out otherwise.
pub fn check(device: mtl.Device, queue: mtl.Queue) !void {
    const lib = try mtl.Library.fromSource(device, check_source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, "tf_attn_layout", false);
    defer pipe.deinit();
    const x = try device.buffer(128 * 64 * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    const ok = try device.buffer(32 * 4, mtl.ResourceOptions.shared);
    defer ok.deinit();
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(pipe);
    enc.setBuffer(x, 0, 0);
    enc.setBuffer(ok, 0, 1);
    enc.dispatchThreads(mtl.Size.of(32, 1, 1), mtl.Size.of(32, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("attention layout check failed: {s}", .{msg});
        return error.GpuFailed;
    }
    for (ok.slice(i32, 32)) |v| if (v != 1) {
        std.log.err("decode attention: the tensor op's results are not in the simdgroup-matrix layout on {s}", .{device.name()});
        return error.AttentionLayout;
    };
}

test "the rewrite finds every line it changes once in a generated attention kernel" {
    const sources = @import("kernel_sources");
    inline for (sources.nemotron.all) |k| {
        if (comptime std.mem.startsWith(u8, k.key, "attn_partial")) {
            const text = try rewrite(std.testing.allocator, k.source);
            defer std.testing.allocator.free(text);
            try std.testing.expect(std.mem.indexOf(u8, text, "(i & 4)") == null);
        }
    }
}
