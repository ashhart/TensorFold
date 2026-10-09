//! The tail attention before M5: results read in the simdgroup-matrix layout, checked on the GPU first.
const std = @import("std");
const mtl = @import("metal");

/// A lane's 16 x N results before M5: rows fm, fm + 8; columns fn + (i & 1) + 8 ((i >> 1) % (N / 8)).
const rewrites = [_][2][]const u8{
    .{ "  const short fn = ((qid & 2) | (lane & 1)) * 4;\n", "  const short fn = ((lane & 8) >> 1) | ((lane & 1) << 1);\n" },
    .{
        "    for (int q = 0; q < 16; q++) {\n      const int row = fm + (q & 1) * 8;\n      if (row >= G) continue;\n      const auto src = POA + (baseA + row) * D + (q >> 1) * 16 + fn;\n      for (int j = 0; j < 4; j++) { Olo[4 * q + j] = src[j]; Ohi[4 * q + j] = src[128 + j]; }\n",
        "    for (int q = 0; q < 32; q++) {\n      const int row = fm + (q / 16) * 8;\n      if (row >= G) continue;\n      const auto src = POA + (baseA + row) * D + fn + 8 * (q % 16);\n      for (int j = 0; j < 2; j++) { Olo[2 * q + j] = src[j]; Ohi[2 * q + j] = src[128 + j]; }\n",
    },
    .{
        "      const int key = kt + (i >> 3) * 16 + fn + (i & 3);\n      s[i] = key < ((i & 4) ? n1 : n0) ? sraw[i] * scale[0] : -INFINITY;\n",
        "      const int key = kt + (i >> 4) * 32 + fn + (i & 1) + 8 * ((i >> 1) & 3);\n      s[i] = key < ((i & 8) ? n1 : n0) ? sraw[i] * scale[0] : -INFINITY;\n",
    },
    .{ "{ if (i & 4) x1 = max(x1, s[i]); else x0 = max(x0, s[i]); }", "{ if (i & 8) x1 = max(x1, s[i]); else x0 = max(x0, s[i]); }" },
    .{ "fast::exp(s[i] - ((i & 4) ? nm1 : nm0));", "fast::exp(s[i] - ((i & 8) ? nm1 : nm0));" },
    .{
        "      y0 += (p[b * 8] + p[b * 8 + 1]) + (p[b * 8 + 2] + p[b * 8 + 3]);\n      y1 += (p[b * 8 + 4] + p[b * 8 + 5]) + (p[b * 8 + 6] + p[b * 8 + 7]);\n",
        "      const int o = (b >> 1) * 16 + (b & 1) * 4;\n      y0 += (p[o] + p[o + 1]) + (p[o + 2] + p[o + 3]);\n      y1 += (p[o + 8] + p[o + 9]) + (p[o + 10] + p[o + 11]);\n",
    },
    .{
        "    for (int f = 0; f < TK / 16; f++)\n      for (int i = 0; i < 4; i++) {\n        myP[fm * TK + f * 16 + fn + i] = half(p[f * 8 + i]);\n        myP[(fm + 8) * TK + f * 16 + fn + i] = half(p[f * 8 + 4 + i]);\n      }\n",
        "    for (int i = 0; i < TK / 2; i++) myP[(fm + ((i & 8) ? 8 : 0)) * TK + (i >> 4) * 32 + fn + (i & 1) + 8 * ((i >> 1) & 3)] = half(p[i]);\n",
    },
    .{ "{ const float f = (i & 4) ? f1 : f0; Olo[i] *= f; Ohi[i] *= f; }", "{ const float f = (i & 32) ? f1 : f0; Olo[i] *= f; Ohi[i] *= f; }" },
    .{
        "  for (int q = 0; q < 16; q++) {\n    device float* dst = PO + (base + fm + (q & 1) * 8) * D + (q >> 1) * 16 + fn;\n    *(device float4*)dst = float4(Olo[4 * q], Olo[4 * q + 1], Olo[4 * q + 2], Olo[4 * q + 3]);\n    *(device float4*)(dst + 128) = float4(Ohi[4 * q], Ohi[4 * q + 1], Ohi[4 * q + 2], Ohi[4 * q + 3]);\n",
        "  for (int q = 0; q < 32; q++) {\n    device float* dst = PO + (base + fm + (q / 16) * 8) * D + fn + 8 * (q % 16);\n    *(device float2*)dst = float2(Olo[2 * q], Olo[2 * q + 1]);\n    *(device float2*)(dst + 128) = float2(Ohi[2 * q], Ohi[2 * q + 1]);\n",
    },
};

/// The attention source with the tail kernel's result indexing in the simdgroup-matrix layout; the caller frees it.
pub fn rewrite(a: std.mem.Allocator, source: []const u8) ![]u8 {
    const begin = std.mem.indexOf(u8, source, "kernel void q27_attn_tail(") orelse return error.AttentionSource;
    const end = begin + (std.mem.indexOf(u8, source[begin..], "kernel void q27_attn_merge(") orelse return error.AttentionSource);
    var tail = try a.dupe(u8, source[begin..end]);
    defer a.free(tail);
    for (rewrites) |r| {
        if (std.mem.count(u8, tail, r[0]) != 1) return error.AttentionSource;
        const next = try std.mem.replaceOwned(u8, a, tail, r[0], r[1]);
        a.free(tail);
        tail = next;
    }
    return std.mem.concat(a, u8, &.{ source[0..begin], tail, source[end..] });
}

/// The tail kernel's ops (scores 16x32 over 256, outputs 16x128 over 64), each element where it is read.
const check_source =
    \\#include <metal_stdlib>
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace metal;
    \\using namespace mpp::tensor_ops;
    \\[[kernel]] void tf_q27_attn_layout(const device bfloat* X [[buffer(0)]], device int* OK [[buffer(1)]], uint lane [[thread_index_in_simdgroup]]) {
    \\  constexpr int TK = 64, D = 256;
    \\  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tQ((device bfloat*)X, dextents<int32_t, 2>(D, 16));
    \\  threadgroup bfloat KV[32 * D];
    \\  threadgroup half Ps[16 * TK];
    \\  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tK32(KV, dextents<int32_t, 2>(D, 32));
    \\  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tVh(KV, dextents<int32_t, 2>(128, TK));
    \\  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(Ps, dextents<int32_t, 2>(TK, 16));
    \\  constexpr auto dS = matmul2d_descriptor(16, 32, D, false, true, false, matmul2d_descriptor::mode::multiply);
    \\  constexpr auto dO = matmul2d_descriptor(16, 128, TK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    \\  matmul2d<dS, execution_simdgroup> opS;
    \\  matmul2d<dO, execution_simdgroup> opO;
    \\  auto S = opS.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK32), float>();
    \\  auto O = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(tVh), float>();
    \\  const int fm = ((lane & 16) >> 2) | ((lane >> 1) & 3), fn = ((lane & 8) >> 1) | ((lane & 1) << 1);
    \\  bool ok = true;
    \\  for (int i = 0; i < 16; i++) {
    \\    auto at = S.get_multidimensional_index(i);
    \\    ok = ok && at[1] == fm + ((i & 8) ? 8 : 0) && at[0] == fn + (i & 1) + 8 * ((i >> 1) & 3);
    \\  }
    \\  for (int i = 0; i < 64; i++) {
    \\    auto at = O.get_multidimensional_index(i);
    \\    ok = ok && at[1] == fm + ((i & 32) ? 8 : 0) && at[0] == fn + (i & 1) + 8 * ((i >> 1) % 16);
    \\  }
    \\  OK[lane] = ok ? 1 : 0;
    \\}
;

/// The layout the rewrite reads, checked on this GPU: error.AttentionLayout when results lie otherwise.
pub fn check(device: mtl.Device, queue: mtl.Queue) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const lib = try mtl.Library.fromSource(device, check_source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, "tf_q27_attn_layout", false);
    defer pipe.deinit();
    const x = try device.buffer(256 * 16 * 2, mtl.ResourceOptions.shared);
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
        std.log.err("qwen27 attention layout check failed: {s}", .{msg});
        return error.GpuFailed;
    }
    for (ok.slice(i32, 32)) |v| if (v != 1) {
        std.log.err("qwen27 attention: the tensor op's results are not in the simdgroup-matrix layout on {s}", .{device.name()});
        return error.AttentionLayout;
    };
}

test "the rewrite changes every M5-indexed line of the tail kernel once and leaves the other kernels alone" {
    const source = @import("qwen_runtime_sources").attention_mpp;
    const text = try rewrite(std.testing.allocator, source);
    defer std.testing.allocator.free(text);
    const begin = std.mem.indexOf(u8, text, "kernel void q27_attn_tail(").?;
    const end = begin + std.mem.indexOf(u8, text[begin..], "kernel void q27_attn_merge(").?;
    try std.testing.expect(std.mem.indexOf(u8, text[begin..end], "(i & 4)") == null);
    try std.testing.expect(std.mem.indexOf(u8, text[begin..end], "float4(Olo") == null);
    try std.testing.expectEqualStrings(source[0..begin], text[0..begin]);
    try std.testing.expectEqualStrings(source[source.len - (text.len - end) ..], text[end..]);
}
