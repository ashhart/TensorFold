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

/// The prefix kernels before M5: scores 16 x 64 (row half at i & 16), outputs 16 x 128, sums in the tail's order.
const prefix_rewrites = [_][2][]const u8{
    .{ "  const short fn = ((qid & 2) | (lane & 1)) * 4;\n", "  const short fn = ((lane & 8) >> 1) | ((lane & 1) << 1);\n" },
    .{
        "      const int key = kt + (i >> 3) * 16 + fn + (i & 3);\n      s[i] = key < ((i & 4) ? e1 : e0) ? S[i] * scale[0] : -INFINITY;\n",
        "      const int key = kt + fn + (i & 1) + 8 * ((i >> 1) & 7);\n      s[i] = key < ((i & 16) ? e1 : e0) ? S[i] * scale[0] : -INFINITY;\n",
    },
    .{ "{ if (i & 4) x1 = max(x1, s[i]); else x0 = max(x0, s[i]); }", "{ if (i & 16) x1 = max(x1, s[i]); else x0 = max(x0, s[i]); }" },
    .{ "fast::exp(s[i] - ((i & 4) ? nm1 : nm0));", "fast::exp(s[i] - ((i & 16) ? nm1 : nm0));" },
    .{
        "      y0 += (p[b * 8] + p[b * 8 + 1]) + (p[b * 8 + 2] + p[b * 8 + 3]);\n      y1 += (p[b * 8 + 4] + p[b * 8 + 5]) + (p[b * 8 + 6] + p[b * 8 + 7]);\n",
        "      y0 += (p[b * 4] + p[b * 4 + 1]) + (p[b * 4 + 2] + p[b * 4 + 3]);\n      y1 += (p[b * 4 + 16] + p[b * 4 + 17]) + (p[b * 4 + 18] + p[b * 4 + 19]);\n",
    },
    .{ "    for (int i = 0; i < 64; i++) O[i] *= (i & 4) ? f1 : f0;\n", "    for (int i = 0; i < 64; i++) O[i] *= (i & 32) ? f1 : f0;\n" },
    .{
        "      const int b = d >> 4, u = d & 3;\n",
        "      const int at = ((d >> 3) << 1) | (d & 1); // element 2 (d / 8) + d % 2 (+ 16) of the lane with fn = d % 8 rounded to 2\n",
    },
    .{
        "        if ((i >> 2) == b && (i & 3) == u) { va = p[(i >> 2) * 8 + (i & 3)]; vb = p[(i >> 2) * 8 + 4 + (i & 3)]; }\n      const short g = (d >> 2) & 3;\n",
        "        if (i == at) { va = p[i]; vb = p[16 + i]; }\n      const short g = (d >> 1) & 3;\n",
    },
    .{
        "      for (int t = 0; t < 8; t++)\n        for (int x = 0; x < 4; x++) {\n          const float val = float(vk[t * 16 + x]);\n          if (key < e0) O[t * 8 + x] += pa * val;\n          if (key < e1) O[t * 8 + 4 + x] += pb * val;\n",
        "      for (int t = 0; t < 16; t++)\n        for (int x = 0; x < 2; x++) {\n          const float val = float(vk[t * 8 + x]);\n          if (key < e0) O[t * 2 + x] += pa * val;\n          if (key < e1) O[32 + t * 2 + x] += pb * val;\n",
    },
    .{
        "  for (int q = 0; q < 16; q++) {\n    if ((q & 1) ? q1 >= G * NQ : q0 >= G * NQ) continue;\n    device float* dst = PO + (base + tile * 16 + fm + (q & 1) * 8) * D + (hi ? 128 : 0) + (q >> 1) * 16 + fn;\n    *(device float4*)dst = float4(O[4 * q], O[4 * q + 1], O[4 * q + 2], O[4 * q + 3]);\n",
        "  for (int q = 0; q < 32; q++) {\n    if ((q / 16) ? q1 >= G * NQ : q0 >= G * NQ) continue;\n    device float* dst = PO + (base + tile * 16 + fm + (q / 16) * 8) * D + (hi ? 128 : 0) + fn + 8 * (q % 16);\n    *(device float2*)dst = float2(O[2 * q], O[2 * q + 1]);\n",
    },
};

/// Whether cached keys take the batched prefix kernels: M5 always; before M5 unless TF_QWEN27_SHARED_PREFIX=0.
pub fn sharedPrefix(units: bool) !bool {
    return units or try @import("sys").environment.read(bool, "TF_QWEN27_SHARED_PREFIX", true);
}

/// The attention kernels as this chip builds them (before M5: prefix and tail rewritten, prompt kernels if `prompt`).
pub fn library(a: std.mem.Allocator, units: bool, prompt: bool) ![]u8 {
    return compose(a, units, prompt, @import("qwen_runtime_sources"));
}

/// As `library`, its K/V kernels reading pages of the core pool (TF_QWEN27_METAL_KV_PAGES), under main's names.
pub fn pagedLibrary(a: std.mem.Allocator, units: bool, prompt: bool) ![]u8 {
    return compose(a, units, prompt, Paged);
}

const Paged = struct {
    const twins = @import("kernel_sources").qwen27_paged;
    pub const glue = @import("qwen_runtime_sources").glue;
    pub const attention_mpp = twins.attention_mpp;
    pub const attention_io = twins.attention_io;
    pub const attention_prompt = twins.attention_prompt;
    pub const attention_prompt_sg = twins.attention_prompt_sg;
};

fn compose(a: std.mem.Allocator, units: bool, prompt: bool, comptime source: type) ![]u8 {
    if (units) return a.dupe(u8, source.glue ++ source.attention_mpp ++ source.attention_io ++ source.attention_prompt);
    const tail = try rewrite(a, source.glue ++ source.attention_mpp ++ source.attention_io);
    if (!prompt) return tail;
    defer a.free(tail);
    return std.mem.concat(a, u8, &.{ tail, source.attention_prompt, source.attention_prompt_sg });
}

/// `library`, or `pagedLibrary`, behind the KV-group defines attention_mpp.metal reads: Q27_G over each of Q27_KVH.
pub fn grouped(a: std.mem.Allocator, units: bool, prompt: bool, group: u32, kv_heads: u32, paged: bool) ![]u8 {
    const body = try if (paged) pagedLibrary(a, units, prompt) else library(a, units, prompt);
    defer a.free(body);
    var defines: [64]u8 = undefined;
    return std.mem.concat(a, u8, &.{ try std.fmt.bufPrint(&defines, "#define Q27_G {d}\n#define Q27_KVH {d}\n", .{ group, kv_heads }), body });
}

/// `grouped` with Q27_JOIN: pack_b, the tail and the merges take a plan's pairs (Attention.initJoin, Plan.pair).
pub fn joined(a: std.mem.Allocator, units: bool, prompt: bool, group: u32, kv_heads: u32, paged: bool) ![]u8 {
    const body = try grouped(a, units, prompt, group, kv_heads, paged);
    defer a.free(body);
    return std.mem.concat(a, u8, &.{ "#define Q27_JOIN 1\n", body });
}

/// The attention source, its prefix and tail kernels indexing results in the simdgroup-matrix layout; caller frees.
pub fn rewrite(a: std.mem.Allocator, source: []const u8) ![]u8 {
    const tail = try region(a, source, "kernel void q27_attn_tail(", "kernel void q27_attn_merge(", &rewrites);
    defer a.free(tail);
    return region(a, tail, "kernel void q27_attn_prefix(", "#define Q27_PREFIX_AS(", &prefix_rewrites);
}

fn region(a: std.mem.Allocator, source: []const u8, from: []const u8, to: []const u8, pairs: []const [2][]const u8) ![]u8 {
    const begin = std.mem.indexOf(u8, source, from) orelse return error.AttentionSource;
    const end = begin + (std.mem.indexOf(u8, source[begin..], to) orelse return error.AttentionSource);
    var part = try a.dupe(u8, source[begin..end]);
    defer a.free(part);
    for (pairs) |r| {
        if (std.mem.count(u8, part, r[0]) != 1) return error.AttentionSource;
        const next = try std.mem.replaceOwned(u8, a, part, r[0], r[1]);
        a.free(part);
        part = next;
    }
    return std.mem.concat(a, u8, &.{ source[0..begin], part, source[end..] });
}

/// The tail and prefix kernels' ops (scores 16x32 and 16x64 over 256, outputs 16x128 over 64), each element where read.
pub const check_source =
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
    \\  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tK64((device bfloat*)X, dextents<int32_t, 2>(D, TK));
    \\  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(Ps, dextents<int32_t, 2>(TK, 16));
    \\  constexpr auto dS = matmul2d_descriptor(16, 32, D, false, true, false, matmul2d_descriptor::mode::multiply);
    \\  constexpr auto dO = matmul2d_descriptor(16, 128, TK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    \\  constexpr auto dW = matmul2d_descriptor(16, TK, D, false, true, false, matmul2d_descriptor::mode::multiply);
    \\  matmul2d<dS, execution_simdgroup> opS;
    \\  matmul2d<dO, execution_simdgroup> opO;
    \\  matmul2d<dW, execution_simdgroup> opW;
    \\  auto S = opS.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK32), float>();
    \\  auto Wd = opW.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK64), half>();
    \\  auto Wp = opO.template get_left_input_cooperative_tensor<half, bfloat, float>(Wd);
    \\  auto O = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(tVh), float>();
    \\  const int fm = ((lane & 16) >> 2) | ((lane >> 1) & 3), fn = ((lane & 8) >> 1) | ((lane & 1) << 1);
    \\  bool ok = true;
    \\  for (int i = 0; i < 16; i++) {
    \\    auto at = S.get_multidimensional_index(i);
    \\    ok = ok && at[1] == fm + ((i & 8) ? 8 : 0) && at[0] == fn + (i & 1) + 8 * ((i >> 1) & 3);
    \\  }
    \\  for (int i = 0; i < 32; i++) {
    \\    auto at = Wd.get_multidimensional_index(i);
    \\    auto in = Wp.get_multidimensional_index(i);
    \\    ok = ok && at[1] == fm + ((i & 16) ? 8 : 0) && at[0] == fn + (i & 1) + 8 * ((i >> 1) & 7) && in[0] == at[0] && in[1] == at[1];
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
    const x = try device.buffer(256 * 64 * 2, mtl.ResourceOptions.shared);
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

test "the rewrite changes every M5-indexed line of the tail kernel once and leaves the kernels after it alone" {
    const source = @import("qwen_runtime_sources").attention_mpp;
    const text = try rewrite(std.testing.allocator, source);
    defer std.testing.allocator.free(text);
    const begin = std.mem.indexOf(u8, text, "kernel void q27_attn_tail(").?;
    const end = begin + std.mem.indexOf(u8, text[begin..], "kernel void q27_attn_merge(").?;
    try std.testing.expect(std.mem.indexOf(u8, text[begin..end], "(i & 4)") == null);
    try std.testing.expect(std.mem.indexOf(u8, text[begin..end], "float4(Olo") == null);
    try std.testing.expectEqualStrings(source[source.len - (text.len - end) ..], text[end..]);
}

test "the rewrite changes every M5-indexed line of the prefix kernel once" {
    const source = @import("qwen_runtime_sources").attention_mpp;
    const text = try rewrite(std.testing.allocator, source);
    defer std.testing.allocator.free(text);
    const begin = std.mem.indexOf(u8, text, "kernel void q27_attn_prefix(").?;
    const end = begin + std.mem.indexOf(u8, text[begin..], "#define Q27_PREFIX_AS(").?;
    try std.testing.expect(std.mem.indexOf(u8, text[begin..end], "(i & 4)") == null);
    try std.testing.expect(std.mem.indexOf(u8, text[begin..end], "float4(O[") == null);
    try std.testing.expect(std.mem.indexOf(u8, text[begin..end], "* 16 + fn") == null);
    try std.testing.expectEqualStrings(source[0..begin], text[0..begin]);
}

test "both attention builds, with and without the prompt kernels, open with the KV-group defines" {
    const a = std.testing.allocator;
    const defines = "#define Q27_G 4\n#define Q27_KVH 4\n";
    for ([_]bool{ false, true }) |units| for ([_]bool{ false, true }) |prompt| {
        const text = try grouped(a, units, prompt, 4, 4, false);
        defer a.free(text);
        const body = try library(a, units, prompt);
        defer a.free(body);
        try std.testing.expect(std.mem.startsWith(u8, text, defines));
        try std.testing.expectEqualStrings(body, text[defines.len..]);
    };
}
