//! Byte-code projection arithmetic and its Metal kernels (TF_Q27_FIVE_BYTES).
const std = @import("std");
const mtl = @import("metal");
const quant = @import("quant_gpu.zig");
const Kernels = quant.Kernels;
const Ref = quant.Ref;
const projection = @import("projection.zig");
const profile = @import("core").gpu_profile;

/// tf_lane's arithmetic on a `bytes` matrix (same codes, sums, K split and order: same bits), all rows at once.
pub fn product(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, y: Ref, rows: u32, stride: u32) !void {
    if (rows == 0 or p.n % 32 != 0 or p.group != 64) return error.BadCoreProjectionRows;
    const sk: u32 = p.slices;
    const key = [3]u32{ p.n, p.k, sk };
    const entry = try k.byte_pipelines.getOrPut(k.allocator, key);
    if (!entry.found_existing) {
        errdefer _ = k.byte_pipelines.remove(key);
        const text = try std.fmt.allocPrint(k.allocator, "#define TF_N {d}\n#define TF_K {d}\n#define TF_SK {d}\n{s}", .{ p.n, p.k, sk, byte_source });
        defer k.allocator.free(text);
        const library = try mtl.Library.fromSource(k.device, text, mtl.CompileOptions.mlx());
        defer library.deinit();
        entry.value_ptr.* = try mtl.Pipeline.init(k.device, library, "tf_byte", false);
        if (k.byte_sums == null) k.byte_sums = try mtl.Pipeline.init(k.device, library, "tf_byte_sums", false);
    }
    const groups = p.k / 64;
    const need = @as(usize, groups) * rows * 4;
    if (need > k.sums_bytes) {
        const b = try k.device.buffer(need, mtl.ResourceOptions.private | mtl.ResourceOptions.untracked);
        if (k.sums_stage) |old| old.deinit();
        k.sums_stage = b;
        k.sums_bytes = need;
    }
    var span = profile.begin(k.profiler, input_encoder, .dense);
    span.shape = .{ .n = p.n, .k = p.k, .sk = sk, .bits = 8 };
    errdefer span.cancel();
    const e = span.encoder;
    e.setPipeline(k.byte_sums.?);
    e.setBuffer(x.buffer, x.offset, 0);
    e.setValue([3]i32{ @intCast(rows), @intCast(p.k), @intCast(groups) }, 1);
    e.setBuffer(k.sums_stage.?, 0, 2);
    e.dispatchThreads(mtl.Size.of(rows, groups, 1), mtl.Size.of(64, 1, 1));
    e.barrier();
    e.setPipeline(entry.value_ptr.*);
    e.setBuffer(x.buffer, x.offset, 0);
    e.setBuffer(p.words.buffer, p.words.offset, 1);
    e.setBuffer(p.pairs.buffer, p.pairs.offset, 2);
    e.setValue([2]i32{ @intCast(rows), @intCast(stride) }, 3);
    e.setBuffer(y.buffer, y.offset, 4);
    e.setBuffer(k.sums_stage.?, 0, 5);
    e.dispatchGroups(mtl.Size.of(p.n / 32, (rows + 15) / 16, 1), mtl.Size.of(32 * sk, 1, 1));
    e.barrier();
    try span.end();
}

/// tf_lane (TF_SIMD_FRAGS 0, group 64) on device byte codes [N][K], sums from tf_byte_sums, 16-row blocks on y.
const byte_source =
    \\#include <metal_stdlib>
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace metal;
    \\using namespace mpp::tensor_ops;
    \\constant constexpr int NT = 32, GS = 64, KG = TF_K / GS, NF = NT / 16;
    \\kernel void tf_byte_sums(const device bfloat* X [[buffer(0)]], constant int3& d [[buffer(1)]], device float* XS [[buffer(2)]],
    \\    uint2 t [[thread_position_in_grid]]) {
    \\  if (int(t.x) >= d.x || int(t.y) >= d.z) return;
    \\  float acc = 0.0f;
    \\  for (int i = 0; i < GS; i++) acc += float(X[(size_t)t.x * d.y + t.y * GS + i]);
    \\  XS[(size_t)t.y * d.x + t.x] = acc;
    \\}
    \\[[kernel]] void tf_byte(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
    \\    const device bfloat* SB [[buffer(2)]], const device int* mdims [[buffer(3)]], device bfloat* Y [[buffer(4)]],
    \\    const device float* XS [[buffer(5)]],
    \\    uint sgi [[simdgroup_index_in_threadgroup]], uint lanei [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
    \\  const int r0 = int(tg.y) * 16, rows = mdims[0], M = min(rows - r0, 16), ys = mdims[1];
    \\  if (M <= 0) return;
    \\  const int lane = int(lanei), sg = int(sgi);
    \\  const short qid = short(lane) >> 2;
    \\  const short fm = (qid & 4) | ((short(lane) >> 1) & 3);
    \\  const short fn = ((qid & 2) | (short(lane) & 1)) * 4;
    \\  const int n0 = int(tg.x) * NT;
    \\  const int g_begin = (sg * KG) / TF_SK, g_end = ((sg + 1) * KG) / TF_SK;
    \\  constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
    \\  matmul2d<desc, execution_simdgroup> op;
    \\  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (size_t)r0 * TF_K, dextents<int32_t, 2>(TF_K, M));
    \\  tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> tB((device uint8_t*)W + (size_t)n0 * TF_K, dextents<int32_t, 2>(TF_K, NT));
    \\  float C[NF * 8];
    \\  for (int i = 0; i < NF * 8; i++) C[i] = 0.0f;
    \\  for (int g = g_begin; g < g_end; g++) {
    \\    float s[NF][4], bb[NF][4];
    \\    for (int f = 0; f < NF; f++)
    \\      for (int j = 0; j < 4; j++) {
    \\        const size_t at = ((size_t)g * TF_N + n0 + f * 16 + fn + j) * 2;
    \\        s[f][j] = float(SB[at]);
    \\        bb[f][j] = float(SB[at + 1]);
    \\      }
    \\    auto a = tA.slice(g * GS, 0);
    \\    auto b = tB.slice(g * GS, 0);
    \\    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    \\    op.run(a, b, P);
    \\    const float xs0 = fm < M ? XS[(size_t)g * rows + r0 + fm] : 0.0f;
    \\    const float xs1 = fm + 8 < M ? XS[(size_t)g * rows + r0 + fm + 8] : 0.0f;
    \\    for (int f = 0; f < NF; f++)
    \\      for (int r = 0; r < 2; r++)
    \\        for (int j = 0; j < 4; j++) {
    \\          const int i = f * 8 + r * 4 + j;
    \\          C[i] = fma(s[f][j], P[i], fma(bb[f][j], r ? xs1 : xs0, C[i]));
    \\        }
    \\  }
    \\  threadgroup float part[(TF_SK > 1 ? TF_SK - 1 : 1) * NF * 8 * 32];
    \\  if (TF_SK > 1) {
    \\    if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[i];
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (sg == 0) for (int s2 = 1; s2 < TF_SK; s2++) for (int i = 0; i < NF * 8; i++) C[i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
    \\  }
    \\  if (sg == 0) {
    \\    Y += (size_t)r0 * ys;
    \\    for (int f = 0; f < NF; f++)
    \\      for (int r = 0; r < 2; r++) {
    \\        const int m = fm + 8 * r;
    \\        const int nn = n0 + f * 16 + fn;
    \\        if (m < M) for (int j = 0; j < 4; j++) Y[(size_t)m * ys + nn + j] = bfloat(C[f * 8 + r * 4 + j]);
    \\      }
    \\  }
    \\}
;
