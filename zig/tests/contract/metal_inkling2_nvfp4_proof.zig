//! Shared GPU exhaustive NVFP4 2-product vs 3-product parity proofs and control fixtures.
const std = @import("std");
const mtl = @import("metal");
const c = @import("contract").spec;
const rig_ = @import("rig.zig");
const src = @import("inkling2_metal");

pub fn fillSwapAb(x: []u16, w: []u8, scales: []u8, amax: []f32, denominator: []f32) void {
    @memset(x, c.toBf16(1.0));
    @memset(w, 0x66);
    @memset(scales, 0x01);
    @memset(amax, 0x1.0p-118);
    denominator[0] = 1.0;
}

fn runProof(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, base_src: []const u8, extra: []const u8, entry: [:0]const u8, guard_msg: []const u8, fail_msg: []const u8) !void {
    const text = try std.mem.concat(a, u8, &.{ base_src, "\n", extra });
    defer a.free(text);
    const pipe = try rig_.pipeline(device, text, entry);
    defer pipe.deinit();
    var r: rig_.Rig = .{ .a = a, .device = device, .queue = queue };
    defer r.deinit();
    var w_blk: [256 * 2]u32 = undefined;
    var s_blk: [256]u8 = undefined;
    for (0..256) |i| {
        w_blk[2 * i] = 0x7654_3210;
        w_blk[2 * i + 1] = 0xfedc_ba98;
        s_blk[i] = @intCast(i);
    }
    var extra_phi: [4096]f32 = undefined;
    const edge = [_]f32{ 0x1.0p-125, 0x1.0p-118, 0x1.0p-117, 0x1.0p-116, 0x1.0p117, 0x1.0p118, 0x1.0p122, -0x1.0p-118, (65.0 / 64.0) / 1.5, 2688.125 / 2695.75, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32), -0.0, 0.0, 1.0 };
    for (&extra_phi, 0..) |*v, i| v.* = if (i < edge.len) edge[i] else blk: {
        const am = (i * 17) & 127;
        const dm = (i * 31) & 127;
        const num = 1.0 + @as(f32, @floatFromInt(am)) / 128.0;
        const den = 1.0 + @as(f32, @floatFromInt(dm)) / 128.0;
        break :blk (num / den) * (if (i & 1 != 0) @as(f32, 1.0) else -1.0);
    };
    const rw = try r.input(std.mem.sliceAsBytes(&w_blk));
    const rs = try r.input(&s_blk);
    const rp = try r.input(std.mem.sliceAsBytes(&extra_phi));
    const st = try r.scratch(16);
    @memset(r.bytes(st, 16), 0);
    const n_phi: u32 = 65536 + extra_phi.len;
    const pass = r.begin();
    for ([_]rig_.Ref{ rw, rs, rp, st }, 0..) |b, i| pass.e.setBuffer(b.buf, b.off, i);
    pass.e.setValue(n_phi, 4);
    pass.e.setPipeline(pipe);
    pass.e.dispatchGroups(mtl.Size.of((n_phi + 31) / 32, 32, 1), mtl.Size.of(32, 8, 1));
    try r.finish(pass);
    if (!try r.intact()) {
        std.debug.print("{s}\n", .{guard_msg});
        return error.Nvfp4TwoProdParityMismatch;
    }
    const stats = std.mem.bytesAsSlice(u32, r.bytes(st, 16));
    if (stats[0] != 0 or stats[1] < 12_600_000 or stats[2] == 0 or stats[3] != 1) {
        std.debug.print("{s} (stats={any})\n", .{ fail_msg, stats });
        return error.Nvfp4TwoProdParityMismatch;
    }
}

pub fn assertBlockProof(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !void {
    const extra =
        \\[[kernel, max_total_threads_per_threadgroup(256)]] void tf_test_nv_two_prod(
        \\    device const uchar* W [[buffer(0)]], device const uchar* S [[buffer(1)]],
        \\    device const float* extra_phi [[buffer(2)]], device atomic_uint* stats [[buffer(3)]],
        \\    constant uint& n_phi [[buffer(4)]], uint2 tid [[thread_position_in_grid]]) {
        \\  const uint pi = tid.x, sc = tid.y;
        \\  if (pi >= n_phi || sc >= 256u) return;
        \\  const float phi = (pi < 65536u) ? float(as_type<bfloat>(ushort(pi))) : extra_phi[pi - 65536u];
        \\  float ref_out[16], got_out[16];
        \\  ink2_nv_block<false>(W, S, phi, sc, ref_out);
        \\  const bool safe = ink2_nv_two_prod_safe(phi);
        \\  if (safe) ink2_nv_block<true>(W, S, 0.5f * phi, sc, got_out);
        \\  else ink2_nv_block<false>(W, S, phi, sc, got_out);
        \\  atomic_fetch_add_explicit(&stats[safe ? 1u : 2u], 1u, memory_order_relaxed);
        \\  for (uint j = 0u; j < 16u; ++j) {
        \\    if (as_type<uint>(ref_out[j]) != as_type<uint>(got_out[j]))
        \\      atomic_fetch_add_explicit(&stats[0], 1u, memory_order_relaxed);
        \\  }
        \\  if (pi == 65544u && sc == 7u) {
        \\    float unguarded[16];
        \\    ink2_nv_block<true>(W, S, 0.5f * phi, sc, unguarded);
        \\    if (as_type<uint>(unguarded[3]) != as_type<uint>(ref_out[3]))
        \\      atomic_fetch_add_explicit(&stats[3], 1u, memory_order_relaxed);
        \\  }
        \\}
    ;
    return runProof(a, device, queue, src.experts, extra, "tf_test_nv_two_prod", "GPU NVFP4 2-product verifier wrote past its buffer guards", "NVFP4 2-product GPU decode differs from 3-product reference");
}

pub fn assertQuadProof(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !void {
    const extra =
        \\[[kernel, max_total_threads_per_threadgroup(256)]] void tf_test_nv_quad_two_prod(
        \\    device const uchar* W [[buffer(0)]], device const uchar* S [[buffer(1)]],
        \\    device const float* extra_phi [[buffer(2)]], device atomic_uint* stats [[buffer(3)]],
        \\    constant uint& n_phi [[buffer(4)]], uint2 tid [[thread_position_in_grid]]) {
        \\  const uint pi = tid.x, sc = tid.y;
        \\  if (pi >= n_phi || sc >= 256u) return;
        \\  const float phi = (pi < 65536u) ? float(as_type<bfloat>(ushort(pi))) : extra_phi[pi - 65536u];
        \\  const bool safe = ink2_nv_two_prod_safe(phi);
        \\  atomic_fetch_add_explicit(&stats[safe ? 1u : 2u], 1u, memory_order_relaxed);
        \\  for (uint q = 0u; q < 4u; ++q) {
        \\    const bfloat4 ref_v = ink2_nv_quad(W, S, phi, sc, 4u * q);
        \\    const bfloat4 got_v = safe ? ink2_nv_quad_2p(W, S, 0.5f * phi, sc, 4u * q)
        \\                               : ink2_nv_quad(W, S, phi, sc, 4u * q);
        \\    for (uint j = 0u; j < 4u; ++j) {
        \\      if (as_type<ushort>(ref_v[j]) != as_type<ushort>(got_v[j]))
        \\        atomic_fetch_add_explicit(&stats[0], 1u, memory_order_relaxed);
        \\    }
        \\  }
        \\  if (pi == 65544u && sc == 7u) {
        \\    const bfloat4 ref_tie = ink2_nv_quad(W, S, phi, sc, 0u);
        \\    const bfloat4 got_tie = ink2_nv_quad_2p(W, S, 0.5f * phi, sc, 0u);
        \\    if (as_type<ushort>(ref_tie[3]) != as_type<ushort>(got_tie[3]))
        \\      atomic_fetch_add_explicit(&stats[3], 1u, memory_order_relaxed);
        \\  }
        \\}
    ;
    return runProof(a, device, queue, src.experts_units, extra, "tf_test_nv_quad_two_prod", "GPU NVFP4 quad verifier wrote past its buffer guards", "ink2_nv_quad_2p differs from ink2_nv_quad");
}
