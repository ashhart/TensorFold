//! The prompt kernels' fragment layout: nax.h inlined for this GPU, and checked on it.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");

/// A prompt-kernel source with nax.h inlined; a GPU without tensor units takes nax.h's simdgroup-matrix layout.
pub fn source(device: mtl.Device, a: std.mem.Allocator, text: []const u8) ![]u8 {
    const src = try std.mem.replaceOwned(u8, a, text, "#include \"../nax.h\"", ks.nax);
    if (device.tensorUnits()) return src;
    defer a.free(src);
    return std.mem.concat(a, u8, &.{ "#define TF_SIMD_FRAGS 1\n", src });
}

/// The op both ways the prompt kernels use it (a transposed right operand, a plain one) through nax.h's fragments.
const check_source =
    \\#include "../nax.h"
    \\using namespace tfp;
    \\[[kernel]] void tf_frag_check(const device bfloat* A [[buffer(0)]], const device bfloat* B [[buffer(1)]],
    \\    const device bfloat* E [[buffer(2)]], device float* C [[buffer(3)]], uint lane [[thread_index_in_simdgroup]]) {
    \\  const short2 home = frag_home(ushort(lane));
    \\  frag<bfloat> a, b0, b1, e0, e1;
    \\  frag_get(a, A, 16, 0, 0, home);
    \\  frag_get_t(b0, B, 16, 0, 0, home);
    \\  frag_get_t(b1, B, 16, 16, 0, home);
    \\  frag_get(e0, E, 32, 0, 0, home);
    \\  frag_get(e1, E, 32, 0, 16, home);
    \\  frag<float> c0 = frag<float>(0), c1 = frag<float>(0), d0 = frag<float>(0), d1 = frag<float>(0);
    \\  mma_16x32<false, true>(c0, c1, a, b0, b1);
    \\  mma_16x32<false, false>(d0, d1, a, e0, e1);
    \\  frag_put(c0, C, 32, 0, 0, home);
    \\  frag_put(c1, C, 32, 0, 16, home);
    \\  frag_put(d0, C + 512, 32, 0, 0, home);
    \\  frag_put(d1, C + 512, 32, 0, 16, home);
    \\}
;

/// Small integers a check reads (exact products and sums in any order).
fn value(i: usize, salt: usize) f32 {
    return @floatFromInt(@as(i32, @intCast((i * 7 + salt * 3) % 5)) - 2);
}

/// nax.h's layout checked on this GPU, A B^T and A E against the host's sums: error.FragLayout when it is wrong.
pub fn check(device: mtl.Device, queue: mtl.Queue, a: std.mem.Allocator) !void {
    const text = try source(device, a, check_source);
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, "tf_frag_check", false);
    defer pipe.deinit();
    const sizes = [4]usize{ 16 * 16 * 2, 32 * 16 * 2, 16 * 32 * 2, 2 * 16 * 32 * 4 };
    var bufs: [4]mtl.Buffer = undefined;
    var made: usize = 0;
    defer for (bufs[0..made]) |x| x.deinit();
    while (made < 4) : (made += 1) bufs[made] = try device.buffer(sizes[made], mtl.ResourceOptions.shared);
    for (0..3) |m| for (bufs[m].slice(u16, sizes[m] / 2), 0..) |*x, i| {
        x.* = @intCast(@as(u32, @bitCast(value(i, m))) >> 16);
    };
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(pipe);
    for (bufs, 0..) |x, j| enc.setBuffer(x, 0, j);
    enc.dispatchThreads(mtl.Size.of(32, 1, 1), mtl.Size.of(32, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("fragment check failed: {s}", .{msg});
        return error.GpuFailed;
    }
    const out = bufs[3].slice(f32, 1024);
    for (0..16) |i| for (0..32) |j| {
        var c: f32 = 0;
        var d: f32 = 0;
        for (0..16) |k| {
            c += value(i * 16 + k, 0) * value(j * 16 + k, 1);
            d += value(i * 16 + k, 0) * value(k * 32 + j, 2);
        }
        if (out[i * 32 + j] != c or out[512 + i * 32 + j] != d) {
            std.log.err("prompt kernels: nax.h's fragment layout is wrong on {s}", .{device.name()});
            return error.FragLayout;
        }
    };
}
