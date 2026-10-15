//! The n-gram table's weight_scale on the looked-up rows, as oMLX's runtime applies it: bf16(row * scale).
const std = @import("std");
const mtl = @import("metal");
const replay = @import("replay.zig");

const Buf = replay.Buf;

const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\kernel void tf_ple_scale(device bfloat* x [[buffer(0)]], constant float& s [[buffer(1)]],
    \\                         constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    \\  if (i < n) x[i] = bfloat(float(x[i]) * s);
    \\}
;

/// A scale other than 1 and its kernel, compiled at load.
pub const Scale = struct {
    pipe: mtl.Pipeline,
    value: f32,

    /// Multiplies `n` bf16 values at `x` in place, then a barrier unless the encoder is serial.
    pub fn apply(s: Scale, enc: mtl.ComputeEncoder, x: Buf, n: usize, serial: bool) void {
        const count: u32 = @intCast(n);
        enc.setPipeline(s.pipe);
        enc.setBuffer(x.b, x.off, 0);
        enc.setBytes(std.mem.asBytes(&s.value), 1);
        enc.setBytes(std.mem.asBytes(&count), 2);
        enc.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
        if (!serial) enc.barrier();
    }
};

/// The checkpoint's n-gram weight_scale as one bf16 or f32 value; null when there is none or it is 1.
pub fn of(r: *replay.Run, map: std.json.ObjectMap) !?Scale {
    const name = for (map.keys()) |k| {
        if (std.mem.endsWith(u8, k, ".ple.ple_embedding.ngram_embedding.weight_scale")) break k;
    } else return null;
    const e = try r.entry(name);
    const b = try r.load(name);
    defer b.b.deinit();
    const bytes = b.b.contents()[0..e.len];
    const value: f32 = switch (e.len) {
        2 => @bitCast(@as(u32, std.mem.readInt(u16, bytes[0..2], .little)) << 16),
        4 => @bitCast(std.mem.readInt(u32, bytes[0..4], .little)),
        else => return error.NgramTableScale,
    };
    if (!std.math.isFinite(value) or value == 0) return error.NgramTableScale;
    if (value == 1) return null;
    return try compile(r.device, value);
}

/// The kernel for `value`; a compile failure refuses the load.
pub fn compile(device: mtl.Device, value: f32) !Scale {
    const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    return .{ .pipe = try mtl.Pipeline.init(device, lib, "tf_ple_scale", false), .value = value };
}

test "the scale rounds each row value once to bf16 after the product, as bf16 row * bf16 scale does" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var r = replay.Run{ .arena = arena.allocator(), .device = device, .queue = try device.queue() };
    const scale: f32 = @bitCast(@as(u32, 0x3951) << 16); // oQ4e's weight_scale
    const s = try compile(device, scale);
    const n = 4099;
    const buf: Buf = .{ .b = try r.buffer(n * 2) };
    const x = std.mem.bytesAsSlice(u16, buf.b.contents()[0 .. n * 2]);
    var rng = std.Random.DefaultPrng.init(7);
    for (x) |*v| { // normal bf16 values of both signs whose products stay normal
        const exp: u16 = 32 + rng.random().uintLessThan(u16, 96);
        v.* = (rng.random().int(u16) & 0x807f) | exp << 7;
    }
    const want = try std.testing.allocator.alloc(u16, n);
    defer std.testing.allocator.free(want);
    for (x, want) |v, *w| w.* = bf16(@as(f32, @bitCast(@as(u32, v) << 16)) * scale);
    const cb = r.queue.commandBuffer();
    const enc = cb.compute(.serial);
    s.apply(enc, buf, n, true);
    enc.end();
    cb.commit();
    cb.wait();
    try std.testing.expectEqualSlices(u16, want, x);
}

/// f32 to bf16, round to nearest even.
fn bf16(f: f32) u16 {
    const u: u32 = @bitCast(f);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
