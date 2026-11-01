//! Directional gradients of the actual packed MoE and gated pocket against its serving operation.
const std = @import("std");
const cuda = @import("cuda");
const train = @import("cuda_train.zig");
const hooks = @import("cuda_learn_hooks.zig");
const engine = @import("cuda_engine.zig");
fn bf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}
fn b16(v: f64) u16 {
    const f: f32 = @floatCast(v);
    const bits: u32 = @bitCast(f);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
fn read(e: *engine.Engine, ptr: u64, bytes: []u8) !void {
    try e.stream.synchronize();
    try e.d.check(e.d.api.cuMemcpyDtoH_v2(bytes.ptr, ptr, bytes.len), "MoE gradient read");
}
fn forward(e: *engine.Engine, layer: usize, x: u64, out: u64, g: []const f32, host: []u16) !f64 {
    const m = e.layers[layer].mlp.moe;
    try e.routedRows(m.router, m.experts, x, out, 1, false, 0);
    try e.pockets.addGated(e.stream, layer, x, out, 1, 0, e.routed.wts + e.g.top_k * 4, e.g.slots());
    try read(e, out, std.mem.sliceAsBytes(host));
    var value: f64 = 0;
    for (host, g) |v, p| value += bf(v) * p;
    return value;
}
pub fn run(a: std.mem.Allocator, t: *train.Trainer, layer: usize) !void {
    const e = t.e;
    const D = e.g.hidden;
    const x = try a.dupe(u16, t.normed(layer, 0));
    defer a.free(x);
    const moved = try a.alloc(u16, D);
    defer a.free(moved);
    const out = try a.alloc(u16, D);
    defer a.free(out);
    const g = try a.alloc(f32, D);
    defer a.free(g);
    const dx = try a.alloc(f32, D);
    defer a.free(dx);
    const direction = try a.alloc(f32, D);
    defer a.free(direction);
    var rng = std.Random.DefaultPrng.init(0x35b);
    for (g) |*v| v.* = rng.random().floatNorm(f32) / @sqrt(@as(f32, @floatFromInt(D)));
    var input = try cuda.DeviceBuffer.fromHost(e.d, std.mem.sliceAsBytes(x));
    defer input.free();
    var output = try cuda.DeviceBuffer.alloc(e.d, D * 2);
    defer output.free();
    var grad = try cuda.DeviceBuffer.fromHost(e.d, std.mem.sliceAsBytes(g));
    defer grad.free();
    defer @import("qwen_vision").completion.drain(e.d, e.stream);
    _ = try forward(e, layer, input.ptr, output.ptr, g, out);
    const kept: hooks.LayerKeep = .{ .y = input.ptr, .ids = e.routed.picks, .wts = e.routed.wts };
    // Borrow the real backward scratch; only the adjoint input changes for this diagnostic.
    const previous = t.grad;
    t.grad = grad.ptr;
    defer t.grad = previous;
    try @import("cuda_moe_train.zig").backward(t, layer, kept, e.pockets.of(layer), 1);
    try read(e, t.dy, std.mem.sliceAsBytes(dx));
    var norm: f64 = 0;
    var size: f64 = 0;
    for (dx, x) |v, q| {
        norm += @as(f64, v) * v;
        size += bf(q) * bf(q);
    }
    norm = @sqrt(norm);
    if (!std.math.isFinite(norm) or norm <= 0) return error.BadMoeGradient;
    for ([_]bool{ false, true }) |mixed| {
        var rn: f64 = 0;
        for (direction) |*v| {
            v.* = rng.random().floatNorm(f32);
            rn += @as(f64, v.*) * v.*;
        }
        var wanted: f64 = 0;
        for (direction, dx) |*v, p| {
            v.* = @floatCast(if (mixed) (p / norm + v.* / @sqrt(rn)) / std.math.sqrt2 else p / norm);
            wanted += @as(f64, p) * v.*;
        }
        for ([_]f64{ 0.002, 0.004, 0.008 }) |fraction| {
            var numerator: f64 = 0;
            var denominator: f64 = 0;
            for (1..7) |m| for ([_]f64{ -1, 1 }) |sign| {
                const step = sign * @as(f64, @floatFromInt(m)) * fraction * @sqrt(size);
                for (moved, x, direction) |*v, q, p| v.* = b16(bf(q) + step * p);
                try input.upload(0, std.mem.sliceAsBytes(moved));
                const value = try forward(e, layer, input.ptr, output.ptr, g, out);
                numerator += step * value;
                denominator += step * step;
            };
            const fd = numerator / denominator;
            const relative = @abs(fd - wanted) / @max(@abs(wanted), 1e-30);
            std.debug.print("RESULT MoE layer {d} gated pocket mixed={any} step={d}: fd {e}, adjoint {e}, relative {e}\n", .{ layer, mixed, fraction, fd, wanted, relative });
            if (!std.math.isFinite(relative) or relative >= 0.15) return error.MoeGradientMismatch;
        }
    }
}
