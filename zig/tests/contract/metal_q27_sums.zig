//! Numeric FP32 gates supplement the mixed-output replacement checks with independent sum panels.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const glue = @import("metal_q27_glue.zig");
const ref = contract.q27_stack_sums;
const c = contract.spec;
const g = contract.gates;
fn none(_: c.Shape) usize {
    return 0;
}
pub const Probe = struct {
    raw: glue.Probe,
    rig: *rig_.Rig = undefined,
    kind: ref.Kind,
    bad: bool,
    name: []const u8,
    pub fn init(a: std.mem.Allocator, d: mtl.Device, q: mtl.Queue, kind: ref.Kind, bad: bool, name: []const u8) Probe {
        return .{ .raw = glue.Probe.init(a, d, q, if (kind == .xs) .mlp_xs else .pre, false, name), .kind = kind, .bad = bad, .name = name };
    }
    pub fn deinit(p: *Probe) void {
        p.raw.deinit();
    }
    fn index(p: *const Probe) usize {
        return switch (p.kind) {
            .norm => 0,
            .xs => 1,
            .conv => 2,
            .columns => unreachable,
        };
    }
    fn outType(p: *const Probe) c.Elem {
        return if (p.kind == .xs) .f32 else .bf16;
    }
    pub fn kernel(p: *Probe) g.Kernel {
        p.rig = &p.raw.rig;
        return .{ .spec = .{
            .name = p.name,
            .accum = .f32,
            .out = p.outType(),
            .depth = none,
            .served = if (p.kind == .xs) &.{.{ .rows = 4, .n = 64, .k = 64 }} else &.{.{ .rows = 4, .n = 128, .k = 0 }},
            .replaces = "the original GDN pre or MLP output on identical inputs",
            .nan = .{ .stated = "non-finite output bytes equal the original mixed-output kernel" },
            .sums = .{ .exact = true, .f16 = 100, .bf16 = if (p.kind == .conv) 100 else 0, .why = "BF16 output or cancellation can hide a BF16 sum; half accumulation must still fail" },
        }, .ctx = p, .vt = &vt, .control = if (p.bad) .sums else null };
    }
    const vt = blk: {
        var v = g.hooks(Probe);
        v.output = output;
        break :blk v;
    };
    fn words(p: *Probe, at: usize, n: usize) []u16 {
        return @alignCast(std.mem.bytesAsSlice(u16, p.raw.rig.bytes(p.raw.ins.items[at], n * 2)));
    }
    fn panel(p: *Probe) void {
        const rows = p.raw.shape.rows;
        if (p.kind == .xs) {
            const n = p.raw.shape.n;
            const fused = p.words(0, rows * 2 * n);
            for (0..rows) |r| for (0..n) |i| {
                fused[r * 2 * n + i] = c.toBf16(256);
                fused[r * 2 * n + n + i] = c.toBf16(@floatCast(ref.term(.xs, i % 64) / 256));
            };
        } else {
            const channels = 80 * 128;
            const projection = p.words(0, rows * channels);
            for (projection, 0..) |*v, i| v.* = c.toBf16(if (p.kind == .conv) -0x1p20 else if (i % 128 == 0) 40 else 32);
            const history = p.words(3, 3 * 3 * channels);
            for (history, 0..) |*v, i| v.* = c.toBf16(if (p.kind == .conv) (if ((i / channels) % 3 == 0) @as(f32, 0x1p20) else 64) else 0);
            const weights = p.words(4, channels * 4);
            for (weights, 0..) |*v, i| v.* = c.toBf16(if (p.kind == .conv or i % 4 == 0) 1 else 0);
            const windows = std.mem.bytesAsSlice(u32, p.raw.rig.bytes(p.raw.ins.items[7], rows * 4 * 4));
            for (windows, 0..) |*v, i| v.* = if (p.kind == .conv) @intCast(i % 4) else if (i % 4 == 0) 3 else 0;
        }
    }
    fn control(p: *Probe) !void {
        const a = p.raw.a;
        const source = if (p.kind == .xs) try tf.qwen27.quant_gpu.specializedSource(a, .{ .kind = .mlp_rows, .k = 64, .n = 64, .xs = true }) else try a.dupe(u8, sources.core_deltanet);
        defer a.free(source);
        const symbol = if (p.kind == .xs) "q27_mlp_rows" else "tf_delta_gdn_pre_vectors";
        const from, const to = switch (p.kind) {
            .conv => .{ "result += w[tap] * taps[tap][part];", "result = float(half(result + w[tap] * taps[tap][part]));" },
            .norm => .{ "square = simd_sum(square);", "square = float(simd_sum(half(square)));" },
            .xs => .{ "acc += x[part];", "acc = float(half(acc + x[part]));" },
            .columns => unreachable,
        };
        const at = std.mem.indexOf(u8, source, symbol).?;
        if (std.mem.count(u8, source[at..], from) != 1) return error.MissingPrecisionControl;
        const tail = try std.mem.replaceOwned(u8, a, source[at..], from, to);
        defer a.free(tail);
        const text = try std.mem.concat(a, u8, &.{ source[0..at], tail });
        defer a.free(text);
        const pipe = try rig_.pipeline(p.raw.rig.device, text, symbol);
        p.raw.pipe.?.deinit();
        p.raw.pipe = pipe;
    }
    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        var actual = s;
        if (p.kind != .xs) actual.k = 0;
        if (!try p.raw.prepare(actual, fill, seed)) return false;
        if (fill == .stress) p.panel();
        if (p.bad) try p.control();
        return true;
    }
    pub fn run(p: *Probe) !void {
        try p.raw.run();
    }
    fn output(ctx: *anyopaque) ![]const u8 {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        return p.raw.rig.bytes(p.raw.outs.items[p.index()], p.raw.lens.items[p.index()]);
    }
    pub fn replaced(p: *Probe) ![]const u8 {
        _ = try p.raw.replaced();
        return p.raw.rig.bytes(p.raw.bases.items[p.index()], p.raw.lens.items[p.index()]);
    }
    pub fn count(p: *Probe) usize {
        return p.raw.lens.items[p.index()] / c.size(p.outType());
    }
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        return p.raw.check(.nonfinite);
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        for (want, 0..) |*v, i| v.* = p.value(i, .none, .sequential);
        @memset(mag, 0);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*v, i| v.* = p.value(i, acc, order);
    }
    fn value(p: *Probe, i: usize, acc: c.Accum, order: g.Order) f64 {
        if (p.kind == .xs and i % ((p.raw.shape.rows + 15) / 16 * 16) >= p.raw.shape.rows) return 0;
        return ref.output(p.kind, i, acc, order);
    }
};
