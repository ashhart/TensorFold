//! CPU numeric instrument for the actual native engine graph, storage and weights. It is never a serving backend.
const std = @import("std");
const cfg = @import("../../src/families/qwen4/config.zig");
const m = @import("math.zig");
pub const B = @import("../../src/families/qwen4/engine_test.zig").B;
pub const Model = @import("../../src/families/qwen4/model.zig").Model(B);
pub const Scratch = @import("../../src/families/qwen4/arena.zig").Arena(B);
pub const Tensor = Scratch.Tensor;
pub fn bits(x: Tensor) []u16 {
    std.debug.assert(x.kind == .bf16);
    return @alignCast(std.mem.bytesAsSlice(u16, x.buffer[x.offset..][0 .. x.rows * x.width * 2]));
}
pub fn vector(bytes: B.Buffer) []f32 {
    return std.mem.bytesAsSlice(f32, bytes);
}
pub fn row(x: Tensor, r: usize) []u16 {
    return bits(x)[r * x.width ..][0..x.width];
}
pub fn view(comptime V: type, v: anytype) []V {
    return @alignCast(std.mem.bytesAsSlice(V, v.buffer[v.offset..][0..v.len]));
}
pub fn norm(g: *G, n: Model.Norm, x: []const u16, group: usize, eps: f32) ![]u16 {
    const out = try g.a.alloc(u16, x.len);
    const w = vector(n.buffer);
    var start: usize = 0;
    while (start < x.len) : (start += group) {
        var ss: f32 = 0;
        for (x[start..][0..group]) |v| ss += m.f(v) * m.f(v);
        const inv = 1 / @sqrt(ss / @as(f32, @floatFromInt(group)) + eps);
        for (start..start + group) |i| out[i] = m.bf(m.f(x[i]) * inv * w[i]);
    }
    return out;
}
extern fn tf_weight_only_cpu([*]const u16, [*]const u8, [*]const u8, [*]const u8, [*]u16, @import("core").weight_only.Params, c_int, c_int) void;
extern fn tf_hc_activation_cpu([*]const u16, [*]u16, c_int, c_int, c_int, c_int, c_int) void;
extern fn tf_hc_pointwise_cpu([*]const u16, [*]const u16, [*]const u16, [*]u16, c_int, c_int, c_int, c_int, c_int, c_int, c_int) void;
pub const G = struct {
    pub const Storage = B;
    pub const kind = @import("../../src/families/qwen4/native_admission.zig").Backend.cuda;
    probability_bf16: bool = true,
    trace: ?@import("../../src/families/qwen4/node_trace.zig").Sink(B) = null,
    b: *B,
    c: *const cfg.Config,
    arena: ?std.heap.ArenaAllocator = null,
    a: std.mem.Allocator = undefined,
    pub fn begin(g: *G) !void {
        if (g.arena != null) return error.NumericPassAlreadyActive;
        g.arena = .init(std.testing.allocator);
        g.a = g.arena.?.allocator();
    }
    pub fn submit(g: *G) !void {
        g.discard();
    }
    pub fn discard(g: *G) void {
        if (g.arena) |*arena| arena.deinit();
        g.arena = null;
    }
    pub fn deinitDrained(g: *G) void {
        g.discard();
    }
    pub fn project(_: *G, w: Model.Matrix, x: Tensor, y: Tensor) !void {
        var p = w.params;
        p.rows = @intCast(x.rows);
        tf_weight_only_cpu(bits(x).ptr, w.weight.ptr, w.scales.ptr, w.biases.ptr, bits(y).ptr, p, 0, 0);
    }
    pub fn hcNorm(g: *G, n: Model.Norm, x: Tensor, sums: Tensor, y: Tensor, eps: f32) !void {
        const partials = std.mem.bytesAsSlice(f32, sums.buffer[sums.offset..][0..try sums.bytes()]);
        for (0..x.rows) |r| {
            for (0..sums.width) |part| {
                var sum: f32 = 0;
                for (row(x, r)[part * 256 ..][0..256]) |bits_| sum += m.f(bits_) * m.f(bits_);
                partials[r * sums.width + part] = sum;
            }
            @memcpy(row(y, r), try norm(g, n, row(x, r), g.c.hidden, eps));
        }
    }
    pub fn hcActivation(_: *G, x: Tensor, y: Tensor, streams: usize, injection: bool) !void {
        tf_hc_activation_cpu(bits(x).ptr, bits(y).ptr, @intCast(bits(x).len), @intCast(streams), @intFromBool(injection), 0, 0);
    }
    pub fn hcMix(g: *G, up: Tensor, x: Tensor, y: Tensor, _: Tensor) !void {
        for (0..x.rows) |r| for (row(y, r), 0..) |*out, d| {
            var sum: f32 = 0;
            for (0..g.c.streams) |s| {
                const i = s * y.width + d;
                sum += m.round(m.round(m.sig(m.f(row(up, r)[i]))) * m.f(row(x, r)[i]));
            }
            out.* = m.bf(sum / @as(f32, @floatFromInt(g.c.streams)));
        };
    }
    pub fn writeback(g: *G, x: Tensor, branch: Tensor, injection: Tensor, y: Tensor) !void {
        tf_hc_pointwise_cpu(bits(x).ptr, bits(branch).ptr, bits(injection).ptr, bits(y).ptr, @intCast(x.rows), @intCast(g.c.hidden), @intCast(g.c.streams), 0, 1, 0, 0);
    }
    pub fn swiglu(_: *G, gate: Tensor, up: Tensor, y: Tensor) !void {
        for (bits(gate), bits(up), bits(y)) |a, b, *out| out.* = m.bf(m.round(m.silu(m.f(a))) * m.f(b));
    }
    pub fn delta(g: *G, w: Model.Linear, state: anytype, layer: usize, x: Tensor, y: Tensor, _: Tensor) !void {
        try @import("numeric_mixers.zig").delta(g, w, state, layer, x, y);
    }
    pub fn indexed(g: *G, _: *Scratch, w: Model.Indexed, state: anytype, layer: usize, x: Tensor, y: Tensor) !void {
        try @import("numeric_mixers.zig").indexed(g, w, state, layer, x, y);
    }
    pub fn ple(g: *G, _: *Scratch, w: Model.Ple, state: anytype, layer: usize, projections: Tensor, h: Tensor, gated: Tensor, normed: Tensor, _: Tensor) !void {
        try @import("numeric_ple.zig").apply(g, w, state, layer, projections, h, gated, normed);
    }
};
pub const Loader = struct {
    pub fn load(_: Loader, _: std.mem.Allocator, b: *B, c: *const cfg.Config, _: *cfg.Problem) !G {
        return .{ .b = b, .c = c };
    }
};

pub const MetalLoader = struct {
    pub const reference = @import("../../src/families/qwen4/device_fixture.zig").metal_reference;
    pub fn load(_: @This(), _: std.mem.Allocator, b: *B, c: *const cfg.Config, _: *cfg.Problem) !G {
        return .{ .b = b, .c = c, .probability_bf16 = false };
    }
};
