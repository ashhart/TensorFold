//! A MoE layer's weights: both routers as bf16 packed for core/moe_route, the experts as the tiles read them.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const cp = core.checkpoint_metal;
const tile = core.tile_qmm;
const route = core.moe_route;
const Config = @import("config.zig").Config;
const widths = @import("widths.zig");

/// Logit columns a row: the routed experts, the shared expert's gate, padding to the logits kernel's blocks of 4.
pub fn columns(c: Config) usize {
    return (c.experts + 1 + 3) / 4 * 4;
}

pub const Experts = struct {
    /// [columns, hidden] bf16: the routed experts' router rows, the shared gate's, zero rows; packed by route.pack.
    router: mtl.Buffer,
    gate: tile.Matrix,
    up: tile.Matrix,
    down: tile.Matrix,
    shared_gate: tile.Matrix,
    shared_up: tile.Matrix,
    shared_down: tile.Matrix,
};

fn checked(ckpt: *const cp.Checkpoint, name: []const u8, dtype: cp.DType, shape: []const usize) !cp.Tensor {
    const t = try ckpt.get(name);
    if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape) or t.offset % 4 != 0) {
        std.log.err("qwen27: {s}: unexpected MoE tensor dtype, shape or alignment", .{name});
        return error.BadNativeTensor;
    }
    return t;
}

pub const Triple = struct { w: cp.Tensor, s: cp.Tensor, b: cp.Tensor };

/// `module`'s weight, scales and biases: `count` matrices [n, k] (count 0: one of rank 2) at width `w`.
fn triple(a: std.mem.Allocator, ckpt: *const cp.Checkpoint, module: []const u8, count: usize, n: usize, k: usize, w: widths.Width) !Triple {
    var shape: [3]usize = .{ count, n, k * w.bits / 32 };
    const lead: usize = if (count == 0) 1 else 0;
    var out: Triple = undefined;
    inline for (.{ "w", "s", "b" }, .{ ".weight", ".scales", ".biases" }, 0..) |field, suffix, i| {
        const name = try std.mem.concat(a, u8, &.{ module, suffix });
        defer a.free(name);
        if (i == 1) shape[2] = k / w.group;
        @field(out, field) = try checked(ckpt, name, if (i == 0) .u32 else .bf16, shape[lead..]);
    }
    return out;
}

fn matrix(t: Triple, n: usize, k: usize, bits: u8) tile.Matrix {
    return .{ .w = t.w.buffer, .w_off = t.w.offset, .scales = t.s.buffer, .s_off = t.s.offset, .biases = t.b.buffer, .b_off = t.b.offset, .n = n, .k = k, .bits = bits };
}

pub fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @truncate((u >> 16) | 0x40);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

pub fn float(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// Rows of an affine matrix as bf16: each weight f32(scale) * q + f32(bias), rounded to nearest even.
pub fn dequantize(t: Triple, rows: usize, k: usize, w: widths.Width, out: []u16) void {
    const words = t.w.host(u32);
    const scales = t.s.host(u16);
    const biases = t.b.host(u16);
    const per = 32 / @as(usize, w.bits);
    const mask = (@as(u32, 1) << @intCast(w.bits)) - 1;
    for (0..rows) |r| for (0..k) |j| {
        const q = (words[r * k / per + j / per] >> @intCast(w.bits * (j % per))) & mask;
        const g = r * (k / w.group) + j / w.group;
        out[r * k + j] = bf16(float(scales[g]) * @as(f32, @floatFromInt(q)) + float(biases[g]));
    };
}

/// Layer `layer`'s routed and shared experts under `prefix` ("language_model." or ""); the packed router is owned.
pub fn load(a: std.mem.Allocator, device: mtl.Device, ckpt: *const cp.Checkpoint, prefix: []const u8, c: Config, ws: *const widths.Widths, layer: usize, owned: *std.ArrayList(mtl.Buffer)) !Experts {
    if (@as(usize, ws.router.bits) * c.hidden % 32 != 0 or 32 % ws.router.bits != 0) return error.BadNativeTensor;
    const base = try std.fmt.allocPrint(a, "{s}model.layers.{d}.mlp.", .{ prefix, layer });
    defer a.free(base);
    var name: [160]u8 = undefined;
    const at = struct {
        fn module(buf: []u8, p: []const u8, leaf: []const u8) ![]const u8 {
            return std.fmt.bufPrint(buf, "{s}{s}", .{ p, leaf });
        }
    }.module;
    const e = c.experts;
    const xw = ws.experts;
    const gate = try triple(a, ckpt, try at(&name, base, "switch_mlp.gate_proj"), e, c.moe_intermediate, c.hidden, xw);
    const up = try triple(a, ckpt, try at(&name, base, "switch_mlp.up_proj"), e, c.moe_intermediate, c.hidden, xw);
    const down = try triple(a, ckpt, try at(&name, base, "switch_mlp.down_proj"), e, c.hidden, c.moe_intermediate, xw);
    const sgate = try triple(a, ckpt, try at(&name, base, "shared_expert.gate_proj"), 0, c.shared_intermediate, c.hidden, xw);
    const sup = try triple(a, ckpt, try at(&name, base, "shared_expert.up_proj"), 0, c.shared_intermediate, c.hidden, xw);
    const sdown = try triple(a, ckpt, try at(&name, base, "shared_expert.down_proj"), 0, c.hidden, c.shared_intermediate, xw);
    const router = try triple(a, ckpt, try at(&name, base, "gate"), 0, e, c.hidden, ws.router);
    const shared_router = try triple(a, ckpt, try at(&name, base, "shared_expert_gate"), 0, 1, c.hidden, ws.router);
    const cols = columns(c);
    const rows = try a.alloc(u16, cols * c.hidden);
    defer a.free(rows);
    @memset(rows, 0);
    dequantize(router, e, c.hidden, ws.router, rows[0 .. e * c.hidden]);
    dequantize(shared_router, 1, c.hidden, ws.router, rows[e * c.hidden ..][0..c.hidden]);
    const buffer = try device.buffer(cols * c.hidden * 2, mtl.ResourceOptions.shared);
    owned.append(a, buffer) catch |err| {
        buffer.deinit();
        return err;
    };
    route.pack(rows, buffer.slice(u16, cols * c.hidden), cols, c.hidden);
    return .{
        .router = buffer,
        .gate = matrix(gate, c.moe_intermediate, c.hidden, xw.bits),
        .up = matrix(up, c.moe_intermediate, c.hidden, xw.bits),
        .down = matrix(down, c.hidden, c.moe_intermediate, xw.bits),
        .shared_gate = matrix(sgate, c.shared_intermediate, c.hidden, xw.bits),
        .shared_up = matrix(sup, c.shared_intermediate, c.hidden, xw.bits),
        .shared_down = matrix(sdown, c.hidden, c.shared_intermediate, xw.bits),
    };
}

test "bf16 rounds to nearest even and dequantize reads codes low bits first" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(@bitCast(@as(u32, 0x3f808000)))); // a tie rounds to even
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16(@bitCast(@as(u32, 0x3f818000))));
    try std.testing.expectEqual(@as(usize, 260), columns(.{ .hidden = 0, .intermediate = 0, .layers = 0, .vocab = 0, .heads = 0, .kv_heads = 0, .head_dim = 0, .k_heads = 0, .v_heads = 0, .dk = 0, .dv = 0, .conv_kernel = 0, .eps = 0, .rope_dims = 0, .rope_theta = 0, .experts = 256 }));
}
