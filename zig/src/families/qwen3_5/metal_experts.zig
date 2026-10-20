//! A qwen3_5_moe layer's experts on Metal: both routers as bf16 packed for core/moe_route, the experts as the tiles
//! read them. The 27B's engine loads them in its one layer loop (qwen27/gpu_weights.zig).
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const cp = core.checkpoint_metal;
const tile = core.tile_qmm;
const route = core.moe_route;

/// An affine width: `bits` a weight, a scale and a bias per `group` weights.
pub const Width = struct { bits: u8, group: u16 };

/// The checkpoint's expert sizes (config.json's num_experts, hidden_size and the two expert widths).
pub const Shape = struct { experts: usize, hidden: usize, moe_width: usize, shared_width: usize };

/// The routed and shared experts' width and the two routers', each one width in every layer.
pub const Widths = struct { experts: Width, router: Width };

/// Logit columns a row: the routed experts, the shared expert's gate, padding to the logits kernel's blocks of 4.
pub fn columns(experts: usize) usize {
    return (experts + 1 + 3) / 4 * 4;
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
        std.log.err("qwen3_5_moe: {s}: unexpected MoE tensor dtype, shape or alignment", .{name});
        return error.BadNativeTensor;
    }
    return t;
}

pub const Triple = struct { w: cp.Tensor, s: cp.Tensor, b: cp.Tensor };

/// `module`'s weight, scales and biases: `count` matrices [n, k] (count 0: one of rank 2) at width `w`.
fn triple(a: std.mem.Allocator, ckpt: *const cp.Checkpoint, module: []const u8, count: usize, n: usize, k: usize, w: Width) !Triple {
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
pub fn dequantize(t: Triple, rows: usize, k: usize, w: Width, out: []u16) void {
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
pub fn load(a: std.mem.Allocator, device: mtl.Device, ckpt: *const cp.Checkpoint, prefix: []const u8, s: Shape, ws: Widths, layer: usize, owned: *std.ArrayList(mtl.Buffer)) !Experts {
    if (@as(usize, ws.router.bits) * s.hidden % 32 != 0 or 32 % ws.router.bits != 0) return error.BadNativeTensor;
    const base = try std.fmt.allocPrint(a, "{s}model.layers.{d}.mlp.", .{ prefix, layer });
    defer a.free(base);
    var name: [160]u8 = undefined;
    const at = struct {
        fn module(buf: []u8, p: []const u8, leaf: []const u8) ![]const u8 {
            return std.fmt.bufPrint(buf, "{s}{s}", .{ p, leaf });
        }
    }.module;
    const e = s.experts;
    const xw = ws.experts;
    const gate = try triple(a, ckpt, try at(&name, base, "switch_mlp.gate_proj"), e, s.moe_width, s.hidden, xw);
    const up = try triple(a, ckpt, try at(&name, base, "switch_mlp.up_proj"), e, s.moe_width, s.hidden, xw);
    const down = try triple(a, ckpt, try at(&name, base, "switch_mlp.down_proj"), e, s.hidden, s.moe_width, xw);
    const sgate = try triple(a, ckpt, try at(&name, base, "shared_expert.gate_proj"), 0, s.shared_width, s.hidden, xw);
    const sup = try triple(a, ckpt, try at(&name, base, "shared_expert.up_proj"), 0, s.shared_width, s.hidden, xw);
    const sdown = try triple(a, ckpt, try at(&name, base, "shared_expert.down_proj"), 0, s.hidden, s.shared_width, xw);
    const router = try triple(a, ckpt, try at(&name, base, "gate"), 0, e, s.hidden, ws.router);
    const shared_router = try triple(a, ckpt, try at(&name, base, "shared_expert_gate"), 0, 1, s.hidden, ws.router);
    const cols = columns(e);
    const rows = try a.alloc(u16, cols * s.hidden);
    defer a.free(rows);
    @memset(rows, 0);
    dequantize(router, e, s.hidden, ws.router, rows[0 .. e * s.hidden]);
    dequantize(shared_router, 1, s.hidden, ws.router, rows[e * s.hidden ..][0..s.hidden]);
    const buffer = try device.buffer(cols * s.hidden * 2, mtl.ResourceOptions.shared);
    owned.append(a, buffer) catch |err| {
        buffer.deinit();
        return err;
    };
    route.pack(rows, buffer.slice(u16, cols * s.hidden), cols, s.hidden);
    return .{
        .router = buffer,
        .gate = matrix(gate, s.moe_width, s.hidden, xw.bits),
        .up = matrix(up, s.moe_width, s.hidden, xw.bits),
        .down = matrix(down, s.hidden, s.moe_width, xw.bits),
        .shared_gate = matrix(sgate, s.shared_width, s.hidden, xw.bits),
        .shared_up = matrix(sup, s.shared_width, s.hidden, xw.bits),
        .shared_down = matrix(sdown, s.hidden, s.shared_width, xw.bits),
    };
}

test "bf16 rounds to nearest even and dequantize reads codes low bits first" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(@bitCast(@as(u32, 0x3f808000)))); // a tie rounds to even
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16(@bitCast(@as(u32, 0x3f818000))));
    try std.testing.expectEqual(@as(usize, 260), columns(256));
}
