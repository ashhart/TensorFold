//! Header-only large matrices and real small-vector bytes for source preflight; no CUDA interface exists here.
const std = @import("std");
const core = @import("core");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_weights.zig");
const DType = core.safetensors.DType;
const Tensor = core.safetensors.Tensor;

pub const Bad = enum { none, input_norm, post_norm, final_norm, q_norm, k_norm, gdn_norm, conv, a_log, dt_bias, weight_shape, weight_type, missing };

pub const Fixture = struct {
    g: geometry.Geometry,
    source: bool,
    bad: Bad = .none,
    scale: f32 = 1,
    reads: usize = 0,
    matrices: usize = 0,
    endpoints: usize = 0,
    vector: [10240]u8 = undefined,
    small: [192]u8 = undefined,
    conv: [81920]u8 = @splat(0),

    fn tensor(dtype: DType, shape: []const usize, bytes: []const u8) Tensor {
        var out: Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = bytes };
        @memcpy(out.shape[0..shape.len], shape);
        return out;
    }

    fn layer(name: []const u8) !usize {
        const at = (std.mem.indexOf(u8, name, ".layers.") orelse return error.UnexpectedTensor) + 8;
        const end = std.mem.indexOfScalarPos(u8, name, at, '.') orelse return error.UnexpectedTensor;
        return std.fmt.parseInt(usize, name[at..end], 10);
    }

    fn slot(name: []const u8) ?geometry.Slot {
        for (std.enums.values(geometry.Slot)) |s| {
            if (std.mem.indexOf(u8, name, geometry.slot_modules.get(s)) != null) return s;
        }
        return null;
    }

    pub fn has(f: *Fixture, name: []const u8) bool {
        const s = slot(name) orelse return false;
        const i = layer(name) catch return false;
        return std.mem.endsWith(u8, name, ".weight_packed") and f.g.kinds.of(i, s) == .nvfp4;
    }

    pub fn expect(f: *Fixture, name: []const u8, dtype: DType, shape: []const usize) !Tensor {
        const got = try f.get(name);
        if (!got.is(dtype, shape)) return error.UnexpectedTensor;
        return got;
    }

    pub fn get(f: *Fixture, name: []const u8) !Tensor {
        f.reads += 1;
        if (std.mem.indexOf(u8, name, "embed_tokens") != null or std.mem.indexOf(u8, name, "lm_head") != null) {
            f.endpoints += 1;
            return error.UnexpectedTensor;
        }
        if (slot(name)) |s| {
            const i = try layer(name);
            const nk = weights.shape(f.g, s);
            const kind = f.g.kinds.of(i, s);
            const codes = std.mem.endsWith(u8, name, ".weight_packed") or std.mem.endsWith(u8, name, ".weight");
            if (codes) {
                if (f.bad == .missing) return error.MissingTensor;
                f.matrices += 1;
                const dtype: DType = if (f.bad == .weight_type) .f16 else if (kind == .bf16) .bf16 else .u8;
                return tensor(dtype, &.{ nk[0] + @intFromBool(f.bad == .weight_shape), nk[1] / (if (kind == .bf16) @as(usize, 1) else 2) }, &.{});
            }
            if (std.mem.endsWith(u8, name, ".weight_scale")) return tensor(.f8_e4m3, &.{ nk[0], nk[1] / 16 }, &.{});
            std.mem.writeInt(u32, f.small[0..4], @bitCast(f.scale), .little);
            return tensor(.f32, &.{}, f.small[0..4]);
        }
        if (std.mem.endsWith(u8, name, "conv1d.weight")) {
            f.conv[0] = @intFromBool(f.bad == .conv);
            return tensor(.bf16, if (f.source) &.{ 10240, 1, 4 } else &.{ 10240, 4, 1 }, &f.conv);
        }
        const a_log = std.mem.endsWith(u8, name, "A_log");
        const dt = std.mem.endsWith(u8, name, "dt_bias");
        if (a_log or dt) {
            const v: u32 = if (a_log) 0x3f800000 else 0x3f000000;
            const corrupt = (a_log and f.bad == .a_log) or (dt and f.bad == .dt_bias);
            if (f.source) {
                for (0..48) |i| std.mem.writeInt(u32, f.small[i * 4 ..][0..4], v | @as(u32, @intFromBool(corrupt and i == 0)), .little);
                return tensor(.f32, &.{48}, &f.small);
            }
            for (0..48) |i| std.mem.writeInt(u16, f.small[i * 2 ..][0..2], @intCast(v >> 16), .little);
            return tensor(.bf16, &.{48}, f.small[0..96]);
        }
        const input = std.mem.endsWith(u8, name, "input_layernorm.weight");
        const post = std.mem.endsWith(u8, name, "post_attention_layernorm.weight");
        const q = std.mem.endsWith(u8, name, "q_norm.weight");
        const k = std.mem.endsWith(u8, name, "k_norm.weight");
        const gdn = std.mem.endsWith(u8, name, "linear_attn.norm.weight");
        const n: usize = if (q or k) 256 else if (gdn) 128 else 5120;
        for (0..n) |i| std.mem.writeInt(u16, f.vector[i * 2 ..][0..2], if (f.source and !gdn) 0 else 0x3f80, .little);
        const bad = (input and f.bad == .input_norm) or (post and f.bad == .post_norm) or
            (q and f.bad == .q_norm) or (k and f.bad == .k_norm) or (gdn and f.bad == .gdn_norm) or
            (!input and !post and !q and !k and !gdn and f.bad == .final_norm);
        if (bad) f.vector[0] ^= 1;
        return tensor(.bf16, &.{n}, f.vector[0 .. n * 2]);
    }
};
