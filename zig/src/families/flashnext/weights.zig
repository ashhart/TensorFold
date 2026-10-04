//! Header-checked borrowed checkpoint views preserve the declared affine format and tensor dtype.
const std = @import("std");
const ckpt = @import("../../core/checkpoint.zig");
const st = @import("../../core/safetensors.zig");
const cfg = @import("config.zig");
const affine = @import("affine.zig");

pub const Matrix = union(enum) {
    raw: ckpt.Tensor,
    quantized: struct { weight: ckpt.Tensor, scales: ckpt.Tensor, biases: ckpt.Tensor, shape: affine.Shape, spec: affine.Spec },
};

fn entry(t: ckpt.Tensor) st.Entry {
    return .{ .dtype = t.dtype, .rank = t.rank, .shape = t.shape, .begin = 0, .end = t.bytes.len };
}

pub const Reader = struct {
    checkpoint: *ckpt.Checkpoint,
    config: *const cfg.Config,

    fn tensor(r: Reader, path: []const u8, suffix: []const u8) !ckpt.Tensor {
        var name: [256]u8 = undefined;
        return r.checkpoint.get(try std.fmt.bufPrint(&name, "{s}.{s}", .{ path, suffix }));
    }

    pub fn plain(r: Reader, name: []const u8, dtype: st.DType, shape: []const usize) !ckpt.Tensor {
        return r.checkpoint.expect(name, dtype, shape);
    }

    pub fn floating(r: Reader, name: []const u8, shape: []const usize) !ckpt.Tensor {
        const t = try r.checkpoint.get(name);
        if (t.dtype != .bf16 and t.dtype != .f16 and t.dtype != .f32) return error.InvalidFlashFloatingTensor;
        if (!t.is(t.dtype, shape)) return error.InvalidFlashFloatingTensor;
        return t;
    }

    pub fn matrix(r: Reader, path: []const u8, experts: usize, n: usize, k: usize) !Matrix {
        const weight = try r.tensor(path, "weight");
        const rank: usize = if (experts == 1) 2 else 3;
        if (experts == 0 or n == 0 or k == 0 or weight.rank != rank) return error.InvalidFlashMatrix;
        if (weight.dtype != .u32) {
            if (weight.dtype != .bf16 and weight.dtype != .f16 and weight.dtype != .f32) return error.InvalidFlashMatrix;
            if (weight.shape[rank - 2] != n or weight.shape[rank - 1] != k or (rank == 3 and weight.shape[0] != experts)) return error.InvalidFlashMatrix;
            return .{ .raw = weight };
        }
        const spec = (try r.config.quantization(path)) orelse return error.DisabledPackedProjection;
        const scales = try r.tensor(path, "scales");
        const biases = try r.tensor(path, "biases");
        const shape = try affine.matrix(entry(weight), entry(scales), entry(biases), spec);
        if (shape.experts != experts or shape.n != n or shape.k != k) return error.InvalidFlashMatrix;
        return .{ .quantized = .{ .weight = weight, .scales = scales, .biases = biases, .shape = shape, .spec = spec } };
    }

    pub fn bodyReadout(r: Reader, embedding: bool) !Matrix {
        return r.matrix(if (embedding) "language_model.model.embed_tokens" else "language_model.lm_head", 1, r.config.vocab, r.config.hidden);
    }

    pub fn hc(r: Reader, path: []const u8, inject: bool) !HyperConnection {
        const wide = try r.config.wide();
        var norm_name: [256]u8 = undefined;
        return .{
            .norm = try r.floating(try std.fmt.bufPrint(&norm_name, "{s}.hc_norm.weight", .{path}), &.{wide}),
            .down = try r.matrixPath(path, "input_mix_weight_down", r.config.hc_lowrank, wide),
            .up = try r.matrixPath(path, "input_mix_weight_up", wide, r.config.hc_lowrank),
            .inject = if (inject) try r.matrixPath(path, "block_inject_weight", r.config.hc_count, wide) else null,
        };
    }

    fn matrixPath(r: Reader, path: []const u8, suffix: []const u8, n: usize, k: usize) !Matrix {
        var name: [256]u8 = undefined;
        return r.matrix(try std.fmt.bufPrint(&name, "{s}.{s}", .{ path, suffix }), 1, n, k);
    }
};

pub const HyperConnection = struct { norm: ckpt.Tensor, down: Matrix, up: Matrix, inject: ?Matrix };
