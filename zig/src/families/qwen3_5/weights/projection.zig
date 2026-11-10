//! The Python loader's `_packed`, `_float`, `_conv` and `_halves`, and torch's `1 + w` norms: read and checked.

const std = @import("std");
const quant = @import("core").quant;
const table = @import("table.zig");
const host = @import("host.zig");

const convert = quant.convert;
const modelopt = @import("modelopt.zig");
const Tensor = table.Tensor;
const Table = table.Table;
pub const Error = table.Error || convert.Error || error{UnexpectedTensor};

pub const join = quant.join;

fn refuse(key: []const u8, what: []const u8) error{UnexpectedTensor} {
    std.log.err("{s}: {s}", .{ key, what });
    return error.UnexpectedTensor;
}

/// `_float`: the tensor widened to fp32.
pub fn float(a: std.mem.Allocator, t: *const Table, key: []const u8) Error!Tensor {
    const src = try t.get(key);
    if (!convert.isFloat(src.dtype)) return refuse(key, "not a floating tensor");
    return convert.float32(a, src);
}

/// A zero-centred RMS weight as torch's Qwen3_5RMSNorm applies it: `1.0 + float(w)` in fp32, not rounded back.
pub fn centred(a: std.mem.Allocator, t: *const Table, key: []const u8) Error!Tensor {
    const w = try float(a, t, key);
    const out = try a.alloc(f32, w.numel());
    for (out, 0..) |*o, i| o.* = 1.0 + convert.load(.f32, w.bytes, i);
    return .{ .dtype = .f32, .rank = w.rank, .shape = w.shape, .bytes = std.mem.sliceAsBytes(out) };
}

/// `_conv`: a depthwise conv weight as fp32 (channels, kernel); a trailing 1 is squeezed, HF's middle 1 for bf16.
pub fn conv(a: std.mem.Allocator, t: *const Table, key: []const u8) Error!Tensor {
    var w = try float(a, t, key);
    if (t.quant == .bf16) return squeezed(w) orelse refuse(key, "a bf16 conv weight must be (channels, 1, kernel)");
    if (w.rank == 3 and w.shape[2] == 1) {
        w.rank = 2;
    } else if (w.rank == 3 and w.shape[1] == 1) {
        return refuse(key, "still in the unsanitized (channels, 1, kernel) layout");
    }
    if (w.rank != 2) return refuse(key, "conv weight must be (channels, kernel)");
    return w;
}

/// HF's Conv1d weight (channels, 1, kernel) as (channels, kernel), the same bytes; null for any other layout.
pub fn squeezed(w: Tensor) ?Tensor {
    if (w.rank != 3 or w.shape[1] != 1) return null;
    var out = w;
    out.shape[1] = w.shape[2];
    out.shape[2] = 1;
    out.rank = 2;
    return out;
}

/// `_packed`: a projection in whichever format its tensors are in; a bf16 one's are BF16 as stored, or fp32 widened.
pub fn projection(a: std.mem.Allocator, t: *const Table, key: []const u8) !host.Projection {
    if (modelopt.algo(t, key)) |what| return modelopt.read(a, t, key, what);
    if (t.quant == .bf16 and t.widen) {
        _ = try quant.bf16.read(a, t, key); // the same dtype and rank checks
        return .{ .dense = try quant.dense.read(a, t, key) };
    }
    if (t.quant == .bf16) return .{ .bf16 = try quant.bf16.read(a, t, key) };
    return host.read(a, t, key);
}

/// `_halves`: a fused [embedding | hidden] projection split in two along K, words and group tables alike.
pub fn halves(a: std.mem.Allocator, fused: host.Projection) ![2]host.Projection {
    return host.halves(a, fused);
}
