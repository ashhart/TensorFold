//! The Python loader's `_packed`, `_float`, `_conv` and `_halves`: a tensor or a projection's, read and checked.

const std = @import("std");
const quant = @import("core").quant;
const table = @import("table.zig");
const host = @import("host.zig");

const convert = quant.convert;
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

/// `_conv`: a depthwise conv weight as fp32 (channels, kernel); a trailing 1 is squeezed.
pub fn conv(a: std.mem.Allocator, t: *const Table, key: []const u8) Error!Tensor {
    var w = try float(a, t, key);
    if (w.rank == 3 and w.shape[2] == 1) {
        w.rank = 2;
    } else if (w.rank == 3 and w.shape[1] == 1) {
        return refuse(key, "still in the unsanitized (channels, 1, kernel) layout");
    }
    if (w.rank != 2) return refuse(key, "conv weight must be (channels, kernel)");
    return w;
}

/// `_packed`: a projection in whichever format its tensors are in.
pub fn projection(a: std.mem.Allocator, t: *const Table, key: []const u8) !host.Projection {
    return quant.read(a, t, key);
}

/// `_halves`: a fused [embedding | hidden] projection split in two along K, words and group tables alike.
pub fn halves(a: std.mem.Allocator, fused: host.Projection) ![2]host.Projection {
    return quant.halves(a, fused);
}
