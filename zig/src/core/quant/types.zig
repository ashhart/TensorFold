//! What the format modules share: format tags, device decoder ids and the uploaded tensor; nothing here is a backend's.

const std = @import("std");
const st = @import("../safetensors.zig");

pub const Tensor = st.Tensor;
pub const DType = st.DType;

/// The weight formats the backend reads.
pub const Format = enum { mlx, dense };

/// The decoder a format's kernels instantiate the shared tiles with: each backend names the kernels of its decoders.
pub const Decoder = enum(u8) { mlx = 0, dense = 1 };

/// The type of a group table, numbered as the kernels number it.
pub const Tables = enum(u8) { f32 = 0, bf16 = 1, f16 = 2 };

/// The config objects a format reads its quantization from: `quantization` of the text tower or the root, and the root.
pub const Sources = struct { root: std.json.ObjectMap, quantization: std.json.ObjectMap };

/// `parts` joined in `buf`: a tensor's key from its base and suffix.
pub fn join(buf: []u8, parts: []const []const u8) []const u8 {
    var n: usize = 0;
    for (parts) |p| {
        @memcpy(buf[n..][0..p.len], p);
        n += p.len;
    }
    return buf[0..n];
}

/// A tensor the loader refuses, logged under its key.
pub fn refuse(key: []const u8, what: []const u8) error{UnexpectedTensor} {
    std.log.err("{s}: {s}", .{ key, what });
    return error.UnexpectedTensor;
}

/// One uploaded tensor: its device address and the element type and shape of the bytes there.
pub const Buf = struct {
    ptr: u64,
    len: usize,
    dtype: DType,
    rank: u8,
    shape: [st.max_rank]usize,

    pub fn dim(b: Buf, i: usize) usize {
        return if (i < b.rank) b.shape[i] else 1;
    }
};

/// How a backend puts a host tensor on its device: `put` copies it, returns where it landed, keeps what it must free.
pub const Uploader = struct {
    ctx: *anyopaque,
    put: *const fn (ctx: *anyopaque, t: Tensor) anyerror!Buf,

    pub fn tensor(u: Uploader, t: Tensor) !Buf {
        return u.put(u.ctx, t);
    }
};
