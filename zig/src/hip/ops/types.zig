//! The types the launches share: buffer kinds, a device tensor, an MLX affine matrix and the address casts.

const runtime = @import("../runtime.zig");

/// act.hpp's numbering: the activation and fp32 buffers the torch-op kernels read and write.
pub const Kind = enum(c_int) {
    f32 = 0,
    f16 = 1,
    bf16 = 2,

    pub fn size(k: Kind) usize {
        return if (k == .f32) 4 else 2;
    }

    /// The attention kernel's cache numbering: 0 fp16, 1 bf16, 2 fp32.
    pub fn cache(k: Kind) c_int {
        return switch (k) {
            .f16 => 0,
            .bf16 => 1,
            .f32 => 2,
        };
    }

    /// The affine kernels' table numbering: 0 fp32, 1 bf16, 2 fp16.
    pub fn table(k: Kind) c_int {
        return switch (k) {
            .f32 => 0,
            .bf16 => 1,
            .f16 => 2,
        };
    }
};

pub const Error = runtime.Error || error{ OutOfDeviceMemory, BadShape };

/// A device buffer of `kind` values.
pub const Tensor = struct { ptr: u64, kind: Kind };

/// One MLX affine matrix (N, K): packed words (N, K * bits / 32), scale and bias (N, K / group) of `tables`.
pub const Affine = struct {
    words: u64,
    scale: u64,
    bias: u64,
    tables: Kind,
    n: u32,
    k: u32,
    bits: u8,
    group: u16,
    /// A tensor-parallel slice along K: the product stays fp32, one rank's share of a sum.
    partial: bool = false,

    pub fn check(a: Affine) Error!void {
        const ok_bits = switch (a.bits) {
            2, 3, 4, 5, 6, 8 => true,
            else => false,
        };
        if (!ok_bits or (a.group != 32 and a.group != 64 and a.group != 128)) return error.BadShape;
        if (a.k % a.group != 0 or (@as(u64, a.k) * a.bits) % 32 != 0) return error.BadShape;
    }
};

pub fn p(addr: u64) ?*anyopaque {
    return @ptrFromInt(addr);
}

pub fn f(addr: u64) ?[*]f32 {
    return @ptrFromInt(addr);
}

pub fn i(addr: u64) ?[*]i32 {
    return @ptrFromInt(addr);
}

pub fn int(v: anytype) c_int {
    return @intCast(v);
}
