//! The types the launches share: buffer kinds, a device tensor and the address casts.

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
};

pub const Error = runtime.Error || error{ OutOfDeviceMemory, BadShape };

/// A device buffer of `kind` values.
pub const Tensor = struct { ptr: u64, kind: Kind };

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
