//! Compatible rank-two base projections and the storage layouts a runtime overlay admits.
const std = @import("std");
const st = @import("safetensors.zig");

pub const Affine = struct {
    bits: u8,
    group: usize,
};

pub const Storage = union(enum) {
    dense: st.DType,
    affine: Affine,
    unsupported,
};

pub const Target = struct {
    module: []const u8,
    n: usize,
    k: usize,
    rank: usize = 2,
    storage: Storage,

    pub fn validate(t: Target) !void {
        if (t.module.len == 0 or t.rank != 2 or t.n == 0 or t.k == 0) return error.UnsupportedGeometry;
        _ = std.math.mul(usize, t.n, t.k) catch return error.UnsupportedGeometry;
        switch (t.storage) {
            .dense => |dtype| switch (dtype) {
                .f32, .bf16, .f16 => {},
                else => return error.UnsupportedQuantization,
            },
            .affine => |q| {
                if (q.bits != 4) return error.UnsupportedQuantization;
                if (q.group != 32 and q.group != 64 and q.group != 128) return error.UnsupportedQuantization;
                if (t.k % q.group != 0) return error.UnsupportedGeometry;
            },
            .unsupported => return error.UnsupportedQuantization,
        }
    }
};
