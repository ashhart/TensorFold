//! The affine bits and group a checkpoint declares; no checkpoint file is read, so any backend's module takes it.
const std = @import("std");

pub const Spec = struct {
    bits: u8,
    group: usize,

    pub fn init(bits: usize, group: usize) !Spec {
        if (bits != 2 and bits != 3 and bits != 4 and bits != 5 and bits != 6 and bits != 8) return error.UnsupportedAffineBits;
        if (group != 32 and group != 64 and group != 128) return error.UnsupportedAffineGroup;
        return .{ .bits = @intCast(bits), .group = group };
    }

    pub fn words(s: Spec, k: usize) !usize {
        if (k == 0 or k % s.group != 0) return error.InvalidAffineWidth;
        const count = try std.math.mul(usize, k, s.bits);
        if (count % 32 != 0) return error.InvalidAffineWidth;
        return count / 32;
    }
};
