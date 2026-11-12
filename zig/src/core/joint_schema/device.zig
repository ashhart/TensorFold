//! The head's weights as a device backend uploads them: one 256-aligned buffer, staged here, and its BF16 scalars.
const std = @import("std");
const tensors = @import("tensors.zig");

/// The head's scalars as its BF16 module computes them: exp(clamp(p, ln 100)) and sigmoid(gate), each rounded.
pub const Scalars = struct { prior_scale: f32, joint_scale: f32, gate: f32 };

fn r16(v: f32) f32 {
    const u: u32 = @bitCast(v);
    return @bitCast(((u +% 0x7fff +% ((u >> 16) & 1)) >> 16) << 16);
}

fn scalar(t: tensors.Tensor) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[0..2], .little)) << 16);
}

pub fn scalars(w: *const tensors.Weights) Scalars {
    const cap: f32 = @floatCast(@log(@as(f64, 100)));
    const capped = struct {
        fn f(p: f32, c: f32) f32 {
            return r16(@exp(r16(@min(p, c))));
        }
    }.f;
    return .{
        .prior_scale = capped(scalar(w.prior_logit_scale), cap),
        .joint_scale = capped(scalar(w.joint_logit_scale), cap),
        .gate = r16(1 / (1 + @exp(-scalar(w.residual_gate)))),
    };
}

/// Each tensor's bytes, 256-aligned, summed: the one buffer a backend's upload allocates.
pub fn bytes(a: std.mem.Allocator, host: *const tensors.Weights) !usize {
    var size: Sizer = .{};
    _ = try tensors.fillOf(u64, a, host.cfg, &size);
    return size.at;
}

/// The shapes the host loader checked give each tensor's bytes.
const Sizer = struct {
    at: usize = 0,

    pub fn take(s: *Sizer, _: []const u8, shape: []const usize) !u64 {
        var n: usize = 2;
        for (shape) |d| n *= d;
        const here = s.at;
        s.at += std.mem.alignForward(usize, n, 256);
        return here;
    }
};

/// The head's tensors as device addresses: each named tensor of `file` copied into `staging`, addressed from `base`.
pub fn stage(a: std.mem.Allocator, host: *const tensors.Weights, file: anytype, staging: []u8, base: u64) !tensors.WeightsOf(u64) {
    var put: Put(@TypeOf(file)) = .{ .file = file, .staging = staging, .base = base };
    return tensors.fillOf(u64, a, host.cfg, &put);
}

fn Put(comptime F: type) type {
    return struct {
        file: F,
        staging: []u8,
        base: u64,
        at: usize = 0,

        pub fn take(p: *@This(), name: []const u8, _: []const usize) !u64 {
            const t = p.file.get(name) orelse return error.MissingTensor;
            @memcpy(p.staging[p.at..][0..t.bytes.len], t.bytes);
            const here = p.at;
            p.at += std.mem.alignForward(usize, t.bytes.len, 256);
            return p.base + here;
        }
    };
}

test "joint schema weights: Clef-Flash's scalars as its BF16 head rounds them" {
    const two: [2]u8 = .{ 0x13, 0x40 }; // bf16 2.296875
    const minus: [2]u8 = .{ 0x00, 0xc0 }; // bf16 -2
    var w: tensors.Weights = undefined;
    w.prior_logit_scale = .{ .dtype = .bf16, .rank = 0, .shape = @splat(1), .bytes = &two };
    w.joint_logit_scale = w.prior_logit_scale;
    w.residual_gate = .{ .dtype = .bf16, .rank = 0, .shape = @splat(1), .bytes = &minus };
    const s = scalars(&w);
    try std.testing.expectEqual(@as(f32, 9.9375), s.prior_scale);
    try std.testing.expectEqual(@as(f32, 9.9375), s.joint_scale);
    try std.testing.expectEqual(@as(f32, 0.119140625), s.gate);
}
