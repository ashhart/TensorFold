//! The joint schema head's weights on the device: every BF16 tensor in one buffer, uploaded once at open.
const std = @import("std");
const js = @import("core").joint_schema;
const runtime = @import("../runtime.zig");
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;

/// The head's tensors as device addresses, in the host loader's structure.
pub const Dev = js.tensors.WeightsOf(u64);

/// The head's scalars as its BF16 module computes them: exp(clamp(p, ln 100)) and sigmoid(gate), each rounded.
pub const Scalars = struct { prior_scale: f32, joint_scale: f32, gate: f32 };

fn r16(v: f32) f32 {
    const u: u32 = @bitCast(v);
    return @bitCast(((u +% 0x7fff +% ((u >> 16) & 1)) >> 16) << 16);
}

fn scalar(t: js.tensors.Tensor) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[0..2], .little)) << 16);
}

pub fn scalars(w: *const js.tensors.Weights) Scalars {
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

/// Each tensor's bytes, 256-aligned, summed: the one buffer `upload` allocates.
pub fn bytes(a: std.mem.Allocator, host: *const js.tensors.Weights) !usize {
    var size: Sizer = .{};
    _ = try js.tensors.fillOf(u64, a, host.cfg, &size);
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

pub const Weights = struct {
    buf: DeviceBuffer,
    arena: std.heap.ArenaAllocator,
    w: Dev,
    s: Scalars,

    /// Uploads the mapped head `file` (checked by the host loader) and keeps the scalars.
    pub fn upload(gpa: std.mem.Allocator, r: *const runtime.Runtime, file: anytype, host: *const js.tensors.Weights) !Weights {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const total = try bytes(arena.allocator(), host);
        const staging = try gpa.alloc(u8, total);
        defer gpa.free(staging);
        @memset(staging, 0);
        var buf = try DeviceBuffer.alloc(r, total);
        errdefer buf.free();
        var put: Put(@TypeOf(file)) = .{ .file = file, .staging = staging, .base = buf.base() };
        const w = try js.tensors.fillOf(u64, arena.allocator(), host.cfg, &put);
        try buf.upload(0, staging);
        return .{ .buf = buf, .arena = arena, .w = w, .s = scalars(host) };
    }

    pub fn deinit(w: *Weights) void {
        w.buf.free();
        w.arena.deinit();
        w.* = undefined;
    }
};

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
    var w: js.tensors.Weights = undefined;
    w.prior_logit_scale = .{ .dtype = .bf16, .rank = 0, .shape = @splat(1), .bytes = &two };
    w.joint_logit_scale = w.prior_logit_scale;
    w.residual_gate = .{ .dtype = .bf16, .rank = 0, .shape = @splat(1), .bytes = &minus };
    const s = scalars(&w);
    try std.testing.expectEqual(@as(f32, 9.9375), s.prior_scale);
    try std.testing.expectEqual(@as(f32, 9.9375), s.joint_scale);
    try std.testing.expectEqual(@as(f32, 0.119140625), s.gate);
}
