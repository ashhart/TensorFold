//! Idol's Gaussians from the Gaussian decoder's rows (idol-sdk's gaussian.py), the 3DGS PLY, and the camera.
const std = @import("std");
const dc = @import("decoder_config.zig");

const G = dc.per_anchor;

/// Activated Gaussians, z-up inside [-0.5, 0.5]^3: xyz [n, 3], colour [n, 3], scale [n, 3], quaternion wxyz [n, 4].
pub const Gaussians = struct {
    xyz: []f32,
    features_dc: []f32,
    scaling: []f32,
    rotation: []f32,
    opacity: []f32,

    pub fn alloc(gpa: std.mem.Allocator, n: usize) !Gaussians {
        const all = try gpa.alloc(f32, n * 14);
        return .{ .xyz = all[0 .. 3 * n], .features_dc = all[3 * n .. 6 * n], .scaling = all[6 * n .. 9 * n], .rotation = all[9 * n .. 13 * n], .opacity = all[13 * n .. 14 * n] };
    }

    pub fn free(g: Gaussians, gpa: std.mem.Allocator) void {
        gpa.free(g.xyz.ptr[0 .. g.opacity.len * 14]);
    }

    pub fn count(g: Gaussians) usize {
        return g.opacity.len;
    }
};

/// torch's F.softplus (beta 1, threshold 20).
pub fn softplus(x: f32) f32 {
    return if (x > 20) x else std.math.log1p(@exp(x));
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// inverse_softplus(torch.tensor(scaling_bias)) and inverse_sigmoid(torch.tensor(opacity_bias)), in fp32.
pub fn offsets(rep: dc.Representation) [2]f32 {
    const s: f32 = @floatCast(rep.scaling_bias);
    const o: f32 = @floatCast(rep.opacity_bias);
    return .{ s + @log(-std.math.expm1(-s)), @log(o / (1 - o)) };
}

/// The activation's fixed inputs: representation_config, base_offset_scale and points_offset_perturbation [32, 3].
pub const Params = struct { rep: dc.Representation, base_offset_scale: f32, perturbation: []const f32 };

/// to_representation then Gaussian(): raw rows [anchors, 480] at anchors [anchors, 3] into `out` (anchors x 32).
pub fn activate(p: Params, raw: []const f32, anchors: []const f32, out: Gaussians) void {
    const n = anchors.len / 3;
    std.debug.assert(raw.len == n * dc.raw_width and out.count() == n * G and p.perturbation.len == G * 3);
    const off = offsets(p.rep);
    const r = p.rep;
    const lr = [5]f32{ @floatCast(r.lr_xyz), @floatCast(r.lr_features_dc), @floatCast(r.lr_scaling), @floatCast(r.lr_rotation), @floatCast(r.lr_opacity) };
    const size: f32 = @floatCast(r.perturbe_size);
    const filter2: f32 = @floatCast(r.filter * r.filter);
    for (0..n) |a| {
        const line = raw[a * dc.raw_width ..][0..dc.raw_width];
        for (0..G) |g| {
            const i = a * G + g;
            const scale = softplus(line[448 + g] + p.base_offset_scale);
            for (0..3) |ax| {
                const o = line[g * 3 + ax] * lr[0] + p.perturbation[g * 3 + ax];
                const offset = ((std.math.tanh(o) * 0.5) * size) * scale;
                out.xyz[i * 3 + ax] = (offset + anchors[a * 3 + ax]) * 1.0 + -0.5;
                out.features_dc[i * 3 + ax] = line[96 + g * 3 + ax] * lr[1];
                const s = softplus(line[192 + g * 3 + ax] * lr[2] + off[0]);
                out.scaling[i * 3 + ax] = @sqrt(s * s + filter2);
            }
            var q: [4]f32 = undefined;
            var sq: f32 = 0;
            for (&q, 0..) |*v, c| {
                v.* = line[288 + g * 4 + c] * lr[3] + @as(f32, if (c == 0) 1 else 0);
                sq += v.* * v.*;
            }
            const norm = @max(@sqrt(sq), 1e-12);
            for (q, 0..) |v, c| out.rotation[i * 4 + c] = v / norm;
            out.opacity[i] = sigmoid(line[416 + g] * lr[4] + off[1]);
        }
    }
}

pub const properties = [_][]const u8{ "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3" };
pub const row_bytes = properties.len * 4;

/// The header plyfile writes for `n` vertices of the 17 float properties.
pub fn header(buf: []u8, n: usize) ![]const u8 {
    var w: std.Io.Writer = .fixed(buf);
    try w.print("ply\nformat binary_little_endian 1.0\nelement vertex {d}\n", .{n});
    for (properties) |name| try w.print("property float {s}\n", .{name});
    try w.writeAll("end_header\n");
    return w.buffered();
}

/// One PLY row as save_ply writes it: (x, -z, y), clamped logit opacity and log scale, the quaternion turned in f64.
pub fn row(g: Gaussians, i: usize, out: *[properties.len]f32) void {
    const eps: f32 = 1e-6;
    const hi: f32 = @floatCast(@as(f64, 1) - 1e-6);
    const xyz = g.xyz[i * 3 ..][0..3];
    out[0] = xyz[0];
    out[1] = -xyz[2];
    out[2] = xyz[1];
    out[3] = 0;
    out[4] = 0;
    out[5] = 0;
    @memcpy(out[6..9], g.features_dc[i * 3 ..][0..3]);
    const o = std.math.clamp(g.opacity[i], eps, hi);
    out[9] = @log(o / (1 - o));
    for (0..3) |a| out[10 + a] = @log(@max(g.scaling[i * 3 + a], eps));
    const aw: f64 = @sqrt(@as(f64, 0.5));
    const ax = aw;
    const ay: f64 = 0;
    const az: f64 = 0;
    const b = g.rotation[i * 4 ..][0..4];
    const bw: f64 = b[0];
    const bx: f64 = b[1];
    const by: f64 = b[2];
    const bz: f64 = b[3];
    out[13] = @floatCast(aw * bw - ax * bx - ay * by - az * bz);
    out[14] = @floatCast(aw * bx + ax * bw + ay * bz - az * by);
    out[15] = @floatCast(aw * by - ax * bz + ay * bw + az * bx);
    out[16] = @floatCast(aw * bz + ax * by - ay * bx + az * bw);
}

/// The whole PLY file in memory: header then every row, little-endian float32.
pub fn ply(gpa: std.mem.Allocator, g: Gaussians) ![]u8 {
    var head: [1024]u8 = undefined;
    const h = try header(&head, g.count());
    const bytes = try gpa.alloc(u8, h.len + g.count() * row_bytes);
    @memcpy(bytes[0..h.len], h);
    for (0..g.count()) |i| {
        var r: [properties.len]f32 = undefined;
        row(g, i, &r);
        for (r, 0..) |v, c| std.mem.writeInt(u32, bytes[h.len + i * row_bytes + c * 4 ..][0..4], @bitCast(v), .little);
    }
    return bytes;
}

/// decode_camera: yaw and pitch (radians), radius, vertical field of view (degrees) from the camera latent [5].
pub const Camera = struct { yaw: f64, pitch: f64, radius: f64, fov: f64 };

pub fn camera(c: [5]f32) Camera {
    const norm = @sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
    var d = [3]f64{ 0, 0, 1 };
    if (norm > 1e-6) for (&d, 0..) |*v, i| {
        v.* = c[i] / norm;
    };
    const inv: f64 = c[3];
    const hw: f64 = c[4];
    const deg = 180.0 / std.math.pi;
    const radius: f64 = if (inv < 1e-3) 1000.0 else 1.0 / inv;
    const fov = if (inv < 1e-3) deg * 2 * std.math.atan(hw / radius) else deg * 2 * std.math.atan(hw * inv);
    return .{ .yaw = std.math.atan2(d[0], d[1]), .pitch = std.math.asin(std.math.clamp(d[2], -1.0, 1.0)), .radius = radius, .fov = fov };
}

test "the activation offsets are torch's fp32 values for the released biases" {
    const o = offsets(dc.released);
    try std.testing.expectEqual(@as(f32, -5.519460201263428), o[0]);
    try std.testing.expectEqual(@as(f32, -2.1972246170043945), o[1]);
}

test "zero rows give unit quaternions, the bias opacity and the scale floor" {
    const a = std.testing.allocator;
    var raw: [dc.raw_width]f32 = @splat(0);
    const pert: [G * 3]f32 = @splat(0);
    const g = try Gaussians.alloc(a, G);
    defer g.free(a);
    activate(.{ .rep = dc.released, .base_offset_scale = 0, .perturbation = &pert }, &raw, &.{ 0.5, 0.5, 0.5 }, g);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), g.opacity[5], 1e-7);
    try std.testing.expectEqual(@as(f32, 1), g.rotation[5 * 4]);
    try std.testing.expectEqual(@as(f32, 0), g.xyz[7]);
    try std.testing.expect(@abs(g.scaling[0] - @sqrt(0.004 * 0.004 + 0.0009 * 0.0009)) < 1e-6);
    raw[288] = std.math.nan(f32);
    activate(.{ .rep = dc.released, .base_offset_scale = 0, .perturbation = &pert }, &raw, &.{ 0.5, 0.5, 0.5 }, g);
    try std.testing.expect(std.math.isNan(g.rotation[0]));
}

test "a PLY row: y-up position, the quaternion turned about x, clamped logit and log" {
    var xyz = [3]f32{ 0.1, 0.2, 0.3 };
    var f = [3]f32{ 1, 2, 3 };
    var s = [3]f32{ 0, 1, 2 };
    var q = [4]f32{ 1, 0, 0, 0 };
    var o = [1]f32{1};
    const g: Gaussians = .{ .xyz = &xyz, .features_dc = &f, .scaling = &s, .rotation = &q, .opacity = &o };
    var r: [properties.len]f32 = undefined;
    row(g, 0, &r);
    try std.testing.expectEqualSlices(f32, &.{ 0.1, -0.3, 0.2, 0, 0, 0, 1, 2, 3 }, r[0..9]);
    try std.testing.expectApproxEqAbs(@as(f32, 13.802318), r[9], 1e-5);
    try std.testing.expectApproxEqAbs(@log(@as(f32, 1e-6)), r[10], 1e-6);
    try std.testing.expectEqual(@as(f32, @floatCast(@sqrt(@as(f64, 0.5)))), r[13]);
    try std.testing.expectEqual(r[13], r[14]);
    var head: [1024]u8 = undefined;
    const h = try header(&head, 262144);
    try std.testing.expect(std.mem.startsWith(u8, h, "ply\nformat binary_little_endian 1.0\nelement vertex 262144\nproperty float x\n"));
    try std.testing.expect(std.mem.endsWith(u8, h, "property float rot_3\nend_header\n"));
}

test "the camera: orthographic below an inverse distance of 1e-3, else perspective" {
    const p = camera(.{ 0, 1, 0, 0.5, 0.25 });
    try std.testing.expectApproxEqAbs(@as(f64, 0), p.yaw, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), p.radius, 1e-12);
    try std.testing.expectApproxEqAbs(2 * std.math.atan(@as(f64, 0.125)) * 180 / std.math.pi, p.fov, 1e-9);
    const o = camera(.{ 0, 0, 0, 0, 1 });
    try std.testing.expectEqual(@as(f64, 1000), o.radius);
    try std.testing.expectApproxEqAbs(std.math.pi / 2.0, o.pitch, 1e-12);
}
