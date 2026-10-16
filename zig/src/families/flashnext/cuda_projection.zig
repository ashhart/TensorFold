//! Owned affine or native lane projections; packed device views borrow these exact buffers.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

pub const Projection = struct {
    w: cuda.DeviceBuffer,
    s: cuda.DeviceBuffer,
    b: cuda.DeviceBuffer,
    n: usize,
    k: usize,
    native: ?cuda.qmmf.Weight = null,

    pub fn free(self: *Projection) void {
        self.w.free();
        self.s.free();
        self.b.free();
    }

    pub fn empty(d: *const cuda.Driver) Projection {
        const z: cuda.DeviceBuffer = .{ .d = d, .ptr = 0, .len = 0 };
        return .{ .w = z, .s = z, .b = z, .n = 0, .k = 0 };
    }
};

/// Repack the checkpoint's BF16 bits without rounding or widening the resident weight.
pub fn dense(gpa: std.mem.Allocator, driver: *const cuda.Driver, t: core.safetensors.Tensor) !Projection {
    if (t.rank != 2 or t.dtype != .bf16 or t.dim(0) == 0 or t.dim(1) == 0 or t.dim(1) % 64 != 0)
        return error.UnexpectedTensor;
    const n = t.dim(0);
    const k = t.dim(1);
    const count = try std.math.mul(usize, n, k);
    if (t.bytes.len != try std.math.mul(usize, count, 2)) return error.UnexpectedTensor;
    const input = try gpa.alloc(u16, count);
    defer gpa.free(input);
    for (input, 0..) |*v, i| v.* = std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little);
    const tiled = try gpa.alloc(u16, cuda.nvfp4.padded(n) * k);
    defer gpa.free(tiled);
    cuda.qmmf.packBf16(tiled, input, n, k);
    var p = Projection.empty(driver);
    p.w = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(tiled));
    p.n = n;
    p.k = k;
    p.native = cuda.modelopt.bf16Weight(p.w.ptr, n, k);
    return p;
}

/// E4M3 codes and UE8M0 group-32 scales are permuted only; calibration metadata is not applied as another scale.
pub fn mxfp8(gpa: std.mem.Allocator, driver: *const cuda.Driver, w: core.safetensors.Tensor, s: core.safetensors.Tensor) !Projection {
    if (w.rank != 2 or w.dtype != .f8_e4m3 or w.dim(0) == 0 or w.dim(1) == 0 or w.dim(1) % 64 != 0)
        return error.UnexpectedTensor;
    const n = w.dim(0);
    const k = w.dim(1);
    const count = try std.math.mul(usize, n, k);
    if (w.bytes.len != count or !s.is(.u8, &.{ n, k / 32 }) or s.bytes.len != count / 32)
        return error.UnexpectedTensor;
    const codes = try gpa.alloc(u8, cuda.mxfp8.padded(n) * k);
    defer gpa.free(codes);
    const scales = try gpa.alloc(u8, codes.len / 32);
    defer gpa.free(scales);
    cuda.mxfp8.packCodes(codes, w.bytes, n, k);
    cuda.mxfp8.packScales(scales, s.bytes, n, k);
    var p = Projection.empty(driver);
    p.w = try cuda.DeviceBuffer.fromHost(driver, codes);
    errdefer p.w.free();
    p.s = try cuda.DeviceBuffer.fromHost(driver, scales);
    p.n = n;
    p.k = k;
    p.native = cuda.mxfp8.weight(p.w.ptr, p.s.ptr, n, k);
    return p;
}
