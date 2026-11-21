//! Owned affine, native lane or block-FP8 projections; packed device views borrow these exact buffers.
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
    /// rows [at, at + n) of the output held as BF16 beside a block-FP8 `native` whose rows there are zero
    tail: ?Tail = null,
    tail_w: ?cuda.DeviceBuffer = null,

    pub const Tail = struct { native: cuda.qmmf.Weight, at: usize, n: usize };

    pub fn free(self: *Projection) void {
        self.w.free();
        self.s.free();
        self.b.free();
        if (self.tail_w) |*t| t.free();
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

/// The FP8G lane's two planes: the packed codes [padded n, k] and their per-64-input fp32 scale tiles.
pub const Planes = struct {
    codes: []u8,
    scales: []f32,

    pub fn deinit(self: *Planes, gpa: std.mem.Allocator) void {
        gpa.free(self.codes);
        gpa.free(self.scales);
        self.* = undefined;
    }
};

/// The FP8G lane's planes: `nh` block-FP8 rows, then `nt` rows of zero codes and a block scale of one.
pub fn fp8gPlanes(gpa: std.mem.Allocator, codes: []const u8, scale_inv: []const f32, nh: usize, k: usize, nt: usize) !Planes {
    const fp8 = cuda.fp8;
    const n = nh + nt;
    const kb = k / 128;
    if (nh == 0 or k % 128 != 0 or codes.len != try std.math.mul(usize, nh, k) or scale_inv.len != ((nh + 127) / 128) * kb)
        return error.UnexpectedTensor;
    // one scale a 128-row block: a tail row may not share its block with an FP8 row, its own rows may be partial
    if (nt != 0 and nh % 128 != 0) return error.UnexpectedTensor;
    const all = try gpa.alloc(u8, try std.math.mul(usize, n, k));
    defer gpa.free(all);
    @memcpy(all[0 .. nh * k], codes);
    @memset(all[nh * k ..], 0);
    const inv = try gpa.alloc(f32, ((n + 127) / 128) * kb);
    defer gpa.free(inv);
    @memcpy(inv[0..scale_inv.len], scale_inv);
    @memset(inv[scale_inv.len..], 1.0);
    const plane = try gpa.alloc(u8, fp8.padded(n) * k);
    errdefer gpa.free(plane);
    fp8.packCodes(plane, all, n, k);
    const scales = try gpa.alloc(f32, fp8.padded(n) * (k / fp8.group));
    errdefer gpa.free(scales);
    fp8.packScales(scales, inv, n, k);
    return .{ .codes = plane, .scales = scales };
}

/// Block-FP8 rows and an optional BF16 tail: the FP8G weight over both, and the tail's own BF16 weight.
pub fn fp8g(gpa: std.mem.Allocator, driver: *const cuda.Driver, codes: []const u8, scale_inv: []const f32, nh: usize, k: usize, tail_bf16: ?[]const u8, nt: usize) !Projection {
    if (nt != 0 and tail_bf16 == null) return error.UnexpectedTensor;
    var planes = try fp8gPlanes(gpa, codes, scale_inv, nh, k, nt);
    defer planes.deinit(gpa);
    const n = nh + nt;
    var p = Projection.empty(driver);
    p.w = try cuda.DeviceBuffer.fromHost(driver, planes.codes);
    errdefer p.w.free();
    p.s = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(planes.scales));
    errdefer p.s.free();
    p.n = n;
    p.k = k;
    p.native = cuda.fp8.weight(p.w.ptr, p.s.ptr, n, k);
    if (tail_bf16) |t| {
        var edge = try dense(gpa, driver, .{ .dtype = .bf16, .rank = 2, .shape = .{ nt, k, 1, 1, 1 }, .bytes = t });
        p.tail_w = edge.w;
        edge.s.free();
        edge.b.free();
        p.tail = .{ .native = edge.native.?, .at = nh, .n = nt };
    }
    return p;
}
