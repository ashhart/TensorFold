//! Borrow native projection tensors from the shared checkpoint reader with declared affine shapes and dtypes checked.
const std = @import("std");
const core = @import("core");
const affine = @import("affine.zig");

pub const Tensor = core.checkpoint_host.Tensor;
pub const DType = core.checkpoint_host.DType;

pub const Source = struct {
    ptr: *anyopaque,
    getFn: *const fn (ptr: *anyopaque, name: []const u8) anyerror!Tensor,

    pub fn get(s: Source, name: []const u8) !Tensor {
        return s.getFn(s.ptr, name);
    }

    pub fn from(ck: *core.Checkpoint) Source {
        return .{ .ptr = ck, .getFn = readCheckpoint };
    }

    fn readCheckpoint(ptr: *anyopaque, name: []const u8) anyerror!Tensor {
        const ck: *core.Checkpoint = @ptrCast(@alignCast(ptr));
        for (ck.files.items) |*file| if (file.get(name) != null) return ck.get(name);
        return error.MissingTensor;
    }
};

pub const Linear = union(enum) {
    dense: Tensor,
    quantized: struct { weight: Tensor, scales: Tensor, biases: Tensor, format: affine.Spec },
};

pub fn floating(dtype: DType) bool {
    return dtype == .f16 or dtype == .bf16 or dtype == .f32;
}

pub fn shape(t: Tensor, dims: []const usize) !void {
    if (t.rank > core.safetensors.max_rank or t.rank != dims.len or !std.mem.eql(usize, t.shape[0..t.rank], dims)) return error.ProjectionShape;
    var bytes: usize = t.dtype.size();
    for (dims) |dim| bytes = try std.math.mul(usize, bytes, dim);
    if (t.bytes.len != bytes) return error.ProjectionShape;
}

pub fn validate(weight: Tensor, scales: ?Tensor, biases: ?Tensor, format: ?affine.Spec, n: usize, k: usize) !Linear {
    if (n == 0 or k == 0) return error.ProjectionShape;
    const f = format orelse {
        if (scales != null or biases != null) return error.UndeclaredAffine;
        if (!floating(weight.dtype)) return error.ProjectionDType;
        try shape(weight, &.{ n, k });
        return .{ .dense = weight };
    };
    try f.check();
    const bits = try std.math.mul(usize, k, f.bits);
    if (k % f.group_size != 0 or bits % 32 != 0) return error.ProjectionShape;
    if (weight.dtype != .u32 and weight.dtype != .i32) return error.ProjectionDType;
    const s = scales orelse return error.MissingAffine;
    const b = biases orelse return error.MissingAffine;
    if (!floating(s.dtype) or !floating(b.dtype)) return error.ProjectionDType;
    try shape(weight, &.{ n, bits / 32 });
    try shape(s, &.{ n, k / f.group_size });
    try shape(b, &.{ n, k / f.group_size });
    return .{ .quantized = .{ .weight = weight, .scales = s, .biases = b, .format = f } };
}

fn optional(s: Source, name: []const u8) !?Tensor {
    return s.get(name) catch |err| switch (err) {
        error.MissingTensor => null,
        else => return err,
    };
}

pub fn linear(a: std.mem.Allocator, src: Source, formats: *const affine.Formats, module: []const u8, n: usize, k: usize) !Linear {
    const wn = try std.mem.concat(a, u8, &.{ module, ".weight" });
    defer a.free(wn);
    const sn = try std.mem.concat(a, u8, &.{ module, ".scales" });
    defer a.free(sn);
    const bn = try std.mem.concat(a, u8, &.{ module, ".biases" });
    defer a.free(bn);
    const w = try src.get(wn);
    return validate(w, try optional(src, sn), try optional(src, bn), try formats.resolve(module), n, k);
}

fn fixture(dtype: DType, rows: usize, cols: usize, bytes: []const u8) Tensor {
    return .{ .dtype = dtype, .rank = 2, .shape = .{ rows, cols, 1, 1 }, .bytes = bytes };
}

test "native packed words and mixed scale dtypes remain borrowed unchanged" {
    const words: [24]u8 = @splat(0x59);
    const scales: [4]u8 = @splat(0x18);
    const biases: [8]u8 = @splat(0x42);
    const q = (try validate(fixture(.u32, 2, 3, &words), fixture(.f16, 2, 1, &scales), fixture(.f32, 2, 1, &biases), .{ .bits = 3, .group_size = 32 }, 2, 32)).quantized;
    try std.testing.expectEqual(DType.f16, q.scales.dtype);
    try std.testing.expectEqual(DType.f32, q.biases.dtype);
    try std.testing.expect(q.weight.bytes.ptr == &words and q.scales.bytes.ptr == &scales and q.biases.bytes.ptr == &biases);
    try std.testing.expectEqualSlices(u8, &words, q.weight.bytes);
}

test "dense projections preserve native f16 bf16 and f32" {
    const bytes: [32]u8 = @splat(0x33);
    for ([_]DType{ .f16, .bf16, .f32 }) |dtype| {
        const t = fixture(dtype, 2, 4, bytes[0 .. 8 * dtype.size()]);
        const got = (try validate(t, null, null, null, 2, 4)).dense;
        try std.testing.expectEqual(dtype, got.dtype);
        try std.testing.expect(got.bytes.ptr == t.bytes.ptr);
    }
}

test "packed shapes require matching metadata and declared affine format" {
    const words: [16]u8 = @splat(0);
    const metadata: [4]u8 = @splat(0);
    const w = fixture(.u32, 2, 2, &words);
    const s = fixture(.bf16, 2, 1, &metadata);
    try std.testing.expectError(error.UndeclaredAffine, validate(w, s, s, null, 2, 32));
    try std.testing.expectError(error.MissingAffine, validate(w, s, null, .{ .bits = 2, .group_size = 32 }, 2, 32));
    try std.testing.expectError(error.ProjectionShape, validate(w, s, s, .{ .bits = 4, .group_size = 32 }, 2, 32));
    try std.testing.expectError(error.ProjectionShape, validate(w, s, s, .{ .bits = 2, .group_size = 64 }, 2, 32));
}

test "wrong native tensor dtype and truncated bytes refuse" {
    const bytes: [32]u8 = @splat(0);
    try std.testing.expectError(error.ProjectionDType, validate(fixture(.u32, 2, 4, &bytes), null, null, null, 2, 4));
    try std.testing.expectError(error.ProjectionShape, validate(fixture(.bf16, 2, 4, bytes[0..14]), null, null, null, 2, 4));
    try std.testing.expectError(error.ProjectionDType, validate(fixture(.bf16, 2, 4, bytes[0..16]), null, null, .{ .bits = 4, .group_size = 32 }, 2, 32));
}

test "source lookup applies a dense module override without converting its bytes" {
    const Fake = struct {
        bytes: [16]u8 = @splat(0x71),
        fn get(ptr: *anyopaque, name: []const u8) anyerror!Tensor {
            const s: *@This() = @ptrCast(@alignCast(ptr));
            if (std.mem.eql(u8, name, "language_model.model.layers.0.linear_attn.in_proj_a.weight")) return fixture(.f16, 2, 4, &s.bytes);
            return error.MissingTensor;
        }
    };
    var src = Fake{};
    var formats = try affine.Formats.init(std.testing.allocator, "{\"quantization\":{\"bits\":4,\"layers.0.linear_attn.in_proj_a\":false}}");
    defer formats.deinit();
    const got = try linear(std.testing.allocator, .{ .ptr = &src, .getFn = Fake.get }, &formats, "language_model.model.layers.0.linear_attn.in_proj_a", 2, 4);
    try std.testing.expect(got.dense.bytes.ptr == &src.bytes);
    try std.testing.expectEqual(DType.f16, got.dense.dtype);
}

test "shared checkpoint source preserves borrowed bytes and used tensor accounting" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header = "{\"model.layers.0.linear_attn.in_proj_a.weight\":{\"dtype\":\"F16\",\"shape\":[2,4],\"data_offsets\":[0,16]}}";
    const image = try a.alloc(u8, 8 + header.len + 16);
    defer a.free(image);
    std.mem.writeInt(u64, image[0..8], header.len, .little);
    @memcpy(image[8..][0..header.len], header);
    @memset(image[8 + header.len ..], 0x35);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    var ck = try core.Checkpoint.openModel(a, io, dir);
    defer ck.close();
    const src = Source.from(&ck);
    var formats = try affine.Formats.init(a, "{\"quantization\":{\"bits\":4,\"layers.0.linear_attn.in_proj_a\":false}}");
    defer formats.deinit();
    const q = (try linear(a, src, &formats, "model.layers.0.linear_attn.in_proj_a", 2, 4)).dense;
    const direct = try ck.get("model.layers.0.linear_attn.in_proj_a.weight");
    try std.testing.expect(q.bytes.ptr == direct.bytes.ptr);
    try std.testing.expectEqual(DType.f16, q.dtype);
    for (q.bytes) |b| try std.testing.expectEqual(@as(u8, 0x35), b);
    try std.testing.expectEqual(@as(usize, 1), ck.used.count());
    try std.testing.expectError(error.MissingTensor, src.get("missing.weight"));
}

test "invalid affine spec or integer scale cannot reach quantized arithmetic" {
    const bytes: [16]u8 = @splat(0);
    const w = fixture(.u32, 2, 2, &bytes);
    const integer_scale = fixture(.u32, 2, 1, bytes[0..8]);
    try std.testing.expectError(error.BadAffine, validate(w, null, null, .{ .bits = 4, .group_size = 0 }, 2, 32));
    try std.testing.expectError(error.ProjectionDType, validate(w, integer_scale, integer_scale, .{ .bits = 2, .group_size = 32 }, 2, 32));
    var invalid_rank = fixture(.bf16, 2, 4, &bytes);
    invalid_rank.rank = 255;
    try std.testing.expectError(error.ProjectionShape, validate(invalid_rank, null, null, null, 2, 4));
}
