//! NVFP4 as ModelOpt stores it: e2m1 codes two to a byte, an e4m3 scale per 16 inputs, an fp32 scale per tensor.

const std = @import("std");
const ckpt = @import("checkpoint.zig");
const quant = @import("quant/quant.zig");
const DType = ckpt.DType;

/// Inputs sharing one e4m3 scale.
pub const group = quant.nvfp4.group;

/// A stored tensor's dtype and shape, as a header lists it.
pub const Meta = struct {
    dtype: DType,
    shape: []const usize,
};

pub const Dims = struct { n: usize, k: usize };

/// The linear's [n, k] from codes U8 [n, k/2], scales F8_E4M3 [n, k/16] and fp32 scalars; else error.BadNvfp4.
pub fn dims(weight: Meta, scale: Meta, scale2: Meta, input_scale: ?Meta) !Dims {
    if (weight.dtype != .u8 or weight.shape.len != 2) return error.BadNvfp4;
    if (scale.dtype != .f8_e4m3 or scale.shape.len != 2) return error.BadNvfp4;
    const n = weight.shape[0];
    const k = 2 * weight.shape[1];
    if (k % group != 0 or scale.shape[0] != n or scale.shape[1] * group != k) return error.BadNvfp4;
    if (!scalar(scale2)) return error.BadNvfp4;
    if (input_scale) |s| if (!scalar(s)) return error.BadNvfp4;
    return .{ .n = n, .k = k };
}

fn scalar(m: Meta) bool {
    return m.dtype == .f32 and (m.shape.len == 0 or (m.shape.len == 1 and m.shape[0] == 1));
}

/// Whether `module` (a linear's name without ".weight") is stored as NVFP4: its tensors say, not config.json.
pub fn stored(ck: *const ckpt.Checkpoint, module: []const u8) bool {
    var buf: [256]u8 = undefined;
    const name = std.fmt.bufPrint(&buf, "{s}.weight_scale", .{module}) catch return false;
    for (ck.files.items) |*f| if (f.names.contains(name)) return true;
    return false;
}

/// One NVFP4 linear's stored bytes, as the checkpoint maps them.
pub const Linear = struct {
    n: usize,
    k: usize,
    codes: []const u8, // [n, k/2]: input 2j in byte j's low nibble
    scales: []const u8, // [n, k/16] e4m3
    global: f32, // weight_scale_2

    /// `module`'s four tensors, marked as used; input_scale is the W4A4 math's static scale, unread by W4A16.
    pub fn read(ck: *ckpt.Checkpoint, module: []const u8) !Linear {
        var buf: [256]u8 = undefined;
        const w = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{module}));
        const s = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{module}));
        const g = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight_scale_2", .{module}));
        const a = try ck.get(try std.fmt.bufPrint(&buf, "{s}.input_scale", .{module}));
        const d = dims(meta(&w), meta(&s), meta(&g), meta(&a)) catch |e| {
            std.log.err("{s}: not ModelOpt NVFP4 ({t} {any}, {t} {any})", .{ module, w.dtype, w.shape[0..w.rank], s.dtype, s.shape[0..s.rank] });
            return e;
        };
        return .{ .n = d.n, .k = d.k, .codes = w.bytes, .scales = s.bytes, .global = std.mem.bytesToValue(f32, g.bytes[0..4]) };
    }
};

fn meta(t: *const ckpt.Tensor) Meta {
    return .{ .dtype = t.dtype, .shape = t.shape[0..t.rank] };
}

test "dims refuse what is not ModelOpt NVFP4" {
    const w: Meta = .{ .dtype = .u8, .shape = &.{ 4, 16 } };
    const s: Meta = .{ .dtype = .f8_e4m3, .shape = &.{ 4, 2 } };
    const g: Meta = .{ .dtype = .f32, .shape = &.{} };
    const d = try dims(w, s, g, g);
    try std.testing.expectEqual(Dims{ .n = 4, .k = 32 }, d);
    try std.testing.expectError(error.BadNvfp4, dims(.{ .dtype = .bf16, .shape = &.{ 4, 16 } }, s, g, g));
    try std.testing.expectError(error.BadNvfp4, dims(w, .{ .dtype = .f8_e4m3, .shape = &.{ 4, 3 } }, g, g));
    try std.testing.expectError(error.BadNvfp4, dims(w, .{ .dtype = .u8, .shape = &.{ 4, 2 } }, g, g));
    try std.testing.expectError(error.BadNvfp4, dims(w, s, .{ .dtype = .f32, .shape = &.{2} }, null));
}

test "a linear read from a checkpoint folder decodes as core/quant's reference" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // n = 2, k = 32: codes 2 x 16 bytes, scales 2 x 2, two fp32 scalars
    const header =
        \\{"l.weight":{"dtype":"U8","shape":[2,16],"data_offsets":[0,32]},
        \\ "l.weight_scale":{"dtype":"F8_E4M3","shape":[2,2],"data_offsets":[32,36]},
        \\ "l.weight_scale_2":{"dtype":"F32","shape":[],"data_offsets":[36,40]},
        \\ "l.input_scale":{"dtype":"F32","shape":[],"data_offsets":[40,44]}}
    ;
    var data: [8 + header.len + 44]u8 = undefined;
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    const body = data[8 + header.len ..];
    for (body[0..32], 0..) |*b, i| b.* = @intCast((i * 7 + 3) & 0xff); // every nibble value appears
    @memcpy(body[32..36], &[_]u8{ 0x38, 0x40, 0x30, 0xb8 }); // 1.0, 2.0, 0.5, -1.0
    @memcpy(body[36..40], std.mem.asBytes(&@as(f32, 0.25)));
    @memcpy(body[40..44], std.mem.asBytes(&@as(f32, 3.0)));
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &data });
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    var ck = try ckpt.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    try std.testing.expect(stored(&ck, "l"));
    try std.testing.expect(!stored(&ck, "m"));
    const l = try Linear.read(&ck, "l");
    try std.testing.expectEqual(@as(usize, 0), ck.unused());
    try std.testing.expectEqual(@as(f32, 0.25), l.global);
    var out: [64]f64 = undefined;
    try quant.nvfp4.reference(.{
        .weight = .{ .dtype = .u8, .rank = 2, .shape = .{ l.n, l.k / 2, 1, 1, 1 }, .bytes = l.codes },
        .scale = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ l.n, l.k / group, 1, 1, 1 }, .bytes = l.scales },
        .global = &[_]f32{l.global},
    }, &out);
    // byte 0 is 3: input 0 is code 3 (1.5), input 1 code 0; inputs 16.. take the second scale (2.0)
    try std.testing.expectEqual(@as(f64, 1.5 * 1.0 * 0.25), out[0]);
    try std.testing.expectEqual(@as(f64, 0.0), out[1]);
    try std.testing.expectEqual(quant.nvfp4.e2m1(body[8] & 0xf) * 2.0 * 0.25, out[16]);
    try std.testing.expectEqual(quant.nvfp4.e2m1(body[16 + 15] >> 4) * -1.0 * 0.25, out[32 + 31]); // row 1, scale -1.0
}
