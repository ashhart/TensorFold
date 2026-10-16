//! The MLX 4-bit group-32 token table: host dequant matches glue._embed, and the same bytes go to the GPU.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

const Tensor = core.safetensors.Tensor;

pub const weight_name = "language_model.model.embed_tokens.weight";
pub const scales_name = "language_model.model.embed_tokens.scales";
pub const biases_name = "language_model.model.embed_tokens.biases";

const Triple = struct { weight: Tensor, scales: Tensor, biases: Tensor };

fn tensorAt(obj: std.json.ObjectMap, data: []const u8, name: []const u8, dtype: core.safetensors.DType) !Tensor {
    const o = (obj.get(name) orelse return error.MissingTensor).object;
    const got = core.safetensors.DType.parse(o.get("dtype").?.string) orelse return error.UnsupportedDType;
    if (got != dtype) return error.UnexpectedTensor;
    const shape = o.get("shape").?.array.items;
    if (shape.len == 0 or shape.len > 4) return error.UnexpectedTensor;
    const offs = o.get("data_offsets").?.array.items;
    const begin: usize = @intCast(offs[0].integer);
    const end: usize = @intCast(offs[1].integer);
    if (end < begin or end > data.len) return error.BadSafetensors;
    var t: Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = data[begin..end] };
    for (shape, 0..) |d, i| t.shape[i] = @intCast(d.integer);
    return t;
}

/// The three embedding tensors inside one safetensors header. Other tensors, including rank above 4, are ignored.
pub fn readTriple(gpa: std.mem.Allocator, header: []const u8, data: []const u8) !Triple {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, header, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    return .{
        .weight = try tensorAt(obj, data, weight_name, .u32),
        .scales = try tensorAt(obj, data, scales_name, .bf16),
        .biases = try tensorAt(obj, data, biases_name, .bf16),
    };
}

/// The shard that holds the embedding table, mapped until `close`.
pub const Mapped = struct {
    io: std.Io,
    file: std.Io.File,
    map: std.Io.File.MemoryMap,
    weight: Tensor,
    scales: Tensor,
    biases: Tensor,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Mapped {
        var file = try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const len: usize = @intCast(try file.length(io));
        if (len < 8) return error.BadSafetensors;
        var map = try std.Io.File.MemoryMap.create(io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
        errdefer map.destroy(io);
        const header_len: usize = @intCast(std.mem.readInt(u64, map.memory[0..8], .little));
        if (header_len > len - 8) return error.BadSafetensors;
        const triple = try readTriple(gpa, map.memory[8..][0..header_len], map.memory[8 + header_len ..]);
        return .{ .io = io, .file = file, .map = map, .weight = triple.weight, .scales = triple.scales, .biases = triple.biases };
    }

    /// One named tensor in this shard. Rank 1 to 4.
    pub fn lookup(self: *const Mapped, gpa: std.mem.Allocator, name: []const u8, dtype: core.safetensors.DType) !Tensor {
        const header_len: usize = @intCast(std.mem.readInt(u64, self.map.memory[0..8], .little));
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, self.map.memory[8..][0..header_len], .{});
        defer parsed.deinit();
        return tensorAt(parsed.value.object, self.map.memory[8 + header_len ..], name, dtype);
    }

    pub fn close(self: *Mapped) void {
        self.map.destroy(self.io);
        self.file.close(self.io);
        self.* = undefined;
    }
};

pub const Table = struct {
    w: cuda.DeviceBuffer,
    s: cuda.DeviceBuffer,
    b: cuda.DeviceBuffer,
    rows: usize,
    dims: usize,

    /// Copies the three tensors. `dims` is hidden size (eight values per stored word).
    pub fn upload(d: *const cuda.Driver, weight: Tensor, scales: Tensor, biases: Tensor) !Table {
        if (weight.dtype != .u32 or scales.dtype != .bf16 or biases.dtype != .bf16) return error.UnsupportedQuantization;
        if (weight.rank != 2 or scales.rank != 2 or biases.rank != 2) return error.UnexpectedTensor;
        const rows = weight.dim(0);
        const dims = weight.dim(1) * 8;
        if (dims == 0 or dims % 32 != 0 or scales.dim(0) != rows or biases.dim(0) != rows) return error.UnexpectedTensor;
        if (scales.dim(1) != dims / 32 or biases.dim(1) != dims / 32) return error.UnexpectedTensor;
        var w = try cuda.DeviceBuffer.fromHost(d, weight.bytes);
        errdefer w.free();
        var s = try cuda.DeviceBuffer.fromHost(d, scales.bytes);
        errdefer s.free();
        var b = try cuda.DeviceBuffer.fromHost(d, biases.bytes);
        errdefer b.free();
        return .{ .w = w, .s = s, .b = b, .rows = rows, .dims = dims };
    }

    pub fn deinit(self: *Table) void {
        self.w.free();
        self.s.free();
        self.b.free();
        self.* = undefined;
    }
};

/// fp32 to bf16, round to nearest even, the rounding Triton uses.
fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

fn f32FromBf16(b: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, b[2 * i ..][0..2], .little)) << 16);
}

/// One token row: each group of 32 is `nibble * scale + bias`, stored bf16. Same order as glue._embed.
pub fn dequant(weight: []const u8, scales: []const u8, biases: []const u8, dims: usize, token: usize, out: []u16) !void {
    if (dims == 0 or dims % 32 != 0 or out.len != dims) return error.UnexpectedTensor;
    const groups = dims / 32;
    const words = dims / 8;
    const woff = token * words * 4;
    const soff = token * groups * 2;
    if (woff + words * 4 > weight.len or soff + groups * 2 > scales.len or soff + groups * 2 > biases.len) return error.UnexpectedTensor;
    for (0..groups) |g| {
        const scale = f32FromBf16(scales[soff..], g);
        const bias = f32FromBf16(biases[soff..], g);
        for (0..4) |i| {
            const word = std.mem.readInt(u32, weight[woff + (g * 4 + i) * 4 ..][0..4], .little);
            for (0..8) |j| {
                const q: f32 = @floatFromInt((word >> @intCast(j * 4)) & 0xF);
                out[g * 32 + i * 8 + j] = toBf16(q * scale + bias);
            }
        }
    }
}

test "embedding bytes are read when another tensor has rank above 4" {
    const header =
        \\{"high":{"dtype":"U8","shape":[1,1,1,1,1],"data_offsets":[0,1]},
        \\ "language_model.model.embed_tokens.weight":{"dtype":"U32","shape":[1,4],"data_offsets":[1,17]},
        \\ "language_model.model.embed_tokens.scales":{"dtype":"BF16","shape":[1,1],"data_offsets":[17,19]},
        \\ "language_model.model.embed_tokens.biases":{"dtype":"BF16","shape":[1,1],"data_offsets":[19,21]}}
    ;
    var data: [21]u8 = @splat(0);
    data[1] = 0x21;
    std.mem.writeInt(u16, data[17..19], 0x3f80, .little);
    const triple = try readTriple(std.testing.allocator, header, &data);
    try std.testing.expectEqual(@as(usize, 4), triple.weight.dim(1));
    var out: [32]u16 = undefined;
    try dequant(triple.weight.bytes, triple.scales.bytes, triple.biases.bytes, 32, 0, &out);
    try std.testing.expectEqual(@as(u16, 0x3f80), out[0]);
}

test "one embedding group dequants nibbles with scale and bias" {
    var weight: [16]u8 = @splat(0);
    std.mem.writeInt(u32, weight[0..4], 0x21, .little);
    var scales: [2]u8 = undefined;
    std.mem.writeInt(u16, &scales, 0x3f80, .little);
    const biases: [2]u8 = @splat(0);
    var out: [32]u16 = undefined;
    try dequant(&weight, &scales, &biases, 32, 0, &out);
    try std.testing.expectEqual(@as(u16, 0x3f80), out[0]);
    try std.testing.expectEqual(@as(u16, 0x4000), out[1]);
    try std.testing.expectEqual(@as(u16, 0), out[2]);
}
