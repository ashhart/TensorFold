//! Safetensors reader and atomic writer for Folded Memory Weights: numeric tensors and metadata, no plaintext.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

/// v8: every value whole as computed (BF16 KV, F32 recurrent changes, prototypes, floats); older banks were not.
pub const format_tag = "tf-living-memory-v8";
const inexact_tags = [_][]const u8{ "tf-living-memory-v7", "tf-living-memory-v6", "tf-living-memory-v4", "tf-living-memory-v3" };

/// Whether `tag` names a format that stored quantized values: such a bank is refused, never read approximately.
pub fn inexactFormat(tag: []const u8) bool {
    for (inexact_tags) |t| if (std.mem.eql(u8, tag, t)) return true;
    return false;
}
pub const file_name = "living-weights.safetensors";
pub const delta_file_name = "living-weights.delta";
const journal = @import("journal.zig");
pub const delta_magic = journal.delta_magic;
pub const DeltaOp = journal.DeltaOp;
pub const DeltaFrame = journal.DeltaFrame;
pub const appendDeltaFrame = journal.appendDeltaFrame;
pub const readDeltaFrames = journal.readDeltaFrames;

pub const DType = enum {
    u8,
    i8,
    bf16,
    u32,
    f32,

    pub fn tag(self: DType) []const u8 {
        return switch (self) {
            .u8 => "U8",
            .i8 => "I8",
            .bf16 => "BF16",
            .u32 => "U32",
            .f32 => "F32",
        };
    }

    pub fn parse(s: []const u8) ?DType {
        if (std.mem.eql(u8, s, "U8")) return .u8;
        if (std.mem.eql(u8, s, "I8")) return .i8;
        if (std.mem.eql(u8, s, "BF16")) return .bf16;
        if (std.mem.eql(u8, s, "U32")) return .u32;
        if (std.mem.eql(u8, s, "F32")) return .f32;
        return null;
    }

    pub fn size(self: DType) usize {
        return switch (self) {
            .u8, .i8 => 1,
            .bf16 => 2,
            .u32, .f32 => 4,
        };
    }
};

pub const TensorSpec = struct {
    name: []const u8,
    dtype: DType,
    shape: []const usize,
    bytes: []const u8,
};

pub const MetaKV = struct {
    key: []const u8,
    value: []const u8,
};

/// Atomically writes a valid `.safetensors` file (8-byte LE header len + JSON header + raw tensor payload).
pub fn writeFileAtomic(
    gpa: Allocator,
    io: Io,
    path: []const u8,
    meta: []const MetaKV,
    tensors: []const TensorSpec,
) !u64 {
    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(gpa);

    try hdr.appendSlice(gpa, "{\"__metadata__\":{");
    for (meta, 0..) |kv, i| {
        if (i > 0) try hdr.append(gpa, ',');
        try appendJsonString(gpa, &hdr, kv.key);
        try hdr.append(gpa, ':');
        try appendJsonString(gpa, &hdr, kv.value);
    }
    try appendCrc(gpa, &hdr, meta.len > 0, tensors);
    try hdr.append(gpa, '}');

    var offset: usize = 0;
    for (tensors) |t| {
        var numel: usize = 1;
        for (t.shape) |d| numel *= d;
        if (t.bytes.len != numel * t.dtype.size()) return error.BadTensorByteLength;
        try hdr.append(gpa, ',');
        try appendJsonString(gpa, &hdr, t.name);
        try hdr.appendSlice(gpa, ":{\"dtype\":\"");
        try hdr.appendSlice(gpa, t.dtype.tag());
        try hdr.appendSlice(gpa, "\",\"shape\":[");
        for (t.shape, 0..) |d, si| {
            if (si > 0) try hdr.append(gpa, ',');
            var num_buf: [32]u8 = undefined;
            const s = try std.fmt.bufPrint(&num_buf, "{d}", .{d});
            try hdr.appendSlice(gpa, s);
        }
        const end_off = offset + t.bytes.len;
        var off_buf: [96]u8 = undefined;
        const off_s = try std.fmt.bufPrint(&off_buf, "],\"data_offsets\":[{d},{d}]}}", .{ offset, end_off });
        try hdr.appendSlice(gpa, off_s);
        offset = end_off;
    }
    try hdr.append(gpa, '}');

    // Pad header with spaces to 8-byte alignment for clean tensor alignment.
    while ((hdr.items.len % 8) != 0) try hdr.append(gpa, ' ');

    const tmp_path = try std.fmt.allocPrint(gpa, "{s}.tmp", .{path});
    defer gpa.free(tmp_path);

    {
        const f = try Io.Dir.cwd().createFile(io, tmp_path, .{ .permissions = private_file });
        errdefer Io.Dir.cwd().deleteFile(io, tmp_path) catch {};
        defer f.close(io);

        var len_le: [8]u8 = undefined;
        std.mem.writeInt(u64, &len_le, @intCast(hdr.items.len), .little);
        try f.writeStreamingAll(io, &len_le);
        try f.writeStreamingAll(io, hdr.items);
        for (tensors) |t| {
            if (t.bytes.len > 0) try f.writeStreamingAll(io, t.bytes);
        }
        try f.sync(io); // the bytes are on disk before the name points at them
    }
    try Io.Dir.cwd().rename(tmp_path, Io.Dir.cwd(), path, io);
    try syncParent(io, path);
    return @intCast(8 + hdr.items.len + offset);
}

/// The rename itself made durable: the directory holding `path` synced (POSIX; Windows has no directory sync).
pub fn syncParent(io: Io, path: []const u8) !void {
    if (comptime @import("builtin").os.tag == .windows) return;
    // iterate opens a real descriptor (not O_PATH), which fsync takes
    var d = try Io.Dir.cwd().openDir(io, std.fs.path.dirname(path) orelse ".", .{ .iterate = true });
    defer d.close(io);
    const handle: Io.File = .{ .handle = d.handle, .flags = .{ .nonblocking = false } };
    try handle.sync(io);
}

fn appendJsonString(gpa: Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    try out.append(gpa, '"');
    for (s) |ch| switch (ch) {
        '"' => try out.appendSlice(gpa, "\\\""),
        '\\' => try out.appendSlice(gpa, "\\\\"),
        '\n' => try out.appendSlice(gpa, "\\n"),
        '\r' => try out.appendSlice(gpa, "\\r"),
        '\t' => try out.appendSlice(gpa, "\\t"),
        else => {
            if (ch < 0x20) {
                var buf: [6]u8 = undefined;
                const esc = try std.fmt.bufPrint(&buf, "\\u{x:0>4}", .{ch});
                try out.appendSlice(gpa, esc);
            } else {
                try out.append(gpa, ch);
            }
        },
    };
    try out.append(gpa, '"');
}

pub const LoadedTensor = struct {
    dtype: DType,
    rank: u8,
    shape: [5]usize,
    bytes: []const u8,
};

pub const LoadedFile = struct {
    gpa: Allocator,
    arena: std.heap.ArenaAllocator,
    raw: []u8,
    meta: std.StringHashMapUnmanaged([]const u8) = .empty,
    tensors: std.StringHashMapUnmanaged(LoadedTensor) = .empty,

    pub fn deinit(self: *LoadedFile) void {
        self.meta.deinit(self.gpa);
        self.tensors.deinit(self.gpa);
        self.arena.deinit();
        self.gpa.free(self.raw);
        self.* = undefined;
    }

    pub fn getMeta(self: *const LoadedFile, key: []const u8) ?[]const u8 {
        return self.meta.get(key);
    }

    pub fn getTensor(self: *const LoadedFile, name: []const u8) ?LoadedTensor {
        return self.tensors.get(name);
    }

    pub fn copyF32(self: *const LoadedFile, gpa: Allocator, name: []const u8, expected_len: usize) ![]f32 {
        const t = self.getTensor(name) orelse return error.MissingTensor;
        if (t.dtype != .f32 or t.bytes.len != expected_len * 4) return error.BadTensorShape;
        const out = try gpa.alloc(f32, expected_len);
        @memcpy(std.mem.sliceAsBytes(out), t.bytes);
        return out;
    }

    pub fn copyU8(self: *const LoadedFile, gpa: Allocator, name: []const u8, expected_len: usize) ![]u8 {
        const t = self.getTensor(name) orelse return error.MissingTensor;
        if (t.dtype != .u8 or t.bytes.len != expected_len) return error.BadTensorShape;
        const out = try gpa.alloc(u8, expected_len);
        @memcpy(out, t.bytes);
        return out;
    }

    pub fn copyI8(self: *const LoadedFile, gpa: Allocator, name: []const u8, expected_len: usize) ![]i8 {
        const t = self.getTensor(name) orelse return error.MissingTensor;
        if (t.dtype != .i8 or t.bytes.len != expected_len) return error.BadTensorShape;
        const out = try gpa.alloc(i8, expected_len);
        @memcpy(std.mem.sliceAsBytes(out), t.bytes);
        return out;
    }

    pub fn copyBf16(self: *const LoadedFile, gpa: Allocator, name: []const u8, expected_len: usize) ![]u16 {
        const t = self.getTensor(name) orelse return error.MissingTensor;
        if (t.dtype != .bf16 or t.bytes.len != expected_len * 2) return error.BadTensorShape;
        const out = try gpa.alloc(u16, expected_len);
        @memcpy(std.mem.sliceAsBytes(out), t.bytes);
        return out;
    }

    pub fn copyU32(self: *const LoadedFile, gpa: Allocator, name: []const u8) ![]u32 {
        const t = self.getTensor(name) orelse return error.MissingTensor;
        if (t.dtype != .u32 or (t.bytes.len % 4) != 0) return error.BadTensorShape;
        const n = t.bytes.len / 4;
        const out = try gpa.alloc(u32, n);
        @memcpy(std.mem.sliceAsBytes(out), t.bytes);
        return out;
    }
};

/// Serializes a complete valid `.safetensors` payload into an owned byte buffer (used by delta frames).
pub fn serializeToBytes(
    gpa: Allocator,
    meta: []const MetaKV,
    tensors: []const TensorSpec,
) ![]u8 {
    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(gpa);

    try hdr.appendSlice(gpa, "{\"__metadata__\":{");
    for (meta, 0..) |kv, i| {
        if (i > 0) try hdr.append(gpa, ',');
        try appendJsonString(gpa, &hdr, kv.key);
        try hdr.append(gpa, ':');
        try appendJsonString(gpa, &hdr, kv.value);
    }
    try appendCrc(gpa, &hdr, meta.len > 0, tensors);
    try hdr.append(gpa, '}');

    var offset: usize = 0;
    for (tensors) |t| {
        var numel: usize = 1;
        for (t.shape) |d| numel *= d;
        if (t.bytes.len != numel * t.dtype.size()) return error.BadTensorByteLength;
        try hdr.append(gpa, ',');
        try appendJsonString(gpa, &hdr, t.name);
        try hdr.appendSlice(gpa, ":{\"dtype\":\"");
        try hdr.appendSlice(gpa, t.dtype.tag());
        try hdr.appendSlice(gpa, "\",\"shape\":[");
        for (t.shape, 0..) |d, si| {
            if (si > 0) try hdr.append(gpa, ',');
            var num_buf: [32]u8 = undefined;
            const s = try std.fmt.bufPrint(&num_buf, "{d}", .{d});
            try hdr.appendSlice(gpa, s);
        }
        const end_off = offset + t.bytes.len;
        var off_buf: [96]u8 = undefined;
        const off_s = try std.fmt.bufPrint(&off_buf, "],\"data_offsets\":[{d},{d}]}}", .{ offset, end_off });
        try hdr.appendSlice(gpa, off_s);
        offset = end_off;
    }
    try hdr.append(gpa, '}');
    while ((hdr.items.len % 8) != 0) try hdr.append(gpa, ' ');

    const out = try gpa.alloc(u8, 8 + hdr.items.len + offset);
    errdefer gpa.free(out);
    std.mem.writeInt(u64, out[0..8], @intCast(hdr.items.len), .little);
    @memcpy(out[8 .. 8 + hdr.items.len], hdr.items);
    var pos: usize = 8 + hdr.items.len;
    for (tensors) |t| {
        if (t.bytes.len > 0) {
            @memcpy(out[pos .. pos + t.bytes.len], t.bytes);
            pos += t.bytes.len;
        }
    }
    return out;
}

/// Checks quickly whether `<dir>/living-weights.safetensors` exists and carries a supported Folded Memory format tag.
pub fn isFoldedMemoryFile(gpa: Allocator, io: Io, path: []const u8) bool {
    return probeFoldedMemoryFile(gpa, io, path) catch false;
}

/// `isFoldedMemoryFile` for a load: false when absent or not a folded-memory file, an error when it cannot be read.
pub fn probeFoldedMemoryFile(gpa: Allocator, io: Io, path: []const u8) !bool {
    const f = Io.Dir.cwd().openFile(io, path, .{}) catch |e| return if (e == error.FileNotFound) false else e;
    defer f.close(io);
    const len = try f.length(io);
    if (len < 16) return false;
    var prefix: [8]u8 = undefined;
    if (try f.readPositionalAll(io, &prefix, 0) != 8) return false;
    const hlen = std.mem.readInt(u64, &prefix, .little);
    if (hlen == 0 or hlen > 16 * 1024 * 1024 or 8 + hlen > len) return false;
    const hdr = try gpa.alloc(u8, @intCast(hlen));
    defer gpa.free(hdr);
    if (try f.readPositionalAll(io, hdr, 8) != hdr.len) return false;
    if (std.mem.indexOf(u8, hdr, format_tag) != null) return true;
    for (inexact_tags) |t| if (std.mem.indexOf(u8, hdr, t) != null) return true; // found, so its load refuses it
    return false;
}

/// Reads and validates a `living-weights.safetensors` file.
pub fn readFile(gpa: Allocator, io: Io, path: []const u8) !LoadedFile {
    const f = try Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const total_len = std.math.cast(usize, try f.length(io)) orelse return error.BadSafetensors;
    if (total_len < 16) return error.BadSafetensors;

    const raw = try gpa.alloc(u8, total_len);
    errdefer gpa.free(raw);
    const got = try f.readPositionalAll(io, raw, 0);
    if (got != total_len) return error.BadSafetensors;
    return parseOwnedBytes(gpa, raw);
}

/// Parses an owned safetensors byte buffer (takes ownership of `raw` on success; caller frees on error).
pub fn parseOwnedBytes(gpa: Allocator, raw: []u8) !LoadedFile {
    const total_len = raw.len;
    if (total_len < 16) return error.BadSafetensors;
    const hlen_u64 = std.mem.readInt(u64, raw[0..8], .little);
    const hlen = std.math.cast(usize, hlen_u64) orelse return error.BadSafetensors;
    if (hlen == 0 or hlen > 100_000_000 or 8 + hlen > total_len) return error.BadSafetensors;

    const data_base = 8 + hlen;
    const data_len = total_len - data_base;

    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, raw[8..data_base], .{});
    if (parsed != .object) return error.BadSafetensors;

    var meta: std.StringHashMapUnmanaged([]const u8) = .empty;
    errdefer meta.deinit(gpa);
    var tensors: std.StringHashMapUnmanaged(LoadedTensor) = .empty;
    errdefer tensors.deinit(gpa);

    var it = parsed.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) {
            if (kv.value_ptr.* != .object) return error.BadSafetensors;
            var mit = kv.value_ptr.object.iterator();
            while (mit.next()) |mkv| {
                if (mkv.value_ptr.* == .string) {
                    try meta.put(gpa, mkv.key_ptr.*, mkv.value_ptr.string);
                }
            }
            continue;
        }
        if (kv.value_ptr.* != .object) return error.BadSafetensors;
        const obj = kv.value_ptr.object;
        const dt_val = obj.get("dtype") orelse return error.BadSafetensors;
        if (dt_val != .string) return error.BadSafetensors;
        const dtype = DType.parse(dt_val.string) orelse return error.UnsupportedDType;

        const sh_val = obj.get("shape") orelse return error.BadSafetensors;
        if (sh_val != .array or sh_val.array.items.len > 5) return error.BadSafetensors;
        var shape: [5]usize = @splat(1);
        var numel: usize = 1;
        for (sh_val.array.items, 0..) |d, i| {
            if (d != .integer or d.integer < 0) return error.BadSafetensors;
            shape[i] = @intCast(d.integer);
            numel = std.math.mul(usize, numel, shape[i]) catch return error.BadSafetensors;
        }

        const off_val = obj.get("data_offsets") orelse return error.BadSafetensors;
        if (off_val != .array or off_val.array.items.len != 2) return error.BadSafetensors;
        const b_v = off_val.array.items[0];
        const e_v = off_val.array.items[1];
        if (b_v != .integer or e_v != .integer or b_v.integer < 0 or e_v.integer < b_v.integer) return error.BadSafetensors;
        const begin: usize = @intCast(b_v.integer);
        const end: usize = @intCast(e_v.integer);
        if (end > data_len or end - begin != numel * dtype.size()) return error.BadSafetensors;

        try tensors.put(gpa, kv.key_ptr.*, .{
            .dtype = dtype,
            .rank = @intCast(sh_val.array.items.len),
            .shape = shape,
            .bytes = raw[data_base + begin .. data_base + end],
        });
    }

    // the data section's CRC, when the writer recorded it: a flipped byte is refused before any value is used
    if (meta.get("data_crc32")) |want| {
        const got = std.hash.Crc32.hash(raw[data_base..]);
        if ((std.fmt.parseInt(u32, want, 10) catch return error.LivingWeightsCorrupt) != got) return error.LivingWeightsCorrupt;
    }
    return .{
        .gpa = gpa,
        .arena = arena,
        .raw = raw,
        .meta = meta,
        .tensors = tensors,
    };
}

/// Learned memory is private: the base, its temporary and the journal are created readable by the owner alone.
pub const private_file: Io.File.Permissions = if (@hasDecl(Io.File.Permissions, "fromMode")) .fromMode(0o600) else .default_file;

/// `"data_crc32"` over every tensor's bytes in order, which is the data section the header describes.
fn appendCrc(gpa: Allocator, hdr: *std.ArrayList(u8), comma: bool, tensors: []const TensorSpec) !void {
    var crc = std.hash.Crc32.init();
    for (tensors) |t| crc.update(t.bytes);
    var buf: [40]u8 = undefined;
    try hdr.appendSlice(gpa, try std.fmt.bufPrint(&buf, "{s}\"data_crc32\":\"{d}\"", .{ if (comma) "," else "", crc.final() }));
}

test "store: atomic safetensors write + read roundtrip with metadata and all dtypes" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    const path = try std.fs.path.join(gpa, &.{ dir, file_name });
    defer gpa.free(path);

    const f32_vals = [_]f32{ 1.25, -2.5, 0.0, 42.0 };
    const i8_vals = [_]i8{ -127, 0, 64, 127 };
    const u32_vals = [_]u32{ 10, 20, 30 };
    const bf16_vals = [_]u16{ 0x3f80, 0x4000 };

    const meta = [_]MetaKV{
        .{ .key = "format", .value = format_tag },
        .{ .key = "model_family", .value = "qwen3_5" },
    };
    const tensors = [_]TensorSpec{
        .{ .name = "t.f32", .dtype = .f32, .shape = &.{ 2, 2 }, .bytes = std.mem.sliceAsBytes(&f32_vals) },
        .{ .name = "t.i8", .dtype = .i8, .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&i8_vals) },
        .{ .name = "t.u32", .dtype = .u32, .shape = &.{3}, .bytes = std.mem.sliceAsBytes(&u32_vals) },
        .{ .name = "t.bf16", .dtype = .bf16, .shape = &.{2}, .bytes = std.mem.sliceAsBytes(&bf16_vals) },
    };
    const written = try writeFileAtomic(gpa, io, path, &meta, &tensors);
    try std.testing.expect(written > 64);
    try std.testing.expect(isFoldedMemoryFile(gpa, io, path));

    var loaded = try readFile(gpa, io, path);
    defer loaded.deinit();
    try std.testing.expectEqualStrings(format_tag, loaded.getMeta("format").?);
    const got_f32 = try loaded.copyF32(gpa, "t.f32", 4);
    defer gpa.free(got_f32);
    try std.testing.expectEqualSlices(f32, &f32_vals, got_f32);
    const got_i8 = try loaded.copyI8(gpa, "t.i8", 4);
    defer gpa.free(got_i8);
    try std.testing.expectEqualSlices(i8, &i8_vals, got_i8);
    const got_u32 = try loaded.copyU32(gpa, "t.u32");
    defer gpa.free(got_u32);
    try std.testing.expectEqualSlices(u32, &u32_vals, got_u32);
    const got_bf16 = try loaded.copyBf16(gpa, "t.bf16", 2);
    defer gpa.free(got_bf16);
    try std.testing.expectEqualSlices(u16, &bf16_vals, got_bf16);
}
