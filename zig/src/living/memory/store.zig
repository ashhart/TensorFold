//! Self-contained safetensors reader & atomic writer for Folded Memory Weights (`living-weights.safetensors`).
//! Stores ONLY pure numerical weight tensors + structural metadata (ZERO raw plaintext!).
const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const format_tag = "tf-living-memory-v6";
pub const format_tag_v4 = "tf-living-memory-v4";
pub const format_tag_v3 = "tf-living-memory-v3";
pub const file_name = "living-weights.safetensors";
pub const delta_file_name = "living-weights.delta";
pub const delta_magic = [4]u8{ 'T', 'F', 'D', '6' };

pub const DeltaOp = enum(u8) {
    add_unit = 1,
    forget_parent = 2,
    supersede_parent = 3,
    reinforce_parent = 4,
};

pub const DeltaFrame = struct {
    op: DeltaOp,
    payload: []u8,

    pub fn deinit(self: *DeltaFrame, gpa: Allocator) void {
        gpa.free(self.payload);
        self.* = undefined;
    }
};

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
        const f = try Io.Dir.cwd().createFile(io, tmp_path, .{});
        errdefer Io.Dir.cwd().deleteFile(io, tmp_path) catch {};
        defer f.close(io);

        var len_le: [8]u8 = undefined;
        std.mem.writeInt(u64, &len_le, @intCast(hdr.items.len), .little);
        try f.writeStreamingAll(io, &len_le);
        try f.writeStreamingAll(io, hdr.items);
        for (tensors) |t| {
            if (t.bytes.len > 0) try f.writeStreamingAll(io, t.bytes);
        }
    }
    try Io.Dir.cwd().rename(tmp_path, Io.Dir.cwd(), path, io);
    return @intCast(8 + hdr.items.len + offset);
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

/// Appends one CRC32-protected binary frame to `<model_dir>/living-weights.delta` in O(1 unit) (< 1 ms!).
/// Layout: [4B magic "TFD6"][1B op][3B zero][8B payload_len LE][4B crc32 LE][4B zero][payload...]
pub fn appendDeltaFrame(io: Io, delta_path: []const u8, op: DeltaOp, payload: []const u8) !u64 {
    const f = try Io.Dir.cwd().createFile(io, delta_path, .{ .truncate = false, .read = true });
    defer f.close(io);
    const end_off = try f.length(io);

    var hdr: [24]u8 = @splat(0);
    @memcpy(hdr[0..4], &delta_magic);
    hdr[4] = @backingInt(op);
    std.mem.writeInt(u64, hdr[8..16], @intCast(payload.len), .little);
    std.mem.writeInt(u32, hdr[16..20], std.hash.Crc32.hash(payload), .little);

    try f.writePositionalAll(io, &hdr, end_off);
    if (payload.len > 0) {
        try f.writePositionalAll(io, payload, end_off + hdr.len);
    }
    return end_off + hdr.len + payload.len;
}

/// Reads all valid CRC-verified frames from `living-weights.delta`.
/// Crash-safe: any truncated or corrupted tail frame stops iteration cleanly without failing startup!
pub fn readDeltaFrames(gpa: Allocator, io: Io, delta_path: []const u8) ![]DeltaFrame {
    const f = Io.Dir.cwd().openFile(io, delta_path, .{}) catch return try gpa.alloc(DeltaFrame, 0);
    defer f.close(io);
    const total_len = std.math.cast(usize, try f.length(io)) orelse return try gpa.alloc(DeltaFrame, 0);
    if (total_len < 24) return try gpa.alloc(DeltaFrame, 0);

    const raw = try gpa.alloc(u8, total_len);
    defer gpa.free(raw);
    const got = try f.readPositionalAll(io, raw, 0);
    if (got != total_len) return try gpa.alloc(DeltaFrame, 0);

    var frames: std.ArrayList(DeltaFrame) = .empty;
    errdefer {
        for (frames.items) |*fr| fr.deinit(gpa);
        frames.deinit(gpa);
    }

    var off: usize = 0;
    while (off + 24 <= raw.len) {
        if (!std.mem.eql(u8, raw[off .. off + 4], &delta_magic)) break;
        const op_raw = raw[off + 4];
        const op: DeltaOp = switch (op_raw) {
            1 => .add_unit,
            2 => .forget_parent,
            3 => .supersede_parent,
            4 => .reinforce_parent,
            else => break,
        };
        const plen_u64 = std.mem.readInt(u64, raw[off + 8 ..][0..8], .little);
        const plen = std.math.cast(usize, plen_u64) orelse break;
        if (plen > 256 * 1024 * 1024 or off + 24 + plen > raw.len) break;
        const expected_crc = std.mem.readInt(u32, raw[off + 16 ..][0..4], .little);
        const slice = raw[off + 24 .. off + 24 + plen];
        if (std.hash.Crc32.hash(slice) != expected_crc) break;

        try frames.append(gpa, .{
            .op = op,
            .payload = try gpa.dupe(u8, slice),
        });
        off += 24 + plen;
    }
    return frames.toOwnedSlice(gpa);
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
    return std.mem.indexOf(u8, hdr, format_tag) != null or
        std.mem.indexOf(u8, hdr, format_tag_v4) != null or
        std.mem.indexOf(u8, hdr, format_tag_v3) != null;
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

    return .{
        .gpa = gpa,
        .arena = arena,
        .raw = raw,
        .meta = meta,
        .tensors = tensors,
    };
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

test "store(v6): crash-safe CRC delta journal append, replay, and truncated-tail recovery" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    const dpath = try std.fs.path.join(gpa, &.{ dir, delta_file_name });
    defer gpa.free(dpath);

    const vals = [_]f32{ 3.5, -1.25 };
    const meta = [_]MetaKV{.{ .key = "uid", .value = "7" }};
    const tensors = [_]TensorSpec{
        .{ .name = "u.t", .dtype = .f32, .shape = &.{2}, .bytes = std.mem.sliceAsBytes(&vals) },
    };
    const blob = try serializeToBytes(gpa, &meta, &tensors);
    defer gpa.free(blob);

    _ = try appendDeltaFrame(io, dpath, .add_unit, blob);
    var sup: [8]u8 = undefined;
    std.mem.writeInt(u32, sup[0..4], 2, .little);
    std.mem.writeInt(u32, sup[4..8], 7, .little);
    _ = try appendDeltaFrame(io, dpath, .supersede_parent, &sup);

    // Simulate a sudden power-loss partial frame appended at the tail
    {
        const f = try Io.Dir.cwd().createFile(io, dpath, .{ .truncate = false, .read = true });
        defer f.close(io);
        const end = try f.length(io);
        try f.writePositionalAll(io, "TFD6\x01\x00\x00\x00\x99\x00\x00", end);
    }

    const frames = try readDeltaFrames(gpa, io, dpath);
    defer {
        for (frames) |*fr| fr.deinit(gpa);
        gpa.free(frames);
    }
    try std.testing.expectEqual(@as(usize, 2), frames.len);
    try std.testing.expectEqual(DeltaOp.add_unit, frames[0].op);
    try std.testing.expectEqual(DeltaOp.supersede_parent, frames[1].op);

    // Parse the safetensors blob inside frame 0
    var lf = try parseOwnedBytes(gpa, try gpa.dupe(u8, frames[0].payload));
    defer lf.deinit();
    try std.testing.expectEqualStrings("7", lf.getMeta("uid").?);
    const got = try lf.copyF32(gpa, "u.t", 2);
    defer gpa.free(got);
    try std.testing.expectEqualSlices(f32, &vals, got);
}
