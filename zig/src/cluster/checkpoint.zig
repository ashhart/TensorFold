//! A safetensors checkpoint as the planner sees it: files, and every tensor's dtype, shape, byte range and role.
const std = @import("std");
const roles = @import("roles.zig");

const Allocator = std.mem.Allocator;

pub const DType = enum(u8) { bf16, f16, f32, f64, i64, i32, i16, i8, u8, u16, u32, u64, f8_e4m3, f8_e5m2, f8_e8m0, boolean };

pub fn dtypeOf(text: []const u8) ?DType {
    const table = [_]struct { []const u8, DType }{
        .{ "BF16", .bf16 },  .{ "F16", .f16 },         .{ "F32", .f32 },         .{ "F64", .f64 },         .{ "I64", .i64 },  .{ "I32", .i32 },
        .{ "I16", .i16 },    .{ "I8", .i8 },           .{ "U8", .u8 },           .{ "U16", .u16 },         .{ "U32", .u32 },  .{ "U64", .u64 },
        .{ "F8_E4M3", .f8_e4m3 }, .{ "F8_E5M2", .f8_e5m2 }, .{ "F8_E8M0", .f8_e8m0 }, .{ "BOOL", .boolean },
    };
    for (table) |row| if (std.mem.eql(u8, text, row[0])) return row[1];
    return null;
}

pub fn size(d: DType) u64 {
    return switch (d) {
        .bf16, .f16, .i16, .u16 => 2,
        .f32, .i32, .u32 => 4,
        .f64, .i64, .u64 => 8,
        .i8, .u8, .f8_e4m3, .f8_e5m2, .f8_e8m0, .boolean => 1,
    };
}

pub const max_rank = 4;

pub const Tensor = struct {
    name: []const u8,
    file: u32,
    /// Absolute offset of the tensor's first byte in its file.
    start: u64,
    bytes: u64,
    dtype: DType,
    rank: u8 = 0,
    shape: [max_rank]u64 = @splat(0),
    class: roles.Class = .{ .role = .other },

    pub fn rows(t: Tensor) u64 {
        return if (t.rank == 0) 1 else t.shape[0];
    }
};

pub const File = struct {
    name: []const u8,
    size: u64,
    /// The Hub's SHA-256 of the whole file, when the manifest gives it.
    sha256: ?[32]u8 = null,
};

pub const Checkpoint = struct {
    files: []File,
    tensors: []Tensor,

    pub fn bytes(c: Checkpoint) u64 {
        var sum: u64 = 0;
        for (c.tensors) |t| sum += t.bytes;
        return sum;
    }

    pub fn fileIndex(c: Checkpoint, name: []const u8) ?u32 {
        for (c.files, 0..) |f, i| if (std.mem.eql(u8, f.name, name)) return @intCast(i);
        return null;
    }
};

pub const Error = error{ BadHeader, BadIndex, BadDType, Inconsistent };

/// One safetensors header (the u64 length, then JSON); `bytes` holds at least 8 + that length.
pub fn parseHeader(a: Allocator, file: u32, bytes: []const u8) ![]Tensor {
    if (bytes.len < 8) return error.BadHeader;
    const n = std.mem.readInt(u64, bytes[0..8], .little);
    if (n > bytes.len - 8) return error.BadHeader;
    const root = std.json.parseFromSliceLeaky(std.json.Value, a, bytes[8..][0..n], .{}) catch return error.BadHeader;
    if (root != .object) return error.BadHeader;
    var out: std.ArrayList(Tensor) = .empty;
    var it = root.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
        const v = kv.value_ptr.*;
        if (v != .object) return error.BadHeader;
        const dt = dtypeOf(str(v.object.get("dtype")) orelse return error.BadHeader) orelse return error.BadDType;
        const offs = (v.object.get("data_offsets") orelse return error.BadHeader);
        if (offs != .array or offs.array.items.len != 2) return error.BadHeader;
        const begin = uint(offs.array.items[0]) orelse return error.BadHeader;
        const end = uint(offs.array.items[1]) orelse return error.BadHeader;
        if (end < begin or end > std.math.maxInt(u64) - 8 - n) return error.BadHeader;
        var t: Tensor = .{ .name = try a.dupe(u8, kv.key_ptr.*), .file = file, .start = 8 + n + begin, .bytes = end - begin, .dtype = dt };
        const shape = v.object.get("shape") orelse return error.BadHeader;
        if (shape != .array or shape.array.items.len > max_rank) return error.BadHeader;
        // saturating: a zero dimension anywhere still gives 0, and the end check keeps bytes below the overflow cap
        var want: u64 = size(dt);
        for (shape.array.items, 0..) |d, i| {
            t.shape[i] = uint(d) orelse return error.BadHeader;
            want *|= t.shape[i];
        }
        t.rank = @intCast(shape.array.items.len);
        if (want != t.bytes) return error.Inconsistent;
        t.class = roles.classify(t.name);
        try out.append(a, t);
    }
    return out.toOwnedSlice(a);
}

/// The index's weight_map: tensor name to file name.
pub fn parseIndex(a: Allocator, text: []const u8) !std.StringArrayHashMapUnmanaged([]const u8) {
    const root = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return error.BadIndex;
    if (root != .object) return error.BadIndex;
    const map = root.object.get("weight_map") orelse return error.BadIndex;
    if (map != .object) return error.BadIndex;
    var out: std.StringArrayHashMapUnmanaged([]const u8) = .empty;
    var it = map.object.iterator();
    while (it.next()) |kv| try out.put(a, kv.key_ptr.*, str(kv.value_ptr.*) orelse return error.BadIndex);
    return out;
}

/// Read a checkpoint folder's index (or its one model.safetensors) and every file's header, nothing else.
pub fn load(io: std.Io, a: Allocator, dir: []const u8) !Checkpoint {
    const cwd = std.Io.Dir.cwd();
    var names: std.ArrayList([]const u8) = .empty;
    const index_path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    if (cwd.readFileAlloc(io, index_path, a, .limited(1 << 30))) |text| {
        const map = try parseIndex(a, text);
        for (map.values()) |f| {
            for (names.items) |seen| {
                if (std.mem.eql(u8, seen, f)) break;
            } else try names.append(a, f);
        }
    } else |err| switch (err) {
        error.FileNotFound => try names.append(a, "model.safetensors"),
        else => return err,
    }
    std.mem.sort([]const u8, names.items, {}, lessStr);
    var files: std.ArrayList(File) = .empty;
    var tensors: std.ArrayList(Tensor) = .empty;
    for (names.items, 0..) |f, i| {
        const path = try std.fs.path.join(a, &.{ dir, f });
        const file = try cwd.openFile(io, path, .{});
        defer file.close(io);
        var head: [8]u8 = undefined;
        if (try file.readPositionalAll(io, &head, 0) != 8) return error.BadHeader;
        const n = std.mem.readInt(u64, &head, .little);
        if (n > 1 << 30) return error.BadHeader;
        const buf = try a.alloc(u8, 8 + n);
        if (try file.readPositionalAll(io, buf, 0) != buf.len) return error.BadHeader;
        try tensors.appendSlice(a, try parseHeader(a, @intCast(i), buf));
        try files.append(a, .{ .name = f, .size = try file.length(io) });
    }
    return .{ .files = files.items, .tensors = tensors.items };
}

fn lessStr(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.order(u8, x, y) == .lt;
}

fn str(v: ?std.json.Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

fn uint(v: std.json.Value) ?u64 {
    return switch (v) {
        .integer => |i| if (i < 0) null else @intCast(i),
        .number_string => |s| std.fmt.parseInt(u64, s, 10) catch null,
        else => null,
    };
}

/// Build a safetensors file image in memory (tests and the loader's fixtures); tensors are laid out in the order given.
pub fn image(a: Allocator, specs: []const struct { name: []const u8, dtype: []const u8, shape: []const u64, fill: u8 }) ![]u8 {
    var json: std.Io.Writer.Allocating = .init(a);
    try json.writer.writeAll("{\"__metadata__\":{\"format\":\"pt\"}");
    var at: u64 = 0;
    for (specs) |s| {
        var n: u64 = size(dtypeOf(s.dtype).?);
        for (s.shape) |d| n *= d;
        try json.writer.print(",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ s.name, s.dtype });
        for (s.shape, 0..) |d, i| try json.writer.print("{s}{d}", .{ if (i == 0) "" else ",", d });
        try json.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ at, at + n });
        at += n;
    }
    try json.writer.writeAll("}");
    const head = json.written();
    const out = try a.alloc(u8, 8 + head.len + at);
    std.mem.writeInt(u64, out[0..8], head.len, .little);
    @memcpy(out[8..][0..head.len], head);
    var p: usize = 8 + head.len;
    for (specs) |s| {
        var n: u64 = size(dtypeOf(s.dtype).?);
        for (s.shape) |d| n *= d;
        @memset(out[p..][0..n], s.fill);
        p += n;
    }
    return out;
}

test "a safetensors header gives absolute byte ranges, dtypes, shapes and roles" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const img = try image(a, &.{
        .{ .name = "model.layers.3.mlp.experts.7.down_proj.weight_packed", .dtype = "U8", .shape = &.{ 8, 4 }, .fill = 1 },
        .{ .name = "model.layers.3.self_attn.o_proj.weight", .dtype = "BF16", .shape = &.{ 4, 4 }, .fill = 2 },
    });
    const ts = try parseHeader(a, 5, img);
    try std.testing.expectEqual(@as(usize, 2), ts.len);
    for (ts) |t| {
        try std.testing.expectEqual(@as(u32, 5), t.file);
        const fill: u8 = if (t.dtype == .u8) 1 else 2;
        for (img[t.start..][0..t.bytes]) |b| try std.testing.expectEqual(fill, b);
    }
    try std.testing.expectEqual(roles.Role.expert, ts[0].class.role);
    try std.testing.expectEqual(@as(?u32, 7), ts[0].class.expert);
    try std.testing.expectEqual(roles.Role.attn_row, ts[1].class.role);
    var bad = try a.dupe(u8, img);
    std.mem.writeInt(u64, bad[0..8], 1 << 40, .little);
    try std.testing.expectError(error.BadHeader, parseHeader(a, 0, bad));
}

test "byte counts and end offsets past 64 bits are refused; a zero dimension still empties a tensor" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { ?anyerror, []const u8 }{
        // 2^62 rows of 4 bytes is 2^64 bytes, which wraps to 0: the size of the empty range [0, 0]
        .{ error.Inconsistent, "{\"t\":{\"dtype\":\"U8\",\"shape\":[4611686018427387904,4],\"data_offsets\":[0,0]}}" },
        // 2^63 elements fit in 64 bits, their 2^64 bytes of BF16 do not
        .{ error.Inconsistent, "{\"t\":{\"dtype\":\"BF16\",\"shape\":[9223372036854775808],\"data_offsets\":[0,0]}}" },
        // the header's own bytes push a start offset of 2^64 - 1 past 64 bits
        .{ error.BadHeader, "{\"t\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[18446744073709551615,18446744073709551615]}}" },
        // a consistent tensor whose start fits but whose end, start + bytes, does not
        .{ error.BadHeader, "{\"t\":{\"dtype\":\"U8\",\"shape\":[18446744073709551615],\"data_offsets\":[0,18446744073709551615]}}" },
        // 2^63 elements, so 2^64 bytes in BF16, before the trailing zero makes the tensor empty
        .{ null, "{\"t\":{\"dtype\":\"BF16\",\"shape\":[16777216,16777216,32768,0],\"data_offsets\":[0,0]}}" },
        // the same empty tensor in 117 header bytes at the last end that fits: 8 + 117 + end is 2^64 - 1
        .{ null, "{\"t\":{\"dtype\":\"BF16\",\"shape\":[16777216,16777216,32768,0],\"data_offsets\":[18446744073709551490,18446744073709551490]}}" },
        // one byte further in 91 header bytes: the start offset 8 + 91 + begin is 2^64
        .{ error.BadHeader, "{\"t\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[18446744073709551517,18446744073709551517]}}" },
    };
    for (cases) |c| {
        const img = try a.alloc(u8, 8 + c[1].len);
        std.mem.writeInt(u64, img[0..8], c[1].len, .little);
        @memcpy(img[8..], c[1]);
        if (c[0]) |err| try std.testing.expectError(err, parseHeader(a, 0, img)) else {
            const t = (try parseHeader(a, 0, img))[0];
            try std.testing.expectEqual(@as(u64, 0), t.bytes);
            try std.testing.expectEqualSlices(u64, &.{ 16777216, 16777216, 32768, 0 }, t.shape[0..t.rank]);
        }
    }
}

test "an index maps tensors to files" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const map = try parseIndex(arena.allocator(), "{\"metadata\":{\"total_size\":3},\"weight_map\":{\"a.weight\":\"model-00001-of-00002.safetensors\",\"b\":\"model-00002-of-00002.safetensors\"}}");
    try std.testing.expectEqualStrings("model-00002-of-00002.safetensors", map.get("b").?);
    try std.testing.expectError(error.BadIndex, parseIndex(arena.allocator(), "{\"weight_map\":3}"));
}
