const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("types.zig");
pub const DType = types.DType;
pub const Tensor = types.Tensor;
pub const elementBytes = types.elementBytes;

test {
    _ = @import("pickle.zig");
}

const Member = struct { local: u64, size: u64, crc: u32 };

pub const Archive = struct {
    arena: std.heap.ArenaAllocator,
    io: std.Io,
    file: std.Io.File,
    members: std.StringHashMapUnmanaged(Member),
    tensors: std.StringHashMapUnmanaged(Tensor),
    /// The archive's top directory with its slash ("<name>/"), as torch.save names it.
    prefix: []const u8,

    pub fn open(gpa: Allocator, io: std.Io, path: []const u8) !Archive {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const members = try readDirectory(a, io, file);
        var pickle_name: ?[]const u8 = null;
        var it = members.iterator();
        while (it.next()) |e| {
            if (!std.mem.endsWith(u8, e.key_ptr.*, "/data.pkl")) continue;
            if (pickle_name != null) return error.BadArchive;
            pickle_name = e.key_ptr.*;
        }
        const name = pickle_name orelse return error.BadArchive;
        const prefix = name[0 .. name.len - "data.pkl".len];
        const order_name = try std.mem.concat(a, u8, &.{ prefix, "byteorder" });
        if (members.get(order_name)) |m| {
            if (m.size > 16) return error.BadArchive;
            const order = try a.alloc(u8, @intCast(m.size));
            try readMember(io, file, m, order);
            if (!std.mem.eql(u8, order, "little")) return error.BigEndianArchive;
        }
        const pm = members.get(name).?;
        if (pm.size > 64 << 20) return error.BadArchive;
        const pickle = try a.alloc(u8, @intCast(pm.size));
        try readMember(io, file, pm, pickle);
        const tensors = try @import("pickle.zig").parse(a, pickle);
        return .{ .arena = arena, .io = io, .file = file, .members = members, .tensors = tensors, .prefix = prefix };
    }

    pub fn close(self: *Archive) void {
        self.file.close(self.io);
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn tensor(self: *const Archive, name: []const u8) !Tensor {
        return self.tensors.get(name) orelse {
            std.log.err("codec.pth has no tensor {s}", .{name});
            return error.MissingTensor;
        };
    }

    pub fn readBytes(self: *const Archive, gpa: Allocator, name: []const u8) ![]u8 {
        const t = try self.tensor(name);
        const geometry = try t.layout();
        if (!geometry.contiguous) return error.NonContiguousTensor;
        const width = elementBytes(t.dtype);
        const size = try std.math.mul(u64, t.storage_numel, width);
        const key = try std.fmt.allocPrint(gpa, "{s}data/{s}", .{ self.prefix, t.storage });
        defer gpa.free(key);
        const m = self.members.get(key) orelse return error.MissingStorage;
        if (m.size != size) return error.BadArchive;
        const all = try gpa.alloc(u8, std.math.cast(usize, size) orelse return error.BadArchive);
        errdefer gpa.free(all);
        try readMember(self.io, self.file, m, all);
        const offset = try std.math.mul(usize, std.math.cast(usize, t.offset) orelse return error.BadArchive, width);
        const bytes = try std.math.mul(usize, geometry.count, width);
        if (offset == 0 and bytes == all.len) return all;
        const out = try gpa.dupe(u8, all[offset..][0..bytes]);
        gpa.free(all);
        return out;
    }

    /// name's float32 elements row-major (caller frees), its shape and its storage member's CRC-32 checked.
    pub fn readF32(self: *const Archive, gpa: Allocator, name: []const u8, shape: []const u64) ![]f32 {
        const t = try self.tensor(name);
        if (t.dtype != .f32) {
            std.log.err("codec.pth tensor {s} is {s}, not float32", .{ name, @tagName(t.dtype) });
            return error.UnexpectedDType;
        }
        if (!std.mem.eql(u64, t.shape, shape)) {
            std.log.err("codec.pth tensor {s} has shape {any}, expected {any}", .{ name, t.shape, shape });
            return error.ShapeMismatch;
        }
        const layout = t.layout() catch return error.BadArchive;
        if (!layout.contiguous) return error.NonContiguousTensor;
        const numel = layout.count;
        if (t.offset > t.storage_numel or numel > t.storage_numel - t.offset) return error.BadArchive;
        const key = try std.fmt.allocPrint(gpa, "{s}data/{s}", .{ self.prefix, t.storage });
        defer gpa.free(key);
        const m = self.members.get(key) orelse return error.MissingStorage;
        if (m.size != std.math.mul(u64, t.storage_numel, 4) catch return error.BadArchive) return error.BadArchive;
        const all = try gpa.alloc(f32, @intCast(t.storage_numel));
        errdefer gpa.free(all);
        try readMember(self.io, self.file, m, std.mem.sliceAsBytes(all));
        if (t.offset == 0 and numel == all.len) return all;
        const out = try gpa.dupe(f32, all[@intCast(t.offset)..][0..@intCast(numel)]);
        gpa.free(all);
        return out;
    }
};

fn readU16(b: []const u8, at: usize) u16 {
    return std.mem.readInt(u16, b[at..][0..2], .little);
}

fn readU32(b: []const u8, at: usize) u32 {
    return std.mem.readInt(u32, b[at..][0..4], .little);
}

fn readU64(b: []const u8, at: usize) u64 {
    return std.mem.readInt(u64, b[at..][0..8], .little);
}

/// A stored member's bytes into `out` (exactly its size), CRC-checked.
fn readMember(io: std.Io, file: std.Io.File, m: Member, out: []u8) !void {
    if (out.len != m.size) return error.BadArchive;
    var head: [30]u8 = undefined;
    if (try file.readPositionalAll(io, &head, m.local) != head.len) return error.ShortRead;
    if (readU32(&head, 0) != 0x04034b50) return error.BadArchive;
    const data: u64 = m.local + @as(u64, head.len) + @as(u64, readU16(&head, 26)) + @as(u64, readU16(&head, 28));
    if (try file.readPositionalAll(io, out, data) != out.len) return error.ShortRead;
    if (std.hash.Crc32.hash(out) != m.crc) return error.ChecksumMismatch;
}

/// The central directory: every member stored (method 0), zip64 records honoured.
fn readDirectory(a: Allocator, io: std.Io, file: std.Io.File) !std.StringHashMapUnmanaged(Member) {
    const size = try file.length(io);
    if (size < 22) return error.BadArchive;
    const tail_len: usize = @intCast(@min(size, 22 + 0xFFFF));
    const tail = try a.alloc(u8, tail_len);
    if (try file.readPositionalAll(io, tail, size - @as(u64, tail_len)) != tail_len) return error.ShortRead;
    var at: usize = tail_len - 22;
    while (readU32(tail, at) != 0x06054b50) {
        if (at == 0) return error.BadArchive;
        at -= 1;
    }
    var count: u64 = readU16(tail, at + 10);
    var cd_size: u64 = readU32(tail, at + 12);
    var cd_off: u64 = readU32(tail, at + 16);
    if (count == 0xFFFF or cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF) {
        if (at < 20 or readU32(tail, at - 20) != 0x07064b50) return error.BadArchive;
        var rec: [56]u8 = undefined;
        if (try file.readPositionalAll(io, &rec, readU64(tail, at - 20 + 8)) != rec.len) return error.ShortRead;
        if (readU32(&rec, 0) != 0x06064b50) return error.BadArchive;
        count = readU64(&rec, 32);
        cd_size = readU64(&rec, 40);
        cd_off = readU64(&rec, 48);
    }
    if (cd_size > 64 << 20 or cd_off > size or cd_size > size - cd_off) return error.BadArchive;
    const cd = try a.alloc(u8, @intCast(cd_size));
    if (try file.readPositionalAll(io, cd, cd_off) != cd.len) return error.ShortRead;
    var members: std.StringHashMapUnmanaged(Member) = .empty;
    var p: usize = 0;
    var n: u64 = 0;
    while (n < count) : (n += 1) {
        if (cd.len - p < 46 or readU32(cd, p) != 0x02014b50) return error.BadArchive;
        const method = readU16(cd, p + 10);
        const crc = readU32(cd, p + 16);
        var csize: u64 = readU32(cd, p + 20);
        var usize_: u64 = readU32(cd, p + 24);
        const nlen: usize = readU16(cd, p + 28);
        const xlen: usize = readU16(cd, p + 30);
        const clen: usize = readU16(cd, p + 32);
        var local: u64 = readU32(cd, p + 42);
        if (cd.len - p - 46 < nlen + xlen + clen) return error.BadArchive;
        const name = cd[p + 46 ..][0..nlen];
        var x = p + 46 + nlen;
        const xend = x + xlen;
        while (xend - x >= 4) {
            const id = readU16(cd, x);
            const len: usize = readU16(cd, x + 2);
            if (xend - x - 4 < len) return error.BadArchive;
            if (id == 0x0001) {
                var f = x + 4;
                const fend = f + len;
                if (usize_ == 0xFFFFFFFF) {
                    if (fend - f < 8) return error.BadArchive;
                    usize_ = readU64(cd, f);
                    f += 8;
                }
                if (csize == 0xFFFFFFFF) {
                    if (fend - f < 8) return error.BadArchive;
                    csize = readU64(cd, f);
                    f += 8;
                }
                if (local == 0xFFFFFFFF) {
                    if (fend - f < 8) return error.BadArchive;
                    local = readU64(cd, f);
                }
            }
            x += 4 + len;
        }
        if (method != 0 or csize != usize_) {
            std.log.err("codec.pth member {s} is compressed; torch.save stores members", .{name});
            return error.CompressedMember;
        }
        if (local > size or usize_ > size - local) return error.BadArchive;
        try members.put(a, name, .{ .local = local, .size = usize_, .crc = crc });
        p += 46 + nlen + xlen + clen;
    }
    return members;
}

fn readWrapped() !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "codec.pth", .data = "" });
    var arc: Archive = .{ .arena = .init(gpa), .io = io, .file = try tmp.dir.openFile(io, "codec.pth", .{}), .members = .empty, .tensors = .empty, .prefix = "codec/" };
    defer arc.close();
    const a = arc.arena.allocator();
    try arc.members.put(a, "codec/data/0", .{ .local = 0, .size = 4, .crc = 0 });
    try arc.tensors.put(a, "w", .{ .dtype = .f32, .storage = "0", .storage_numel = (1 << 62) + 1, .offset = 0, .shape = &.{1}, .stride = &.{1} });
    try std.testing.expectError(error.BadArchive, arc.readF32(gpa, "w", &.{1}));
}

// In a child: if readF32's `t.storage_numel * 4` is unchecked, a safe build panics and only this test fails
test "a storage whose byte count wraps past 2^64 is refused, not sized by the wrapped count" {
    try std.testing.expectEqual(@import("sys").isolate.Outcome.passed, try @import("sys").isolate.run(readWrapped, .{}));
}

test "shared checkpoint readers preserve offset views and reject corrupt storage" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var payload: [42]u8 = @splat(0);
    std.mem.writeInt(u32, payload[0..4], 0x04034b50, .little);
    for ([_]f32{ -2, 0.25, 9 }, 0..) |value, i|
        std.mem.writeInt(u32, payload[30 + i * 4 ..][0..4], @bitCast(value), .little);
    try tmp.dir.writeFile(io, .{ .sub_path = "weights.pt", .data = &payload });
    var arc: Archive = .{ .arena = .init(a), .io = io, .file = try tmp.dir.openFile(io, "weights.pt", .{}), .members = .empty, .tensors = .empty, .prefix = "weights/" };
    defer arc.close();
    const keep = arc.arena.allocator();
    try arc.members.put(keep, "weights/data/0", .{ .local = 0, .size = 12, .crc = std.hash.Crc32.hash(payload[30..]) });
    try arc.tensors.put(keep, "view", .{ .dtype = .f32, .storage = "0", .storage_numel = 3, .offset = 1, .shape = &.{2}, .stride = &.{1} });
    const floats = try arc.readF32(a, "view", &.{2});
    defer a.free(floats);
    try std.testing.expectEqualSlices(f32, &.{ 0.25, 9 }, floats);
    const bytes = try arc.readBytes(a, "view");
    defer a.free(bytes);
    try std.testing.expectEqualSlices(u8, payload[34..], bytes);
    arc.members.getPtr("weights/data/0").?.crc ^= 1;
    try std.testing.expectError(error.ChecksumMismatch, arc.readBytes(a, "view"));
    try std.testing.expectError(error.ChecksumMismatch, arc.readF32(a, "view", &.{2}));
}
