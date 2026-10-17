//! NumPy .npy files for the parity tools: little-endian f4 / f2 / i4 / i8 arrays in C order, version 1 headers.
const std = @import("std");

pub const DType = enum { f4, f2, i4, i8 };

pub const Array = struct {
    dtype: DType,
    shape: [4]usize,
    rank: usize,
    data: []align(8) u8,

    pub fn numel(a: Array) usize {
        var n: usize = 1;
        for (a.shape[0..a.rank]) |d| n *= d;
        return n;
    }

    pub fn deinit(a: Array, gpa: std.mem.Allocator) void {
        gpa.free(a.data);
    }

    /// The values as f32 (f4, f2, i4 or i8 converted); caller frees.
    pub fn toF32(a: Array, gpa: std.mem.Allocator) ![]f32 {
        const n = a.numel();
        const out = try gpa.alloc(f32, n);
        for (out, 0..) |*o, i| o.* = switch (a.dtype) {
            .f4 => @bitCast(std.mem.readInt(u32, a.data[i * 4 ..][0..4], .little)),
            .f2 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, a.data[i * 2 ..][0..2], .little)))),
            .i4 => @floatFromInt(std.mem.readInt(i32, a.data[i * 4 ..][0..4], .little)),
            .i8 => @floatFromInt(std.mem.readInt(i64, a.data[i * 8 ..][0..8], .little)),
        };
        return out;
    }

    /// The values as i32 (i4 or i8); caller frees.
    pub fn toI32(a: Array, gpa: std.mem.Allocator) ![]i32 {
        const n = a.numel();
        const out = try gpa.alloc(i32, n);
        for (out, 0..) |*o, i| o.* = switch (a.dtype) {
            .i4 => std.mem.readInt(i32, a.data[i * 4 ..][0..4], .little),
            .i8 => @intCast(std.mem.readInt(i64, a.data[i * 8 ..][0..8], .little)),
            else => return error.NotIntegers,
        };
        return out;
    }
};

pub fn read(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Array {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(16 << 30));
    defer gpa.free(bytes);
    if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.NotNpy;
    const major = bytes[6];
    var hlen: usize = undefined;
    var off: usize = undefined;
    if (major == 1) {
        hlen = std.mem.readInt(u16, bytes[8..10], .little);
        off = 10;
    } else {
        hlen = std.mem.readInt(u32, bytes[8..12], .little);
        off = 12;
    }
    const header = bytes[off .. off + hlen];
    var a: Array = .{ .dtype = .f4, .shape = @splat(1), .rank = 0, .data = undefined };
    if (std.mem.indexOf(u8, header, "'<f4'") != null) a.dtype = .f4 else if (std.mem.indexOf(u8, header, "'<f2'") != null) a.dtype = .f2 else if (std.mem.indexOf(u8, header, "'<i4'") != null) a.dtype = .i4 else if (std.mem.indexOf(u8, header, "'<i8'") != null) a.dtype = .i8 else return error.UnsupportedDtype;
    if (std.mem.indexOf(u8, header, "'fortran_order': True") != null) return error.FortranOrder;
    const sp = std.mem.indexOf(u8, header, "'shape': (") orelse return error.BadHeader;
    const end = std.mem.indexOfScalarPos(u8, header, sp, ')') orelse return error.BadHeader;
    var it = std.mem.tokenizeAny(u8, header[sp + 10 .. end], ", ");
    while (it.next()) |tok| {
        if (a.rank == 4) return error.RankTooHigh;
        a.shape[a.rank] = try std.fmt.parseInt(usize, tok, 10);
        a.rank += 1;
    }
    const esize: usize = switch (a.dtype) {
        .f4, .i4 => 4,
        .f2 => 2,
        .i8 => 8,
    };
    const body = bytes[off + hlen ..];
    if (body.len < a.numel() * esize) return error.ShortNpy;
    a.data = try gpa.alignedAlloc(u8, .@"8", a.numel() * esize);
    @memcpy(a.data, body[0 .. a.numel() * esize]);
    return a;
}

/// Writes f32 (or i32) values with the given shape.
pub fn write(gpa: std.mem.Allocator, io: std.Io, path: []const u8, comptime T: type, values: []const T, shape: []const usize) !void {
    const descr = switch (T) {
        f32 => "<f4",
        i32 => "<i4",
        else => @compileError("npy.write: f32 or i32"),
    };
    var hb: [256]u8 = undefined;
    var sb: [128]u8 = undefined;
    var sl: usize = 0;
    for (shape, 0..) |d, i| {
        const w = try std.fmt.bufPrint(sb[sl..], "{d}{s}", .{ d, if (shape.len == 1 or i + 1 < shape.len) "," else "" });
        sl += w.len;
    }
    const dict = try std.fmt.bufPrint(&hb, "{{'descr': '{s}', 'fortran_order': False, 'shape': ({s}), }}", .{ descr, sb[0..sl] });
    var total = 10 + dict.len + 1;
    const pad = (64 - total % 64) % 64;
    total += pad;
    var head: [512]u8 = undefined;
    @memcpy(head[0..6], "\x93NUMPY");
    head[6] = 1;
    head[7] = 0;
    std.mem.writeInt(u16, head[8..10], @intCast(total - 10), .little);
    @memcpy(head[10..][0..dict.len], dict);
    @memset(head[10 + dict.len ..][0..pad], ' ');
    head[total - 1] = '\n';
    const bytes = std.mem.sliceAsBytes(values);
    const all = try gpa.alloc(u8, total + bytes.len);
    defer gpa.free(all);
    @memcpy(all[0..total], head[0..total]);
    @memcpy(all[total..], bytes);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = all });
}
