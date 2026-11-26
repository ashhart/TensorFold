//! Safetensors admission, exact tensor IO and residual serialization for the native encoder.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Tensor = struct { name: []const u8, file: []const u8, dtype: []const u8, shape: []u64, off: u64, len: u64 };

fn add(a: u64, b: u64) !u64 {
    return std.math.add(u64, a, b) catch error.InvalidSafetensors;
}

fn mul(a: u64, b: u64) !u64 {
    return std.math.mul(u64, a, b) catch error.InvalidSafetensors;
}

fn dtypeBytes(dtype: []const u8) !u64 {
    const one = [_][]const u8{ "BOOL", "U8", "I8", "F8_E4M3", "F8_E5M2" };
    const two = [_][]const u8{ "I16", "U16", "F16", "BF16" };
    const four = [_][]const u8{ "I32", "U32", "F32" };
    const eight = [_][]const u8{ "I64", "U64", "F64" };
    for (one) |d| if (std.mem.eql(u8, dtype, d)) return 1;
    for (two) |d| if (std.mem.eql(u8, dtype, d)) return 2;
    for (four) |d| if (std.mem.eql(u8, dtype, d)) return 4;
    for (eight) |d| if (std.mem.eql(u8, dtype, d)) return 8;
    return error.UnsupportedDtype;
}

pub fn parseFile(a: Allocator, io: std.Io, dir: []const u8, fn_name: []const u8, list: *std.ArrayList(Tensor)) !void {
    const f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ dir, fn_name }), .{});
    defer f.close(io);
    const file_size = (try f.stat(io)).size;
    var lb: [8]u8 = undefined;
    if ((try f.readPositionalAll(io, &lb, 0)) != lb.len) return error.InvalidSafetensors;
    const hlen = std.mem.readInt(u64, &lb, .little);
    const data_base = try add(8, hlen);
    if (hlen == 0 or hlen > (64 << 20) or data_base > file_size) return error.InvalidSafetensors;
    const hbytes = try a.alloc(u8, @intCast(hlen));
    defer a.free(hbytes);
    if ((try f.readPositionalAll(io, hbytes, 8)) != hbytes.len) return error.InvalidSafetensors;
    var parsed = try std.json.parseFromSlice(std.json.Value, a, hbytes, .{ .duplicate_field_behavior = .@"error" });
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidSafetensors;
    var it = parsed.value.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) {
            if (kv.value_ptr.* != .object) return error.InvalidSafetensors;
            continue;
        }
        if (kv.value_ptr.* != .object) return error.InvalidSafetensors;
        const obj = kv.value_ptr.object;
        const dt = obj.get("dtype") orelse return error.InvalidSafetensors;
        const sh = obj.get("shape") orelse return error.InvalidSafetensors;
        const dof = obj.get("data_offsets") orelse return error.InvalidSafetensors;
        if (dt != .string or sh != .array or dof != .array or dof.array.items.len != 2) return error.InvalidSafetensors;
        if (dof.array.items[0] != .integer or dof.array.items[1] != .integer) return error.InvalidSafetensors;
        const first = dof.array.items[0].integer;
        const last = dof.array.items[1].integer;
        // JSON integers outside i64 never reach a lossy float-to-integer cast: their representation is refused above.
        if (first < 0 or last < first) return error.InvalidSafetensors;
        const o0: u64 = @intCast(first);
        const o1: u64 = @intCast(last);
        const len = o1 - o0;
        if (try add(data_base, o1) > file_size) return error.InvalidSafetensors;
        const shape = try a.alloc(u64, sh.array.items.len);
        var elements: u64 = 1;
        for (sh.array.items, 0..) |sv, i| {
            if (sv != .integer or sv.integer < 0) return error.InvalidSafetensors;
            shape[i] = @intCast(sv.integer);
            elements = try mul(elements, shape[i]);
        }
        if (try mul(elements, try dtypeBytes(dt.string)) != len) return error.InvalidSafetensors;
        try list.append(a, .{
            .name = try a.dupe(u8, kv.key_ptr.*),
            .file = try a.dupe(u8, fn_name),
            .dtype = try a.dupe(u8, dt.string),
            .shape = shape,
            .off = try add(data_base, o0),
            .len = len,
        });
    }
}

pub fn readBytes(a: Allocator, io: std.Io, dir: []const u8, t: Tensor) ![]u8 {
    const f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ dir, t.file }), .{});
    defer f.close(io);
    const len = std.math.cast(usize, t.len) orelse return error.InvalidSafetensors;
    const buf = try a.alloc(u8, len);
    errdefer a.free(buf);
    if ((try f.readPositionalAll(io, buf, t.off)) != len) return error.InvalidSafetensors;
    return buf;
}

fn bf16ToF32(u: u16) f32 {
    return @bitCast(@as(u32, u) << 16);
}

fn f16BitsToF32(u: u16) f32 {
    return @floatCast(@as(f16, @bitCast(u)));
}

pub fn decodeF32(a: Allocator, raw: []const u8, dtype: []const u8, r: u32, c: u32) ![]f32 {
    const out = try a.alloc(f32, std.math.mul(usize, r, c) catch return error.InvalidSafetensors);
    errdefer a.free(out);
    if (std.mem.eql(u8, dtype, "BF16")) {
        for (out, 0..) |*v, i| v.* = bf16ToF32(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little));
    } else if (std.mem.eql(u8, dtype, "F16")) {
        for (out, 0..) |*v, i| v.* = f16BitsToF32(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little));
    } else if (std.mem.eql(u8, dtype, "F32")) {
        for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
    } else return error.UnsupportedDtype;
    return out;
}

/// The residual header uses the same JSON escaping for ordinary and adversarial names and dtype strings.
pub fn residualHeader(a: Allocator, list: []const Tensor) ![]u8 {
    var hdr: std.ArrayList(u8) = .empty;
    errdefer hdr.deinit(a);
    try hdr.appendSlice(a, "{\"__metadata__\":{\"format\":\"tfq3\"}");
    var off: u64 = 0;
    for (list) |t| {
        try hdr.print(a, ",{f}:{{\"dtype\":{f},\"shape\":[", .{ std.json.fmt(t.name, .{}), std.json.fmt(t.dtype, .{}) });
        for (t.shape, 0..) |dim, i| {
            if (i > 0) try hdr.append(a, ',');
            try hdr.print(a, "{d}", .{dim});
        }
        const end = try add(off, t.len);
        try hdr.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ off, end });
        off = end;
    }
    try hdr.append(a, '}');
    while ((hdr.items.len & 7) != 0) try hdr.append(a, ' ');
    return hdr.toOwnedSlice(a);
}

pub fn writeResidual(a: Allocator, io: std.Io, src_dir: []const u8, out_dir: []const u8, list: []const Tensor, hdr: []const u8) !void {
    const f = try std.Io.Dir.cwd().createFile(io, try std.fs.path.join(a, &.{ out_dir, "residual.safetensors" }), .{});
    defer f.close(io);
    var hb: [8]u8 = undefined;
    std.mem.writeInt(u64, &hb, hdr.len, .little);
    try f.writeStreamingAll(io, &hb);
    try f.writeStreamingAll(io, hdr);
    for (list) |t| {
        const bytes = try readBytes(a, io, src_dir, t);
        defer a.free(bytes);
        try f.writeStreamingAll(io, bytes);
    }
}
