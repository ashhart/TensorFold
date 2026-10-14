//! GLM-5.3-Flash's two-rank protocol: the frames rank 0 sends rank 1 (each a call rank 1 replays) and their
//! little-endian encoding.

const std = @import("std");
const lanes = @import("lanes");

pub const Op = enum(u32) { prefill = 1, verify = 2, keep = 3, draft = 4, release = 5, stop = 6, ready = 7, image = 8, evict = 9, room = 10 };

pub const Writer = struct {
    buf: std.ArrayList(u8) = .empty,
    gpa: std.mem.Allocator,
    pub fn int(w: *Writer, v: anytype) !void {
        const T = @TypeOf(v);
        const x: u64 = switch (@typeInfo(T)) {
            .comptime_int => @intCast(v),
            .int => |i| if (i.signedness == .signed) @bitCast(@as(i64, v)) else @as(u64, v),
            .bool => @intFromBool(v),
            else => @compileError("Writer.int: an integer"),
        };
        try w.buf.appendSlice(w.gpa, std.mem.asBytes(&x));
    }
    pub fn tokens(w: *Writer, t: []const u32) !void {
        try w.int(t.len);
        try w.buf.appendSlice(w.gpa, std.mem.sliceAsBytes(t));
    }
};

pub const Reader = struct {
    bytes: []const u8,
    at: usize = 0,
    pub fn int(r: *Reader) !u64 {
        if (r.at + 8 > r.bytes.len) return error.ShortFrame;
        const v = std.mem.readInt(u64, r.bytes[r.at..][0..8], .little);
        r.at += 8;
        return v;
    }
    pub fn raw(r: *Reader, n: usize) ![]const u8 {
        if (r.at + n > r.bytes.len) return error.ShortFrame;
        r.at += n;
        return r.bytes[r.at - n .. r.at];
    }
    pub fn tokens(r: *Reader) ![]const u32 {
        const n: usize = @intCast(try r.int());
        if (r.at + 4 * n > r.bytes.len) return error.ShortFrame;
        const t: []align(1) const u32 = std.mem.bytesAsSlice(u32, r.bytes[r.at..][0 .. 4 * n]);
        r.at += 4 * n;
        return @alignCast(t);
    }
};

pub fn writeSampling(w: *Writer, s: ?lanes.Sampling) !void {
    const r = s orelse return w.int(0);
    try w.int(1);
    try w.int(r.seed);
    try w.int(@as(u64, @bitCast(r.temperature)));
    try w.int(r.top_k);
    try w.int(@as(u64, @bitCast(r.top_p)));
    try w.int(@as(u64, @bitCast(r.min_p)));
}

pub fn readSampling(r: *Reader) !?lanes.Sampling {
    if (try r.int() == 0) return null;
    return .{ .seed = try r.int(), .temperature = @bitCast(try r.int()), .top_k = @intCast(try r.int()), .top_p = @bitCast(try r.int()), .min_p = @bitCast(try r.int()) };
}
