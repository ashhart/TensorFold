//! A safetensors file's header, read by core (`headerSize`, `parseHeader`) before any tensor is sliced (host-tested).
const std = @import("std");
const st = @import("core").safetensors;

pub const Tensor = struct { name: []const u8, dtype: []const u8, shape: [8]u64, rank: usize, bytes: []const u8 };

pub const Error = error{ BadSafetensors, TensorRank, UnknownDtype } || std.mem.Allocator.Error;

/// The header's own spelling of a core dtype ("BF16", "F8_E4M3").
pub fn dtypeName(d: st.DType) []const u8 {
    return switch (d) {
        inline else => |tag| comptime blk: {
            const s = @tagName(tag);
            var b: [s.len]u8 = undefined;
            for (s, 0..) |ch, i| b[i] = std.ascii.toUpper(ch);
            const out = b;
            break :blk &out;
        },
    };
}

/// Appends `file`'s tensors to `out` as core reads the header; their names live in the returned arena.
pub fn parse(gpa: std.mem.Allocator, file: []const u8, out: *std.ArrayList(Tensor)) Error!std.heap.ArenaAllocator {
    if (file.len < 8) return error.BadSafetensors;
    const n = try st.headerSize(std.mem.readInt(u64, file[0..8], .little), file.len);
    const data = file[8 + n ..];
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const names = st.parseHeader(arena.allocator(), file[8..][0..n], data.len) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.UnsupportedDType => error.UnknownDtype,
        error.RankTooHigh => error.TensorRank,
        else => error.BadSafetensors,
    };
    try out.ensureUnusedCapacity(gpa, names.count());
    for (names.keys(), names.values()) |name, e| {
        var t: Tensor = .{ .name = name, .dtype = dtypeName(e.dtype), .shape = @splat(1), .rank = e.rank, .bytes = data[e.begin..e.end] };
        for (e.shape[0..e.rank], t.shape[0..e.rank]) |d, *s| s.* = d;
        out.appendAssumeCapacity(t);
    }
    return arena;
}

fn sample(comptime header: []const u8, comptime data_len: usize) [8 + header.len + data_len]u8 {
    var f: [8 + header.len + data_len]u8 = @splat(0);
    std.mem.writeInt(u64, f[0..8], header.len, .little);
    @memcpy(f[8 .. 8 + header.len], header);
    return f;
}

fn expectParse(f: []const u8) Error!usize {
    var out: std.ArrayList(Tensor) = .empty;
    defer out.deinit(std.testing.allocator);
    const names = try parse(std.testing.allocator, f, &out);
    names.deinit();
    return out.items.len;
}

test "a well-formed file gives its tensors" {
    const f = sample("{\"__metadata__\":{},\"a\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[0,12]}}", 12);
    try std.testing.expectEqual(@as(usize, 1), try expectParse(&f));
}

test "a header longer than the file is refused" {
    var f = sample("{}", 0);
    std.mem.writeInt(u64, f[0..8], 1 << 40, .little);
    try std.testing.expectError(error.BadSafetensors, expectParse(&f));
    try std.testing.expectError(error.BadSafetensors, expectParse(f[0..5]));
}

test "offsets past the data, reversed, or not matching the shape are refused" {
    const past = sample("{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}", 8);
    try std.testing.expectError(error.BadSafetensors, expectParse(&past));
    const reversed = sample("{\"a\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[8,0]}}", 8);
    try std.testing.expectError(error.BadSafetensors, expectParse(&reversed));
    const short = sample("{\"a\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,6]}}", 8);
    try std.testing.expectError(error.BadSafetensors, expectParse(&short));
    const negative = sample("{\"a\":{\"dtype\":\"U8\",\"shape\":[-1],\"data_offsets\":[0,0]}}", 0);
    try std.testing.expectError(error.BadSafetensors, expectParse(&negative));
}

test "a missing field or an unknown dtype is refused" {
    const no_shape = sample("{\"a\":{\"dtype\":\"U8\",\"data_offsets\":[0,1]}}", 1);
    try std.testing.expectError(error.BadSafetensors, expectParse(&no_shape));
    const unknown = sample("{\"a\":{\"dtype\":\"Q4\",\"shape\":[1],\"data_offsets\":[0,1]}}", 1);
    try std.testing.expectError(error.UnknownDtype, expectParse(&unknown));
}

test "a rank past core's, overlapping tensors or a header past core's cap are refused; F8_E8M0 is read" {
    const rank_six = sample("{\"a\":{\"dtype\":\"U8\",\"shape\":[1,1,1,1,1,1],\"data_offsets\":[0,1]}}", 1);
    try std.testing.expectError(error.TensorRank, expectParse(&rank_six));
    const overlap = sample("{\"a\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]},\"b\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[1,3]}}", 3);
    try std.testing.expectError(error.BadSafetensors, expectParse(&overlap));
    var capped = sample("{}", 0);
    std.mem.writeInt(u64, capped[0..8], st.max_header_bytes + 1, .little);
    try std.testing.expectError(error.BadSafetensors, expectParse(&capped));
    const scales = sample("{\"s\":{\"dtype\":\"F8_E8M0\",\"shape\":[2,2],\"data_offsets\":[0,4]}}", 4);
    try std.testing.expectEqual(@as(usize, 1), try expectParse(&scales));
}

test "tensors keep the header's order, names, dtype spelling, shape padded with ones and bytes" {
    const f = sample("{\"z\":{\"dtype\":\"F16\",\"shape\":[1,2],\"data_offsets\":[4,8]},\"a\":{\"dtype\":\"F8_E4M3\",\"shape\":[4],\"data_offsets\":[0,4]}}", 8);
    var out: std.ArrayList(Tensor) = .empty;
    defer out.deinit(std.testing.allocator);
    const names = try parse(std.testing.allocator, &f, &out);
    defer names.deinit();
    try std.testing.expectEqualStrings("z", out.items[0].name);
    try std.testing.expectEqualStrings("F16", out.items[0].dtype);
    try std.testing.expectEqualSlices(u64, &.{ 1, 2, 1, 1, 1, 1, 1, 1 }, &out.items[0].shape);
    try std.testing.expect(out.items[0].rank == 2 and out.items[0].bytes.ptr == f[f.len - 4 ..].ptr);
    try std.testing.expectEqualStrings("F8_E4M3", out.items[1].dtype);
}
