//! Tensor-parallel cuts of host tensors, the same for every format: rows, columns and experts, byte for byte.

const std = @import("std");
const st = @import("../safetensors.zig");

const Tensor = st.Tensor;
const Allocator = std.mem.Allocator;

pub const Error = error{UnevenSplit} || Allocator.Error;

/// A tensor-parallel rank of a group.
pub const Rank = struct { rank: usize, world: usize };

/// Rows [from, to) of a tensor's first dimension.
pub const Span = struct { from: usize, to: usize };

/// `size` over `world` equal parts, or `error.UnevenSplit` (logged under `what`).
pub fn even(size: usize, world: usize, what: []const u8) Error!usize {
    if (size % world != 0) {
        std.log.err("{s}: {d} does not split into {d} equal parts", .{ what, size, world });
        return error.UnevenSplit;
    }
    return size / world;
}

fn rowBytes(t: Tensor) usize {
    return t.numel() / t.shape[0] * t.dtype.size();
}

/// `spans` of `t`'s rows joined in order; one span shares the source's bytes.
pub fn takeRows(a: Allocator, t: Tensor, spans: []const Span) Error!Tensor {
    const row = rowBytes(t);
    var rows: usize = 0;
    for (spans) |s| rows += s.to - s.from;
    var out = t;
    out.shape[0] = rows;
    if (spans.len == 1) {
        out.bytes = t.bytes[spans[0].from * row .. spans[0].to * row];
        return out;
    }
    const bytes = try a.alloc(u8, rows * row);
    var at: usize = 0;
    for (spans) |s| {
        const n = (s.to - s.from) * row;
        @memcpy(bytes[at..][0..n], t.bytes[s.from * row ..][0..n]);
        at += n;
    }
    out.bytes = bytes;
    return out;
}

/// Columns [from, to) of a rank-2 tensor, copied.
pub fn takeCols(a: Allocator, t: Tensor, from: usize, to: usize) Error!Tensor {
    const size = t.dtype.size();
    const width = (to - from) * size;
    const bytes = try a.alloc(u8, t.shape[0] * width);
    for (0..t.shape[0]) |r| @memcpy(bytes[r * width ..][0..width], t.bytes[(r * t.shape[1] + from) * size ..][0..width]);
    var out = t;
    out.shape[1] = to - from;
    out.bytes = bytes;
    return out;
}

/// Experts [part * rank, ...) of a stacked tensor (E + 1 experts, the shared one last), and the shared one on rank 0.
pub fn takeExperts(a: Allocator, t: Tensor, r: Rank, part: usize) Error!Tensor {
    const total = t.shape[0] - 1;
    const row = rowBytes(t);
    const mine = t.bytes[r.rank * part * row ..][0 .. part * row];
    if (r.rank != 0) {
        var out = t;
        out.shape[0] = part;
        out.bytes = mine;
        return out;
    }
    const bytes = try a.alloc(u8, (part + 1) * row);
    @memcpy(bytes[0 .. part * row], mine);
    @memcpy(bytes[part * row ..], t.bytes[total * row ..][0..row]);
    var out = t;
    out.shape[0] = part + 1;
    out.bytes = bytes;
    return out;
}

test "rows join their spans and a column cut keeps its rows" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var data: [4 * 3]u16 = undefined;
    for (&data, 0..) |*v, i| v.* = @intCast(i);
    const t: Tensor = .{ .dtype = .u16, .rank = 2, .shape = .{ 4, 3, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&data) };
    const rows = try takeRows(a, t, &.{ .{ .from = 1, .to = 2 }, .{ .from = 3, .to = 4 } });
    try std.testing.expectEqualSlices(u16, &.{ 3, 4, 5, 9, 10, 11 }, @alignCast(std.mem.bytesAsSlice(u16, rows.bytes)));
    const cols = try takeCols(a, t, 1, 3);
    try std.testing.expectEqual(@as(usize, 2), cols.shape[1]);
    try std.testing.expectEqualSlices(u16, &.{ 1, 2, 4, 5, 7, 8, 10, 11 }, @alignCast(std.mem.bytesAsSlice(u16, cols.bytes)));
    try std.testing.expectEqual(@as(usize, 2), try even(4, 2, "test"));
}
