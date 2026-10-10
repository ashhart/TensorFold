//! Weights kept in float: fp32 [N, K], the unquantized projections a conversion leaves (an MTP head's fc).

const std = @import("std");
const types = @import("types.zig");
const slice = @import("slice.zig");
const convert = @import("convert.zig");

const Tensor = types.Tensor;
const Allocator = std.mem.Allocator;
const join = types.join;

pub const id: types.Format = .dense;
pub const decoder: types.Decoder = .dense;

/// A float weight is not named by the config: nothing here declares it.
pub const Config = void;

pub fn detect(a: Allocator, src: types.Sources) Allocator.Error!?Config {
    _ = a;
    _ = src;
    return null;
}

pub const Host = struct { weight: Tensor };

/// Whether `key` holds a float matrix with no table of scales beside it.
pub fn matches(t: anytype, key: []const u8) bool {
    var buf: [256]u8 = undefined;
    const w = t.get(join(&buf, &.{ key, ".weight" })) catch return false;
    return convert.isFloat(w.dtype) and w.rank == 2 and !t.has(join(&buf, &.{ key, ".scales" }));
}

/// The float weight widened to fp32.
pub fn read(a: Allocator, t: anytype, key: []const u8) !Host {
    var buf: [256]u8 = undefined;
    return .{ .weight = try convert.float32(a, try t.get(join(&buf, &.{ key, ".weight" }))) };
}

pub fn rows(h: Host) usize {
    return h.weight.shape[0];
}

pub fn bytes(h: Host) usize {
    return h.weight.bytes.len;
}

/// Both halves along K, copied.
pub fn halves(a: Allocator, p: Host) ![2]Host {
    const half = p.weight.shape[1] / 2;
    return .{
        .{ .weight = try slice.takeCols(a, p.weight, 0, half) },
        .{ .weight = try slice.takeCols(a, p.weight, half, p.weight.shape[1]) },
    };
}

pub fn sliceRows(a: Allocator, p: Host, spans: []const slice.Span) slice.Error!Host {
    return .{ .weight = try slice.takeRows(a, p.weight, spans) };
}

pub fn sliceCols(a: Allocator, p: Host, r: slice.Rank, what: []const u8) slice.Error!Host {
    const width = try slice.even(p.weight.shape[1], r.world, what);
    return .{ .weight = try slice.takeCols(a, p.weight, r.rank * width, (r.rank + 1) * width) };
}

pub fn sliceStack(a: Allocator, p: Host, r: slice.Rank, part: usize) slice.Error!Host {
    return .{ .weight = try slice.takeExperts(a, p.weight, r, part) };
}

pub const Device = struct { weight: types.Buf };

pub fn upload(u: types.Uploader, h: Host) !Device {
    return .{ .weight = try u.tensor(h.weight) };
}

/// What the kernels read: the fp32 weight's address.
pub const Matrix = struct { weight: u64 };

pub const View = struct { n: u32, k: u32, matrix: Matrix };

pub fn view(d: Device, stacked: bool) error{UnsupportedTables}!View {
    const lead: usize = @intFromBool(stacked);
    return .{ .n = @intCast(d.weight.dim(lead)), .k = @intCast(d.weight.dim(lead + 1)), .matrix = .{ .weight = d.weight.ptr } };
}

pub fn elements(h: Host) usize {
    return h.weight.numel();
}

/// The weight as fp64, row by row.
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    if (out.len != h.weight.numel()) return error.UnexpectedTensor;
    for (out, 0..) |*o, i| o.* = convert.load(.f32, h.weight.bytes, i);
}

test "a float weight halves along K and cuts its rows" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const h: Host = .{ .weight = .{ .dtype = .f32, .rank = 2, .shape = .{ 2, 4, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&data) } };
    const pair = try halves(a, h);
    try std.testing.expectEqualSlices(f32, &.{ 3, 4, 7, 8 }, @alignCast(std.mem.bytesAsSlice(f32, pair[1].weight.bytes)));
    const low = try sliceRows(a, h, &.{.{ .from = 1, .to = 2 }});
    var out: [4]f64 = undefined;
    try reference(low, &out);
    try std.testing.expectEqualSlices(f64, &.{ 5, 6, 7, 8 }, &out);
}
