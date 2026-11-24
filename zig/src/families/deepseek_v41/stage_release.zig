//! The rank's release weights as stored bytes: row and column cuts of FP8, BF16 and MXFP4 matrices, never decoded.
const std = @import("std");
const plan = @import("plan.zig");
const names = @import("names.zig");
const Pack = @import("pack.zig").Pack;
const Info = @import("pack.zig").Info;
const Share = @import("partition.zig").Share;

const Io = std.Io;

/// Rows `rows` and bytes `cols` of a row-major tensor whose rows hold `row_bytes` bytes, into `out` (rows x cols.len).
pub fn cut(io: Io, pack: *const Pack, gpa: std.mem.Allocator, info: Info, row_bytes: u64, rows: Share, cols: Share, out: []u8) !void {
    if (cols.lo + cols.len > row_bytes or out.len != @as(u64, rows.len) * cols.len) return error.UnexpectedShape;
    if (cols.lo == 0 and cols.len == row_bytes) return pack.read(io, info, .{ .offset = rows.lo * row_bytes, .len = out.len }, out);
    // whole rows a block at a time, then their columns: one read for many short runs
    const per: u64 = @max(1, (16 << 20) / row_bytes);
    const tmp = try gpa.alloc(u8, @intCast(@min(per, rows.len) * row_bytes));
    defer gpa.free(tmp);
    var r: u64 = 0;
    while (r < rows.len) {
        const k = @min(per, rows.len - r);
        try pack.read(io, info, .{ .offset = (rows.lo + r) * row_bytes, .len = k * row_bytes }, tmp[0 .. k * row_bytes]);
        for (0..k) |i| @memcpy(out[(r + i) * cols.len ..][0..cols.len], tmp[i * row_bytes + cols.lo ..][0..cols.len]);
        r += k;
    }
}

fn div(s: Share, d: u32) Share {
    return .{ .lo = s.lo / d, .len = s.len / d };
}

/// A release group's images: `w` its codes or BF16 values [rows][cols], `s` its FP8 block scales (empty for BF16).
pub const GroupImage = struct {
    w: []align(16) u8,
    s: []align(16) u8,

    pub fn deinit(g: *GroupImage, gpa: std.mem.Allocator) void {
        gpa.free(g.w);
        gpa.free(g.s);
    }
};

pub fn group(gpa: std.mem.Allocator, io: Io, pack: *const Pack, form: plan.Form) !GroupImage {
    switch (form) {
        .fp8_block => |f| {
            const w = try gpa.alignedAlloc(u8, .@"16", @as(usize, f.rows.len) * f.cols.len);
            errdefer gpa.free(w);
            try cut(io, pack, gpa, f.weight, f.k, f.rows, f.cols, w);
            const s = try gpa.alignedAlloc(u8, .@"16", @as(usize, f.rows.len / 32) * (f.cols.len / 32));
            errdefer gpa.free(s);
            try cut(io, pack, gpa, f.scale, (f.k + 31) / 32, div(f.rows, 32), div(f.cols, 32), s);
            return .{ .w = w, .s = s };
        },
        .bf16 => |f| {
            const w = try gpa.alignedAlloc(u8, .@"16", 2 * @as(usize, f.rows.len) * f.cols.len);
            errdefer gpa.free(w);
            try cut(io, pack, gpa, f.weight, 2 * @as(u64, f.k), f.rows, .{ .lo = 2 * f.cols.lo, .len = 2 * f.cols.len }, w);
            return .{ .w = w, .s = try gpa.alignedAlloc(u8, .@"16", 0) };
        },
        .exl3 => return error.UnexpectedTensor,
    }
}

/// A routed projection's images: `w` [E][rows][cols/2] E2M1 pairs, `s` [E][rows][cols/32] E8M0, expert by expert.
pub fn experts(gpa: std.mem.Allocator, io: Io, pack: *const Pack, block: []const u8, x: *const plan.ExpertsPlan) !GroupImage {
    const f = switch (x.form) {
        .mxfp4 => |m| m,
        .exl3 => return error.UnexpectedTensor,
    };
    const wb = @as(usize, f.rows.len) * f.cols.len / 2;
    const sb = @as(usize, f.rows.len) * f.cols.len / 32;
    const w = try gpa.alignedAlloc(u8, .@"16", f.ids.len * wb);
    errdefer gpa.free(w);
    const s = try gpa.alignedAlloc(u8, .@"16", f.ids.len * sb);
    errdefer gpa.free(s);
    for (0..f.ids.len) |slot| {
        const e = f.ids.lo + slot;
        var nb: [160]u8 = undefined;
        var tb: [176]u8 = undefined;
        const prefix = try names.expertName(block, @intCast(e), x.proj, &nb);
        const wi = try pack.need(try std.fmt.bufPrint(&tb, "{s}.weight", .{prefix}));
        try cut(io, pack, gpa, wi, f.k / 2, f.rows, div(f.cols, 2), w[slot * wb ..][0..wb]);
        const si = try pack.need(try std.fmt.bufPrint(&tb, "{s}.scale", .{prefix}));
        try cut(io, pack, gpa, si, f.k / 32, f.rows, div(f.cols, 32), s[slot * sb ..][0..sb]);
    }
    return .{ .w = w, .s = s };
}

test "a cut of a row-major tensor takes the asked rows and byte columns, whole rows in one read" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var data: [6 * 8]u8 = undefined;
    for (&data, 0..) |*x, i| x.* = @intCast(i);
    const spec = [_]@import("pack.zig").Spec{.{ .name = "t.weight", .dtype = "U8", .shape = &.{ 6, 8 }, .fill = 0 }};
    const img = try @import("pack.zig").image(a, &spec);
    defer a.free(img);
    const at = img.len - data.len;
    @memcpy(img[at..], &data);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = img });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var p = try Pack.open(a, io, dir);
    defer p.deinit();
    const info = try p.need("t.weight");
    var out: [2 * 3]u8 = undefined;
    try cut(io, &p, a, info, 8, .{ .lo = 3, .len = 2 }, .{ .lo = 4, .len = 3 }, &out);
    try std.testing.expectEqualSlices(u8, &.{ 28, 29, 30, 36, 37, 38 }, &out);
    var rows: [2 * 8]u8 = undefined;
    try cut(io, &p, a, info, 8, .{ .lo = 1, .len = 2 }, .{ .lo = 0, .len = 8 }, &rows);
    try std.testing.expectEqualSlices(u8, data[8..24], &rows);
    try std.testing.expectError(error.UnexpectedShape, cut(io, &p, a, info, 8, .{ .lo = 0, .len = 1 }, .{ .lo = 6, .len = 3 }, out[0..3]));
}

fn pattern(t: usize, i: usize) u8 {
    return @truncate(t * 37 + i * 7 + i / 251);
}

test "release images: FP8 rows, columns and block scales, BF16 columns and MXFP4 experts cut from their stored bytes" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const pk = @import("pack.zig");
    const specs = [_]pk.Spec{
        .{ .name = "f.weight", .dtype = "F8_E4M3", .shape = &.{ 64, 64 } },
        .{ .name = "f.scale", .dtype = "F8_E8M0", .shape = &.{ 2, 2 } },
        .{ .name = "b.weight", .dtype = "BF16", .shape = &.{ 4, 8 } },
        .{ .name = "blk.ffn.experts.0.w2.weight", .dtype = "I8", .shape = &.{ 64, 32 } },
        .{ .name = "blk.ffn.experts.0.w2.scale", .dtype = "F8_E8M0", .shape = &.{ 64, 2 } },
        .{ .name = "blk.ffn.experts.1.w2.weight", .dtype = "I8", .shape = &.{ 64, 32 } },
        .{ .name = "blk.ffn.experts.1.w2.scale", .dtype = "F8_E8M0", .shape = &.{ 64, 2 } },
    };
    const sizes = [_]usize{ 64 * 64, 4, 2 * 32, 64 * 32, 64 * 2, 64 * 32, 64 * 2 };
    const img = try pk.image(a, &specs);
    defer a.free(img);
    var total: usize = 0;
    for (sizes) |n| total += n;
    var at = img.len - total;
    for (sizes, 0..) |n, t| {
        for (0..n) |i| img[at + i] = pattern(t, i);
        at += n;
    }
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = img });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var p = try Pack.open(a, io, dir);
    defer p.deinit();
    // FP8: rows 32..63, columns 32..63, and the one scale of that 32x32 block
    var g = try group(a, io, &p, .{ .fp8_block = .{ .weight = try p.need("f.weight"), .scale = try p.need("f.scale"), .n = 64, .k = 64, .rows = .{ .lo = 32, .len = 32 }, .cols = .{ .lo = 32, .len = 32 } } });
    defer g.deinit(a);
    for (0..32) |r| for (0..32) |c| try std.testing.expectEqual(pattern(0, (32 + r) * 64 + 32 + c), g.w[r * 32 + c]);
    try std.testing.expectEqualSlices(u8, &.{pattern(1, 3)}, g.s);
    // BF16: rows 1..2, columns 2..5 (bytes 4..11 of each 16-byte row)
    var h = try group(a, io, &p, .{ .bf16 = .{ .weight = try p.need("b.weight"), .n = 4, .k = 8, .rows = .{ .lo = 1, .len = 2 }, .cols = .{ .lo = 2, .len = 4 } } });
    defer h.deinit(a);
    for (0..2) |r| for (0..8) |c| try std.testing.expectEqual(pattern(2, (1 + r) * 16 + 4 + c), h.w[r * 8 + c]);
    try std.testing.expectEqual(@as(usize, 0), h.s.len);
    // MXFP4 down: both experts' columns 32..63 = bytes 16..31 of each row and scale column 1
    const x: plan.ExpertsPlan = .{ .proj = .w2, .form = .{ .mxfp4 = .{ .count = 2, .ids = .{ .lo = 0, .len = 2 }, .n = 64, .k = 64, .rows = .{ .lo = 0, .len = 64 }, .cols = .{ .lo = 32, .len = 32 } } } };
    var e = try experts(a, io, &p, "blk", &x);
    defer e.deinit(a);
    try std.testing.expectEqual(@as(usize, 2 * 64 * 16), e.w.len);
    for (0..2) |ex| for (0..64) |r| {
        for (0..16) |c| try std.testing.expectEqual(pattern(3 + 2 * ex, r * 32 + 16 + c), e.w[(ex * 64 + r) * 16 + c]);
        try std.testing.expectEqual(pattern(4 + 2 * ex, r * 2 + 1), e.s[ex * 64 + r]);
    };
}
