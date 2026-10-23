//! FP8_PB_WO (e4m3 codes, fp32 128 x 128 block scales) as a rank's lane weight: read, cut and packed on the host.

const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const fp8 = cuda.fp8;
const formats = @import("cuda_formats.zig");

/// An FP8_PB_WO projection as stored: e4m3 codes [n, k] and its fp32 128 x 128 block scales, copied out of the file.
pub const Fp8pb = struct {
    n: usize,
    k: usize,
    codes: []const u8,
    scales: []f32,

    /// `module`'s weight and weight_scale ([Rb, 1, Cb, 1] or [Rb, Cb]), checked as formats.of checks them.
    pub fn read(gpa: std.mem.Allocator, ck: *core.Checkpoint, module: []const u8) !Fp8pb {
        var buf: [256]u8 = undefined;
        const w = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{module}));
        const s = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{module}));
        _ = formats.of(.{ .mixed = .fp8_pb_wo }, .{ .weight = formats.metaOf(&w), .scale = formats.metaOf(&s) }) catch return error.BadFp8pb;
        const scales = try gpa.alloc(f32, s.numel());
        for (scales, 0..) |*v, j| v.* = @bitCast(std.mem.readInt(u32, s.bytes[4 * j ..][0..4], .little));
        return .{ .n = w.dim(0), .k = w.dim(1), .codes = w.bytes, .scales = scales };
    }

    pub fn deinit(p: Fp8pb, gpa: std.mem.Allocator) void {
        gpa.free(p.scales);
    }
};

/// Rows [lo, hi) and inputs [klo, khi) of a stored projection: one part of a rank's lane weight.
pub const Share = struct { p: *const Fp8pb, lo: usize, hi: usize, klo: usize, khi: usize };

/// The rows and inputs of `parts` stacked, and the bytes and floats its packed codes and scale tiles take.
pub fn sizes(parts: []const Share) error{BadFp8Share}!struct { n: usize, k: usize, codes: usize, scales: usize } {
    if (parts.len == 0) return error.BadFp8Share;
    const k = parts[0].khi -| parts[0].klo;
    var n: usize = 0;
    for (parts) |s| {
        const inside = s.lo < s.hi and s.hi <= s.p.n and s.klo < s.khi and s.khi <= s.p.k;
        // the lane's 64-input groups and 64-row tiles: a cut off them would mix two groups' scales or rows
        if (!inside or s.khi - s.klo != k or s.klo % fp8.group != 0 or k % fp8.group != 0 or (s.hi - s.lo) % 64 != 0) return error.BadFp8Share;
        n += s.hi - s.lo;
    }
    const npad = fp8.padded(n);
    return .{ .n = n, .k = k, .codes = npad * k, .scales = npad * (k / fp8.group) };
}

/// `parts` packed one after another in 64-row tiles as the lane's one weight (cuda.fp8's order); padding 0 and 1.
pub fn pack(gpa: std.mem.Allocator, parts: []const Share, codes: []u8, scales: []f32) !void {
    const z = try sizes(parts);
    if (codes.len != z.codes or scales.len != z.scales) return error.BadFp8Share;
    @memset(codes, 0);
    @memset(scales, 1.0);
    const kg = z.k / fp8.group;
    var t0: usize = 0; // tiles written
    for (parts) |s| {
        const n = s.hi - s.lo;
        const npad = fp8.padded(n);
        const cut = try gpa.alloc(u8, n * z.k);
        defer gpa.free(cut);
        for (0..n) |r| @memcpy(cut[r * z.k ..][0..z.k], s.p.codes[(s.lo + r) * s.p.k + s.klo ..][0..z.k]);
        const pc = try gpa.alloc(u8, npad * z.k);
        defer gpa.free(pc);
        fp8.packCodes(pc, cut, n, z.k);
        const ps = try gpa.alloc(f32, npad * kg);
        defer gpa.free(ps);
        fp8.packScalesAt(ps, s.p.scales, s.p.n, s.p.k, s.lo, s.klo, n, z.k);
        const tiles = n / 64;
        @memcpy(codes[t0 * 64 * z.k ..][0 .. tiles * 64 * z.k], pc[0 .. tiles * 64 * z.k]);
        @memcpy(scales[t0 * kg * 64 ..][0 .. tiles * kg * 64], ps[0 .. tiles * kg * 64]);
        t0 += tiles;
    }
}

const testing = std.testing;

/// A packed weight's scale for row `r`, input group `g` (cuda.fp8's tiles [npad/64][kg][64]).
fn tileScale(tiles: []const f32, kg: usize, r: usize, g: usize) f32 {
    return tiles[(r / 64 * kg + g) * 64 + r % 64];
}

fn fp8pbOf(codes: []const u8, scales: []f32, n: usize, k: usize) Fp8pb {
    return .{ .n = n, .k = k, .codes = codes, .scales = scales };
}

test "an FP8_PB_WO projection reads its block scales as stored, in either shape" {
    const gpa = testing.allocator;
    var codes: [192 * 256]u8 = undefined;
    for (&codes, 0..) |*c, j| c.* = @truncate(j * 13 + 5);
    const inv = [_]f32{ 1.5, 2.5, 3.5, 4.5 };
    const tensors = [_]core.shard_edit.New{
        .{ .name = "a.weight", .dtype = .f8_e4m3, .shape = &.{ 192, 256 }, .bytes = &codes },
        .{ .name = "a.weight_scale", .dtype = .f32, .shape = &.{ 2, 1, 2, 1 }, .bytes = std.mem.sliceAsBytes(&inv) },
        .{ .name = "b.weight", .dtype = .f8_e4m3, .shape = &.{ 192, 256 }, .bytes = &codes },
        .{ .name = "b.weight_scale", .dtype = .f32, .shape = &.{ 2, 2 }, .bytes = std.mem.sliceAsBytes(&inv) },
        .{ .name = "c.weight", .dtype = .f8_e4m3, .shape = &.{ 192, 256 }, .bytes = &codes },
        .{ .name = "c.weight_scale", .dtype = .f32, .shape = &.{ 1, 2 }, .bytes = std.mem.sliceAsBytes(inv[0..2]) },
    };
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir: [64]u8 = undefined;
    const at = try std.fmt.bufPrint(&dir, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var path: [96]u8 = undefined;
    try core.shard_edit.create(gpa, try std.fmt.bufPrint(&path, "{s}/model.safetensors", .{at}), &.{}, &tensors);
    var ck = try core.Checkpoint.openModel(gpa, testing.io, at);
    defer ck.close();
    inline for (.{ "a", "b" }) |m| {
        const p = try Fp8pb.read(gpa, &ck, m);
        defer p.deinit(gpa);
        try testing.expectEqual(@as(usize, 192), p.n);
        try testing.expectEqual(@as(usize, 256), p.k);
        try testing.expectEqualSlices(u8, &codes, p.codes);
        try testing.expectEqualSlices(f32, &inv, p.scales);
    }
    try testing.expectError(error.BadFp8pb, Fp8pb.read(gpa, &ck, "c")); // one block row for 192 rows
}

test "a rank's FP8_PB_WO share packs as cuda.fp8 packs the rows and inputs it holds, a cut inside a block too" {
    const gpa = testing.allocator;
    const n = 192;
    const k = 256;
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    for (codes, 0..) |*c, j| c.* = @truncate(j * 29 + 3);
    var inv = [_]f32{ 1.5, 2.5, 3.5, 4.5 };
    const p = fp8pbOf(codes, &inv, n, k);
    // rows [64, 192): a q_b or gate/up share starting inside the first 128-row block
    {
        const z = try sizes(&.{.{ .p = &p, .lo = 64, .hi = 192, .klo = 0, .khi = k }});
        const got_c = try gpa.alloc(u8, z.codes);
        defer gpa.free(got_c);
        const got_s = try gpa.alloc(f32, z.scales);
        defer gpa.free(got_s);
        try pack(gpa, &.{.{ .p = &p, .lo = 64, .hi = 192, .klo = 0, .khi = k }}, got_c, got_s);
        const want_c = try gpa.alloc(u8, fp8.padded(128) * k);
        defer gpa.free(want_c);
        fp8.packCodes(want_c, codes[64 * k ..], 128, k);
        try testing.expectEqualSlices(u8, want_c, got_c);
        for (0..128) |r| for (0..k / 64) |g| {
            try testing.expectEqual(inv[((64 + r) / 128) * 2 + g * 64 / 128], tileScale(got_s, k / 64, r, g));
        };
    }
    // inputs [128, 256) of every row: an o_proj or down share, its 192 rows padded to 256
    {
        const share = [_]Share{.{ .p = &p, .lo = 0, .hi = n, .klo = 128, .khi = k }};
        const z = try sizes(&share);
        try testing.expectEqual(@as(usize, 256 * 128), z.codes);
        const got_c = try gpa.alloc(u8, z.codes);
        defer gpa.free(got_c);
        const got_s = try gpa.alloc(f32, z.scales);
        defer gpa.free(got_s);
        try pack(gpa, &share, got_c, got_s);
        const cut = try gpa.alloc(u8, n * 128);
        defer gpa.free(cut);
        for (0..n) |r| @memcpy(cut[r * 128 ..][0..128], codes[r * k + 128 ..][0..128]);
        const want_c = try gpa.alloc(u8, 256 * 128);
        defer gpa.free(want_c);
        fp8.packCodes(want_c, cut, n, 128);
        try testing.expectEqualSlices(u8, want_c, got_c);
        for (0..256) |r| for (0..2) |g| {
            const want: f32 = if (r < n) inv[(r / 128) * 2 + 1] else 1.0;
            try testing.expectEqual(want, tileScale(got_s, 2, r, g));
        };
    }
}

test "stacked shares follow each other in 64-row tiles, the stack's padding rows zero with scale one" {
    const gpa = testing.allocator;
    const k = 128;
    const a_codes = try gpa.alloc(u8, 192 * k);
    defer gpa.free(a_codes);
    for (a_codes, 0..) |*c, j| c.* = @truncate(j * 7 + 1);
    const b_codes = try gpa.alloc(u8, 128 * k);
    defer gpa.free(b_codes);
    for (b_codes, 0..) |*c, j| c.* = @truncate(j * 11 + 2);
    var a_inv = [_]f32{ 2, 3 };
    var b_inv = [_]f32{5};
    const a = fp8pbOf(a_codes, &a_inv, 192, k);
    const b = fp8pbOf(b_codes, &b_inv, 128, k);
    // [gate | up]-like: 64 rows of a from inside its second block, then 128 rows of b: 192 rows padded to 256
    const parts = [_]Share{ .{ .p = &a, .lo = 128, .hi = 192, .klo = 0, .khi = k }, .{ .p = &b, .lo = 0, .hi = 128, .klo = 0, .khi = k } };
    const z = try sizes(&parts);
    try testing.expectEqual(@as(usize, 192), z.n);
    const got_c = try gpa.alloc(u8, z.codes);
    defer gpa.free(got_c);
    const got_s = try gpa.alloc(f32, z.scales);
    defer gpa.free(got_s);
    try pack(gpa, &parts, got_c, got_s);
    const both = try gpa.alloc(u8, 192 * k);
    defer gpa.free(both);
    @memcpy(both[0 .. 64 * k], a_codes[128 * k ..][0 .. 64 * k]);
    @memcpy(both[64 * k ..], b_codes);
    const want_c = try gpa.alloc(u8, 256 * k);
    defer gpa.free(want_c);
    fp8.packCodes(want_c, both, 192, k);
    try testing.expectEqualSlices(u8, want_c, got_c);
    for (0..256) |r| for (0..2) |g| {
        const want: f32 = if (r < 64) 3 else if (r < 192) 5 else 1;
        try testing.expectEqual(want, tileScale(got_s, 2, r, g));
    };
}

test "a share off the lane's 64-input groups or 64-row tiles, or past its tensor, is refused" {
    const codes: [128 * 128]u8 = @splat(0);
    var inv = [_]f32{1};
    const p = fp8pbOf(&codes, &inv, 128, 128);
    try testing.expectError(error.BadFp8Share, sizes(&.{.{ .p = &p, .lo = 0, .hi = 128, .klo = 32, .khi = 128 }}));
    try testing.expectError(error.BadFp8Share, sizes(&.{.{ .p = &p, .lo = 0, .hi = 100, .klo = 0, .khi = 128 }}));
    try testing.expectError(error.BadFp8Share, sizes(&.{.{ .p = &p, .lo = 64, .hi = 192, .klo = 0, .khi = 128 }}));
    try testing.expectError(error.BadFp8Share, sizes(&.{ .{ .p = &p, .lo = 0, .hi = 64, .klo = 0, .khi = 128 }, .{ .p = &p, .lo = 0, .hi = 64, .klo = 0, .khi = 64 } }));
    try testing.expectError(error.BadFp8Share, sizes(&.{}));
    var small: [8]u8 = undefined;
    var tiles: [2]f32 = undefined;
    try testing.expectError(error.BadFp8Share, pack(testing.allocator, &.{.{ .p = &p, .lo = 0, .hi = 64, .klo = 0, .khi = 128 }}, &small, &tiles));
}
