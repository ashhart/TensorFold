//! MXFP8 (e4m3 codes, a UE8M0 power of two per 32 inputs) as a rank's lane weight: read, cut and packed on the host.

const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const mx = cuda.mxfp8;
const formats = @import("cuda_formats.zig");

/// An MXFP8 projection as stored: e4m3 codes [n, k] and its UE8M0 scale bytes [n, k/32], the file's own bytes.
pub const Mxfp8 = struct {
    n: usize,
    k: usize,
    codes: []const u8,
    scales: []const u8,

    /// `module`'s weight and weight_scale, checked as formats.of checks them.
    pub fn read(ck: *core.Checkpoint, module: []const u8) !Mxfp8 {
        var buf: [256]u8 = undefined;
        const w = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{module}));
        const s = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{module}));
        _ = formats.of(.{ .mixed = .mxfp8 }, .{ .weight = formats.metaOf(&w), .scale = formats.metaOf(&s) }) catch return error.BadMxfp8;
        return .{ .n = w.dim(0), .k = w.dim(1), .codes = w.bytes, .scales = s.bytes };
    }
};

/// Rows [lo, hi) and inputs [klo, khi) of a stored projection: one part of a rank's lane weight.
pub const Share = struct { p: *const Mxfp8, lo: usize, hi: usize, klo: usize, khi: usize };

/// The rows and inputs of `parts` stacked, and the bytes its packed codes and scale tiles take.
pub fn sizes(parts: []const Share) error{BadMxfp8Share}!struct { n: usize, k: usize, codes: usize, scales: usize } {
    if (parts.len == 0) return error.BadMxfp8Share;
    const k = parts[0].khi -| parts[0].klo;
    var n: usize = 0;
    for (parts, 1..) |s, j| {
        const inside = s.lo < s.hi and s.hi <= s.p.n and s.klo < s.khi and s.khi <= s.p.k;
        // a cut splits no scale byte's 32 inputs, 64-input stage or 64-row tile; the last part may end inside its tile
        const tiled = (s.hi - s.lo) % 64 == 0 or j == parts.len;
        if (!inside or s.khi - s.klo != k or s.klo % mx.group != 0 or k % 64 != 0 or !tiled) return error.BadMxfp8Share;
        n += s.hi - s.lo;
    }
    const npad = mx.padded(n);
    return .{ .n = n, .k = k, .codes = npad * k, .scales = npad * (k / mx.group) };
}

/// `parts` packed one after another in 64-row tiles as the lane's one weight (cuda.mxfp8's order); padding 0 and 127.
pub fn pack(gpa: std.mem.Allocator, parts: []const Share, codes: []u8, scales: []u8) !void {
    const z = try sizes(parts);
    if (codes.len != z.codes or scales.len != z.scales) return error.BadMxfp8Share;
    @memset(codes, 0);
    @memset(scales, 127);
    const sg = z.k / mx.group; // a row's scale bytes
    var t0: usize = 0; // tiles written
    for (parts) |s| {
        const n = s.hi - s.lo;
        const npad = mx.padded(n);
        const row_sg = s.p.k / mx.group;
        const cut = try gpa.alloc(u8, n * z.k);
        defer gpa.free(cut);
        const sc = try gpa.alloc(u8, n * sg);
        defer gpa.free(sc);
        for (0..n) |r| {
            @memcpy(cut[r * z.k ..][0..z.k], s.p.codes[(s.lo + r) * s.p.k + s.klo ..][0..z.k]);
            @memcpy(sc[r * sg ..][0..sg], s.p.scales[(s.lo + r) * row_sg + s.klo / mx.group ..][0..sg]);
        }
        const pc = try gpa.alloc(u8, npad * z.k);
        defer gpa.free(pc);
        mx.packCodes(pc, cut, n, z.k);
        const ps = try gpa.alloc(u8, npad * sg);
        defer gpa.free(ps);
        mx.packScales(ps, sc, n, z.k);
        const tiles = (n + 63) / 64; // the last part's partial tile with its padding rows
        @memcpy(codes[t0 * 64 * z.k ..][0 .. tiles * 64 * z.k], pc[0 .. tiles * 64 * z.k]);
        @memcpy(scales[t0 * 64 * sg ..][0 .. tiles * 64 * sg], ps[0 .. tiles * 64 * sg]);
        t0 += tiles;
    }
}

/// Rows [lo, hi) of `p` as bf16, each value e4m3 times its power of two; error.NotExact when one is not a bf16.
pub fn bf16Rows(p: Mxfp8, lo: usize, hi: usize, out: []u16) error{ NotExact, BadMxfp8Share }!void {
    if (lo > hi or hi > p.n or out.len != (hi - lo) * p.k) return error.BadMxfp8Share;
    const sg = p.k / mx.group;
    for (lo..hi) |r| for (0..p.k) |c| {
        const e = p.scales[r * sg + c / mx.group];
        if (e == 0xff) return error.NotExact; // UE8M0's NaN
        const v = core.quant.fp8.e4m3(p.codes[r * p.k + c]) * std.math.ldexp(@as(f64, 1), @as(i32, e) - 127);
        const b: u32 = @bitCast(@as(f32, @floatCast(v)));
        if (@as(f64, @as(f32, @bitCast(b & 0xffff0000))) != v) return error.NotExact;
        out[(r - lo) * p.k + c] = @intCast(b >> 16);
    };
}

/// Rows [lo, hi) of `p` widened to bf16 onto the loader's host bytes (an MTP kv_b); inexact: refused by `module`.
pub fn appendWide(L: anytype, p: Mxfp8, module: []const u8, lo: usize, hi: usize) !void {
    const out = try L.gpa.alloc(u16, (hi -| lo) * p.k);
    defer L.gpa.free(out);
    bf16Rows(p, lo, hi, out) catch |e| {
        std.log.err("{s}: its MXFP8 rows hold a value no bf16 does ({t}); the absorbed attention reads bf16", .{ module, e });
        return e;
    };
    try L.host.appendSlice(L.gpa, std.mem.sliceAsBytes(out));
}

/// Rows [lo, hi) of MXFP8 `module` widened to bf16 on the device, for a draft head's quantizer; inexact: refused.
pub fn wideRows(d: *const cuda.Driver, gpa: std.mem.Allocator, ck: *core.Checkpoint, module: []const u8, lo: usize, hi: usize) !cuda.DeviceBuffer {
    const p = try Mxfp8.read(ck, module);
    const out = try gpa.alloc(u16, (hi -| lo) * p.k);
    defer gpa.free(out);
    bf16Rows(p, lo, hi, out) catch |e| {
        std.log.err("{s}: its MXFP8 rows hold a value no bf16 does ({t}); the draft head quantizes bf16 rows", .{ module, e });
        return e;
    };
    return cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(out));
}

const testing = std.testing;

/// e4m3 codes without NaN (0x7f, 0xff), from `seed`.
fn codesOf(out: []u8, seed: u8) void {
    for (out, 0..) |*c, j| {
        const v: u8 = @truncate(j * 29 + seed);
        c.* = if (v & 0x7f == 0x7f) v ^ 1 else v;
    }
}

test "an MXFP8 projection reads its codes and scale bytes as stored, and a wrong scale shape is refused" {
    const gpa = testing.allocator;
    var codes: [128 * 64]u8 = undefined;
    codesOf(&codes, 3);
    var sc: [128 * 2]u8 = undefined;
    for (&sc, 0..) |*s, j| s.* = @truncate(120 + j % 8);
    const tensors = [_]core.shard_edit.New{
        .{ .name = "a.weight", .dtype = .f8_e4m3, .shape = &.{ 128, 64 }, .bytes = &codes },
        .{ .name = "a.weight_scale", .dtype = .u8, .shape = &.{ 128, 2 }, .bytes = &sc },
        .{ .name = "b.weight", .dtype = .f8_e4m3, .shape = &.{ 128, 64 }, .bytes = &codes },
        .{ .name = "b.weight_scale", .dtype = .u8, .shape = &.{ 128, 1 }, .bytes = sc[0..128] },
    };
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir: [64]u8 = undefined;
    const at = try std.fmt.bufPrint(&dir, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var path: [96]u8 = undefined;
    try core.shard_edit.create(gpa, try std.fmt.bufPrint(&path, "{s}/model.safetensors", .{at}), &.{}, &tensors);
    var ck = try core.Checkpoint.openModel(gpa, testing.io, at);
    defer ck.close();
    const p = try Mxfp8.read(&ck, "a");
    try testing.expectEqual(@as(usize, 128), p.n);
    try testing.expectEqual(@as(usize, 64), p.k);
    try testing.expectEqualSlices(u8, &codes, p.codes);
    try testing.expectEqualSlices(u8, &sc, p.scales);
    try testing.expectError(error.BadMxfp8, Mxfp8.read(&ck, "b"));
}

test "a rank's MXFP8 share packs as cuda.mxfp8 packs the rows and inputs it holds, inputs from inside a 64-input stage" {
    const gpa = testing.allocator;
    const n = 192;
    const k = 256;
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    codesOf(codes, 5);
    const sc = try gpa.alloc(u8, n * k / 32);
    defer gpa.free(sc);
    for (sc, 0..) |*s, j| s.* = @truncate(110 + j % 17);
    const p: Mxfp8 = .{ .n = n, .k = k, .codes = codes, .scales = sc };
    // rows [64, 192) and inputs [32, 160): an o_proj or down share whose origin is a scale byte, not a 64-input stage
    const share = [_]Share{.{ .p = &p, .lo = 64, .hi = 192, .klo = 32, .khi = 160 }};
    const z = try sizes(&share);
    const got_c = try gpa.alloc(u8, z.codes);
    defer gpa.free(got_c);
    const got_s = try gpa.alloc(u8, z.scales);
    defer gpa.free(got_s);
    try pack(gpa, &share, got_c, got_s);
    const cut = try gpa.alloc(u8, 128 * 128);
    defer gpa.free(cut);
    const cut_s = try gpa.alloc(u8, 128 * 4);
    defer gpa.free(cut_s);
    for (0..128) |r| {
        @memcpy(cut[r * 128 ..][0..128], codes[(64 + r) * k + 32 ..][0..128]);
        @memcpy(cut_s[r * 4 ..][0..4], sc[(64 + r) * 8 + 1 ..][0..4]);
    }
    const want_c = try gpa.alloc(u8, mx.padded(128) * 128);
    defer gpa.free(want_c);
    mx.packCodes(want_c, cut, 128, 128);
    const want_s = try gpa.alloc(u8, mx.padded(128) * 4);
    defer gpa.free(want_s);
    mx.packScales(want_s, cut_s, 128, 128);
    try testing.expectEqualSlices(u8, want_c, got_c);
    try testing.expectEqualSlices(u8, want_s, got_s);
}

test "stacked MXFP8 parts follow each other in 64-row tiles, the stack's padding rows zero with scale 127" {
    const gpa = testing.allocator;
    const k = 64;
    var a_codes: [128 * k]u8 = undefined;
    codesOf(&a_codes, 7);
    var b_codes: [64 * k]u8 = undefined;
    codesOf(&b_codes, 9);
    var a_sc: [128 * 2]u8 = @splat(121);
    const b_sc: [64 * 2]u8 = @splat(124);
    a_sc[2 * 100] = 119;
    const a: Mxfp8 = .{ .n = 128, .k = k, .codes = &a_codes, .scales = &a_sc };
    const b: Mxfp8 = .{ .n = 64, .k = k, .codes = &b_codes, .scales = &b_sc };
    // [q | k]-like: rows [64, 128) of a, then all of b: 128 rows, no padding; then a third part pads to 256
    const parts = [_]Share{ .{ .p = &a, .lo = 64, .hi = 128, .klo = 0, .khi = k }, .{ .p = &b, .lo = 0, .hi = 64, .klo = 0, .khi = k }, .{ .p = &a, .lo = 0, .hi = 64, .klo = 0, .khi = k } };
    const z = try sizes(&parts);
    try testing.expectEqual(@as(usize, 192), z.n);
    const got_c = try gpa.alloc(u8, z.codes);
    defer gpa.free(got_c);
    const got_s = try gpa.alloc(u8, z.scales);
    defer gpa.free(got_s);
    try pack(gpa, &parts, got_c, got_s);
    var all: [192 * k]u8 = undefined;
    @memcpy(all[0 .. 64 * k], a_codes[64 * k ..][0 .. 64 * k]);
    @memcpy(all[64 * k ..][0 .. 64 * k], &b_codes);
    @memcpy(all[128 * k ..][0 .. 64 * k], a_codes[0 .. 64 * k]);
    var all_s: [192 * 2]u8 = undefined;
    @memcpy(all_s[0..128], a_sc[128..256]);
    @memcpy(all_s[128..256], &b_sc);
    @memcpy(all_s[256..384], a_sc[0..128]);
    const want_c = try gpa.alloc(u8, 256 * k);
    defer gpa.free(want_c);
    mx.packCodes(want_c, &all, 192, k);
    const want_s = try gpa.alloc(u8, 256 * 2);
    defer gpa.free(want_s);
    mx.packScales(want_s, &all_s, 192, k);
    try testing.expectEqualSlices(u8, want_c, got_c);
    try testing.expectEqualSlices(u8, want_s, got_s);
}

test "a last MXFP8 part may end inside its 64-row tile: [64 | 32] rows pack as the 96 rows packed whole" {
    const gpa = testing.allocator;
    const k = 128;
    var a_codes: [64 * k]u8 = undefined;
    codesOf(&a_codes, 11);
    var b_codes: [32 * k]u8 = undefined;
    codesOf(&b_codes, 13);
    var a_sc: [64 * 4]u8 = undefined;
    for (&a_sc, 0..) |*s, j| s.* = @truncate(118 + j % 9);
    var b_sc: [32 * 4]u8 = undefined;
    for (&b_sc, 0..) |*s, j| s.* = @truncate(121 + j % 5);
    const a: Mxfp8 = .{ .n = 64, .k = k, .codes = &a_codes, .scales = &a_sc };
    const b: Mxfp8 = .{ .n = 32, .k = k, .codes = &b_codes, .scales = &b_sc };
    const parts = [_]Share{ .{ .p = &a, .lo = 0, .hi = 64, .klo = 0, .khi = k }, .{ .p = &b, .lo = 0, .hi = 32, .klo = 0, .khi = k } };
    const z = try sizes(&parts);
    try testing.expectEqual(@as(usize, 96), z.n);
    const got_c = try gpa.alloc(u8, z.codes);
    defer gpa.free(got_c);
    const got_s = try gpa.alloc(u8, z.scales);
    defer gpa.free(got_s);
    try pack(gpa, &parts, got_c, got_s);
    var all_c: [96 * k]u8 = undefined;
    @memcpy(all_c[0 .. 64 * k], &a_codes);
    @memcpy(all_c[64 * k ..], &b_codes);
    var all_s: [96 * 4]u8 = undefined;
    @memcpy(all_s[0 .. 64 * 4], &a_sc);
    @memcpy(all_s[64 * 4 ..], &b_sc);
    const want_c = try gpa.alloc(u8, z.codes);
    defer gpa.free(want_c);
    mx.packCodes(want_c, &all_c, 96, k);
    const want_s = try gpa.alloc(u8, z.scales);
    defer gpa.free(want_s);
    mx.packScales(want_s, &all_s, 96, k);
    try testing.expectEqualSlices(u8, want_c, got_c);
    try testing.expectEqualSlices(u8, want_s, got_s);
}

test "an MXFP8 share off a scale byte, a 64-input stage or a 64-row tile, or past its tensor, is refused" {
    const codes: [128 * 128]u8 = @splat(0);
    const sc: [128 * 4]u8 = @splat(127);
    const p: Mxfp8 = .{ .n = 128, .k = 128, .codes = &codes, .scales = &sc };
    try testing.expectError(error.BadMxfp8Share, sizes(&.{.{ .p = &p, .lo = 0, .hi = 128, .klo = 16, .khi = 80 }}));
    try testing.expectError(error.BadMxfp8Share, sizes(&.{.{ .p = &p, .lo = 0, .hi = 128, .klo = 32, .khi = 128 }}));
    try testing.expectError(error.BadMxfp8Share, sizes(&.{ .{ .p = &p, .lo = 0, .hi = 100, .klo = 0, .khi = 128 }, .{ .p = &p, .lo = 0, .hi = 64, .klo = 0, .khi = 128 } }));
    try testing.expectError(error.BadMxfp8Share, sizes(&.{.{ .p = &p, .lo = 64, .hi = 192, .klo = 0, .khi = 128 }}));
    try testing.expectError(error.BadMxfp8Share, sizes(&.{}));
    var small: [8]u8 = undefined;
    var tiles: [2]u8 = undefined;
    try testing.expectError(error.BadMxfp8Share, pack(testing.allocator, &.{.{ .p = &p, .lo = 0, .hi = 64, .klo = 0, .khi = 128 }}, &small, &tiles));
}

test "MXFP8 rows widen to bf16 exactly, and a value bf16 cannot hold is refused" {
    const codes = [_]u8{ 0x38, 0xb8, 0x3f, 0x01, 0x00, 0x7e } ++ @as([26]u8, @splat(0x30));
    var sc = [_]u8{127};
    const p: Mxfp8 = .{ .n = 1, .k = 32, .codes = &codes, .scales = &sc };
    var out: [32]u16 = undefined;
    try bf16Rows(p, 0, 1, &out);
    try testing.expectEqual(@as(u16, 0x3f80), out[0]); // 1.0
    try testing.expectEqual(@as(u16, 0xbf80), out[1]); // -1.0
    try testing.expectEqual(@as(u16, 0x3ff0), out[2]); // 1.875
    try testing.expectEqual(@as(u16, 0x3b00), out[3]); // 2^-9, e4m3's least subnormal
    try testing.expectEqual(@as(u16, 0), out[4]);
    sc[0] = 0; // 2^-127: e4m3's subnormals fall below bf16's
    try testing.expectError(error.NotExact, bf16Rows(p, 0, 1, &out));
    sc[0] = 0xff;
    try testing.expectError(error.NotExact, bf16Rows(p, 0, 1, &out));
}
