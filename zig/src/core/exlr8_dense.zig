//! EXLR8 dense and vocabulary matrices: the int64[8] records, formats 1-7 as HF weights (out, in), block checksums.

const std = @import("std");
const Io = std.Io;
const st = @import("safetensors.zig");
const codec = @import("exlr8_codec.zig");
const xxh = @import("xxhash64.zig");
const job = @import("exlr8_job.zig");

pub const Format = enum(u8) { trellis = 1, int4 = 2, mxfp8 = 3, fp8_block = 4, fp8_row = 5, bf16 = 6, fp32 = 7 };
/// Which channels a tensor-parallel shard splits: none, outputs, or inputs.
pub const Side = enum(u8) { replicated = 0, output = 1, input = 2 };

/// format, out, in, K (trellis only), side, channels per shard, shared input rotation (0 own), solve method.
pub const Record = struct {
    format: Format,
    out: usize,
    in: usize,
    k: u8,
    side: Side,
    channels: usize,
    rotation: u8,
    solve: u8,

    pub fn parse(t: st.Tensor) !Record {
        if (!t.is(.i64, &.{8})) return error.BadExlr8Record;
        var f: [8]i64 = undefined;
        for (&f, 0..) |*v, i| v.* = std.mem.readInt(i64, t.bytes[8 * i ..][0..8], .little);
        const r: Record = .{
            .format = std.enums.fromInt(Format, std.math.cast(u8, f[0]) orelse 0) orelse return error.BadExlr8Record,
            .out = std.math.cast(usize, f[1]) orelse return error.BadExlr8Record,
            .in = std.math.cast(usize, f[2]) orelse return error.BadExlr8Record,
            .k = std.math.cast(u8, f[3]) orelse return error.BadExlr8Record,
            .side = std.enums.fromInt(Side, std.math.cast(u8, f[4]) orelse 9) orelse return error.BadExlr8Record,
            .channels = std.math.cast(usize, f[5]) orelse return error.BadExlr8Record,
            .rotation = std.math.cast(u8, f[6]) orelse return error.BadExlr8Record,
            .solve = std.math.cast(u8, f[7]) orelse return error.BadExlr8Record,
        };
        if (r.out == 0 or r.in == 0 or (r.format == .trellis) != (r.k != 0) or r.k > 8) return error.BadExlr8Record;
        return r;
    }

    /// The dense kernels' format number (int4 sharded by input has its own layout); trellis has its own kernels.
    pub fn kernelFormat(r: Record) !u32 {
        return switch (r.format) {
            .trellis => error.TrellisHasItsOwnKernels,
            .int4 => if (r.side == .input) job.dense_int4_in else @backingInt(r.format),
            else => @backingInt(r.format),
        };
    }

    /// 128-channel blocks on the sharded side (output rows when replicated): one checksum each.
    pub fn blocks(r: Record) usize {
        return std.math.divCeil(usize, if (r.side == .input) r.in else r.out, 128) catch unreachable;
    }
};

/// The data tensors of each format, in checksum order.
pub fn parts(f: Format) []const []const u8 {
    return switch (f) {
        .trellis => &.{ "trellis", "suh", "svh" },
        .int4 => &.{ "q4", "s4" },
        .mxfp8 => &.{ "fp8", "e8m0" },
        .fp8_block, .fp8_row => &.{ "fp8", "scale" },
        .bf16 => &.{"bf16"},
        .fp32 => &.{"fp32"},
    };
}

pub const Matrix = struct {
    record: Record,
    data: [3]st.Tensor,
    checksum: ?st.Tensor,

    /// M.record and its data tensors, every shape checked against the record.
    pub fn get(f: *const st.File, name: []const u8) !Matrix {
        var buf: [256]u8 = undefined;
        const rec = try Record.parse(f.get(try std.fmt.bufPrint(&buf, "{s}.record", .{name})) orelse return error.MissingExlr8Tensor);
        var m: Matrix = .{ .record = rec, .data = undefined, .checksum = f.get(try std.fmt.bufPrint(&buf, "{s}.checksum", .{name})) };
        for (parts(rec.format), 0..) |p, i| m.data[i] = f.get(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ name, p })) orelse return error.MissingExlr8Tensor;
        try m.check();
        return m;
    }

    fn check(m: Matrix) !void {
        const r = m.record;
        const ok = switch (r.format) {
            .trellis => m.data[0].is(.u8, &.{ std.math.divExact(usize, if (r.side == .input) r.in else r.out, 128) catch return error.BadExlr8Shape, (if (r.side == .input) r.out else r.in) * 16 * @as(usize, r.k) }) and
                m.data[1].is(.f16, &.{r.in}) and m.data[2].is(.f16, &.{r.out}),
            .int4 => if (r.side == .input)
                m.data[0].is(.u8, &.{ r.in / 128, r.out, 64 }) and m.data[1].is(.f16, &.{ r.in / 128, r.out, 4 })
            else
                m.data[0].is(.u8, &.{ r.out, r.in / 2 }) and m.data[1].is(.f16, &.{ r.out, r.in / 32 }),
            .mxfp8 => m.data[0].is(.u8, &.{ r.out, r.in }) and m.data[1].is(.u8, &.{ r.out, r.in / 32 }),
            .fp8_block => m.data[0].is(.u8, &.{ r.out, r.in }) and m.data[1].is(.f32, &.{ try std.math.divCeil(usize, r.out, 128), try std.math.divCeil(usize, r.in, 128) }),
            .fp8_row => m.data[0].is(.u8, &.{ r.out, r.in }) and m.data[1].is(.f32, &.{r.out}),
            .bf16 => m.data[0].is(.bf16, &.{ r.out, r.in }),
            .fp32 => m.data[0].is(.f32, &.{ r.out, r.in }),
        };
        const group: usize = switch (r.format) {
            .trellis => 128,
            .int4, .mxfp8 => if (r.side == .input) 128 else 32,
            else => 1,
        };
        if (!ok or r.in % group != 0 or (r.format == .trellis and r.out % 128 != 0)) return error.BadExlr8Shape;
        if (m.checksum) |c| if (!c.is(.u64, &.{r.blocks()})) return error.BadExlr8Shape;
    }

    fn f16At(t: st.Tensor, i: usize) f32 {
        return @as(f16, @bitCast(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)));
    }
    fn f32At(t: st.Tensor, i: usize) f32 {
        return @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little));
    }

    /// Rows [row0, row0 + n) of the HF weight (out, in) in fp32, for every format but trellis.
    pub fn rows(m: Matrix, row0: usize, n: usize, out: []f32) !void {
        const r = m.record;
        if (r.format == .trellis) return error.TrellisNeedsCodebook;
        if (row0 + n > r.out or out.len != n * r.in) return error.BadExlr8Rows;
        for (0..n) |ro| for (0..r.in) |i| {
            const o = row0 + ro;
            out[ro * r.in + i] = switch (r.format) {
                .int4 => blk: {
                    const q, const s = if (r.side == .input)
                        .{ m.data[0].bytes[((i / 128) * r.out + o) * 64 + (i % 128) / 2], f16At(m.data[1], ((i / 128) * r.out + o) * 4 + (i % 128) / 32) }
                    else
                        .{ m.data[0].bytes[o * (r.in / 2) + i / 2], f16At(m.data[1], o * (r.in / 32) + i / 32) };
                    const nibble: i32 = if (i % 2 == 0) q & 15 else q >> 4;
                    break :blk @as(f32, @floatFromInt(nibble - 8)) * s;
                },
                .mxfp8 => codec.e4m3Value(m.data[0].bytes[o * r.in + i]) * std.math.ldexp(@as(f32, 1), @as(i32, m.data[1].bytes[o * (r.in / 32) + i / 32]) - 127),
                .fp8_block => codec.e4m3Value(m.data[0].bytes[o * r.in + i]) * f32At(m.data[1], (o / 128) * (std.math.divCeil(usize, r.in, 128) catch unreachable) + i / 128),
                .fp8_row => codec.e4m3Value(m.data[0].bytes[o * r.in + i]) * f32At(m.data[1], o),
                .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, m.data[0].bytes[2 * (o * r.in + i) ..][0..2], .little)) << 16),
                .fp32 => f32At(m.data[0], o * r.in + i),
                .trellis => unreachable,
            };
        };
    }

    /// The trellis tiles as E4M3 codes T (in, out); entries are output columns, or input rows on side 2.
    pub fn trellisCodes(m: Matrix, book: *const codec.Codebook, out: []u8) !void {
        const r = m.record;
        if (r.format != .trellis or out.len != r.in * r.out) return error.BadExlr8Rows;
        const entry = m.data[0].dim(1);
        if (r.side == .input) {
            for (0..r.in / 128) |g| try codec.decodeRun(book, r.k, .in_major, m.data[0].bytes[g * entry ..][0..entry], 128, r.out, out[g * 128 * r.out ..][0 .. 128 * r.out]);
            return;
        }
        var cols: [128 * 128]u8 = undefined;
        for (0..r.out / 128) |g| for (0..r.in / 128) |b| {
            const tiles = 64 * codec.tileBytes(r.k);
            try codec.decodeRun(book, r.k, .out_major, m.data[0].bytes[g * entry + b * tiles ..][0..tiles], 128, 128, &cols);
            for (0..128) |row| @memcpy(out[(b * 128 + row) * r.out + g * 128 ..][0..128], cols[row * 128 ..][0..128]);
        };
    }

    pub fn suh(m: Matrix, out: []f32) void {
        for (out, 0..) |*v, i| v.* = f16At(m.data[1], i);
    }
    pub fn svh(m: Matrix, out: []f32) void {
        for (out, 0..) |*v, i| v.* = f16At(m.data[2], i);
    }

    /// Every stored block checksum against xxhash64 of the bytes the format names; false on the first mismatch.
    pub fn verify(m: Matrix) !bool {
        const c = m.checksum orelse return error.NoExlr8Checksums;
        const r = m.record;
        for (0..r.blocks()) |b| {
            var x = xxh.Hasher.init(0);
            switch (r.format) {
                .trellis => {
                    const own: usize, const other: usize = if (r.side == .input) .{ 1, 2 } else .{ 2, 1 };
                    x.update(rowBytes(m.data[0], b, 1));
                    x.update(rowBytes(m.data[own], 128 * b, 128));
                    x.update(m.data[other].bytes);
                },
                .fp8_block => {
                    x.update(rowBytes(m.data[0], 128 * b, 128));
                    x.update(rowBytes(m.data[1], b, 1));
                },
                .int4 => if (r.side == .input) {
                    x.update(rowBytes(m.data[0], b, 1));
                    x.update(rowBytes(m.data[1], b, 1));
                } else for (parts(r.format), 0..) |_, i| x.update(rowBytes(m.data[i], 128 * b, 128)),
                else => for (parts(r.format), 0..) |_, i| x.update(rowBytes(m.data[i], 128 * b, 128)),
            }
            if (x.final() != std.mem.readInt(u64, c.bytes[8 * b ..][0..8], .little)) return false;
        }
        return true;
    }
};

/// Rows [row, row + n) of a tensor along its first dimension, clipped at its end.
fn rowBytes(t: st.Tensor, row: usize, n: usize) []const u8 {
    const stride = t.bytes.len / t.dim(0);
    const end = @min(row + n, t.dim(0));
    return t.bytes[row * stride .. end * stride];
}

/// Tensors stored as shipped (norms, biases, router corrections): every name no matrix owns, in header order.
pub fn plainNames(gpa: std.mem.Allocator, f: *const st.File) ![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(gpa);
    next: for (f.names.keys()) |k| {
        if (std.mem.startsWith(u8, k, "__pad.")) continue;
        const dot = std.mem.lastIndexOfScalar(u8, k, '.') orelse {
            try out.append(gpa, k);
            continue;
        };
        var buf: [256]u8 = undefined;
        const rec = f.get(std.fmt.bufPrint(&buf, "{s}.record", .{k[0..dot]}) catch return error.BadExlr8Name);
        if (rec) |r| {
            const fmt = (Record.parse(r) catch return error.BadExlr8Record).format;
            const suffix = k[dot + 1 ..];
            if (std.mem.eql(u8, suffix, "record") or std.mem.eql(u8, suffix, "checksum")) continue :next;
            for (parts(fmt)) |p| if (std.mem.eql(u8, suffix, p)) continue :next;
        }
        try out.append(gpa, k);
    }
    return out.toOwnedSlice(gpa);
}

/// Every matrix name in a dense or vocabulary file (tensors named M.record), in header order.
pub fn names(gpa: std.mem.Allocator, f: *const st.File) ![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(gpa);
    for (f.names.keys()) |k| if (std.mem.endsWith(u8, k, ".record")) try out.append(gpa, k[0 .. k.len - ".record".len]);
    return out.toOwnedSlice(gpa);
}

test "records parse every field and refuse an unknown format, side or a trellis without a width" {
    var bytes: [64]u8 = undefined;
    const fields = [8]i64{ 1, 16384, 2048, 4, 1, 256, 2, 1 };
    for (fields, 0..) |v, i| std.mem.writeInt(i64, bytes[8 * i ..][0..8], v, .little);
    var t: st.Tensor = .{ .dtype = .i64, .rank = 1, .shape = .{ 8, 1, 1, 1, 1 }, .bytes = &bytes };
    const r = try Record.parse(t);
    try std.testing.expectEqual(Format.trellis, r.format);
    try std.testing.expectEqual(@as(usize, 128), r.blocks());
    try std.testing.expectEqual(@as(u8, 2), r.rotation);
    std.mem.writeInt(i64, bytes[32..40], 2, .little);
    try std.testing.expectEqual(@as(usize, 16), (try Record.parse(t)).blocks());
    std.mem.writeInt(i64, bytes[0..8], 8, .little);
    try std.testing.expectError(error.BadExlr8Record, Record.parse(t));
    std.mem.writeInt(i64, bytes[0..8], 6, .little);
    try std.testing.expectError(error.BadExlr8Record, Record.parse(t));
    t.bytes = bytes[0..56];
    t.shape[0] = 7;
    try std.testing.expectError(error.BadExlr8Record, Record.parse(t));
}

test "an fp32 matrix (format 7, which the GLM-5.3 release does not use) reads as stored and checks its blocks" {
    var vals: [2 * 32]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.25 - 3;
    const data = std.mem.sliceAsBytes(&vals);
    var rec: [64]u8 = undefined;
    for ([8]i64{ 7, 2, 32, 0, 0, 0, 0, 0 }, 0..) |v, i| std.mem.writeInt(i64, rec[8 * i ..][0..8], v, .little);
    var sum: [8]u8 = undefined;
    std.mem.writeInt(u64, &sum, xxh.hash(0, data), .little);
    const m: Matrix = .{
        .record = try Record.parse(.{ .dtype = .i64, .rank = 1, .shape = .{ 8, 1, 1, 1, 1 }, .bytes = &rec }),
        .data = .{ .{ .dtype = .f32, .rank = 2, .shape = .{ 2, 32, 1, 1, 1 }, .bytes = data }, undefined, undefined },
        .checksum = .{ .dtype = .u64, .rank = 1, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = &sum },
    };
    try m.check();
    var out: [64]f32 = undefined;
    try m.rows(0, 2, &out);
    try std.testing.expectEqualSlices(f32, &vals, &out);
    try std.testing.expect(try m.verify());
    sum[0] ^= 1;
    try std.testing.expect(!try m.verify());
}
