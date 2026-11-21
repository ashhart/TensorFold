//! Engram lookups on the CPU: each position's n-gram hash, its rows gathered from the mapped tables (~203 GB).
const std = @import("std");
const c = @import("config.zig");

/// Compressed id of a position that takes no part in an n-gram (an image span's tokens).
pub const dead: i32 = -1;

/// Parts a table may be split into (engram/layerL-partI.safetensors, each under the Hub's per-file limit).
const max_parts = 8;

const Part = struct { map: []align(std.heap.page_size_min) const u8, rows: []const u8, first: usize, count: usize };

/// A table's rows, in one file or in parts of consecutive rows.
const Table = struct {
    parts: [max_parts]Part = undefined,
    n: usize = 0,
    count: usize = 0,

    /// Row `r`'s bytes (r < count).
    fn row(t: *const Table, r: usize) *const [c.engram_row]u8 {
        var i: usize = 0;
        while (i + 1 < t.n and r >= t.parts[i + 1].first) i += 1;
        const p = t.parts[i];
        return p.rows[(r - p.first) * c.engram_row ..][0..c.engram_row];
    }

    fn unmap(t: *const Table) void {
        for (t.parts[0..t.n]) |p| std.posix.munmap(p.map);
    }
};

pub const Hash = struct {
    token_map: []i32, // [vocab] compressed ids
    multipliers: [c.engram_layers.len][c.engram_ngram]i64,
    primes: [c.engram_layers.len][c.engram_cols]i64,
    offsets: [c.engram_layers.len][c.engram_cols]i64,
    pad: i32,
};

pub const Engram = struct {
    gpa: std.mem.Allocator,
    hash: Hash,
    tables: [c.engram_layers.len]Table,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, config: *const c.Config) !Engram {
        var out: Engram = .{ .gpa = gpa, .hash = try readHash(gpa, io, dir), .tables = undefined };
        errdefer gpa.free(out.hash.token_map);
        var opened: usize = 0;
        errdefer for (out.tables[0..opened]) |*t| t.unmap();
        for (&out.tables, c.engram_layers, config.engram_rows) |*t, layer, rows| {
            t.* = try openTable(gpa, dir, layer, rows);
            opened += 1;
        }
        return out;
    }

    pub fn deinit(self: *Engram) void {
        for (&self.tables) |*t| t.unmap();
        self.gpa.free(self.hash.token_map);
    }

    /// The compressed id of `token` (`dead` for an image span's).
    pub fn compress(self: *const Engram, token: u32, text: bool) i32 {
        return if (text) self.hash.token_map[token] else dead;
    }

    /// The table rows position `p` looks up in engram layer `e`, given the compressed ids up to and including it.
    pub fn rowsAt(self: *const Engram, e: usize, ids: []const i32, p: usize, out: *[c.engram_cols]u64) void {
        const h = &self.hash;
        var tokens: [c.engram_ngram]i64 = undefined;
        var blocked = false;
        for (&tokens, 0..) |*t, shift| {
            const src: i32 = if (p >= shift) ids[p - shift] else dead;
            blocked = blocked or p < shift or src == dead;
            t.* = if (blocked) h.pad else src;
        }
        var rolling = tokens[0] *% h.multipliers[e][0];
        for (1..c.engram_ngram) |i| {
            rolling ^= tokens[i] *% h.multipliers[e][i];
            for (0..c.engram_heads) |head| {
                const col = (i - 1) * c.engram_heads + head;
                out[col] = @intCast(@mod(rolling, h.primes[e][col]) + h.offsets[e][col]);
            }
        }
    }

    /// Rows `first..first + count` of a call into `dsts`: all their pages asked for first (madvise), then dequantized.
    pub fn gatherRows(self: *const Engram, ids: []const i32, first: usize, count: usize, dsts: [c.engram_layers.len][]u16) !void {
        const page = std.heap.pageSize();
        for (0..c.engram_layers.len) |e| {
            const t = self.tables[e];
            for (0..count) |r| {
                var rows: [c.engram_cols]u64 = undefined;
                self.rowsAt(e, ids, first + r, &rows);
                for (rows) |row| {
                    if (row >= t.count) return error.EngramRowOutOfRange;
                    const at = @intFromPtr(t.row(row).ptr);
                    const lo = std.mem.alignBackward(usize, at, page);
                    const hi = std.mem.alignForward(usize, at + c.engram_row, page);
                    _ = std.c.madvise(@ptrFromInt(lo), hi - lo, std.c.MADV.WILLNEED);
                }
            }
        }
        for (0..c.engram_layers.len) |e| for (0..count) |r| {
            try self.gather(e, ids, first + r, dsts[e][r * c.engram_cols * c.engram_dim ..][0 .. c.engram_cols * c.engram_dim]);
        };
    }

    /// Position `p`'s 24 rows of engram layer `e`, dequantized into `dst` (bf16 [24 x 256]).
    pub fn gather(self: *const Engram, e: usize, ids: []const i32, p: usize, dst: []u16) !void {
        std.debug.assert(dst.len == c.engram_cols * c.engram_dim);
        var rows: [c.engram_cols]u64 = undefined;
        self.rowsAt(e, ids, p, &rows);
        const t = self.tables[e];
        for (rows, 0..) |r, col| {
            if (r >= t.count) return error.EngramRowOutOfRange;
            const src = t.row(r);
            dequantRow(src, dst[col * c.engram_dim ..][0..c.engram_dim]);
        }
    }
};

/// e4m3fn byte -> f32 (no infinities; NaN codes never appear in the tables).
fn e4m3(b: u8) f32 {
    const sign: f32 = if (b & 0x80 != 0) -1 else 1;
    const e: i32 = (b >> 3) & 0xF;
    const m: f32 = @floatFromInt(b & 7);
    return sign * if (e == 0) m / 8.0 * std.math.pow(f32, 2, -6) else (1 + m / 8.0) * std.math.pow(f32, 2, @floatFromInt(e - 7));
}

const e4m3_table = blk: {
    @setEvalBranchQuota(100000);
    var t: [256]f32 = undefined;
    for (&t, 0..) |*v, i| v.* = e4m3(i);
    break :blk t;
};

/// One stored row (256 e4m3 values, 8 e8m0 scales) to bf16; e4m3 times a power of two is exact in bf16.
fn dequantRow(src: *const [c.engram_row]u8, dst: *[c.engram_dim]u16) void {
    for (0..c.engram_dim / 32) |blk| {
        const scale: f32 = @bitCast(@as(u32, src[c.engram_dim + blk]) << 23);
        for (0..32) |j| {
            const i = blk * 32 + j;
            const v: f32 = e4m3_table[src[i]] * scale;
            dst[i] = @intCast(@as(u32, @bitCast(v)) >> 16);
        }
    }
}

/// Layer `layer`'s table: engram/layerL.safetensors, or its parts engram/layerL-part0.safetensors, -part1, ...
fn openTable(gpa: std.mem.Allocator, dir: []const u8, layer: usize, rows: usize) !Table {
    var t: Table = .{};
    errdefer t.unmap();
    var buf: [64]u8 = undefined;
    const whole = try std.fmt.bufPrint(&buf, "engram/layer{d}.safetensors", .{layer});
    if (try exists(gpa, dir, whole)) {
        t.parts[0] = try mapPart(gpa, dir, whole, 0);
        t.n = 1;
        t.count = t.parts[0].count;
    } else while (t.n < max_parts) : (t.n += 1) {
        const name = try std.fmt.bufPrint(&buf, "engram/layer{d}-part{d}.safetensors", .{ layer, t.n });
        if (!try exists(gpa, dir, name)) break;
        t.parts[t.n] = try mapPart(gpa, dir, name, t.count);
        t.count += t.parts[t.n].count;
    }
    if (t.n == 0) {
        std.log.err("deepseek_v41: no engram table for layer {d} in {s}", .{ layer, dir });
        return error.OpenFailed;
    }
    if (t.count != rows) return error.BadEngramFile;
    return t;
}

fn exists(gpa: std.mem.Allocator, dir: []const u8, name: []const u8) !bool {
    const path = try std.fs.path.joinZ(gpa, &.{ dir, name });
    defer gpa.free(path);
    return std.c.access(path, std.c.F_OK) == 0;
}

/// One file of a table, its rows from `first` on: one tensor, "rows": U8 [count, 264] at offset 0.
fn mapPart(gpa: std.mem.Allocator, dir: []const u8, name: []const u8, first: usize) !Part {
    const path = try std.fs.path.joinZ(gpa, &.{ dir, name });
    defer gpa.free(path);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
    if (fd < 0) {
        std.log.err("deepseek_v41: cannot open {s}", .{path});
        return error.OpenFailed;
    }
    defer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 8) return error.BadEngramFile;
    const size: usize = @intCast(end);
    const map = try std.posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0);
    errdefer std.posix.munmap(map);
    const n = std.mem.readInt(u64, map[0..8], .little);
    if (8 + n > size) return error.BadEngramFile;
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, std.mem.trimEnd(u8, map[8..][0..n], " "), .{});
    defer parsed.deinit();
    const info = parsed.value.object.get("rows") orelse return error.BadEngramFile;
    const shape = info.object.get("shape") orelse return error.BadEngramFile;
    if (shape.array.items.len != 2 or shape.array.items[0].integer <= 0 or shape.array.items[1].integer != c.engram_row) return error.BadEngramFile;
    const count: usize = @intCast(shape.array.items[0].integer);
    if (8 + n + count * c.engram_row > size) return error.BadEngramFile;
    return .{ .map = map, .rows = map[8 + n ..][0 .. count * c.engram_row], .first = first, .count = count };
}

fn readHash(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Hash {
    const path = try std.fs.path.join(gpa, &.{ dir, "engram/hash.safetensors" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(16 << 20));
    defer gpa.free(bytes);
    if (bytes.len < 8) return error.BadEngramHash;
    const n = std.mem.readInt(u64, bytes[0..8], .little);
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, std.mem.trimEnd(u8, bytes[8..][0..n], " "), .{});
    defer parsed.deinit();
    const data = bytes[8 + n ..];
    const T = struct {
        fn get(root: std.json.Value, d: []const u8, name: []const u8, dtype: []const u8, count: usize) ![]const u8 {
            const info = root.object.get(name) orelse return error.BadEngramHash;
            if (!std.mem.eql(u8, info.object.get("dtype").?.string, dtype)) return error.BadEngramHash;
            const offs = info.object.get("data_offsets").?.array.items;
            const a: usize = @intCast(offs[0].integer);
            const b: usize = @intCast(offs[1].integer);
            const size: usize = if (std.mem.eql(u8, dtype, "I64")) 8 else 4;
            if (b - a != count * size or b > d.len) return error.BadEngramHash;
            return d[a..b];
        }
    };
    var h: Hash = undefined;
    const tm = try T.get(parsed.value, data, "token_map", "I32", c.vocab);
    h.token_map = try gpa.alloc(i32, c.vocab);
    for (h.token_map, 0..) |*v, i| v.* = std.mem.readInt(i32, tm[i * 4 ..][0..4], .little);
    const L = c.engram_layers.len;
    const mult = try T.get(parsed.value, data, "multipliers", "I64", L * c.engram_ngram);
    const primes = try T.get(parsed.value, data, "primes", "I64", L * c.engram_cols);
    const offsets = try T.get(parsed.value, data, "offsets", "I64", L * c.engram_cols);
    for (0..L) |e| {
        for (0..c.engram_ngram) |i| h.multipliers[e][i] = std.mem.readInt(i64, mult[(e * c.engram_ngram + i) * 8 ..][0..8], .little);
        for (0..c.engram_cols) |j| {
            h.primes[e][j] = std.mem.readInt(i64, primes[(e * c.engram_cols + j) * 8 ..][0..8], .little);
            h.offsets[e][j] = std.mem.readInt(i64, offsets[(e * c.engram_cols + j) * 8 ..][0..8], .little);
        }
    }
    h.pad = std.mem.readInt(i32, (try T.get(parsed.value, data, "pad", "I32", 1))[0..4], .little);
    return h;
}

test "e4m3 decoding matches the format's corners" {
    try std.testing.expectEqual(@as(f32, 448), e4m3(0x7e));
    try std.testing.expectEqual(@as(f32, -448), e4m3(0xfe));
    try std.testing.expectEqual(@as(f32, 1), e4m3(0x38));
    try std.testing.expectEqual(std.math.pow(f32, 2, -9), e4m3(0x01));
    try std.testing.expectEqual(@as(f32, 0), e4m3(0));
}
