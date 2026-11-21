//! Eight-row block tiles over TensorFold's own GGUF SDOT arithmetic.
const std = @import("std");
const g = @import("core").gguf;
const rows = @import("cpu_rows.zig");
const sdot = @import("cpu_gguf_q4_sdot.zig");
const block_batch = @import("cpu_gguf_block_batch.zig");
pub const layout_version: u32 = 5;

pub const BlockQ4Kx8 = extern struct { rows: [8][144]u8 align(16) };
pub const BlockQ6Kx8 = extern struct { rows: [8][210]u8 align(16) };

pub const Repacked = union(enum) {
    q4_k: []const BlockQ4Kx8,
    q6_k: []const BlockQ6Kx8,
    pub fn deinit(self: Repacked, a: std.mem.Allocator) void {
        switch (self) {
            .q4_k => |buf| a.free(buf),
            .q6_k => |buf| a.free(buf),
        }
    }
};

pub fn releaseSourcePages(bytes: []const u8) void {
    if (comptime @import("builtin").os.tag == .windows or @TypeOf(std.posix.MADV) == void) return;
    const page: usize = std.heap.page_size_min;
    const start = @intFromPtr(bytes.ptr);
    const end = start + bytes.len;
    const aligned_start = (start + page - 1) & ~(page - 1);
    const aligned_end = end & ~(page - 1);
    if (aligned_end <= aligned_start) return;
    const ptr: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(aligned_start);
    std.posix.madvise(ptr, aligned_end - aligned_start, std.posix.MADV.DONTNEED) catch {};
}

// Tile eight complete GGUF blocks at each K position; tail rows retain the raw reader.
fn packInto(comptime T: type, comptime bytes_per_block: usize, src: []const u8, width: usize, n: usize, dst: []T) void {
    const bpr = width / 256;
    for (0..n / 8) |group| for (0..bpr) |block| {
        inline for (0..8) |row| {
            const offset = ((group * 8 + row) * bpr + block) * bytes_per_block;
            @memcpy(&dst[group * bpr + block].rows[row], src[offset..][0..bytes_per_block]);
        }
    };
}

pub fn repackQ4KInto(src: []const u8, width: usize, n: usize, dst: []BlockQ4Kx8) void {
    packInto(BlockQ4Kx8, 144, src, width, n, dst);
}
pub fn repackQ6KInto(src: []const u8, width: usize, n: usize, dst: []BlockQ6Kx8) void {
    packInto(BlockQ6Kx8, 210, src, width, n, dst);
}

pub fn repack(a: std.mem.Allocator, t: g.Tensor) !?Repacked {
    if (t.rank != 2) return error.BadShape;
    const width: usize = @intCast(t.shape[0]);
    const n: usize = @intCast(t.shape[1]);
    if (n < 8 or width == 0 or width % 256 != 0) return null;
    const count = (n / 8) * (width / 256);
    switch (t.kind) {
        .q4_k => {
            if (t.bytes.len != n * (width / 256) * 144) return error.BadShape;
            const out = try a.alloc(BlockQ4Kx8, count);
            repackQ4KInto(t.bytes, width, n, out);
            return .{ .q4_k = out };
        },
        .q6_k => {
            if (t.bytes.len != n * (width / 256) * 210) return error.BadShape;
            const out = try a.alloc(BlockQ6Kx8, count);
            repackQ6KInto(t.bytes, width, n, out);
            return .{ .q6_k = out };
        },
        else => return null,
    }
}

// Preserve our per-row SDOT integer and floating reduction order across prompt rows.
fn projectTiles(comptime T: type, comptime M: usize, blocks: [*]const T, acts: [*]const sdot.Activation256, output: [*]f32, count: usize, bpr: usize, n: usize, rounded: bool, lo: usize, hi: usize) void {
    @setEvalBranchQuota(100000);
    for (lo..hi) |group| {
        var first: usize = 0;
        while (first < count) : (first += M) {
            const batch = @min(M, count - first);
            var totals: [M][8]f32 = @splat(@splat(0));
            for (0..bpr) |block| {
                const tile = &blocks[group * bpr + block];
                inline for (0..8) |row| {
                    const bytes: [*]const u8 = &tile.rows[row];
                    if (M > 1 and batch == M) {
                        const act = acts + first * bpr + block;
                        const values = if (T == BlockQ4Kx8)
                            block_batch.q4k(M, bytes, act, bpr)
                        else
                            block_batch.q6k(M, bytes, act, bpr);
                        inline for (0..M) |input_row| totals[input_row][row] += values[input_row];
                    } else inline for (0..M) |input_row| {
                        if (input_row < batch) {
                            const act = &acts[(first + input_row) * bpr + block];
                            totals[input_row][row] += if (T == BlockQ4Kx8)
                                sdot.q4kBlockDot(bytes, act)
                            else
                                sdot.q6kBlockDot(bytes, act);
                        }
                    }
                }
            }
            for (0..batch) |input_row| inline for (0..8) |row| {
                const value = totals[input_row][row];
                output[(first + input_row) * n + group * 8 + row] = if (rounded) rows.round(value) else value;
            };
        }
    }
}

pub fn projectRangeQ4Kx8(blocks: [*]const BlockQ4Kx8, raw: [*]const u8, acts: [*]const sdot.Activation256, output: [*]f32, count: usize, bpr: usize, n: usize, rounded: bool, lo: usize, hi: usize) void {
    if (count == 1) projectTiles(BlockQ4Kx8, 1, blocks, acts, output, count, bpr, n, rounded, lo, hi) else projectTiles(BlockQ4Kx8, 4, blocks, acts, output, count, bpr, n, rounded, lo, hi);
    if (hi == n / 8 and n % 8 != 0) sdot.projectRangeQ4K(raw, acts, output, count, bpr, n, rounded, n / 8 * 8, n);
}
pub fn projectRangeQ6Kx8(blocks: [*]const BlockQ6Kx8, raw: [*]const u8, acts: [*]const sdot.Activation256, output: [*]f32, count: usize, bpr: usize, n: usize, rounded: bool, lo: usize, hi: usize) void {
    if (count == 1) projectTiles(BlockQ6Kx8, 1, blocks, acts, output, count, bpr, n, rounded, lo, hi) else projectTiles(BlockQ6Kx8, 4, blocks, acts, output, count, bpr, n, rounded, lo, hi);
    if (hi == n / 8 and n % 8 != 0) sdot.projectRangeQ6K(raw, acts, output, count, bpr, n, rounded, n / 8 * 8, n);
}

pub fn projectBatch(a: std.mem.Allocator, rep: Repacked, t: g.Tensor, input: []const f32, output: []f32, count: usize, rounded: bool) !void {
    try sdot.checkTarget();
    if (t.rank != 2 or count == 0) return error.BadShape;
    const width: usize = @intCast(t.shape[0]);
    const n: usize = @intCast(t.shape[1]);
    if (width == 0 or width % 256 != 0 or n == 0 or input.len != count * width or output.len != count * n) return error.BadShape;
    const bpr = width / 256;
    var small: [8]sdot.Activation256 = undefined;
    const allocated = count * bpr > small.len;
    const acts = if (allocated) try a.alloc(sdot.Activation256, count * bpr) else small[0 .. count * bpr];
    defer if (allocated) a.free(acts);
    for (0..count) |row| try sdot.quantize256(input[row * width ..][0..width], acts[row * bpr ..][0..bpr], true);
    switch (rep) {
        .q4_k => |buf| projectRangeQ4Kx8(buf.ptr, t.bytes.ptr, acts.ptr, output.ptr, count, bpr, n, rounded, 0, n / 8),
        .q6_k => |buf| projectRangeQ6Kx8(buf.ptr, t.bytes.ptr, acts.ptr, output.ptr, count, bpr, n, rounded, 0, n / 8),
    }
}

fn expectRepackBitExact(kind: g.TensorKind, width: usize, n: usize, max_count: usize) !void {
    const blk = kind.block();
    const bytes = try std.testing.allocator.alloc(u8, n * (width / 256) * blk.bytes);
    defer std.testing.allocator.free(bytes);
    for (bytes, 0..) |*v, i| v.* = @intCast((i * 73 + 29) % 256);
    for (0..bytes.len / blk.bytes) |b| {
        const part = bytes[b * blk.bytes ..][0..blk.bytes];
        const at: usize = if (kind == .q6_k) 208 else 0;
        std.mem.writeInt(u16, part[at..][0..2], if (b % 3 == 0) 0x3500 else if (b % 3 == 1) 0xb400 else 0x3200, .little);
        if (kind == .q4_k) std.mem.writeInt(u16, part[2..4], if (b % 2 == 0) 0x3100 else 0x3000, .little);
    }
    const t = g.Tensor{ .name = "repack8", .kind = kind, .rank = 2, .shape = .{ width, n, 1, 1 }, .offset = 0, .len = bytes.len, .bytes = bytes };
    const rep = (try repack(std.testing.allocator, t)).?;
    defer rep.deinit(std.testing.allocator);
    const x = try std.testing.allocator.alloc(f32, max_count * width);
    defer std.testing.allocator.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 19) % 41)) - 20)) / 9.0;
    const want = try std.testing.allocator.alloc(f32, max_count * n);
    defer std.testing.allocator.free(want);
    const got = try std.testing.allocator.alloc(f32, max_count * n);
    defer std.testing.allocator.free(got);
    for ([_]bool{ false, true }) |rounded| for ([_]usize{ 1, max_count }) |count| {
        try sdot.projectBatch(std.testing.allocator, t, x[0 .. count * width], want[0 .. count * n], count, rounded);
        try projectBatch(std.testing.allocator, rep, t, x[0 .. count * width], got[0 .. count * n], count, rounded);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want[0 .. count * n]), std.mem.sliceAsBytes(got[0 .. count * n]));
    };
}

test "8-row repacked Q4_K and Q6_K SDOT match 1-row SDOT bit-for-bit including non-multiple-of-8 tails" {
    for ([_]g.TensorKind{ .q4_k, .q6_k }) |kind| for ([_]usize{ 8, 13, 24 }) |n| try expectRepackBitExact(kind, 512, n, 2);
}

test "Q6_K eight-row block tile matches 1-row SDOT bit-for-bit on Ling width 1536" {
    try std.testing.expectEqual(@as(usize, 1680), @sizeOf(BlockQ6Kx8));
    try expectRepackBitExact(.q6_k, 1536, 24, 4);
}

test "Q4_K eight-row block tile matches 1-row SDOT bit-for-bit on Ling expert width 1536" {
    try std.testing.expectEqual(@as(usize, 1152), @sizeOf(BlockQ4Kx8));
    try expectRepackBitExact(.q4_k, 1536, 24, 4);
}

test "eight-row tiles preserve complete GGUF blocks without decoded-weight expansion" {
    const a = std.testing.allocator;
    for ([_]g.TensorKind{ .q4_k, .q6_k }) |kind| {
        const width = 512;
        const n = 19;
        const size = kind.block().bytes;
        const raw = try a.alloc(u8, n * 2 * size);
        defer a.free(raw);
        for (raw, 0..) |*byte, i| byte.* = @truncate(i * 71 + i / 17);
        const t = g.Tensor{ .name = "tile", .kind = kind, .rank = 2, .shape = .{ width, n, 1, 1 }, .offset = 0, .len = raw.len, .bytes = raw };
        const tiled = (try repack(a, t)).?;
        defer tiled.deinit(a);
        for (0..2) |group| for (0..2) |block| for (0..8) |row| {
            const offset = ((group * 8 + row) * 2 + block) * size;
            const got: []const u8 = switch (tiled) {
                .q4_k => |tiles| &tiles[group * 2 + block].rows[row],
                .q6_k => |tiles| &tiles[group * 2 + block].rows[row],
            };
            try std.testing.expectEqualSlices(u8, raw[offset..][0..size], got);
        };
        const used = switch (tiled) {
            .q4_k => |tiles| tiles.len * @sizeOf(BlockQ4Kx8),
            .q6_k => |tiles| tiles.len * @sizeOf(BlockQ6Kx8),
        };
        try std.testing.expectEqual(16 * 2 * size, used);
    }
}

test "eight-row SDOT tiles retain the same error against an independent FP64 dot" {
    const reference = @import("cpu_gguf_rows.zig");
    const a = std.testing.allocator;
    for ([_]g.TensorKind{ .q4_k, .q6_k }) |kind| {
        const size = kind.block().bytes;
        const raw = try a.alloc(u8, 8 * size);
        defer a.free(raw);
        for (raw, 0..) |*byte, i| byte.* = @truncate(i * 37 + 19);
        for (0..8) |r| {
            const block = raw[r * size ..][0..size];
            const at: usize = if (kind == .q4_k) 0 else 208;
            std.mem.writeInt(u16, block[at..][0..2], 0x2800, .little);
            if (kind == .q4_k) std.mem.writeInt(u16, block[2..4], 0x2400, .little);
        }
        const t = g.Tensor{ .name = "f64", .kind = kind, .rank = 2, .shape = .{ 256, 8, 1, 1 }, .offset = 0, .len = raw.len, .bytes = raw };
        const tiled = (try repack(a, t)).?;
        defer tiled.deinit(a);
        var input: [256]f32 = undefined;
        for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 47)) - 23)) / 7;
        var act: [1]sdot.Activation256 = undefined;
        try sdot.quantize256(&input, &act, true);
        var actual: [8]f32 = undefined;
        var original: [8]f32 = undefined;
        try projectBatch(a, tiled, t, &input, &actual, 1, false);
        try sdot.project(a, t, &input, &original, false);
        for (0..8) |r| {
            var weights: [256]f32 = undefined;
            try reference.decodeBlock(kind, raw[r * size ..][0..size], &weights);
            var expected: f64 = 0;
            var magnitude: f64 = 0;
            for (weights, act[0].values) |w, q| {
                const term = @as(f64, w) * @as(f64, @floatFromInt(q)) * @as(f64, act[0].scale);
                expected += term;
                magnitude += @abs(term);
            }
            const err = @abs(@as(f64, actual[r]) - expected);
            try std.testing.expect(err <= @abs(@as(f64, original[r]) - expected));
            try std.testing.expect(err <= 0.00001 * @max(1, magnitude));
        }
    }
}
