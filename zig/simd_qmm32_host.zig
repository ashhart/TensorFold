//! Actual projection launchers against a recording facade, including 32-row tails and strided mixed members.
const std = @import("std");
const row = @import("projection");
const mtl = @import("metal");
const t = std.testing;
const sq = row.simd;
const bits = row.simd_bits;

fn covered(groups: mtl.Size, nt: usize, rt: usize, n: usize, rows: usize) !void {
    // Independent rectangle coverage: tile edges must cover each valid element once, never omit a tail.
    const hits = try t.allocator.alloc(u8, n * rows);
    defer t.allocator.free(hits);
    @memset(hits, 0);
    for (0..groups.height) |y| for (0..groups.width) |x| {
        for (0..8 * rt) |r| for (0..8 * nt) |c| {
            const rr = y * 8 * rt + r;
            const cc = x * 8 * nt + c;
            if (rr < rows and cc < n) hits[rr * n + cc] += 1;
        };
    };
    for (hits) |count| try t.expectEqual(@as(u8, 1), count);
}

test "simd32: real contiguous and strided calls read weights for 32 rows within the launch cap" {
    mtl.limit = 1024;
    var p = try sq.Pipelines.load(.{});
    defer p.deinit();
    var b = try bits.Pipelines.load(.{});
    defer b.deinit();
    for ([_]usize{ 48, 64, 264, 272, 5120, 6408, 6416, 6432, 17408, 248320 }) |n| {
        for ([_]usize{ 17, 31, 32, 33, 63, 64, 65 }) |rows| {
            const cl = try sq.call(&p, n, 192, rows, false, true, 0);
            const st = try bits.call(&b, 4, n, 192, rows, n + 8, false, 0);
            try t.expect(std.mem.endsWith(u8, cl.pipeline.symbol(), "_4"));
            try t.expect(std.mem.endsWith(u8, st.pipeline.symbol(), "_4"));
            try t.expectEqual((rows + 31) / 32, cl.groups.height);
            try t.expectEqual(cl.groups, st.groups);
            const s = sq.splits(n);
            const nt = sq.tiles(n, 4, s);
            try t.expect(s * 4 * nt * 64 * 4 <= 32768);
            try t.expectEqual(@as(i32, @intCast(n + 8)), st.args.ld);
            if (n < 7000) try covered(cl.groups, nt, 4, n, rows);
        }
    }
}

test "simd32: old widths and other bit formats keep their original tile height and resource bound" {
    mtl.limit = 1024;
    var p = try sq.Pipelines.load(.{});
    defer p.deinit();
    var b = try bits.Pipelines.load(.{});
    defer b.deinit();
    for ([_]usize{ 48, 5120, 17408 }) |n| {
        for (1..17) |rows| {
            const cl = try sq.call(&p, n, 5120, rows, false, false, 0);
            const rt: u32 = @intCast(@min(2, (rows + 7) / 8));
            try t.expectEqual(rt, sq.rowTiles(rows));
            try t.expect(sq.splits(n) * rt * sq.tiles(n, rt, sq.splits(n)) * 64 * 4 <= 16384);
            try t.expectEqual((rows + 8 * rt - 1) / (8 * rt), cl.groups.height);
        }
        for (bits.bit_widths) |width| {
            if (width == 4) continue;
            const cl = try bits.call(&b, width, n, 5120, 32, n, false, 0);
            try t.expect(std.mem.endsWith(u8, cl.pipeline.symbol(), "_2"));
            try t.expectEqual(@as(usize, 2), cl.groups.height);
        }
        const one = try sq.call(&p, n, 5120, 1, true, false, 0);
        try t.expect(std.mem.indexOf(u8, one.pipeline.symbol(), "scalar") != null);
    }
}

test "simd32: threads adapt without changing K chunks and strided encode keeps every offset" {
    mtl.limit = 224;
    defer mtl.limit = 1024;
    var p = try sq.Pipelines.load(.{});
    defer p.deinit();
    var b = try bits.Pipelines.load(.{});
    defer b.deinit();
    const cl = try sq.call(&p, 5120, 5120, 32, false, false, 10240);
    const st = try bits.call(&b, 4, 5120, 5120, 32, 10496, false, 10240);
    try t.expectEqual(@as(usize, 224), cl.threads);
    try t.expectEqual(cl.threads, st.threads);
    try t.expectEqual(@as(u32, 8), sq.chunks(5120, 10240));
    var record: mtl.Record = .{};
    bits.encode(.{ .record = &record }, st, .{ .w = .{ .id = 1 }, .w_off = 64, .scales = .{ .id = 2 }, .s_off = 128, .biases = .{ .id = 3 }, .b_off = 192 }, .{ .id = 4 }, 256, .{ .id = 5 }, 512);
    try t.expectEqual(@as(usize, 10496), record.ld);
    try t.expectEqual(@as(usize, 32), record.rows);
    try t.expectEqual([6]usize{ 256, 64, 128, 192, 0, 512 }, record.offsets);
    try t.expectEqual(st.groups, record.groups);
    try t.expectEqual(st.threads, record.threads);
}

test "simd32: the actual source control couples only rows past the old tile height" {
    const control = @import("tests/contract/metal_simd32_controls.zig");
    const sources = @import("kernel_sources");
    for ([_][]const u8{ sources.core_simd_qmm, sources.core_simd_qmm_bits }) |text| {
        const bad = try control.source(t.allocator, text);
        defer t.allocator.free(bad);
        try t.expect(std.mem.indexOf(u8, bad, "xr0[rt] = min(rb + 8 * (rt % 2) + fn") != null);
    }
    try t.expect(std.mem.indexOf(u8, control.frozen, "TF_SQ_MMA(8, 4, 4)") == null);
    try t.expect(std.mem.indexOf(u8, control.frozen_strided, "TF_SQW(4, 8, 4, 4)") == null);
}
