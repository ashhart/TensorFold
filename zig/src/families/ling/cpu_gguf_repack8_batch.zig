//! Multi-row and worker-count regression tests for TensorFold's eight-row block tiles.
const std = @import("std");
const g = @import("core").gguf;
const act_mod = @import("cpu_gguf_sdot_act.zig");
const repack8 = @import("cpu_gguf_repack8.zig");
const Pool = @import("cpu_gguf_pool.zig").Pool;

fn fillQ4K(a: std.mem.Allocator, width: usize, n: usize, seed: u64) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const bpr = width / 256;
    const raw = try a.alloc(u8, n * bpr * 144);
    for (0..n * bpr) |bi| {
        const b = raw[bi * 144 ..][0..144];
        const d: f16 = @floatCast(0.01 + rng.float(f32) * 0.25);
        const m: f16 = @floatCast(0.005 + rng.float(f32) * 0.15);
        std.mem.writeInt(u16, b[0..2], @bitCast(d), .little);
        std.mem.writeInt(u16, b[2..4], @bitCast(m), .little);
        for (b[4..]) |*byte| byte.* = rng.int(u8);
    }
    return raw;
}

test "multi-row Q4_K eight-row tiles matches 1-row byte for byte across counts, rounding, and thread counts" {
    if (!act_mod.has_dotprod) return error.SkipZigTest;
    const a = std.testing.allocator;
    const width: usize = 512;
    const n: usize = 27;
    const raw = try fillQ4K(a, width, n, 0x9e3779b97f4a7c15);
    defer a.free(raw);
    const t = g.Tensor{ .name = "q4_batch", .kind = .q4_k, .rank = 2, .shape = .{ width, n, 1, 1 }, .offset = 0, .len = raw.len, .bytes = raw };
    const rep = (try repack8.repack(a, t)).?;
    defer a.free(rep.q4_k);

    var prng = std.Random.DefaultPrng.init(0x517cc1b727220a95);
    const rng = prng.random();
    const max_rows: usize = 8;
    const x_all = try a.alloc(f32, max_rows * width);
    defer a.free(x_all);
    for (x_all) |*v| v.* = (rng.float(f32) - 0.5) * 4.0;

    for ([_]bool{ false, true }) |rounded| {
        const ref = try a.alloc(f32, max_rows * n);
        defer a.free(ref);
        for (0..max_rows) |r| {
            try repack8.projectBatch(a, rep, t, x_all[r * width ..][0..width], ref[r * n ..][0..n], 1, rounded);
        }

        for ([_]usize{ 1, 2, 3, 4, 5, 7, 8 }) |count| {
            const got = try a.alloc(f32, count * n);
            defer a.free(got);
            try repack8.projectBatch(a, rep, t, x_all[0 .. count * width], got, count, rounded);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ref[0 .. count * n]), std.mem.sliceAsBytes(got));

            for ([_]usize{ 2, 3, 4, 8 }) |threads| {
                const pool = try Pool.create(a, std.testing.io, threads);
                defer pool.destroy();
                @memset(got, 0);
                try pool.projectSdotRepacked(a, rep, t, x_all[0 .. count * width], got, count, rounded);
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ref[0 .. count * n]), std.mem.sliceAsBytes(got));
            }
        }
    }
}

fn fillQ6K(a: std.mem.Allocator, width: usize, n: usize, seed: u64) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const bpr = width / 256;
    const raw = try a.alloc(u8, n * bpr * 210);
    for (0..n * bpr) |bi| {
        const b = raw[bi * 210 ..][0..210];
        for (b[0..208]) |*byte| byte.* = rng.int(u8);
        const d: f16 = @floatCast(0.01 + rng.float(f32) * 0.25);
        std.mem.writeInt(u16, b[208..210], @bitCast(d), .little);
    }
    return raw;
}

test "multi-row Q6_K eight-row tiles matches 1-row byte for byte across counts, rounding, and thread counts" {
    if (!act_mod.has_dotprod) return error.SkipZigTest;
    const a = std.testing.allocator;
    const width: usize = 512;
    const n: usize = 27;
    const raw = try fillQ6K(a, width, n, 0xd1b54a32d192ed03);
    defer a.free(raw);
    const t = g.Tensor{ .name = "q6_batch", .kind = .q6_k, .rank = 2, .shape = .{ width, n, 1, 1 }, .offset = 0, .len = raw.len, .bytes = raw };
    const rep = (try repack8.repack(a, t)).?;
    defer a.free(rep.q6_k);

    var prng = std.Random.DefaultPrng.init(0x94d049bb133111eb);
    const rng = prng.random();
    const max_rows: usize = 8;
    const x_all = try a.alloc(f32, max_rows * width);
    defer a.free(x_all);
    for (x_all) |*v| v.* = (rng.float(f32) - 0.5) * 4.0;

    for ([_]bool{ false, true }) |rounded| {
        const ref = try a.alloc(f32, max_rows * n);
        defer a.free(ref);
        for (0..max_rows) |r| {
            try repack8.projectBatch(a, rep, t, x_all[r * width ..][0..width], ref[r * n ..][0..n], 1, rounded);
        }

        for ([_]usize{ 1, 2, 3, 4, 5, 7, 8 }) |count| {
            const got = try a.alloc(f32, count * n);
            defer a.free(got);
            try repack8.projectBatch(a, rep, t, x_all[0 .. count * width], got, count, rounded);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ref[0 .. count * n]), std.mem.sliceAsBytes(got));

            for ([_]usize{ 2, 3, 4, 8 }) |threads| {
                const pool = try Pool.create(a, std.testing.io, threads);
                defer pool.destroy();
                @memset(got, 0);
                try pool.projectSdotRepacked(a, rep, t, x_all[0 .. count * width], got, count, rounded);
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ref[0 .. count * n]), std.mem.sliceAsBytes(got));
            }
        }
    }
}
