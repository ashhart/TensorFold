//! glm_mm's width 12 against its bf16 tiles: every bucket, K slice, prompt block and escape gives the same bytes.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");

const pool = 136;

/// Width 12's raw blocks by its rule, on the host: an infinity or NaN, two weights under the window, or one 16+ under.
fn rawBlocks(w: []const u16) usize {
    var raw: usize = 0;
    var b: usize = 0;
    while (b < w.len) : (b += 64) {
        var top: i32 = 0;
        var special = false;
        for (w[b..][0..64]) |v| {
            const e: i32 = (v >> 7) & 0xFF;
            top = @max(top, e);
            special = special or e == 255;
        }
        const base = @max(top - 14, 1);
        var under: usize = 0;
        var depth: i32 = 0;
        for (w[b..][0..64]) |v| {
            const e: i32 = (v >> 7) & 0xFF;
            if (e != 0 and e < base) {
                under += 1;
                depth = base - e;
            }
        }
        if (special or under > 1 or (under == 1 and depth > 15)) raw += 1;
    }
    return raw;
}

/// Weights like a checkpoint's (a spread of exponents a block) with planted zeros, subnormals and every escape.
fn weights(a: std.mem.Allocator, n: usize, k: usize, seed: u64) !struct { w: []u16, raw: usize } {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const w = try a.alloc(u16, n * k);
    for (w) |*v| {
        const f = r.floatNorm(f32) * 0.02;
        v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
    }
    var raw: usize = 0;
    const blocks = n * (k / 64);
    for (0..blocks) |b| {
        const at = (b / (k / 64)) * k + (b % (k / 64)) * 64;
        const top: u16 = 0x3C00; // 2^-7's exponent field, 120
        switch (b % 9) {
            0 => w[at + 3] = 0x8000, // -0
            1 => w[at + 5] = 0x0011, // a subnormal
            2 => { // one weight under the window, 5 binades under its base: the slot's flag 1
                w[at] = top;
                w[at + 17] = (@as(u16, 120 - 14 - 5) << 7) | 0x2A;
            },
            3 => if (b % 27 == 3) { // one 20 binades under: raw
                w[at] = top;
                w[at + 40] = (@as(u16, 120 - 14 - 20) << 7) | 0x81 | 0x8000;
                raw += 1;
            },
            4 => if (b % 27 == 4) { // two under: raw
                w[at] = top;
                w[at + 1] = (@as(u16, 120 - 16) << 7) | 0x11;
                w[at + 63] = (@as(u16, 120 - 17) << 7) | 0x22;
                raw += 1;
            },
            5 => if (b % 27 == 5) { // tiny weights alone: the base floors at 1
                for (w[at..][0..64], 0..) |*v, i| v.* = @intCast((i % 13) << 7 | (i % 128));
            },
            6 => if (b % 54 == 6) {
                w[at + 9] = 0x7F80; // an infinity: raw
                raw += 1;
            },
            else => {},
        }
    }
    return .{ .w = w, .raw = raw };
}

fn shape(gpu: check.Gpu, modules: *const glm.mm.Kernels, s: cuda.Stream, n: usize, k: usize) !void {
    const a = gpu.gpa;
    const made = try weights(a, n, k, n * 31 + k);
    defer a.free(made.w);
    const hx = try a.alloc(u16, pool * k);
    defer a.free(hx);
    var prng = std.Random.DefaultPrng.init(k);
    for (hx) |*v| v.* = @truncate(@as(u32, @bitCast(prng.random().floatNorm(f32))) >> 16);
    var source = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(made.w));
    defer source.free();
    var t16 = try cuda.DeviceBuffer.alloc(gpu.d, n * k * 2);
    defer t16.free();
    var t12 = try cuda.DeviceBuffer.alloc(gpu.d, n * k * 2);
    defer t12.free();
    var count = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer count.free();
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(hx));
    defer x.free();
    defer s.synchronize() catch @panic("glm-mm-12 retains device work until completion");
    try count.fill8(0, s.handle);
    try glm.mm.tile(modules, s, source.ptr, t16.ptr, n, k);
    const sizes = glm.mm.packedBytes(n, k, 12);
    try glm.mm.tile12(modules, s, source.ptr, t12.ptr, count.ptr, n, k, (n * k * 2 - sizes[0] - sizes[1]) / glm.mm.raw_bytes);
    try s.synchronize();
    var raw: [1]i32 = undefined;
    try count.download(0, std.mem.sliceAsBytes(&raw));
    const want = rawBlocks(made.w);
    const room = (n * k * 2 - sizes[0] - sizes[1]) / glm.mm.raw_bytes;
    try check.expect(want <= room, "glm-mm-12: {d}x{d} plants {d} raw blocks, room for {d}", .{ n, k, want, room });
    try check.expect(raw[0] == want and want >= made.raw, "glm-mm-12: {d}x{d} kept {d} raw blocks, the rule {d}, planted {d}", .{ n, k, raw[0], want, made.raw });
    const w16: glm.mm.Tiled = .{ .w = t16.ptr, .n = n, .k = k };
    const w12: glm.mm.Tiled = .{ .w = t12.ptr, .s = t12.ptr + sizes[0], .bits = 12, .n = n, .k = k };
    var part = try cuda.DeviceBuffer.alloc(gpu.d, @max(16, glm.mm.partBytes(pool, w16)));
    defer part.free();
    var o16 = try cuda.DeviceBuffer.alloc(gpu.d, pool * n * 4);
    defer o16.free();
    var o12 = try cuda.DeviceBuffer.alloc(gpu.d, pool * n * 4);
    defer o12.free();
    const g16 = try a.alloc(u8, o16.len);
    defer a.free(g16);
    const g12 = try a.alloc(u8, o12.len);
    defer a.free(g12);
    for ([_]bool{ false, true }) |prompt| for ([_]bool{ false, true }) |wide| for ([_]usize{ 1, 2, 3, 5, 8, 16, 17, 33, 64, 65, 129 }) |rows| {
        try o16.fill8(0xa5, s.handle);
        try o12.fill8(0x5a, s.handle);
        try glm.mm.matmul(modules, s, x.ptr, k, w16, o16.ptr, part.ptr, rows, wide, prompt);
        try glm.mm.matmul(modules, s, x.ptr, k, w12, o12.ptr, part.ptr, rows, wide, prompt);
        try s.synchronize();
        try o16.download(0, g16);
        try o12.download(0, g12);
        const bytes = rows * n * @as(usize, if (wide) 4 else 2);
        try check.sameBytes("glm-mm-12: width 12 against bf16 tiles", g12[0..bytes], g16[0..bytes]);
    };
}

/// Width 12's shapes: one K slice, two and eight, decode buckets and prompt blocks.
pub fn run(gpu: check.Gpu, modules: *const glm.mm.Kernels, s: cuda.Stream) !void {
    for ([_][2]usize{ .{ 64, 512 }, .{ 128, 2048 }, .{ 832, 4096 }, .{ 12288, 512 } }) |dims| try shape(gpu, modules, s, dims[0], dims[1]);
    check.pass("glm-mm-12: width 12 equals the bf16 tiles' bytes at every bucket, slice, prompt block and escape", .{});
}
