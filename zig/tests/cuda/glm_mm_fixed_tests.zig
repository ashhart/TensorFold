//! Prompt K policies: each GPU row alone and batched, guarded tails, and exact dyadic FP64 references.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
pub const Gpu = check.Gpu;
const pool = 136;
const guard = 128;
const marker = 0xa5;
const counts = [_]usize{ 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129 };
fn bf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
fn value(u: u16) f64 {
    return @as(f32, @bitCast(@as(u32, u) << 16));
}
fn integer(bits: u8, col: usize, k: usize) i32 {
    const half: usize = if (bits == 16) 8 else @as(usize, 1) << @intCast(bits - 1);
    return @as(i32, @intCast((col * 17 + k * 13 + k / 64) % (2 * half))) - @as(i32, @intCast(half));
}
fn scale(col: usize, group: usize) f32 {
    return 1.0 / @as(f32, @floatFromInt(@as(u32, 1) << @as(u5, @intCast(4 + (col + group) % 3))));
}
fn pack(dst: []u32, codes: [16]u32, bits: u8) void {
    @memset(dst, 0);
    for (codes, 0..) |code, i| {
        if (bits == 6 and i == 15) {
            for (0..3) |j| dst[j] |= ((code >> @as(u5, @intCast(2 * j))) & 3) << 30;
        } else {
            const per: usize = 32 / bits;
            dst[i / per] |= code << @as(u5, @intCast((i % per) * bits));
        }
    }
}
fn guards(bytes: []const u8, used: usize) !void {
    for (bytes[0..guard]) |byte| try check.expect(byte == marker, "glm-mm-fixed: prefix guard", .{});
    for (bytes[guard + used ..]) |byte| try check.expect(byte == marker, "glm-mm-fixed: tail guard", .{});
}
fn reference(a: std.mem.Allocator, x: []const u16, w: []const u16, n: usize, k: usize, stride: usize, wide: bool) ![]u8 {
    const size: usize = if (wide) 4 else 2;
    const out = try a.alloc(u8, pool * n * size);
    errdefer a.free(out);
    for (0..pool) |row| for (0..n) |col| {
        var sum: f64 = 0;
        for (0..k) |j| sum += value(x[row * stride + j]) * value(w[col * k + j]);
        if (wide) std.mem.writeInt(u32, out[(row * n + col) * size ..][0..4], @bitCast(@as(f32, @floatCast(sum))), .little) else std.mem.writeInt(u16, out[(row * n + col) * size ..][0..2], bf(@floatCast(sum)), .little);
    };
    return out;
}
fn shape(gpu: check.Gpu, modules: *const glm.mm.Kernels, s: cuda.Stream, bits: u8, n: usize, k: usize) !void {
    const a = gpu.gpa;
    const stride = k + 8;
    const hw = try a.alloc(u16, n * k);
    defer a.free(hw);
    const hs = try a.alloc(u16, n * (k / 64));
    defer a.free(hs);
    const words = try a.alloc(u32, if (bits == 16) 4 else n * k * bits / 32);
    defer a.free(words);
    for (0..n) |col| {
        for (0..k / 64) |group| hs[col * (k / 64) + group] = @bitCast(@as(f16, @floatCast(scale(col, group))));
        for (0..k) |j| hw[col * k + j] = bf(@as(f32, @floatFromInt(integer(bits, col, j))) * scale(col, j / 64));
        if (bits != 16) for (0..k / 16) |chunk| {
            var codes: [16]u32 = undefined;
            for (&codes, 0..) |*code, i| code.* = @intCast(integer(bits, col, chunk * 16 + i) + (@as(i32, 1) << @as(u5, @intCast(bits - 1))));
            const at = (col * (k / 16) + chunk) * (bits / 2);
            pack(words[at..][0 .. bits / 2], codes, bits);
        };
    }
    const hx = try a.alloc(u16, pool * stride);
    defer a.free(hx);
    @memset(hx, 0x7fc0);
    for (0..pool) |row| for (0..k) |j| {
        const q = @as(i32, @intCast((row * 5 + j * 3 + 1) % 15)) - 7;
        hx[row * stride + j] = bf(@as(f32, @floatFromInt(q)) / 32);
    };
    var source = try cuda.DeviceBuffer.fromHost(gpu.d, if (bits == 16) std.mem.sliceAsBytes(hw) else std.mem.sliceAsBytes(words));
    defer source.free();
    var scales = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(hs));
    defer scales.free();
    var tiled = try cuda.DeviceBuffer.alloc(gpu.d, n * k * @as(usize, if (bits == 16) 2 else bits) / @as(usize, if (bits == 16) 1 else 8));
    defer tiled.free();
    var tiled_scales = try cuda.DeviceBuffer.alloc(gpu.d, hs.len * 2);
    defer tiled_scales.free();
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(hx));
    defer x.free();
    const tw: glm.mm.Tiled = .{ .w = tiled.ptr, .s = if (bits == 16) 0 else tiled_scales.ptr, .bits = bits, .n = n, .k = k };
    var part = try cuda.DeviceBuffer.alloc(gpu.d, @max(16, glm.mm.partBytes(pool, tw)));
    defer part.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, 2 * guard + pool * n * 4);
    defer out.free();
    var solo = try cuda.DeviceBuffer.alloc(gpu.d, out.len);
    defer solo.free();
    defer s.synchronize() catch @panic("GLM prompt gate retains device work until completion");
    if (bits == 16) try glm.mm.tile(modules, s, source.ptr, tiled.ptr, n, k) else try glm.mm.tilePacked(modules, s, bits, source.ptr, scales.ptr, tiled.ptr, tiled_scales.ptr, n, k);
    const got = try a.alloc(u8, out.len);
    defer a.free(got);
    const alone = try a.alloc(u8, solo.len);
    defer a.free(alone);
    for ([_]bool{ false, true }) |wide| {
        const size: usize = if (wide) 4 else 2;
        const want = try reference(a, hx, hw, n, k, stride, wide);
        defer a.free(want);
        for ([_]bool{ false, true }) |fixed| {
            var mk = modules.*;
            mk.prompt_fixed_k = fixed;
            try solo.fill8(marker, s.handle);
            for (0..pool) |row| try glm.mm.matmul(&mk, s, x.ptr + row * stride * 2, stride, tw, solo.ptr + guard + row * n * size, part.ptr, 1, wide, true);
            try s.synchronize();
            try solo.download(0, alone);
            try guards(alone, want.len);
            try check.sameBytes("glm-mm-fixed: solo against FP64", alone[guard..][0..want.len], want);
            for (counts) |rows| for ([_]usize{ 0, 1, 7 }) |offset| {
                try out.fill8(marker, s.handle);
                try glm.mm.matmul(&mk, s, x.ptr + offset * stride * 2, stride, tw, out.ptr + guard, part.ptr, rows, wide, true);
                try s.synchronize();
                try out.download(0, got);
                const bytes = rows * n * size;
                const at = offset * n * size;
                try guards(got, bytes);
                try check.sameBytes("glm-mm-fixed: batched against FP64", got[guard..][0..bytes], want[at..][0..bytes]);
                try check.sameBytes("glm-mm-fixed: batched against each row alone", got[guard..][0..bytes], alone[guard + at ..][0..bytes]);
            };
        }
    }
}
pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm_mm);
    defer module.unload();
    const kernels = try glm.mm.Kernels.load(module);
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_][2]usize{ .{ 64, 512 }, .{ 128, 2048 } }) |dims| try shape(gpu, &kernels, s, bits, dims[0], dims[1]);
    check.pass("glm-mm-fixed: both policies, bf16/4/6/8 bits, FP32/bf16 output, every solo/offset/bucket/tail cell", .{});
}
