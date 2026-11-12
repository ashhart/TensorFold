//! -Dnative-fp-decode: the hardware FP4/FP8 pair decode equals the software one on every input but NaN-byte pairs.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const pairs = 65536;
const bytes = 256;

/// An e4m3 pair whose either byte is a NaN code (0x7F, 0xFF): hardware NaN, software +-480.
fn nanPair(i: usize) bool {
    return i & 0x7F == 0x7F or (i >> 8) & 0x7F == 0x7F;
}

pub fn run(gpu: Gpu) !void {
    if (!cuda.kernels.available) return error.BuiltWithoutKernels;
    try pairs4(gpu);
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.prompt16);
    defer mod.unload();
    const f = mod.function("tf_native_fp_check") catch {
        check.pass("native-fp: skipped, built without -Dnative-fp-decode", .{});
        return;
    };
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, (pairs + bytes) * 3 * 4);
    defer out.free();
    var a: cuda.launch.Args = .{};
    a.add(out.ptr);
    try cuda.launch.launch(f, .{ .grid = .{ .x = (pairs + bytes) / 256 }, .block = .{ .x = 256 } }, stream, &a);
    try stream.synchronize();
    const raw = try check.download(gpu, out);
    defer gpu.gpa.free(raw);
    const w = std.mem.bytesAsSlice(u32, raw);
    if (w[0] == 0xFFFFFFFF and w[1] == 0xFFFFFFFF) {
        check.pass("native-fp: skipped, this GPU's image has no family-specific (sm_120a+) native decode", .{});
        return;
    }
    var nan: usize = 0;
    for (0..pairs) |i| {
        const hw = w[3 * i];
        if (nanPair(i)) {
            nan += 1;
            // the hardware's NaN half is 0x7FFF; the software's +-480 differs, the other half still matches
            try check.expect(hw != w[3 * i + 1], "native-fp: e4m3 pair {x:0>4} has a NaN byte yet matches", .{i});
            continue;
        }
        try check.expect(hw == w[3 * i + 1] and hw == w[3 * i + 2], "native-fp: e4m3 pair {x:0>4}: {x:0>8} vs {x:0>8} {x:0>8}", .{ i, hw, w[3 * i + 1], w[3 * i + 2] });
    }
    try check.expect(nan == 1020, "native-fp: {d} NaN-byte pairs, expected 1020", .{nan});
    for (0..bytes) |b| {
        const j = pairs + b;
        try check.expect(w[3 * j] == w[3 * j + 1], "native-fp: e2m1 byte {x:0>2}: {x:0>8} vs {x:0>8}", .{ b, w[3 * j], w[3 * j + 1] });
    }
    check.pass("native-fp: e2m1x2 == fp4pair on all 256 bytes; e4m3x2 == fp8pair on {d} finite pairs ({d} NaN-byte excluded)", .{ pairs - nan, nan });
}

/// qmmf's FP4 decode: e2m1_pairs equals fp4pair on all 2^32 words at both nibble offsets, scales included.
fn pairs4(gpu: Gpu) !void {
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.qmmf);
    defer mod.unload();
    const f = mod.function("tf_qmmf_native_fp4_check") catch {
        check.pass("native-fp qmmf: skipped, built without -Dnative-fp-decode", .{});
        return;
    };
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const init = [3]u64{ 0, std.math.maxInt(u64), 0 };
    var out = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&init));
    defer out.free();
    var a: cuda.launch.Args = .{};
    a.add(out.ptr);
    try cuda.launch.launch(f, .{ .grid = .{ .x = 4096 }, .block = .{ .x = 256 } }, stream, &a);
    try stream.synchronize();
    const raw = try check.download(gpu, out);
    defer gpu.gpa.free(raw);
    const r = std.mem.bytesAsSlice(u64, raw);
    if (r[0] == std.math.maxInt(u64) and r[2] == std.math.maxInt(u64)) {
        check.pass("native-fp qmmf: skipped, this GPU's image has no family-specific (sm_120a+) native decode", .{});
        return;
    }
    const want: u64 = (1 << 33) + 16 * 256;
    try check.expect(r[2] == want, "native-fp qmmf: {d} checks, expected {d}", .{ r[2], want });
    try check.expect(r[0] == 0, "native-fp qmmf: {d} mismatches, first {x}", .{ r[0], r[1] });
    check.pass("native-fp qmmf: e2m1_pairs == fp4pair on 2^32 words x 2 offsets, 16 codes x 256 scales", .{});
}
