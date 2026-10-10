//! Micro-benchmark of an add_rmsnorm / rmsnorm SPIR-V kernel: back-to-back launches over `rows` rows of n = 2688.
const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const stop = @import("xpu").stop;

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn main(init: std.process.Init) !void {
    stop.install();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return error.Usage;
    const rows: u32 = if (args.len > 3) try std.fmt.parseInt(u32, args[3], 10) else 1;
    const local: u32 = if (args.len > 4) try std.fmt.parseInt(u32, args[4], 10) else 64;
    const n: u32 = 2688;
    const io = init.io;
    const spv = try std.Io.Dir.cwd().readFileAlloc(io, args[1], init.gpa, .unlimited);
    defer init.gpa.free(spv);
    const name = try init.gpa.dupeSentinel(u8, args[2], 0);
    defer init.gpa.free(name);
    var r = try rt.open();
    defer r.deinit();
    var m = try r.module(spv);
    var k = try m.kernel(name.ptr, .{ local, 1, 1 });
    const x = try r.alloc(@as(usize, rows) * n * 2);
    const d = try r.alloc(@as(usize, rows) * n * 2);
    const w = try r.alloc(n * 2);
    const y = try r.alloc(@as(usize, rows) * n * 2);
    const zeros = try init.gpa.alloc(u8, @as(usize, rows) * n * 2);
    defer init.gpa.free(zeros);
    if (std.c.getenv("BENCH_ZEROS") != null) {
        @memset(zeros, 0x3f);
    } else { // random bf16 values of magnitude 2^-7 .. 2^-4 (random bit patterns would hit inf and nan)
        var prng = std.Random.DefaultPrng.init(7);
        prng.random().bytes(zeros);
        for (0..zeros.len / 2) |i| zeros[2 * i + 1] = (zeros[2 * i + 1] & 0x8f) | 0x3c;
    }
    try r.upload(x, zeros);
    try r.upload(d, zeros);
    try r.upload(w, zeros[0 .. n * 2]);
    try r.sync();
    const add = std.mem.startsWith(u8, args[2], "add");
    try k.setBuffer(0, x);
    if (add) {
        try k.setBuffer(1, d);
        try k.setBuffer(2, w);
        try k.setBuffer(3, y);
        try k.setU32(4, n);
        try k.setF32(5, 1e-5);
    } else {
        try k.setBuffer(1, w);
        try k.setBuffer(2, y);
        try k.setU32(3, n);
        try k.setF32(4, 1e-5);
    }
    const count: u32 = 2000;
    var best: u64 = std.math.maxInt(u64);
    for (0..5) |_| {
        const t0 = nowNs();
        for (0..count) |_| try k.launch(.{ rows, 1, 1 });
        try r.sync();
        best = @min(best, nowNs() - t0);
    }
    std.debug.print("{s}: {d} rows, local {d}: {d:.2} us/launch (queued)\n", .{ args[2], rows, local, @as(f64, @floatFromInt(best)) / 1e3 / @as(f64, @floatFromInt(count)) });
}
