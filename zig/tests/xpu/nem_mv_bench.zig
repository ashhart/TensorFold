//! Micro-benchmark of a qmv4_bf16-shaped 4-bit matvec kernel from a SPIR-V file over ROWS rows; GB/s of weights.
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
    if (args.len < 5) return error.Usage;
    const rows: u32 = try std.fmt.parseInt(u32, args[3], 10);
    const in: u32 = try std.fmt.parseInt(u32, args[4], 10);
    const rpg: u32 = if (args.len > 5) try std.fmt.parseInt(u32, args[5], 10) else 1;
    const local: u32 = if (args.len > 6) try std.fmt.parseInt(u32, args[6], 10) else 16;
    const spv = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], init.gpa, .unlimited);
    defer init.gpa.free(spv);
    const name = try init.gpa.dupeSentinel(u8, args[2], 0);
    defer init.gpa.free(name);
    var r = try rt.open();
    defer r.deinit();
    var m = try r.module(spv);
    var k = try m.kernel(name.ptr, .{ local, 1, 1 });
    // several distinct matrices, round robin, so that the weights come from memory and not from the L2
    const nm: usize = 8;
    const wb = @as(usize, rows) * in / 2;
    const sb = @as(usize, rows) * in / 64 * 2;
    var w: [8]rt.Buffer = undefined;
    var s: [8]rt.Buffer = undefined;
    var b: [8]rt.Buffer = undefined;
    const fill = try init.gpa.alloc(u8, wb);
    defer init.gpa.free(fill);
    if (std.c.getenv("BENCH_ZEROS") != null) @memset(fill, 0x5b) else {
        var prng = std.Random.DefaultPrng.init(7);
        prng.random().bytes(fill);
    } // random weights, scales, biases and activations (constant data runs faster)
    for (0..nm) |i| {
        w[i] = try r.alloc(wb);
        s[i] = try r.alloc(sb);
        b[i] = try r.alloc(sb);
        try r.upload(w[i], fill);
        try r.upload(s[i], fill[0..sb]);
        try r.upload(b[i], fill[0..sb]);
        try r.sync();
    }
    const x = try r.alloc(@as(usize, in) * 2);
    const y = try r.alloc(@as(usize, rows) * 2);
    try r.upload(x, fill[0 .. @as(usize, in) * 2]);
    try r.sync();
    const count: u32 = 400;
    var best: u64 = std.math.maxInt(u64);
    for (0..4) |_| {
        const t0 = nowNs();
        for (0..count) |c| {
            const i = c % nm;
            try k.setBuffer(0, w[i]);
            try k.setBuffer(1, s[i]);
            try k.setBuffer(2, b[i]);
            try k.setBuffer(3, x);
            try k.setBuffer(4, y);
            try k.setU32(5, in);
            try k.launch(.{ rows / rpg, 1, 1 });
        }
        try r.sync();
        best = @min(best, nowNs() - t0);
    }
    const us = @as(f64, @floatFromInt(best)) / 1e3 / @as(f64, @floatFromInt(count));
    std.debug.print("{s}: {d} rows x {d}: {d:.1} us/launch, {d:.0} GB/s (weights+scales+biases {d:.1} MB)\n", .{ args[2], rows, in, us, @as(f64, @floatFromInt(wb + 2 * sb)) / 1e3 / us, @as(f64, @floatFromInt(wb + 2 * sb)) / 1e6 });
}
