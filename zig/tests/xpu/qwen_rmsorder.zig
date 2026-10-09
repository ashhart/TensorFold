//! The 512-item RMSNorm kernels must equal the 64-item rmsnorm bit for bit. usage: xpu-qwen_rmsorder-test

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");

const spv = @import("xpu").kernels.qwen_basic;
const rows: u32 = 64;
const n: u32 = 5120;

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    var r = try rt.open();
    defer r.deinit();
    var m = try r.module(spv);
    var old = try m.kernel("rmsnorm", .{ 64, 1, 1 });
    var new = try m.kernel("rmsnorm512", .{ 512, 1, 1 });
    var fused = try m.kernel("add_rmsnorm", .{ 512, 1, 1 });
    var add = try m.kernel("add_bf16", .{ 64, 1, 1 });
    const gpa = std.heap.page_allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    const x = try gpa.alloc(u16, rows * n);
    const d = try gpa.alloc(u16, rows * n);
    const w = try gpa.alloc(u16, n);
    const toBf = struct {
        fn f(v: f32) u16 {
            return @intCast(@as(u32, @bitCast(v)) >> 16);
        }
    }.f;
    for (x) |*v| v.* = toBf(rnd.floatNorm(f32) * 3);
    for (d) |*v| v.* = toBf(rnd.floatNorm(f32) * 0.5);
    for (w) |*v| v.* = toBf(1.0 + rnd.floatNorm(f32) * 0.1);
    const bx = try r.alloc(x.len * 2);
    const bd = try r.alloc(x.len * 2);
    const bw = try r.alloc(w.len * 2);
    const y0 = try r.alloc(x.len * 2);
    const y1 = try r.alloc(x.len * 2);
    const x2 = try r.alloc(x.len * 2);
    const y2 = try r.alloc(x.len * 2);
    try r.upload(bx, std.mem.sliceAsBytes(x));
    try r.upload(bd, std.mem.sliceAsBytes(d));
    try r.upload(bw, std.mem.sliceAsBytes(w));
    try r.upload(x2, std.mem.sliceAsBytes(x));
    try r.sync();
    for (0..rows) |row| {
        const off = row * n * 2;
        const sx = @import("xpu").exl3.at(bx, off);
        const sy0 = @import("xpu").exl3.at(y0, off);
        const sy1 = @import("xpu").exl3.at(y1, off);
        try old.setBuffer(0, sx);
        try old.setBuffer(1, bw);
        try old.setBuffer(2, sy0);
        try old.setU32(3, n);
        try old.setF32(4, 1e-6);
        try old.launch(.{ 1, 1, 1 });
        try new.setBuffer(0, sx);
        try new.setBuffer(1, bw);
        try new.setBuffer(2, sy1);
        try new.setU32(3, n);
        try new.setF32(4, 1e-6);
        try new.launch(.{ 1, 1, 1 });
    }
    // fused: x2 = bf16(x + d), y2 = norm(x2), against add_bf16 then the old rmsnorm
    try add.setBuffer(0, bx);
    try add.setBuffer(1, bd);
    try add.setU32(2, rows * n);
    try add.launch(.{ rows * n / 64, 1, 1 });
    const y3 = try r.alloc(x.len * 2);
    for (0..rows) |row| {
        const off = row * n * 2;
        const e = @import("xpu").exl3;
        try old.setBuffer(0, e.at(bx, off));
        try old.setBuffer(1, bw);
        try old.setBuffer(2, e.at(y3, off));
        try old.setU32(3, n);
        try old.setF32(4, 1e-6);
        try old.launch(.{ 1, 1, 1 });
        try fused.setBuffer(0, e.at(x2, off));
        try fused.setBuffer(1, e.at(bd, off));
        try fused.setBuffer(2, bw);
        try fused.setBuffer(3, e.at(y2, off));
        try fused.setU32(4, n);
        try fused.setF32(5, 1e-6);
        try fused.launch(.{ 1, 1, 1 });
    }
    const a = try gpa.alloc(u16, x.len);
    const b = try gpa.alloc(u16, x.len);
    var bad: usize = 0;
    try r.download(std.mem.sliceAsBytes(a), y0);
    try r.download(std.mem.sliceAsBytes(b), y1);
    try r.sync();
    for (a, b) |p, q| bad += @intFromBool(p != q);
    std.debug.print("rmsnorm512 vs rmsnorm: {d} of {d} values differ\n", .{ bad, a.len });
    var bad2: usize = 0;
    try r.download(std.mem.sliceAsBytes(a), y3);
    try r.download(std.mem.sliceAsBytes(b), y2);
    try r.sync();
    for (a, b) |p, q| bad2 += @intFromBool(p != q);
    std.debug.print("add_rmsnorm vs add_bf16 + rmsnorm: {d} of {d} values differ\n", .{ bad2, a.len });
    if (bad + bad2 != 0) return error.Differ;
    std.debug.print("rmsorder ok\n", .{});
}
