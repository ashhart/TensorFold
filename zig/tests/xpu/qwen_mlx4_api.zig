//! qwen_mlx4.zig wrapper: .exact rows bit-identical to the single-row kernel; .fast self-consistent in m.

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const mx = @import("qwen_xpu").mlx4;

const alloc = std.heap.page_allocator;

fn toBf(f: f32) u16 {
    const u: u32 = @bitCast(f);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

fn upload(r: *rt.Runtime, bytes: []const u8) !rt.Buffer {
    const b = try r.alloc(@max(bytes.len, 16));
    try r.upload(b, bytes);
    try r.sync();
    return b;
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    var r = try rt.open();
    defer r.deinit();
    const in: u32 = 5120;
    const rows: u32 = 4096;
    var mr = try mx.Rows.init(&r, in, rows);
    defer mr.deinit();
    var rng = std.Random.DefaultPrng.init(3);
    const rnd = rng.random();
    const w = try alloc.alloc(u32, @as(usize, rows) * in / 8);
    rnd.bytes(std.mem.sliceAsBytes(w));
    const sc = try alloc.alloc(u16, @as(usize, rows) * in / 64);
    const bi = try alloc.alloc(u16, sc.len);
    for (sc, bi) |*s, *b| {
        s.* = toBf((rnd.float(f32) - 0.5) * 0.02);
        b.* = toBf((rnd.float(f32) - 0.5) * 0.2);
    }
    const x = try alloc.alloc(u16, 16 * in);
    for (x) |*v| v.* = toBf(rnd.floatNorm(f32));
    var bw = try upload(&r, std.mem.sliceAsBytes(w));
    var bs = try upload(&r, std.mem.sliceAsBytes(sc));
    var bb = try upload(&r, std.mem.sliceAsBytes(bi));
    var bx = try upload(&r, std.mem.sliceAsBytes(x));
    var by = try r.alloc(16 * rows * 2);
    const ref = try alloc.alloc(u16, 16 * rows);
    const got = try alloc.alloc(u16, 16 * rows);
    var bad_exact: usize = 0;
    var bad_fast: usize = 0;
    var max_rel: f64 = 0;
    // each row alone through the exact m = 1 path
    for (0..16) |i| {
        const xi: rt.Buffer = .{ .rt = bx.rt, .ptr = @ptrFromInt(@intFromPtr(bx.ptr.?) + i * in * 2), .len = bx.len - i * in * 2 };
        try mr.matvec(.exact, bw, bs, bb, xi, by, in, rows, 1, @intCast(i * rows), false);
    }
    try r.download(std.mem.sliceAsBytes(ref), by);
    try r.sync();
    var fast_ref: [16][]u16 = undefined;
    for (0..16) |i| fast_ref[i] = try alloc.alloc(u16, rows);
    // fast: each row alone first (m = 1)
    for (0..16) |i| {
        const xi: rt.Buffer = .{ .rt = bx.rt, .ptr = @ptrFromInt(@intFromPtr(bx.ptr.?) + i * in * 2), .len = bx.len - i * in * 2 };
        try mr.matvec(.fast, bw, bs, bb, xi, by, in, rows, 1, 0, false);
        try r.download(std.mem.sliceAsBytes(fast_ref[i]), by);
        try r.sync();
    }
    for (1..17) |mm| {
        try mr.matvec(.exact, bw, bs, bb, bx, by, in, rows, @intCast(mm), 0, false);
        try r.download(std.mem.sliceAsBytes(got), by);
        try r.sync();
        for (0..mm) |i| bad_exact += @intFromBool(!std.mem.eql(u16, got[i * rows ..][0..rows], ref[i * rows ..][0..rows]));
        try mr.matvec(.fast, bw, bs, bb, bx, by, in, rows, @intCast(mm), 0, false);
        try r.download(std.mem.sliceAsBytes(got), by);
        try r.sync();
        for (0..mm) |i| {
            bad_fast += @intFromBool(!std.mem.eql(u16, got[i * rows ..][0..rows], fast_ref[i]));
            for (got[i * rows ..][0..rows], ref[i * rows ..][0..rows]) |a, b| max_rel = @max(max_rel, @abs(@as(f64, bf(a)) - bf(b)) / @max(@abs(@as(f64, bf(b))), 1.0));
        }
    }
    // gate + up + SwiGLU: m rows bit-identical to the single-row kernel on each row
    var bw2 = try upload(&r, std.mem.sliceAsBytes(w[0 .. w.len / 2]));
    var bs2 = try upload(&r, std.mem.sliceAsBytes(sc[0 .. sc.len / 2]));
    var bb2 = try upload(&r, std.mem.sliceAsBytes(bi[0 .. bi.len / 2]));
    var bu = try upload(&r, std.mem.sliceAsBytes(w[w.len / 2 ..]));
    var bsu = try upload(&r, std.mem.sliceAsBytes(sc[sc.len / 2 ..]));
    var bbu = try upload(&r, std.mem.sliceAsBytes(bi[bi.len / 2 ..]));
    const grows: u32 = rows / 2;
    var bact = try r.alloc(16 * grows * 2);
    const gref = try alloc.alloc(u16, 16 * grows);
    for (0..16) |i| {
        const xi: rt.Buffer = .{ .rt = bx.rt, .ptr = @ptrFromInt(@intFromPtr(bx.ptr.?) + i * in * 2), .len = bx.len - i * in * 2 };
        const ai: rt.Buffer = .{ .rt = bact.rt, .ptr = @ptrFromInt(@intFromPtr(bact.ptr.?) + i * grows * 2), .len = bact.len - i * grows * 2 };
        try mr.gateUp(bw2, bs2, bb2, bu, bsu, bbu, xi, ai, in, grows, 1);
    }
    try r.download(std.mem.sliceAsBytes(gref), bact);
    try r.sync();
    var bad_gu: usize = 0;
    const ggot = try alloc.alloc(u16, 16 * grows);
    for (1..17) |mm| {
        try mr.gateUp(bw2, bs2, bb2, bu, bsu, bbu, bx, bact, in, grows, @intCast(mm));
        try r.download(std.mem.sliceAsBytes(ggot), bact);
        try r.sync();
        for (0..mm) |i| bad_gu += @intFromBool(!std.mem.eql(u16, ggot[i * grows ..][0..grows], gref[i * grows ..][0..grows]));
    }
    bw2.free();
    bs2.free();
    bb2.free();
    bu.free();
    bsu.free();
    bbu.free();
    bact.free();
    bw.free();
    bs.free();
    bb.free();
    bx.free();
    by.free();
    std.debug.print("exact: {d} windows differ from the single row; gate+up+SwiGLU: {d}; fast: {d} rows differ between windows; fast vs exact max relative difference {e:.2}\n", .{ bad_exact, bad_gu, bad_fast, max_rel });
    if (bad_exact != 0 or bad_gu != 0 or bad_fast != 0 or max_rel > 0.02) return error.Failed;
    std.debug.print("qwen_mlx4 api OK\n", .{});
}
