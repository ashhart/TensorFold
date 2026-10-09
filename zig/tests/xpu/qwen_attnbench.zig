//! Decode attention kernels at long context on random bf16 K/V, time and GB/s. usage: xpu-qwen_attnbench-test

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const stop = @import("xpu").stop;

const spv = @import("xpu").kernels.qwen_attn;
const kv_dim: usize = 4 * 256;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    stop.install();
    var r = try rt.open();
    defer r.deinit();
    var m = try r.module(spv);
    var part = try m.kernel("attn_partial", .{ 128, 1, 1 });
    var merge = try m.kernel("attn_merge", .{ 256, 1, 1 });
    const max_len: usize = 8192;
    const gpa = std.heap.page_allocator;
    const host = try gpa.alloc(u16, max_len * kv_dim);
    var prng = std.Random.DefaultPrng.init(7);
    for (host) |*v| v.* = @as(u16, @truncate(prng.random().int(u32))) & 0x3fff | 0x3c00 & 0x3f80; // small finite bf16 values
    const kc = try r.alloc(max_len * kv_dim * 2);
    const vc = try r.alloc(max_len * kv_dim * 2);
    try r.upload(kc, std.mem.sliceAsBytes(host));
    try r.upload(vc, std.mem.sliceAsBytes(host));
    const q = try r.alloc(24 * 256 * 2);
    try r.upload(q, std.mem.sliceAsBytes(host[0 .. 24 * 256]));
    const qg = try r.alloc(24 * 512 * 2);
    try r.upload(qg, std.mem.sliceAsBytes(host[0 .. 24 * 512]));
    const po = try r.alloc(128 * 24 * 256 * 4);
    const pm = try r.alloc(128 * 24 * 4);
    const pl = try r.alloc(128 * 24 * 4);
    const att = try r.alloc(24 * 256 * 2);
    try r.sync();
    for ([_]u32{ 128, 512, 1024, 2048, 4096, 8192 }) |len| {
        const nch = (len + 63) / 64;
        var best: u64 = std.math.maxInt(u64);
        for (0..5) |_| {
            const t0 = nowNs();
            for (0..16) |_| {
                try part.setBuffer(0, q);
                try part.setBuffer(1, kc);
                try part.setBuffer(2, vc);
                try part.setBuffer(3, po);
                try part.setBuffer(4, pm);
                try part.setBuffer(5, pl);
                try part.setU32(6, len);
                try part.setF32(7, 0.0625);
                try part.launch(.{ 4, nch, 1 });
                try merge.setBuffer(0, po);
                try merge.setBuffer(1, pm);
                try merge.setBuffer(2, pl);
                try merge.setBuffer(3, qg);
                try merge.setBuffer(4, att);
                try merge.setU32(5, len);
                try merge.launch(.{ 24, 1, 1 });
            }
            try r.sync();
            best = @min(best, nowNs() - t0);
        }
        const us = @as(f64, @floatFromInt(best)) / 1e3 / 16.0;
        const bytes = @as(f64, @floatFromInt(len)) * kv_dim * 2 * 2;
        std.debug.print("len {d:>5}: {d:>8.1} us per layer ({d:>6.1} GB/s of K+V, {d:>6.2} ms per token over 16 layers)\n", .{ len, us, bytes / us / 1e3, us * 16 / 1e3 });
    }
}
