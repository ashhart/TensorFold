//! Delta-rule window kernels of qwen_rows.cl: gdn_step_r2 vs r2x2 and r2x4 (2 and 4 value rows a sub-group), bitwise.

const std = @import("std");
const rt = @import("rig.zig");
const stop = @import("xpu").stop;

const spv_rows = @import("xpu").kernels.qwen_rows;
const gpa = std.heap.page_allocator;
const VH = 48;
const DV = 128;
const DK = 128;
const KH = 16;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn bfBits(f: f32) u16 {
    const u: u32 = @bitCast(f);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    stop.install();
    var r = try rt.open();
    defer r.deinit();
    var m = try r.module(spv_rows);
    var kernels = [_]rt.Kernel{
        try m.kernel("gdn_step_r2", .{ 64, 1, 1 }),
        try m.kernel("gdn_step_r2x2", .{ 64, 1, 1 }),
        try m.kernel("gdn_step_r2x4", .{ 64, 1, 1 }),
    };
    const nr = [_]u32{ 1, 2, 4 };
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    const max_rows: usize = 2048;
    const qn = try gpa.alloc(f32, max_rows * KH * DK);
    const kn = try gpa.alloc(f32, max_rows * KH * DK);
    for (qn, kn) |*a, *b| {
        a.* = rnd.floatNorm(f32) * 0.09; // unit-ish rows of 128: q scale 1/128, k scale 1/sqrt(128) as the real prep does
        b.* = rnd.floatNorm(f32) * 0.09;
    }
    const vc = try gpa.alloc(u16, max_rows * VH * DV);
    for (vc) |*v| v.* = bfBits(rnd.floatNorm(f32));
    const beta = try gpa.alloc(f32, max_rows * VH);
    const gg = try gpa.alloc(f32, max_rows * VH);
    for (beta, gg) |*b, *g| {
        b.* = 0.05 + 0.9 * rnd.float(f32);
        g.* = 0.6 + 0.4 * rnd.float(f32);
    }
    const st = try gpa.alloc(f32, VH * DV * DK);
    for (st) |*s| s.* = rnd.floatNorm(f32) * 0.05;
    const dqn = try r.alloc(qn.len * 4);
    const dkn = try r.alloc(kn.len * 4);
    const dvc = try r.alloc(vc.len * 2);
    const dbeta = try r.alloc(beta.len * 4);
    const dg = try r.alloc(gg.len * 4);
    const dst = try r.alloc(st.len * 4);
    try r.upload(dqn, std.mem.sliceAsBytes(qn));
    try r.upload(dkn, std.mem.sliceAsBytes(kn));
    try r.upload(dvc, std.mem.sliceAsBytes(vc));
    try r.upload(dbeta, std.mem.sliceAsBytes(beta));
    try r.upload(dg, std.mem.sliceAsBytes(gg));
    try r.upload(dst, std.mem.sliceAsBytes(st));
    var so: [3]rt.Buffer = undefined;
    var yo: [3]rt.Buffer = undefined;
    for (0..3) |i| {
        so[i] = try r.alloc(st.len * 4);
        yo[i] = try r.alloc(max_rows * VH * DV * 2);
    }
    try r.sync();
    const hs = try gpa.alloc(u8, st.len * 4);
    const hy = try gpa.alloc(u8, max_rows * VH * DV * 2);
    const rs = try gpa.alloc(u8, st.len * 4);
    const ry = try gpa.alloc(u8, max_rows * VH * DV * 2);
    for ([_]u32{ 16, 40, 512, 1024, 2048 }) |rows| {
        for (0..3) |i| {
            var best: u64 = std.math.maxInt(u64);
            for (0..5) |_| {
                if (stop.requested()) return error.Interrupted;
                const t0 = nowNs();
                var k = &kernels[i];
                try k.setBuffer(0, dqn);
                try k.setBuffer(1, dkn);
                try k.setBuffer(2, dvc);
                try k.setBuffer(3, dbeta);
                try k.setBuffer(4, dg);
                try k.setBuffer(5, dst);
                try k.setBuffer(6, so[i]);
                try k.setBuffer(7, yo[i]);
                try k.setU32(8, 1);
                try k.setU32(9, rows);
                try k.launch(.{ VH, 32 / nr[i], 1 });
                try r.sync();
                best = @min(best, nowNs() - t0);
            }
            try r.download(if (i == 0) rs else hs, so[i]);
            try r.download(if (i == 0) ry else hy, yo[i]);
            try r.sync();
            const ny = @as(usize, rows) * VH * DV * 2;
            const same = i == 0 or (std.mem.eql(u8, rs, hs) and std.mem.eql(u8, ry[0..ny], hy[0..ny]));
            std.debug.print("rows {d:>4}: {d} row(s) a sub-group {d:>8.3} ms, state and y {s}\n", .{ rows, nr[i], @as(f64, @floatFromInt(best)) / 1e6, if (same) "bit-identical" else "DIFFERENT" });
        }
    }
}
