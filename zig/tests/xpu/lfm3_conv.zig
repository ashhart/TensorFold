//! LFM3 gated short conv (lfm3_conv_rows) against an FP64 replay on random data: a prompt window, then decode steps.

const std = @import("std");
const o = @import("lfm3_ops.zig");
const stop = @import("xpu").stop;

const D: u32 = 2048;
const PD: u32 = 3 * D;

/// FP64 replay of rows of [B | C | x]: y = C * sum_k hist[k] * w[k] over the last 7 B * x; taps reversed when `flip`.
fn refConv(proj: []const f64, w: []const f64, hist: []f64, n: usize, flip: bool, y: []f64) void {
    for (0..D) |c| {
        for (0..n) |r| {
            const p = proj[r * PD ..];
            const bx = p[c] * p[2 * D + c];
            var acc: f64 = 0;
            for (0..6) |k| acc += hist[k * D + c] * w[c * 7 + (if (flip) 6 - k else k)];
            acc += bx * w[c * 7 + @as(usize, if (flip) 0 else 6)];
            y[r * D + c] = p[D + c] * acc;
            for (0..5) |k| hist[k * D + c] = hist[(k + 1) * D + c];
            hist[5 * D + c] = bx;
        }
    }
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    stop.install();
    var r = try o.rt.open();
    defer r.deinit();
    var m = try r.module(@import("xpu").kernels.lfm3);
    var k = try m.kernel("lfm3_conv_rows", .{ 64, 1, 1 });
    var t: o.Tally = .{ .name = "lfm3_conv" };
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();

    const n_pre: u32 = 37;
    const n_dec: u32 = 5;
    const rows = n_pre + n_dec;
    const ph = try o.gpa.alloc(u16, rows * PD);
    o.randBf(ph, rnd, 1.0);
    const wh = try o.gpa.alloc(u16, D * 7);
    o.randBf(wh, rnd, 0.3);
    const proj = try o.upload(&r, ph);
    const cw = try o.upload(&r, wh);
    const zero = try o.gpa.alloc(u16, 6 * D);
    @memset(zero, 0);
    const st = try o.upload(&r, zero);
    const y = try r.alloc(rows * D * 2);

    // the prompt window from an empty state, then single decode steps on the state it leaves (state in place)
    try o.run(&k, .{ D / 64, 1, 1 }, .{ proj, PD, st, st, cw, y, n_pre, D });
    for (0..n_dec) |i| {
        const off = (n_pre + i) * PD * 2;
        const pr: o.Buf = .{ .rt = proj.rt, .ptr = @ptrFromInt(@intFromPtr(proj.ptr.?) + off), .len = proj.len - off };
        const yo = (n_pre + i) * D * 2;
        const yr: o.Buf = .{ .rt = y.rt, .ptr = @ptrFromInt(@intFromPtr(y.ptr.?) + yo), .len = y.len - yo };
        try o.run(&k, .{ D / 64, 1, 1 }, .{ pr, PD, st, st, cw, yr, @as(u32, 1), D });
    }
    const yh = try o.fetch(&r, y, rows * D);
    const sh = try o.fetch(&r, st, 6 * D);

    const pf = try o.widen(o.gpa, ph);
    const wf = try o.widen(o.gpa, wh);
    const hist = try o.gpa.alloc(f64, 6 * D);
    const yf = try o.gpa.alloc(f64, rows * D);
    @memset(hist, 0);
    refConv(pf, wf, hist, rows, false, yf);
    std.debug.print("gated short conv, D {d}: {d} prompt rows then {d} decode steps, against FP64\n", .{ D, n_pre, n_dec });
    t.check("prompt rows y", o.relErr(yh[0 .. n_pre * D], yf[0 .. n_pre * D]));
    t.check("decode steps y", o.relErr(yh[n_pre * D ..], yf[n_pre * D ..]));
    t.check("state after the last step (last six B*x)", o.relErr(sh, hist));

    // the same rows one at a time: the state handoff gives the window's bits
    const st1 = try o.upload(&r, zero);
    const y1 = try r.alloc(rows * D * 2);
    for (0..rows) |i| {
        const off = i * PD * 2;
        const pr: o.Buf = .{ .rt = proj.rt, .ptr = @ptrFromInt(@intFromPtr(proj.ptr.?) + off), .len = proj.len - off };
        const yo = i * D * 2;
        const yr: o.Buf = .{ .rt = y1.rt, .ptr = @ptrFromInt(@intFromPtr(y1.ptr.?) + yo), .len = y1.len - yo };
        try o.run(&k, .{ D / 64, 1, 1 }, .{ pr, PD, st1, st1, cw, yr, @as(u32, 1), D });
    }
    const y1h = try o.fetch(&r, y1, rows * D);
    t.same("all rows stepped one at a time against the window", std.mem.eql(u16, y1h, yh));

    // planted control: the taps in reverse order (oldest input on the newest tap) must not match
    @memset(hist, 0);
    refConv(pf, wf, hist, rows, true, yf);
    t.control("reversed taps", o.relErr(yh, yf));
    try t.finish();
}
