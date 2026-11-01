//! The GEMM-fused GeGLU of prompt rows against the unfused path (GEMM, then g4_geglu), bit for bit (--geglu-check).
const std = @import("std");
const dev = @import("xpu_dev.zig");
const fwd = @import("xpu_forward.zig");
const cfg = @import("xpu_config.zig");
const load_mod = @import("xpu_load.zig");
const alloc = dev.alloc;
const Proj = load_mod.Proj;

const FF = cfg.ff;
const ROWS = 1024;

/// Group sums of a row block as the glue stores them: hi = fp16(256 s), lo = fp16(256 s - hi), s the mean of 64.
fn sums(h: []const f16, rows: usize, k: usize, hi: []f16, lo: []f16) void {
    const g = k / 64;
    for (0..rows) |m| for (0..g) |gi| {
        var s: f32 = 0;
        for (0..64) |j| s += @as(f32, h[m * k + gi * 64 + j]);
        s *= 1.0 / 64.0;
        const hh: f16 = @floatCast(s * 256.0);
        hi[m * g + gi] = hh;
        lo[m * g + gi] = @floatCast(s * 256.0 - @as(f32, hh));
    };
}

const Stats = struct { values: u64 = 0, bad: u64 = 0, nan_bad: u64 = 0, sums_bad: u64 = 0 };

/// One block of `rows` rows through both paths on the weights `w`; mismatches counted in `st`.
fn compare(r: *fwd.Runner, w: Proj, h: []const f16, rows: u32, st: *Stats) !void {
    const k = w.k;
    const g = k / 64;
    const hi = try alloc.alloc(f16, rows * g);
    defer alloc.free(hi);
    const lo = try alloc.alloc(f16, rows * g);
    defer alloc.free(lo);
    sums(h, rows, k, hi, lo);
    const a = &r.a;
    const act_n = try alloc.alloc(u16, rows * FF);
    defer alloc.free(act_n);
    const act_o = try alloc.alloc(u16, rows * FF);
    defer alloc.free(act_o);
    const s_n = try alloc.alloc(u16, rows * FF / 64 * 2);
    defer alloc.free(s_n);
    const s_o = try alloc.alloc(u16, rows * FF / 64 * 2);
    defer alloc.free(s_o);
    try r.d.copyIn(a.h, std.mem.sliceAsBytes(h));
    try r.d.copyIn(a.ahi, std.mem.sliceAsBytes(hi));
    try r.d.copyIn(a.alo, std.mem.sliceAsBytes(lo));
    if (!try r.gateUp(w, rows)) return error.NotFused;
    try r.d.read(std.mem.sliceAsBytes(act_n), a.p);
    try r.d.read(std.mem.sliceAsBytes(s_n[0 .. rows * FF / 64]), a.ahi);
    try r.d.read(std.mem.sliceAsBytes(s_n[rows * FF / 64 ..]), a.alo);
    try r.d.copyIn(a.h, std.mem.sliceAsBytes(h));
    try r.d.copyIn(a.ahi, std.mem.sliceAsBytes(hi));
    try r.d.copyIn(a.alo, std.mem.sliceAsBytes(lo));
    try r.gemm(a.h, a.ahi, a.alo, w, a.p, rows);
    try r.d.launch(r.kp.geglu, .{ .x = rows, .y = FF / 1024 }, .{ a.p, a.h, a.ahi, a.alo });
    try r.d.read(std.mem.sliceAsBytes(act_o), a.h);
    try r.d.read(std.mem.sliceAsBytes(s_o[0 .. rows * FF / 64]), a.ahi);
    try r.d.read(std.mem.sliceAsBytes(s_o[rows * FF / 64 ..]), a.alo);
    for (act_n, act_o) |x, y| {
        st.values += 1;
        if (x != y) {
            st.bad += 1;
            if ((x & 0x7C00) == 0x7C00 and (x & 0x3FF) != 0 and (y & 0x7C00) == 0x7C00 and (y & 0x3FF) != 0) st.nan_bad += 1; // NaN payloads
        }
    }
    for (s_n, s_o) |x, y| st.sums_bad += @intFromBool(x != y);
}

fn report(name: []const u8, st: Stats) void {
    std.debug.print("{s}: {d} activations, {d} differ ({d} are NaN with another payload), {d} of the sums differ\n", .{ name, st.values, st.bad, st.nan_bad, st.sums_bad });
}

/// A synthetic projection: the codes and scales zero, a bias per column, so a row with one value x gives x * bias.
fn synthetic(r: *fwd.Runner, rng: *std.Random) !Proj {
    const n = 2 * FF;
    const k = cfg.hidden;
    const zeros = try alloc.alloc(u8, k / 8 * n * 4);
    defer alloc.free(zeros);
    @memset(zeros, 0);
    const bias = try alloc.alloc(f16, k / 64 * n);
    defer alloc.free(bias);
    for (0..n) |c| {
        const mag = std.math.pow(f32, 2.0, rng.float(f32) * 18.0 - 8.0);
        const v: f16 = switch (c % 64) {
            0 => 0,
            1 => 1,
            2 => -1,
            else => @floatCast(if (rng.boolean()) mag else -mag),
        };
        for (0..k / 64) |gi| bias[gi * n + c] = v;
    }
    return .{
        .w = try r.d.upload(zeros),
        .s = try r.d.upload(zeros[0 .. k / 64 * n * 2]),
        .bg = try r.d.upload(std.mem.sliceAsBytes(bias)),
        .n = n,
        .k = k,
        .wide = true,
    };
}

pub fn run(r: *fwd.Runner) !void {
    var prng = std.Random.DefaultPrng.init(11);
    var rng = prng.random();
    const k = cfg.hidden;
    const h = try alloc.alloc(f16, ROWS * k);
    defer alloc.free(h);
    // 1. every fp16 pattern (NaNs and infinities included) as the one nonzero input of a row, on synthetic biases
    const syn = try synthetic(r, &rng);
    var st: Stats = .{};
    var x: u32 = 0;
    while (x < 65536) : (x += ROWS) {
        @memset(h, 0);
        for (0..ROWS) |m| h[m * k] = @bitCast(@as(u16, @intCast(x + m)));
        try compare(r, syn, h, ROWS, &st);
    }
    report("every fp16 input x biases", st);
    // 2. real weights of several layers, normal inputs of several amplitudes, ragged row counts
    for ([_]usize{ 0, 3, 17, 29, 41 }) |layer| {
        var s2: Stats = .{};
        for ([_]struct { rows: u32, amp: f32 }{ .{ .rows = 1024, .amp = 0.5 }, .{ .rows = 1000, .amp = 3 }, .{ .rows = 17, .amp = 40 }, .{ .rows = 513, .amp = 300 } }) |c| {
            for (h[0 .. c.rows * k]) |*e| e.* = @floatCast(rng.floatNorm(f32) * c.amp);
            try compare(r, r.m.layers[layer].gu, h, c.rows, &s2);
        }
        var nm: [32]u8 = undefined;
        report(try std.fmt.bufPrint(&nm, "layer {d} real weights", .{layer}), s2);
        st.bad += s2.bad;
        st.sums_bad += s2.sums_bad;
    }
    if (st.bad != 0 or st.sums_bad != 0) return error.GegluDiffers;
    std.debug.print("geglu-check: identical\n", .{});
}
