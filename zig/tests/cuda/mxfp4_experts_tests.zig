//! Grouped MXFP4 experts' downs: bf16 experts' bytes on the host-decoded weights, in each form, and float64 bounds.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const grouped = cuda.grouped;
const ex_ = cuda.experts;
const Bf16 = ex_.Bf16Experts;
const Mx = ex_.Mxfp4Experts;

const E = 6; // experts
const D = 256; // outputs
const W = 96; // inputs: three 32-value scale groups
const K = 3; // picks a row

fn bf(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

fn f(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

/// An E2M1 code times 2^(s - 127) as bf16 bits: exact for the scales Mxfp4Experts takes, -0 kept.
fn decoded(code: u4, s: u8) u16 {
    const mag = [8]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const v = mag[code & 7] * @as(f32, @bitCast(@as(u32, s) << 23)); // 2^(s - 127), s >= 1
    const b: u32 = @bitCast(v);
    return @intCast((b >> 16) | (if (code & 8 != 0) @as(u32, 0x8000) else 0));
}

pub fn experts(gpu: Gpu) !void {
    const a = gpu.gpa;
    var rng = std.Random.DefaultPrng.init(0x3f4);
    const r = rng.random();
    const codes = try a.alloc(u8, Mx.codeBytes(E, D, W));
    defer a.free(codes);
    const scales = try a.alloc(u8, Mx.scaleBytes(E, D, W));
    defer a.free(scales);
    for (codes) |*c| c.* = r.int(u8); // every nibble, both zeros among them
    for (scales) |*s| s.* = 118 + r.uintLessThan(u8, 7);
    try Mx.checkScales(scales);
    const host = try a.alloc(u16, E * D * W); // [E][D][W], nn.Linear: the bf16 the kernel must decode to
    defer a.free(host);
    for (host, 0..) |*h, i| {
        const byte = codes[i / 2];
        h.* = decoded(@intCast(if (i % 2 == 0) byte & 15 else byte >> 4), scales[i / 32]);
    }

    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var kx = try Bf16.load(gpu.d, sms);
    defer kx.unload();
    var mx = try Mx.load(gpu.d, sms);
    defer mx.unload();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try grouped.Router.resolve(mod);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();

    var dcodes = try cuda.DeviceBuffer.fromHost(gpu.d, codes);
    defer dcodes.free();
    var dscales = try cuda.DeviceBuffer.fromHost(gpu.d, scales);
    defer dscales.free();
    var src = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
    defer src.free();
    var packed_ = try cuda.DeviceBuffer.alloc(gpu.d, Bf16.bytes(E, D, W));
    defer packed_.free();
    try kx.pack(s, src.ptr, .{ .stride = D * W, .ld = W, .in_out = false }, packed_.ptr, E, D, W);
    try s.synchronize();
    const lm: ex_.Layer = .{ .format = .mxfp4, .up = 0, .down = dcodes.ptr, .down_scale = dscales.ptr, .width = W, .dims = D, .experts = E };
    const lb: ex_.Layer = .{ .format = .bf16, .up = 0, .down = packed_.ptr, .width = W, .dims = D, .experts = E, .relu2 = true };
    const gm: ex_.Grouped = .{ .mxfp4 = &mx };
    const gb: ex_.Grouped = .{ .bf16 = &kx };

    // 37 rows of K pairs: three 16-pair decode items for the busiest expert and a partial prompt tile
    const rows = 37;
    const pairs = rows * K;
    const x = try a.alloc(u16, pairs * W);
    defer a.free(x);
    for (x) |*v| v.* = bf(r.float(f32) * 2 - 1);
    const picks = try a.alloc(i32, pairs);
    defer a.free(picks);
    for (0..rows) |row| {
        for (0..K) |k| {
            var p: i32 = undefined;
            while (true) {
                p = @intCast(r.uintLessThan(usize, E));
                if (std.mem.indexOfScalar(i32, picks[row * K ..][0..k], p) == null) break;
            }
            picks[row * K + k] = p;
        }
    }
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer dx.free();

    // float64: y = x . down, against the fp32 down within 1e-5 of its terms' magnitude
    var worst: f64 = 0;
    var whole: ?[2][]u8 = null;
    defer if (whole) |w| for (w) |b| a.free(b);
    for ([_]bool{ false, true }) |prompt| {
        const got = try downs(gpu, router, gm, s, lm, dx.ptr, picks, pairs, prompt);
        defer for (got) |b| a.free(b);
        const want = try downs(gpu, router, gb, s, lb, dx.ptr, picks, pairs, prompt);
        defer for (want) |b| a.free(b);
        const form = if (prompt) "prompt" else "decode";
        try check.sameBytes(if (prompt) "prompt fp32 down against bf16 experts" else "fp32 down against bf16 experts", got[0], want[0]);
        try check.sameBytes(if (prompt) "prompt bf16 down against bf16 experts" else "bf16 down against bf16 experts", got[1], want[1]);
        const y = std.mem.bytesAsSlice(f32, got[0]);
        for (0..pairs) |pr| {
            const ex: usize = @intCast(picks[pr]);
            for (0..D) |o| {
                var acc: f64 = 0;
                var m: f64 = 0;
                for (0..W) |i| {
                    const t = f(x[pr * W + i]) * f(host[(ex * D + o) * W + i]);
                    acc += t;
                    m += @abs(t);
                }
                worst = @max(worst, @abs(@as(f64, y[pr * D + o]) - acc) / @max(m, 1e-6));
            }
        }
        try check.expect(worst <= 1e-5, "mxfp4 experts {s}: fp32 down off by {e} of its terms' magnitude", .{ form, worst });
        if (whole) |w| {
            try check.sameBytes("prompt fp32 down against decode", got[0], w[0]);
            try check.sameBytes("prompt bf16 down against decode", got[1], w[1]);
        } else whole = .{ try a.dupe(u8, got[0]), try a.dupe(u8, got[1]) };
        check.pass("mxfp4 experts: {d} pairs ({s}) give bf16 experts' bytes on the decoded weights; fp32 within {e}", .{ pairs, form, worst });
    }
    // each row alone gives the bytes it gave beside the others
    for ([_]usize{ 0, 17, rows - 1 }) |row| {
        const got = try downs(gpu, router, gm, s, lm, dx.ptr + row * K * W * 2, picks[row * K ..][0..K], K, false);
        defer for (got) |b| a.free(b);
        try check.sameBytes("row fp32 down", got[0], whole.?[0][row * K * D * 4 ..][0 .. K * D * 4]);
        try check.sameBytes("row bf16 down", got[1], whole.?[1][row * K * D * 2 ..][0 .. K * D * 2]);
    }
    check.pass("mxfp4 experts: a row alone, beside 36 others and in the prompt form gives the same bytes", .{});
}

/// Plan and both downs over `pairs` pair rows; returns fp32 y and bf16 y.
fn downs(gpu: Gpu, router: grouped.Router, g: ex_.Grouped, s: cuda.Stream, l: ex_.Layer, x: u64, picks: []const i32, pairs: usize, prompt: bool) ![2][]u8 {
    const tile: usize = if (prompt) ex_.promptTile(l.format) else 16;
    var dpicks = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(picks));
    defer dpicks.free();
    const bound = grouped.maxItems(pairs, l.experts, tile);
    var bufs: [5]cuda.DeviceBuffer = undefined;
    const big = pairs > grouped.small;
    const sizes = [5]usize{ pairs * 4, bound * 12, 8, (if (big) pairs else 1) * 4, (if (big) (pairs + 1023) / 1024 * l.experts else 1) * 4 };
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| b.free();
    for (sizes, &bufs) |n, *b| {
        b.* = try cuda.DeviceBuffer.alloc(gpu.d, n);
        made += 1;
    }
    const plan: grouped.Plan = .{ .members = bufs[0].ptr, .items = bufs[1].ptr, .counts = bufs[2].ptr, .rank = bufs[3].ptr, .hist = bufs[4].ptr };
    try router.route(s, dpicks.ptr, pairs, l.experts, tile, plan);
    var y32 = try cuda.DeviceBuffer.alloc(gpu.d, pairs * l.dims * 4);
    defer y32.free();
    var y16 = try cuda.DeviceBuffer.alloc(gpu.d, pairs * l.dims * 2);
    defer y16.free();
    for ([_]*cuda.DeviceBuffer{ &y32, &y16 }) |b| try b.fill8(0xff, s.handle); // a call that writes nothing fails
    const call = if (prompt) &ex_.Grouped.prompt else &ex_.Grouped.decode;
    try call(g, s, .down_f32, .{ .x = x, .stride = l.width }, l, plan, y32.ptr, bound, pairs, tile, -1);
    try call(g, s, .down_bf16, .{ .x = x, .stride = l.width }, l, plan, y16.ptr, bound, pairs, tile, -1);
    try s.synchronize();
    var out: [2][]u8 = undefined;
    out[0] = try check.download(gpu, y32);
    errdefer gpu.gpa.free(out[0]);
    out[1] = try check.download(gpu, y16);
    return out;
}
