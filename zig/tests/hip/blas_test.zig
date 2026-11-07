//! Device checks for the native BF16 projection path against exact independent integer dots.
const std = @import("std");
const hip = @import("hip");

fn bf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

fn value(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

test "HIP BLAS device: bias is added before BF16 output rounding" {
    var r = try selectedRuntime();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    var b = try hip.blas.Blas.open(stream);
    defer b.close();
    const x = [_]u16{ bf16(256), bf16(1) };
    const w = [_]u16{ bf16(1), bf16(1) };
    const bias = [_]u16{bf16(-256)};
    var dx = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(&x));
    defer dx.free();
    var dw = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(&w));
    defer dw.free();
    var dy = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(&bias));
    defer dy.free();
    try b.projectAdd(dx.base(), dw.base(), dy.base(), 1, 1, 2);
    try stream.synchronize();
    var got: [1]u16 = undefined;
    try dy.download(0, std.mem.sliceAsBytes(&got));
    try std.testing.expectEqual(bf16(1), got[0]);
    try b.project(dx.base(), dw.base(), dy.base(), 1, 1, 2);
    try stream.synchronize();
    try dy.download(0, std.mem.sliceAsBytes(&got));
    try std.testing.expectEqual(bf16(256), got[0]);
}

test "HIP BLAS device: cancellation dots distinguish FP32 accumulation from FP16 and BF16" {
    const a = std.testing.allocator;
    const k = 1536;
    var r = try selectedRuntime();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    var b = try hip.blas.Blas.open(stream);
    defer b.close();
    const x = try a.alloc(u16, k);
    defer a.free(x);
    const w = try a.alloc(u16, k);
    defer a.free(w);
    @memset(x, bf16(1));
    var half_sum: f16 = 0;
    var brain_sum: f32 = 0;
    for (w, 0..) |*v, j| {
        const f: f32 = if (j < k / 2) 128 else -127;
        v.* = bf16(f);
        half_sum = @floatCast(@as(f32, half_sum) + f);
        brain_sum = @floatCast(value(bf16(brain_sum + f)));
    }
    try std.testing.expect(!std.math.isFinite(half_sum));
    try std.testing.expect(brain_sum != 768);
    var dx = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(x));
    defer dx.free();
    var dw = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(w));
    defer dw.free();
    var dy = try hip.DeviceBuffer.alloc(&r, 2);
    defer dy.free();
    for (0..5) |_| {
        try b.project(dx.base(), dw.base(), dy.base(), 1, 1, k);
        try stream.synchronize();
        var got: [1]u16 = undefined;
        try dy.download(0, std.mem.sliceAsBytes(&got));
        try std.testing.expectEqual(bf16(768), got[0]);
    }
}

test "HIP BLAS device: BF16 projection preserves inputs, guards and exact FP32 sums at served widths" {
    const a = std.testing.allocator;
    var r = try selectedRuntime();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    var b = try hip.blas.Blas.open(stream);
    defer b.close();
    const cases = [_][3]usize{ .{ 1, 1536, 1536 }, .{ 4, 1024, 1176 }, .{ 7, 9216, 1536 }, .{ 2, 1536, 4608 }, .{ 17, 2048, 1536 }, .{ 4, 1536, 2048 }, .{ 4, 1024, 1536 }, .{ 2, 59392, 1536 } };
    var dimensions: [cases.len]hip.blas_fixed.Dimensions = undefined;
    for (cases, &dimensions) |shape, *d| d.* = .{ .n = shape[1], .k = shape[2] };
    var fixed = try hip.blas_fixed.Fixed.open(stream, &dimensions);
    defer fixed.close();
    for (cases) |shape| {
        const m, const n, const k = shape;
        if ((try ctx.memInfo()).free < (m * k + n * k + m * n + 32) * 2 + (128 << 20)) return error.NoDeviceBudget;
        const x = try a.alloc(u16, m * k);
        defer a.free(x);
        const w = try a.alloc(u16, n * k);
        defer a.free(w);
        const out = try a.alloc(u16, m * n + 32);
        defer a.free(out);
        const want = try a.alloc(u16, m * n);
        defer a.free(want);
        for (x, 0..) |*v, i| v.* = bf16(@floatFromInt(@as(i32, @intCast((i * 3 + i / k) % 7)) - 3));
        for (w, 0..) |*v, i| v.* = bf16(@floatFromInt(@as(i32, @intCast((i * 5 + i / k) % 11)) - 5));
        for (0..m) |row| for (0..n) |col| {
            var sum: f64 = 0;
            for (0..k) |j| sum += value(x[row * k + j]) * value(w[col * k + j]);
            want[row * n + col] = bf16(@floatCast(sum));
        };
        var dx = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(x));
        defer dx.free();
        var dw = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(w));
        defer dw.free();
        var dy = try hip.DeviceBuffer.alloc(&r, out.len * 2);
        defer dy.free();
        try dy.fill8(0xa5);
        for (0..10) |repeat| {
            if (repeat < 5) try b.project(dx.base(), dw.base(), dy.base() + 32, m, n, k) else try fixed.project(dx.base(), dw.base(), dy.base() + 32, m, n, k);
            try stream.synchronize();
            try dy.download(0, std.mem.sliceAsBytes(out));
            try std.testing.expectEqualSlices(u16, want, out[16..][0 .. m * n]);
            for (out[0..16]) |v| try std.testing.expectEqual(@as(u16, 0xa5a5), v);
            for (out[16 + m * n ..]) |v| try std.testing.expectEqual(@as(u16, 0xa5a5), v);
        }
        const actual_x = try a.alloc(u16, x.len);
        defer a.free(actual_x);
        const actual_w = try a.alloc(u16, w.len);
        defer a.free(actual_w);
        try dx.download(0, std.mem.sliceAsBytes(actual_x));
        try dw.download(0, std.mem.sliceAsBytes(actual_w));
        try std.testing.expectEqualSlices(u16, x, actual_x);
        try std.testing.expectEqualSlices(u16, w, actual_w);
    }
}

test "HIP BLAS device: products run in the planned workspace, which rocBLAS keeps and never outgrows" {
    const a = std.testing.allocator;
    var r = try selectedRuntime();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    var b = try hip.blas.Blas.open(stream);
    defer b.close();
    var ws = try hip.DeviceBuffer.alloc(&r, hip.blas.workspace_bytes);
    defer ws.free();
    try b.useWorkspace(ws.base(), ws.len);
    const set = try b.workspace();
    try std.testing.expect(!set.managed);
    try std.testing.expectEqual(hip.blas.workspace_bytes, set.bytes);
    // Clef-Flash's projections (n, k) at prompt-piece row counts, the 248,320-row head at a few rows
    const shapes = [_][2]usize{ .{ 8192, 4096 }, .{ 4096, 4096 }, .{ 32, 4096 }, .{ 12288, 4096 }, .{ 4096, 12288 }, .{ 1024, 4096 } };
    const rows = [_]usize{ 1, 2, 3, 7, 17, 64, 255, 512, 513, 1000, 1024 };
    const x = try a.alloc(u16, 1024 * 12288);
    defer a.free(x);
    for (x, 0..) |*v, i| v.* = bf16(@floatFromInt(@as(i32, @intCast(i % 5)) - 2));
    var dx = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(x));
    defer dx.free();
    var dw = try hip.DeviceBuffer.alloc(&r, 248320 * 4096 * 2);
    defer dw.free();
    try dw.fill8(0);
    var dy = try hip.DeviceBuffer.alloc(&r, 1024 * 12288 * 2);
    defer dy.free();
    for (shapes) |s| for (rows) |m| try b.project(dx.base(), dw.base(), dy.base(), m, s[0], s[1]);
    for ([_]usize{ 1, 4, 32 }) |m| try b.project(dx.base(), dw.base(), dy.base(), m, 248320, 4096);
    try stream.synchronize();
    const after = try b.workspace();
    try std.testing.expect(!after.managed);
    try std.testing.expectEqual(hip.blas.workspace_bytes, after.bytes);
}

/// Attention at Clef-Flash's heads: 16 of 64, keys from a short record to the 16,384-row window, query blocks to 64.
const attention_shapes = [_][2]usize{ .{ 245, 1 }, .{ 245, 64 }, .{ 1024, 7 }, .{ 4099, 33 }, .{ 16384, 64 } };

/// A float64 product the FP32 one must match within (k + 1) 2^-24 of its terms' absolute sum, any order.
const Ref = struct { want: []f64, bound: []f64 };

/// How far `got` is from `ref` in units of the bound (over 1 fails).
fn worst(got: []const f32, ref: Ref, view: []const usize) f64 {
    var w: f64 = 0;
    for (view, ref.want, ref.bound) |g, want, bound| {
        const q = @abs(got[g] - want) / (bound + 1e-30);
        if (!std.math.isFinite(q)) return std.math.inf(f64); // a non-finite output fails, whatever @max would keep
        w = @max(w, q);
    }
    return w;
}

fn identity(a: std.mem.Allocator, n: usize) ![]usize {
    const v = try a.alloc(usize, n);
    for (v, 0..) |*x, i| x.* = i;
    return v;
}

/// Float64 scores [heads][queries][keys]; `skew` reads head h's keys from head h + skew (a wrong-stride control).
fn scoreRef(a: std.mem.Allocator, q: []const u16, k: []const u16, h: hip.blas.Heads, alpha: f64, skew: usize) !Ref {
    const w = h.heads * h.head_dim;
    const want = try a.alloc(f64, h.heads * h.queries * h.keys);
    const bound = try a.alloc(f64, want.len);
    for (0..h.heads) |hh| for (0..h.queries) |i| for (0..h.keys) |j| {
        var s: f64 = 0;
        var m: f64 = 0;
        const kh = (hh + skew) % h.heads;
        for (0..h.head_dim) |d| {
            const t = value(q[i * w + hh * h.head_dim + d]) * value(k[j * w + kh * h.head_dim + d]);
            s += t;
            m += @abs(t);
        }
        want[(hh * h.queries + i) * h.keys + j] = s * alpha;
        bound[(hh * h.queries + i) * h.keys + j] = @as(f64, @floatFromInt(h.head_dim + 1)) * 0x1p-24 * m * alpha;
    };
    return .{ .want = want, .bound = bound };
}

/// Float64 contexts [queries][W]; `skew` moves head h to head h + skew's columns, `flip` reads P transposed (controls).
fn contextRef(a: std.mem.Allocator, p: []const f32, v: []const f32, h: hip.blas.Heads, skew: usize, flip: bool) !Ref {
    const w = h.heads * h.head_dim;
    const want = try a.alloc(f64, h.queries * w);
    const bound = try a.alloc(f64, want.len);
    for (0..h.heads) |hh| for (0..h.queries) |i| for (0..h.head_dim) |d| {
        var s: f64 = 0;
        var m: f64 = 0;
        for (0..h.keys) |j| {
            const at = if (flip) (hh * h.keys + j) * h.queries + i else (hh * h.queries + i) * h.keys + j;
            const t = @as(f64, p[at]) * v[j * w + hh * h.head_dim + d];
            s += t;
            m += @abs(t);
        }
        const at = i * w + ((hh + skew) % h.heads) * h.head_dim + d;
        want[at] = s;
        bound[at] = @as(f64, @floatFromInt(h.keys + 2)) * 0x1p-24 * m;
    };
    return .{ .want = want, .bound = bound };
}

/// The output read as [heads][keys][queries], as a wrong transpose would lay it out.
fn transposed(a: std.mem.Allocator, h: hip.blas.Heads) ![]usize {
    const v = try a.alloc(usize, h.heads * h.queries * h.keys);
    for (0..h.heads) |hh| for (0..h.queries) |i| for (0..h.keys) |j| {
        v[(hh * h.queries + i) * h.keys + j] = hh * h.queries * h.keys + j * h.queries + i;
    };
    return v;
}

fn randomBf16(rand: std.Random, out: []u16) void {
    for (out) |*x| x.* = bf16(rand.float(f32) * 4 - 2);
}

test "HIP BLAS device: per-head FP32 scores and contexts against float64 at served shapes, with planted controls" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var r = try selectedRuntime();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    var b = try hip.blas.Blas.open(stream);
    defer b.close();
    var prng = std.Random.DefaultPrng.init(1952);
    const rand = prng.random();
    for (attention_shapes) |shape| {
        _ = arena.reset(.retain_capacity);
        const a = arena.allocator();
        const h: hip.blas.Heads = .{ .keys = shape[0], .queries = shape[1], .heads = 16, .head_dim = 64 };
        const w = h.heads * h.head_dim;
        const q = try a.alloc(u16, h.queries * w);
        const k = try a.alloc(u16, h.keys * w);
        randomBf16(rand, q);
        randomBf16(rand, k);
        const n = h.heads * h.queries * h.keys;
        var dq = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(q));
        defer dq.free();
        var dk = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(k));
        defer dk.free();
        var ds = try hip.DeviceBuffer.alloc(&r, n * 4 + 128);
        defer ds.free();
        try ds.fill8(0xa5);
        try b.scores(dq.base(), dk.base(), ds.base() + 64, h, 0.125);
        try stream.synchronize();
        const raw = try a.alloc(f32, n + 32);
        try ds.download(0, std.mem.sliceAsBytes(raw));
        for (raw[0..16]) |x| try std.testing.expectEqual(@as(u32, 0xa5a5a5a5), @as(u32, @bitCast(x)));
        for (raw[16 + n ..]) |x| try std.testing.expectEqual(@as(u32, 0xa5a5a5a5), @as(u32, @bitCast(x)));
        const s = raw[16..][0..n];
        const ref = try scoreRef(a, q, k, h, 0.125, 0);
        const plain = try identity(a, n);
        const fit = worst(s, ref, plain);
        // planted controls: a wrong head stride and the transposed layout must both fail the same check
        const skewed = worst(s, try scoreRef(a, q, k, h, 0.125, 1), plain);
        const flipped = if (h.queries > 1) worst(s, ref, try transposed(a, h)) else std.math.inf(f64);
        var wide: usize = 0; // FP32 outputs: most are not BF16 values
        for (s) |x| wide += @intFromBool(value(bf16(x)) != x);
        std.debug.print("HIP BLAS scores keys {d} queries {d}: worst {d:.3} of the bound, controls {d:.1} {d:.1}, {d}/{d} not BF16\n", .{ h.keys, h.queries, fit, skewed, flipped, wide, n });
        try std.testing.expect(fit <= 1 and skewed > 1 and flipped > 1 and wide * 2 > n);
        // contexts: FP32 probabilities times FP32 values
        const p = try a.alloc(f32, n);
        for (p) |*x| x.* = rand.float(f32);
        const vb = try a.alloc(u16, h.keys * w);
        randomBf16(rand, vb);
        const v = try a.alloc(f32, vb.len);
        for (v, vb) |*x, y| x.* = @floatCast(value(y));
        var dp = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(p));
        defer dp.free();
        var dv = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(v));
        defer dv.free();
        var dc = try hip.DeviceBuffer.alloc(&r, h.queries * w * 4 + 128);
        defer dc.free();
        try dc.fill8(0xa5);
        try b.context(dp.base(), dv.base(), dc.base() + 64, h);
        try stream.synchronize();
        const craw = try a.alloc(f32, h.queries * w + 32);
        try dc.download(0, std.mem.sliceAsBytes(craw));
        for (craw[0..16]) |x| try std.testing.expectEqual(@as(u32, 0xa5a5a5a5), @as(u32, @bitCast(x)));
        for (craw[16 + h.queries * w ..]) |x| try std.testing.expectEqual(@as(u32, 0xa5a5a5a5), @as(u32, @bitCast(x)));
        const c = craw[16..][0 .. h.queries * w];
        const all = try identity(a, c.len);
        const cfit = worst(c, try contextRef(a, p, v, h, 0, false), all);
        const cskew = worst(c, try contextRef(a, p, v, h, 1, false), all);
        const cflip = if (h.queries > 1) worst(c, try contextRef(a, p, v, h, 0, true), all) else std.math.inf(f64);
        std.debug.print("HIP BLAS context keys {d} queries {d}: worst {d:.3} of the bound, controls {d:.1} {d:.1}\n", .{ h.keys, h.queries, cfit, cskew, cflip });
        try std.testing.expect(cfit <= 1 and cskew > 1 and cflip > 1);
    }
}

fn selectedRuntime() !hip.Runtime {
    var r = try hip.Runtime.open();
    errdefer r.close();
    if (try r.deviceCount() != 1) return error.SelectOneHipDevice;
    return r;
}
