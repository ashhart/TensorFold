//! GPU check of k3_kda on Ling-tiny's KDA shape against kda.zig. Loads Kimi's metallib.
const std = @import("std");
const mtl = @import("metal");
const kda = @import("kda");
const metallib = @import("kimi_metallib");

const KdaArgs = extern struct { heads: u32, head0: u32, proj_stride: u32, gate_stride: u32, out_stride: u32, log_rows: u32, lower_bound: f32, eps: f32 };
const KdaSeg = extern struct { first: u32, rows: u32, kept: u32, flags: u32, slot: u32 };
const KdaState = extern struct { S: u64, conv: u64, log_a: u64, log_k: u64, log_u: u64, log_x: u64 };
const KdaWeights = extern struct { conv_q: u64, conv_k: u64, conv_v: u64, A_log: u64, dt_bias: u64, o_norm: u64 };
const seg_commit: u32 = 1;

fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate((bits >> 16) | 0x40);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn fill(seed: *u64, n: usize, a: std.mem.Allocator) ![]f32 {
    const x = try a.alloc(f32, n);
    for (x) |*v| {
        seed.* = seed.* *% 1103515245 +% 12345;
        v.* = @as(f32, @floatFromInt((seed.* >> 16) & 0x7fff)) / 16384.0 - 1.0;
    }
    return x;
}

fn putBf16(buf: mtl.Buffer, x: []const f32) void {
    const d = buf.slice(u16, x.len);
    for (x, d) |v, *o| o.* = toBf16(v);
}

fn putF32(buf: mtl.Buffer, x: []const f32) void {
    @memcpy(buf.slice(f32, x.len), x);
}

fn roundInputs(proj: []f32, fgate: []f32, braw: []f32, g2: []f32) void {
    for (proj) |*v| v.* = fromBf16(toBf16(v.*));
    for (fgate) |*v| v.* = fromBf16(toBf16(v.*));
    for (braw) |*v| v.* = fromBf16(toBf16(v.*));
    for (g2) |*v| v.* = fromBf16(toBf16(v.*));
}

const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    lib: mtl.Library,
    pipe: mtl.Pipeline,

    fn init() !Gpu {
        const device = try mtl.Device.init();
        errdefer device.deinit();
        const lib = try mtl.Library.fromBytes(device, metallib.bytes);
        errdefer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, "k3_kda", false);
        errdefer pipe.deinit();
        const queue = try device.queue();
        return .{ .device = device, .queue = queue, .lib = lib, .pipe = pipe };
    }

    fn deinit(g: *Gpu) void {
        g.pipe.deinit();
        g.lib.deinit();
        g.queue.deinit();
        g.device.deinit();
    }
};

const Launch = struct {
    s: kda.Shape,
    rows: u32,
    b_proj: mtl.Buffer,
    b_f: mtl.Buffer,
    b_b: mtl.Buffer,
    b_g: mtl.Buffer,
    b_y: mtl.Buffer,
    b_seg: mtl.Buffer,
    b_S: mtl.Buffer,
    b_conv: mtl.Buffer,
    dummy: mtl.Buffer,
    b_cq: mtl.Buffer,
    b_ck: mtl.Buffer,
    b_cv: mtl.Buffer,
    b_A: mtl.Buffer,
    b_dt: mtl.Buffer,
    b_on: mtl.Buffer,

    fn open(g: Gpu, s: kda.Shape, rows: u32, proj: []const f32, fgate: []const f32, braw: []const f32, g2: []const f32, conv_q: []const f32, conv_k: []const f32, conv_v: []const f32, A_log: []const f32, dt_bias: []const f32, o_norm: []const f32) !Launch {
        const H = s.heads;
        const W = s.width();
        const D = kda.dim;
        const shared = mtl.ResourceOptions.shared;
        const b_proj = try g.device.buffer(proj.len * 2, shared);
        const b_f = try g.device.buffer(fgate.len * 2, shared);
        const b_b = try g.device.buffer(braw.len * 2, shared);
        const b_g = try g.device.buffer(g2.len * 2, shared);
        const b_y = try g.device.buffer(rows * W * 2, shared);
        const b_seg = try g.device.buffer(@sizeOf(KdaSeg), shared);
        const b_S = try g.device.buffer(H * D * D * 4, shared);
        const b_conv = try g.device.buffer(3 * 3 * W * 2, shared);
        const dummy = try g.device.buffer(4, shared);
        const b_cq = try g.device.buffer(conv_q.len * 4, shared);
        const b_ck = try g.device.buffer(conv_k.len * 4, shared);
        const b_cv = try g.device.buffer(conv_v.len * 4, shared);
        const b_A = try g.device.buffer(A_log.len * 4, shared);
        const b_dt = try g.device.buffer(dt_bias.len * 4, shared);
        const b_on = try g.device.buffer(o_norm.len * 4, shared);
        putBf16(b_proj, proj);
        putBf16(b_f, fgate);
        putBf16(b_b, braw);
        putBf16(b_g, g2);
        @memset(b_y.slice(u16, rows * W), 0);
        @memset(b_S.slice(f32, H * D * D), 0);
        @memset(b_conv.slice(u16, 3 * 3 * W), 0);
        putF32(b_cq, conv_q);
        putF32(b_ck, conv_k);
        putF32(b_cv, conv_v);
        putF32(b_A, A_log);
        putF32(b_dt, dt_bias);
        putF32(b_on, o_norm);
        return .{
            .s = s,
            .rows = rows,
            .b_proj = b_proj,
            .b_f = b_f,
            .b_b = b_b,
            .b_g = b_g,
            .b_y = b_y,
            .b_seg = b_seg,
            .b_S = b_S,
            .b_conv = b_conv,
            .dummy = dummy,
            .b_cq = b_cq,
            .b_ck = b_ck,
            .b_cv = b_cv,
            .b_A = b_A,
            .b_dt = b_dt,
            .b_on = b_on,
        };
    }

    fn deinit(l: *Launch) void {
        l.b_proj.deinit();
        l.b_f.deinit();
        l.b_b.deinit();
        l.b_g.deinit();
        l.b_y.deinit();
        l.b_seg.deinit();
        l.b_S.deinit();
        l.b_conv.deinit();
        l.dummy.deinit();
        l.b_cq.deinit();
        l.b_ck.deinit();
        l.b_cv.deinit();
        l.b_A.deinit();
        l.b_dt.deinit();
        l.b_on.deinit();
    }

    fn dispatch(l: *Launch, g: Gpu, first: u32, n: u32, flags: u32) !void {
        const H = l.s.heads;
        const W = l.s.width();
        l.b_seg.slice(KdaSeg, 1)[0] = .{ .first = first, .rows = n, .kept = 0, .flags = flags, .slot = 0 };
        const st = KdaState{
            .S = l.b_S.gpuAddress(),
            .conv = l.b_conv.gpuAddress(),
            .log_a = l.dummy.gpuAddress(),
            .log_k = l.dummy.gpuAddress(),
            .log_u = l.dummy.gpuAddress(),
            .log_x = l.dummy.gpuAddress(),
        };
        const w = KdaWeights{
            .conv_q = l.b_cq.gpuAddress(),
            .conv_k = l.b_ck.gpuAddress(),
            .conv_v = l.b_cv.gpuAddress(),
            .A_log = l.b_A.gpuAddress(),
            .dt_bias = l.b_dt.gpuAddress(),
            .o_norm = l.b_on.gpuAddress(),
        };
        const args = KdaArgs{
            .heads = H,
            .head0 = 0,
            .proj_stride = 3 * W,
            .gate_stride = W,
            .out_stride = W,
            .log_rows = 1,
            .lower_bound = l.s.lower_bound,
            .eps = l.s.eps,
        };
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(g.pipe);
        e.setBuffer(l.b_proj, 0, 0);
        e.setBuffer(l.b_f, 0, 1);
        e.setBuffer(l.b_b, 0, 2);
        e.setBuffer(l.b_g, 0, 3);
        e.setBuffer(l.b_y, 0, 4);
        e.setBuffer(l.b_seg, 0, 5);
        e.setValue(st, 6);
        e.setValue(w, 7);
        e.setValue(args, 8);
        const rw = mtl.ResourceUsage.read | mtl.ResourceUsage.write;
        e.useResource(l.b_S, rw);
        e.useResource(l.b_conv, rw);
        e.useResource(l.dummy, mtl.ResourceUsage.read);
        e.useResource(l.b_cq, mtl.ResourceUsage.read);
        e.useResource(l.b_ck, mtl.ResourceUsage.read);
        e.useResource(l.b_cv, mtl.ResourceUsage.read);
        e.useResource(l.b_A, mtl.ResourceUsage.read);
        e.useResource(l.b_dt, mtl.ResourceUsage.read);
        e.useResource(l.b_on, mtl.ResourceUsage.read);
        e.dispatchGroups(mtl.Size.of(H, 1, 1), mtl.Size.of(256, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.debug.print("GPU failed: {s}\n", .{why});
            return error.GpuFailed;
        }
    }

    fn y(l: *Launch) []u16 {
        return l.b_y.slice(u16, l.rows * l.s.width());
    }
};

fn glmTol(y: f64, rowmax: f64) f64 {
    return @abs(y) * std.math.pow(f64, 2, -8) + rowmax * std.math.pow(f64, 2, -8) + 1e-30;
}

fn gateGlmVsMix(mix_y: []const f32, got: []const u16, W: u32, rows: u32) !void {
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        const lo = r * W;
        var rowmax: f64 = 0;
        for (mix_y[lo .. lo + W]) |v| rowmax = @max(rowmax, @abs(@as(f64, v)));
        for (mix_y[lo .. lo + W], got[lo .. lo + W], 0..) |w_y, g_bits, i| {
            const err = @abs(@as(f64, w_y) - @as(f64, fromBf16(g_bits)));
            const tol = glmTol(w_y, rowmax);
            if (err > tol) {
                std.debug.print("GLM miss row {d} col {d} err {d:.6} tol {d:.6} mix {d:.6} gpu {d:.6} rowmax {d:.6}\n", .{ r, i, err, tol, w_y, fromBf16(g_bits), rowmax });
                return error.GlmBound;
            }
        }
    }
}

fn printF64PerRow(tag: []const u8, ref64: []const f64, got: []const u16, W: u32, rows: u32) void {
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        var worst: f64 = 0;
        const lo = r * W;
        for (ref64[lo .. lo + W], got[lo .. lo + W]) |w_y, g_bits|
            worst = @max(worst, @abs(w_y - @as(f64, fromBf16(g_bits))));
        std.debug.print("{s} row {d} vs float64 {d:.6} ({d} heads)\n", .{ tag, r, worst, W / kda.dim });
    }
}

fn printMixVsMixF64(tag: []const u8, mix_y: []const f32, ref64: []const f64, W: u32, rows: u32) void {
    var worst: f64 = 0;
    for (mix_y, ref64) |m, f| worst = @max(worst, @abs(@as(f64, m) - f));
    std.debug.print("{s} mix vs mixF64 worst abs {e} ({d} heads, {d} rows)\n", .{ tag, worst, W / kda.dim, rows });
}

fn check(rows: u32) !void {
    const a = std.testing.allocator;
    const s = kda.Shape{ .heads = 16, .lower_bound = -5, .eps = 1e-6 };
    const H = s.heads;
    const W = s.width();
    const D = kda.dim;
    var seed: u64 = 0x6b63;
    const proj = try fill(&seed, rows * 3 * W, a);
    defer a.free(proj);
    const fgate = try fill(&seed, rows * W, a);
    defer a.free(fgate);
    const braw = try fill(&seed, rows * H, a);
    defer a.free(braw);
    const g2 = try fill(&seed, rows * W, a);
    defer a.free(g2);
    const conv_q = try fill(&seed, W * kda.taps, a);
    defer a.free(conv_q);
    const conv_k = try fill(&seed, W * kda.taps, a);
    defer a.free(conv_k);
    const conv_v = try fill(&seed, W * kda.taps, a);
    defer a.free(conv_v);
    const A_log = try fill(&seed, H, a);
    defer a.free(A_log);
    const dt_bias = try fill(&seed, W, a);
    defer a.free(dt_bias);
    const o_norm = try fill(&seed, D, a);
    defer a.free(o_norm);
    for (o_norm) |*v| v.* = 0.5 + 0.5 * v.*;
    roundInputs(proj, fgate, braw, g2);

    const want = try a.alloc(f32, rows * W);
    defer a.free(want);
    const S_host = try a.alloc(f32, H * D * D);
    defer a.free(S_host);
    @memset(S_host, 0);
    try kda.mix(want, S_host, proj, fgate, braw, g2, conv_q, conv_k, conv_v, A_log, dt_bias, o_norm, s, rows);

    const want64 = try a.alloc(f64, rows * W);
    defer a.free(want64);
    const S64 = try a.alloc(f64, H * D * D);
    defer a.free(S64);
    @memset(S64, 0);
    try kda.mixF64(want64, S64, proj, fgate, braw, g2, conv_q, conv_k, conv_v, A_log, dt_bias, o_norm, s, rows);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var g = try Gpu.init();
    defer g.deinit();
    var l = try Launch.open(g, s, rows, proj, fgate, braw, g2, conv_q, conv_k, conv_v, A_log, dt_bias, o_norm);
    defer l.deinit();
    try l.dispatch(g, 0, rows, seg_commit);
    const got = l.y();
    try gateGlmVsMix(want, got, W, rows);
    printMixVsMixF64("k3_kda", want, want64, W, rows);
    printF64PerRow("k3_kda", want64, got, W, rows);
}

fn takeU32(b: []const u8, off: *usize) u32 {
    const v = std.mem.readInt(u32, b[off.*..][0..4], .little);
    off.* += 4;
    return v;
}

fn takeF32s(alloc: std.mem.Allocator, b: []const u8, off: *usize, n: usize) ![]f32 {
    const x = try alloc.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(x), b[off.* .. off.* + n * 4]);
    off.* += n * 4;
    return x;
}

fn checkDump(bytes: []const u8) !void {
    const a = std.testing.allocator;
    try std.testing.expect(std.mem.eql(u8, bytes[0..8], "TFLINGK1"));
    var off: usize = 8;
    const heads = takeU32(bytes, &off);
    const rows = takeU32(bytes, &off);
    const d = takeU32(bytes, &off);
    const tp = takeU32(bytes, &off);
    try std.testing.expectEqual(kda.dim, d);
    try std.testing.expectEqual(kda.taps, tp);
    const lower = @as(f32, @bitCast(takeU32(bytes, &off)));
    const eps = @as(f32, @bitCast(takeU32(bytes, &off)));
    off += 8;
    const s = kda.Shape{ .heads = heads, .lower_bound = lower, .eps = eps };
    const W = s.width();
    const H = s.heads;
    const D = kda.dim;
    const proj = try takeF32s(a, bytes, &off, rows * 3 * W);
    defer a.free(proj);
    const fgate = try takeF32s(a, bytes, &off, rows * W);
    defer a.free(fgate);
    const braw = try takeF32s(a, bytes, &off, rows * H);
    defer a.free(braw);
    const g2 = try takeF32s(a, bytes, &off, rows * W);
    defer a.free(g2);
    const conv_q = try takeF32s(a, bytes, &off, W * kda.taps);
    defer a.free(conv_q);
    const conv_k = try takeF32s(a, bytes, &off, W * kda.taps);
    defer a.free(conv_k);
    const conv_v = try takeF32s(a, bytes, &off, W * kda.taps);
    defer a.free(conv_v);
    const A_log = try takeF32s(a, bytes, &off, H);
    defer a.free(A_log);
    const dt_bias = try takeF32s(a, bytes, &off, W);
    defer a.free(dt_bias);
    const o_norm = try takeF32s(a, bytes, &off, D);
    defer a.free(o_norm);
    const want_n = rows * W;
    const want64 = try a.alloc(f64, want_n);
    defer a.free(want64);
    @memcpy(std.mem.sliceAsBytes(want64), bytes[off .. off + want_n * 8]);

    roundInputs(proj, fgate, braw, g2);
    const want = try a.alloc(f32, want_n);
    defer a.free(want);
    const S_host = try a.alloc(f32, H * D * D);
    defer a.free(S_host);
    @memset(S_host, 0);
    try kda.mix(want, S_host, proj, fgate, braw, g2, conv_q, conv_k, conv_v, A_log, dt_bias, o_norm, s, rows);
    const mix64 = try a.alloc(f64, want_n);
    defer a.free(mix64);
    const S64 = try a.alloc(f64, H * D * D);
    defer a.free(S64);
    @memset(S64, 0);
    try kda.mixF64(mix64, S64, proj, fgate, braw, g2, conv_q, conv_k, conv_v, A_log, dt_bias, o_norm, s, rows);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var g = try Gpu.init();
    defer g.deinit();
    var l = try Launch.open(g, s, rows, proj, fgate, braw, g2, conv_q, conv_k, conv_v, A_log, dt_bias, o_norm);
    defer l.deinit();
    try l.dispatch(g, 0, rows, seg_commit);
    const got = l.y();
    try gateGlmVsMix(want, got, W, rows);
    printMixVsMixF64("k3_kda dump", want, mix64, W, rows);
    printF64PerRow("k3_kda dump", want64, got, W, rows);
}

fn checkSeam() !void {
    const a = std.testing.allocator;
    const rows: u32 = 4;
    const s = kda.Shape{ .heads = 16, .lower_bound = -5, .eps = 1e-6 };
    const H = s.heads;
    const W = s.width();
    const D = kda.dim;
    var seed: u64 = 0x6b63;
    const proj = try fill(&seed, rows * 3 * W, a);
    defer a.free(proj);
    const fgate = try fill(&seed, rows * W, a);
    defer a.free(fgate);
    const braw = try fill(&seed, rows * H, a);
    defer a.free(braw);
    const g2 = try fill(&seed, rows * W, a);
    defer a.free(g2);
    const conv_q = try fill(&seed, W * kda.taps, a);
    defer a.free(conv_q);
    const conv_k = try fill(&seed, W * kda.taps, a);
    defer a.free(conv_k);
    const conv_v = try fill(&seed, W * kda.taps, a);
    defer a.free(conv_v);
    const A_log = try fill(&seed, H, a);
    defer a.free(A_log);
    const dt_bias = try fill(&seed, W, a);
    defer a.free(dt_bias);
    const o_norm = try fill(&seed, D, a);
    defer a.free(o_norm);
    for (o_norm) |*v| v.* = 0.5 + 0.5 * v.*;
    roundInputs(proj, fgate, braw, g2);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var g = try Gpu.init();
    defer g.deinit();
    var one = try Launch.open(g, s, rows, proj, fgate, braw, g2, conv_q, conv_k, conv_v, A_log, dt_bias, o_norm);
    defer one.deinit();
    try one.dispatch(g, 0, rows, seg_commit);
    var two = try Launch.open(g, s, rows, proj, fgate, braw, g2, conv_q, conv_k, conv_v, A_log, dt_bias, o_norm);
    defer two.deinit();
    try two.dispatch(g, 0, 2, seg_commit);
    try two.dispatch(g, 2, 2, seg_commit);
    try std.testing.expect(std.mem.eql(u16, one.y(), two.y()));
}

test "k3_kda on Ling-tiny (16 heads, D=128, 4 taps) matches kda.mix on two rows" {
    try check(2);
}

test "k3_kda on Ling-tiny (16 heads, D=128, 4 taps) vs mix and mixF64 on four rows" {
    try check(4);
}

test "k3_kda vs kda_inner dump on 16 heads x 2 rows" {
    try checkDump(kda.dump_h16_r2);
}

test "k3_kda vs kda_inner dump on 16 heads x 4 rows" {
    try checkDump(kda.dump_h16_r4);
}

test "k3_kda 4 rows equals 2+2 with S and conv carried" {
    try checkSeam();
}

test "k3_kda 32 rows vs mix, mixF64 computed in the test" {
    // GPU vs mixF64 is printed, not gated; a GLM bound vs mixF64 cannot hold at bf16.
    try check(32);
}
