//! SenseNova-Loop BF16 Metal linear contract probe (extracted from metal_sensenova_linear.zig).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const dense_in = contract.dense;
const lgpu = tf.sensenova_loop.linear_gpu;

pub fn ctlGate(ctl: u8) ?g.Gate {
    return ([_]?g.Gate{ null, .nonfinite, .repeat, .shapes, .sums })[ctl];
}

pub fn injectNonFinite(buf: []u16, rows: usize, k: usize) void {
    const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
    const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
    for (at, put) |pos, val| {
        if (pos[0] < rows) buf[pos[0] * k + pos[1]] = c.toBf16(val);
    }
}

fn depthLinear(s: c.Shape) usize {
    return s.k / 8 + 32;
}

pub const Bf16Variant = enum { mv, gemm, splitk };
pub const Bf16Cfg = struct { v: Bf16Variant, tier: u32 = 1, sk: u32 = 1, ctl: u8 = 0, name: []const u8 };

pub const Bf16Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    sum_pipe: ?mtl.Pipeline = null,
    cfg: Bf16Cfg,
    served_buf: [3]c.Shape = undefined,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    d: ?dense_in.Dense = null,
    w16: []u16 = &.{},
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    scratch: ?rig_.Ref = null,
    y: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, dev: mtl.Device, q: mtl.Queue, lib: mtl.Library, cfg: Bf16Cfg) !Bf16Probe {
        var buf: [64]u8 = undefined;
        const fun: [:0]const u8 = switch (cfg.v) {
            .mv => try std.fmt.bufPrintSentinel(&buf, "sn_bf16_mv_r{d}_sk{d}", .{ cfg.tier, cfg.sk }, 0),
            .gemm => "sn_bf16_gemm_00",
            .splitk => "sn_bf16_gemm_splitk_00",
        };
        const pipe = try mtl.Pipeline.init(dev, lib, fun, false);
        errdefer pipe.deinit();
        const sp = if (cfg.v == .splitk) try mtl.Pipeline.init(dev, lib, "sn_bf16_gemm_splitk_sum", false) else null;
        return .{ .a = a, .rig = .{ .a = a, .device = dev, .queue = q }, .pipe = pipe, .sum_pipe = sp, .cfg = cfg };
    }

    pub fn deinit(p: *Bf16Probe) void {
        if (p.d) |d| d.deinit();
        p.a.free(p.w16);
        p.rig.deinit();
        p.pipe.deinit();
        if (p.sum_pipe) |sp| sp.deinit();
    }

    const vt = g.hooks(Bf16Probe);

    pub fn kernel(p: *Bf16Probe) g.Kernel {
        const t = p.cfg.tier;
        const v = p.cfg.v;
        const ks: usize = if (v == .mv) 8 else 256;
        p.served_buf = switch (v) {
            .mv => .{ .{ .rows = t, .n = 64, .k = 4096 }, .{ .rows = t, .n = 4160, .k = 4096 }, .{ .rows = t, .n = 512, .k = 12288 } },
            .gemm => .{ .{ .rows = 96, .n = 192, .k = 4096 }, .{ .rows = 64, .n = 128, .k = 512 }, .{ .rows = 0, .n = 0, .k = 0 } },
            .splitk => .{ .{ .rows = 65, .n = 192, .k = 4096 }, .{ .rows = 64, .n = 128, .k = 512 }, .{ .rows = 0, .n = 0, .k = 0 } },
        };
        const n_srv: usize = if (v == .mv) 3 else 2;
        return .{
            .spec = .{
                .name = p.cfg.name,
                .accum = .f32,
                .depth = depthLinear,
                .out = .bf16,
                .tiles = .{
                    .{ .size = if (v == .mv) t else 64, .step = 1, .min = 1 },
                    .{ .size = if (v == .mv) (8 / p.cfg.sk) * lgpu.bf16MvCols(t) else if (v == .gemm) 128 else 64, .step = 1, .min = 1 },
                    .{ .size = if (v == .splitk) 1024 else 256, .step = ks, .min = ks },
                },
                .served = p.served_buf[0..n_srv],
            },
            .ctx = p,
            .vt = &vt,
            .control = ctlGate(p.cfg.ctl),
        };
    }

    pub fn prepare(p: *Bf16Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.n == 0 or s.k == 0 or s.k % 8 != 0) return false;
        if (p.cfg.v == .mv and s.rows > 32) return false;
        if (p.cfg.v != .mv and s.k % 256 != 0) return false;
        if (p.cfg.v == .splitk and s.k < 512) return false;
        if (p.d) |d| d.deinit();
        p.a.free(p.w16);
        p.rig.reset();
        p.shape = s;
        const d = try dense_in.Dense.init(p.a, s, 1);
        p.d = d;
        d.fillX(fill, seed, 2);
        p.w16 = try p.a.alloc(u16, s.n * s.k);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x5151);
        const h = s.k / 2;
        for (0..s.n) |j| for (0..s.k) |i| {
            const sv: f32 = if (i < h) @floatFromInt(1 + (i * 7 + j) % 6) else -@as(f32, @floatFromInt(1 + ((i - h) * 7 + j) % 6));
            p.w16[j * s.k + i] = c.toBf16(if (fill != .stress) prng.random().float(f32) * 2 - 1 else sv);
            d.w[j * s.k + i] = c.fromBf16(p.w16[j * s.k + i]);
        };
        p.x = try p.rig.input(std.mem.sliceAsBytes(d.x));
        p.w = try p.rig.input(std.mem.sliceAsBytes(p.w16));
        if (p.cfg.v == .splitk) p.scratch = try p.rig.scratch(4 * s.rows * s.n * 4);
        p.y = try p.rig.output(d.outputs() * 2);
        return true;
    }

    pub fn count(p: *Bf16Probe) usize {
        return p.d.?.outputs();
    }

    pub fn run(p: *Bf16Probe) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        const xr = lgpu.Ref{ .buf = p.x.buf, .off = p.x.off };
        const wr = lgpu.Ref{ .buf = p.w.buf, .off = p.w.off };
        const yr = lgpu.Ref{ .buf = p.y.buf, .off = p.y.off };
        const dims = lgpu.LinearGpu.Dims{ .rows = @intCast(s.rows), .n = @intCast(s.n), .k = @intCast(s.k), .ldy = @intCast(s.n) };
        switch (p.cfg.v) {
            .mv => {
                const no = (8 / p.cfg.sk) * lgpu.bf16MvCols(p.cfg.tier);
                var r0: usize = 0;
                while (r0 < s.rows) : (r0 += p.cfg.tier) {
                    const rm: u32 = @intCast(@min(p.cfg.tier, s.rows - r0));
                    e.setPipeline(p.pipe);
                    e.setBuffer(p.x.buf, p.x.off + r0 * s.k * 2, 0);
                    e.setBuffer(p.w.buf, p.w.off, 1);
                    e.setValue(lgpu.Bf16Args{ .rows = rm, .n = dims.n, .k = dims.k, .ldy = dims.ldy }, 2);
                    e.setBuffer(p.y.buf, p.y.off + r0 * s.n * 2, 3);
                    e.dispatchGroups(mtl.Size.of((s.n + no - 1) / no, 1, 1), mtl.Size.of(256, 1, 1));
                }
            },
            .gemm => lgpu.LinearGpu.dispatchBf16Gemm(p.pipe, e, xr, wr, yr, dims),
            .splitk => lgpu.LinearGpu.dispatchBf16SplitK(
                p.pipe,
                p.sum_pipe.?,
                e,
                xr,
                wr,
                .{ .buf = p.scratch.?.buf, .off = p.scratch.?.off },
                yr,
                dims,
                2,
            ),
        }
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Bf16Probe, want: []f64, mag: []f64) !void {
        return p.d.?.reference(want, mag);
    }

    pub fn simulate(p: *Bf16Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.d.?.simulate(acc, order, out);
    }
};

fn fillRandomBf16(buf: mtl.Buffer, count: usize, seed: u64, scale: f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const dst = buf.slice(u16, count);
    for (dst) |*d| d.* = c.toBf16((rng.float(f32) * 2.0 - 1.0) * scale);
}

fn runBf16Once(
    gpu: *lgpu.LinearGpu,
    q: mtl.Queue,
    mat: lgpu.Bf16Matrix,
    x: lgpu.Ref,
    y: lgpu.Ref,
    rows: u32,
    ldy: u32,
    prompt: bool,
    splits: lgpu.OutSplits,
) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    try gpu.encodeBf16Split(e, mat, x, y, rows, ldy, prompt, null, splits);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("bf16 split gpu err: {s}\n", .{why});
        return error.GpuFailed;
    }
}

fn verifyBf16SplitCase(
    gpu: *lgpu.LinearGpu,
    q: mtl.Queue,
    n: u32,
    k: u32,
    rows: u32,
    prompt: bool,
    split1: u32,
    split2: u32,
) !void {
    const opts = mtl.ResourceOptions.shared;
    const x_buf = try gpu.device.buffer(@as(usize, rows) * k * 2, opts);
    defer x_buf.deinit();
    const w_buf = try gpu.device.buffer(@as(usize, n) * k * 2, opts);
    defer w_buf.deinit();
    fillRandomBf16(x_buf, @as(usize, rows) * k, 0x1111 + rows + n, 1.5);
    fillRandomBf16(w_buf, @as(usize, n) * k, 0x2222 + split1 + split2, 0.25);

    const has2 = split2 > 0;
    const w0 = split1;
    const w1 = (if (has2) split2 else n) - split1;
    const w2 = if (has2) n - split2 else 0;
    const y_ref = try gpu.device.buffer(@as(usize, rows) * n * 2, opts);
    defer y_ref.deinit();
    const y0 = try gpu.device.buffer(@as(usize, rows) * w0 * 2, opts);
    defer y0.deinit();
    const y1 = try gpu.device.buffer(@as(usize, rows) * w1 * 2, opts);
    defer y1.deinit();
    const y2 = if (has2) try gpu.device.buffer(@as(usize, rows) * w2 * 2, opts) else null;
    defer if (y2) |b| b.deinit();
    const y_sep = try gpu.device.buffer(@as(usize, rows) * n * 2, opts);
    defer y_sep.deinit();

    const full_mat = try lgpu.LinearGpu.prepareBf16(w_buf, 0, n, k);
    const xr = lgpu.Ref{ .buf = x_buf };
    try runBf16Once(gpu, q, full_mat, xr, .{ .buf = y_ref }, rows, n, prompt, .{});
    try runBf16Once(gpu, q, full_mat, xr, .{ .buf = y0 }, rows, w0, prompt, .{
        .split1 = split1,
        .y1 = .{ .buf = y1 },
        .ldy1 = w1,
        .split2 = split2,
        .y2 = if (y2) |b| .{ .buf = b } else null,
        .ldy2 = w2,
    });

    // Compare split outputs against full stacked output row by row.
    const ref_s = y_ref.slice(u16, @as(usize, rows) * n);
    const y0_s = y0.slice(u16, @as(usize, rows) * w0);
    const y1_s = y1.slice(u16, @as(usize, rows) * w1);
    const y2_s = if (y2) |b| b.slice(u16, @as(usize, rows) * w2) else &.{};
    for (0..rows) |r| {
        if (!std.mem.eql(u16, ref_s[r * n ..][0..w0], y0_s[r * w0 ..][0..w0])) return error.SplitMismatch;
        if (!std.mem.eql(u16, ref_s[r * n + split1 ..][0..w1], y1_s[r * w1 ..][0..w1])) return error.SplitMismatch;
        if (has2 and !std.mem.eql(u16, ref_s[r * n + split2 ..][0..w2], y2_s[r * w2 ..][0..w2])) return error.SplitMismatch;
    }

    // Also match separate slice matrices bit-for-bit when the same kernel path runs.
    if (!prompt or gpu.tensor_units) {
        const m0 = try lgpu.LinearGpu.prepareBf16(w_buf, 0, w0, k);
        try runBf16Once(gpu, q, m0, xr, .{ .buf = y_sep }, rows, w0, prompt, .{});
        if (!std.mem.eql(u16, y_sep.slice(u16, @as(usize, rows) * w0), y0_s)) return error.SplitSeparateMismatch;
        const m1 = try lgpu.LinearGpu.prepareBf16(w_buf, @as(usize, split1) * k * 2, w1, k);
        try runBf16Once(gpu, q, m1, xr, .{ .buf = y_sep }, rows, w1, prompt, .{});
        if (!std.mem.eql(u16, y_sep.slice(u16, @as(usize, rows) * w1), y1_s)) return error.SplitSeparateMismatch;
        if (has2) {
            const m2 = try lgpu.LinearGpu.prepareBf16(w_buf, @as(usize, split2) * k * 2, w2, k);
            try runBf16Once(gpu, q, m2, xr, .{ .buf = y_sep }, rows, w2, prompt, .{});
            if (!std.mem.eql(u16, y_sep.slice(u16, @as(usize, rows) * w2), y2_s)) return error.SplitSeparateMismatch;
        }
    }
}

pub fn checkBf16Split(a: std.mem.Allocator, dev: mtl.Device, q: mtl.Queue) !void {
    var gpu = lgpu.LinearGpu.init(a, dev);
    defer gpu.deinit();
    try gpu.reserve(128, false);
    for ([_]bool{ false, true }) |prompt| {
        const rows: u32 = if (prompt) 65 else 12;
        try verifyBf16SplitCase(&gpu, q, 10240, 4096, rows, prompt, 8192, 9216);
        try verifyBf16SplitCase(&gpu, q, 12288, 4096, rows, prompt, 8192, 0);
        if (prompt and !dev.tensorUnits()) {
            try verifyBf16SplitCase(&gpu, q, 8192, 4096, rows, true, 4096, 0);
        }
    }
    // Check OutSplits validation errors.
    const opts = mtl.ResourceOptions.shared;
    const dummy = try dev.buffer(4096 * 128 * 2, opts);
    defer dummy.deinit();
    const mat = try lgpu.LinearGpu.prepareBf16(dummy, 0, 256, 128);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    const ref = lgpu.Ref{ .buf = dummy };
    const bad = [_]lgpu.OutSplits{
        .{ .split1 = 0, .y1 = ref, .ldy1 = 256 },
        .{ .split1 = 64, .y1 = ref, .ldy1 = 192 },
        .{ .split1 = 128, .y1 = ref, .ldy1 = 64 },
        .{ .split1 = 128, .y1 = ref, .ldy1 = 128, .split2 = 128, .y2 = ref, .ldy2 = 128 },
        .{ .split1 = 128, .y1 = null, .y2 = ref, .ldy2 = 128 },
    };
    for (bad) |sp| {
        const res = gpu.encodeBf16Split(e, mat, ref, ref, 1, 128, false, null, sp);
        if (res) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
    }
    e.end();
    std.debug.print("PASS encodeBf16Split (10240,4096) & (12288,4096) bit-exact + OutSplits validation\n", .{});
}

fn runAwq(
    gpu: *lgpu.LinearGpu,
    q: mtl.Queue,
    mat: lgpu.AwqMatrix,
    x: lgpu.Ref,
    xs: ?lgpu.Sums,
    y: lgpu.Ref,
    rows: u32,
    ldy: u32,
    prompt: bool,
    sk: ?u32,
    splits: lgpu.OutSplits,
) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    try gpu.encodeAwqSplit(e, mat, x, xs, y, rows, ldy, prompt, sk, splits);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("awq gpu err: {s}\n", .{why});
        return error.GpuFailed;
    }
}

pub fn packBf16AsAwq(gpu: *const lgpu.LinearGpu, w_bf16: []const u16, n: u32, k: u32) !lgpu.AwqMatrix {
    const awq = tf.sensenova_loop.awq;
    const kg = k / 128;
    const layout: awq.Layout = if (gpu.tensor_units) .m5_tile else .m3_row;
    const w = try gpu.allocator.alloc(u32, awq.weightWords(n, k, layout));
    defer gpu.allocator.free(w);
    @memset(w, 0);
    const sb = try gpu.allocator.alloc(awq.Float2, awq.metaCount(n, k));
    defer gpu.allocator.free(sb);
    for (0..n) |j| for (0..kg) |gi| {
        const row = w_bf16[j * k + gi * 128 ..][0..128];
        var min_v: f32 = 0;
        var max_v: f32 = 0;
        for (row) |b| {
            const v: f32 = @floatCast(c.fromBf16(b));
            min_v = @min(min_v, v);
            max_v = @max(max_v, v);
        }
        const scale = (max_v - min_v) / 15.0;
        const zp: f32 = if (scale != 0) @floatFromInt(@as(u32, @intFromFloat(std.math.clamp(@round(-min_v / scale), 0.0, 15.0)))) else 0;
        const bias = if (scale != 0) -zp * scale else min_v;
        sb[if (layout == .m5_tile) gi * n + j else j * kg + gi] = .{ .scale = scale, .bias = bias };
        for (0..16) |w16| {
            var word: u32 = 0;
            for (0..8) |ni| {
                const vf: f32 = @floatCast(c.fromBf16(row[w16 * 8 + ni]));
                const q_val: u32 = if (scale != 0) @intFromFloat(std.math.clamp(@round((vf - bias) / scale), 0.0, 15.0)) else 0;
                word |= q_val << @intCast(ni * 4);
            }
            const w_idx = if (layout == .m5_tile) (((j / 32) * kg + gi) * 32 + (j % 32)) * 16 + w16 else j * (k / 8) + gi * 16 + w16;
            w[w_idx] = word;
        }
    };
    return lgpu.LinearGpu.prepareAwqFromPacked(gpu, .{ .n = n, .k = k, .layout = layout, .w = w, .sb = sb, .input_scale = null });
}

fn verifyAwqSplitCase(gpu: *lgpu.LinearGpu, q: mtl.Queue, n: u32, k: u32, rows: u32, prompt: bool, split1: u32, split2: u32) !void {
    const opts = mtl.ResourceOptions.shared;
    const x_buf = try gpu.device.buffer(@as(usize, rows) * k * 2, opts);
    defer x_buf.deinit();
    const w_buf = try gpu.device.buffer(@as(usize, n) * k * 2, opts);
    defer w_buf.deinit();
    fillRandomBf16(x_buf, @as(usize, rows) * k, 0x5555 + rows + n, 1.5);
    fillRandomBf16(w_buf, @as(usize, n) * k, 0x6666 + split1 + split2, 0.25);

    const has2 = split2 > 0;
    const w0 = split1;
    const w1 = (if (has2) split2 else n) - split1;
    const w2 = if (has2) n - split2 else 0;
    const w_all = w_buf.slice(u16, @as(usize, n) * k);
    var full = try packBf16AsAwq(gpu, w_all, n, k);
    defer full.deinit();
    const y_ref = try gpu.device.buffer(@as(usize, rows) * n * 2, opts);
    defer y_ref.deinit();
    const y0 = try gpu.device.buffer(@as(usize, rows) * w0 * 2, opts);
    defer y0.deinit();
    const y1 = try gpu.device.buffer(@as(usize, rows) * w1 * 2, opts);
    defer y1.deinit();
    const y2 = if (has2) try gpu.device.buffer(@as(usize, rows) * w2 * 2, opts) else null;
    defer if (y2) |b| b.deinit();
    const y_sep = try gpu.device.buffer(@as(usize, rows) * n * 2, opts);
    defer y_sep.deinit();

    const xr = lgpu.Ref{ .buf = x_buf };
    try runAwq(gpu, q, full, xr, null, .{ .buf = y_ref }, rows, n, prompt, null, .{});
    try runAwq(gpu, q, full, xr, null, .{ .buf = y0 }, rows, w0, prompt, null, .{
        .split1 = split1,
        .y1 = .{ .buf = y1 },
        .ldy1 = w1,
        .split2 = split2,
        .y2 = if (y2) |b| .{ .buf = b } else null,
        .ldy2 = w2,
    });
    const ref_s = y_ref.slice(u16, @as(usize, rows) * n);
    const y0_s = y0.slice(u16, @as(usize, rows) * w0);
    const y1_s = y1.slice(u16, @as(usize, rows) * w1);
    const y2_s = if (y2) |b| b.slice(u16, @as(usize, rows) * w2) else &.{};
    for (0..rows) |r| {
        if (!std.mem.eql(u16, ref_s[r * n ..][0..w0], y0_s[r * w0 ..][0..w0])) return error.SplitMismatch;
        if (!std.mem.eql(u16, ref_s[r * n + split1 ..][0..w1], y1_s[r * w1 ..][0..w1])) return error.SplitMismatch;
        if (has2 and !std.mem.eql(u16, ref_s[r * n + split2 ..][0..w2], y2_s[r * w2 ..][0..w2])) return error.SplitMismatch;
    }

    const sk_match: ?u32 = if (prompt) null else lgpu.decodeSk(gpu.tensor_units, true, n, k);
    var m0 = try packBf16AsAwq(gpu, w_all[0 .. @as(usize, w0) * k], w0, k);
    defer m0.deinit();
    try runAwq(gpu, q, m0, xr, null, .{ .buf = y_sep }, rows, w0, prompt, sk_match, .{});
    if (!std.mem.eql(u16, y_sep.slice(u16, @as(usize, rows) * w0), y0_s)) return error.SplitSeparateMismatch;
    var m1 = try packBf16AsAwq(gpu, w_all[@as(usize, split1) * k ..][0 .. @as(usize, w1) * k], w1, k);
    defer m1.deinit();
    try runAwq(gpu, q, m1, xr, null, .{ .buf = y_sep }, rows, w1, prompt, sk_match, .{});
    if (!std.mem.eql(u16, y_sep.slice(u16, @as(usize, rows) * w1), y1_s)) return error.SplitSeparateMismatch;
    if (has2) {
        var m2 = try packBf16AsAwq(gpu, w_all[@as(usize, split2) * k ..][0 .. @as(usize, w2) * k], w2, k);
        defer m2.deinit();
        try runAwq(gpu, q, m2, xr, null, .{ .buf = y_sep }, rows, w2, prompt, sk_match, .{});
        if (!std.mem.eql(u16, y_sep.slice(u16, @as(usize, rows) * w2), y2_s)) return error.SplitSeparateMismatch;
    }
}

fn checkQuantRoundTrip(gpu: *lgpu.LinearGpu, q: mtl.Queue, n: u32, k: u32) !void {
    const opts = mtl.ResourceOptions.shared;
    const x_buf = try gpu.device.buffer(@as(usize, 65) * k * 2, opts);
    defer x_buf.deinit();
    const w_buf = try gpu.device.buffer(@as(usize, n) * k * 2, opts);
    defer w_buf.deinit();
    fillRandomBf16(x_buf, @as(usize, 65) * k, 0x7777 + n, 1.5);
    // Use structured weights (16-level group grid + BF16 noise) plus an all-zero group for scale == 0 coverage.
    var prng = std.Random.DefaultPrng.init(0x8888 + k);
    const rng = prng.random();
    const w_s = w_buf.slice(u16, @as(usize, n) * k);
    for (0..n) |j| {
        for (0..k / 128) |g_idx| {
            const gs: f32 = 0.02 + 0.04 * rng.float(f32);
            const zp: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 14));
            for (0..128) |i| {
                const val: f32 = if (j == 0 and g_idx == 0) 0.0 else blk: {
                    const qv: f32 = @floatFromInt(rng.uintLessThan(u32, 16));
                    break :blk (qv - zp) * gs + (rng.float(f32) - 0.5) * gs * 0.25;
                };
                w_s[j * k + g_idx * 128 + i] = c.toBf16(val);
            }
        }
    }
    const bf = try lgpu.LinearGpu.prepareBf16(w_buf, 0, n, k);
    var aw = try packBf16AsAwq(gpu, w_s, n, k);
    defer aw.deinit();
    const y_bf = try gpu.device.buffer(@as(usize, 65) * n * 2, opts);
    defer y_bf.deinit();
    const y_aw = try gpu.device.buffer(@as(usize, 65) * n * 2, opts);
    defer y_aw.deinit();
    for ([_]struct { rows: u32, prompt: bool }{ .{ .rows = 4, .prompt = false }, .{ .rows = 65, .prompt = true } }) |tc| {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = q.commandBuffer();
        const e = cb.compute(.serial);
        try gpu.encodeBf16(e, bf, .{ .buf = x_buf }, .{ .buf = y_bf }, tc.rows, n, tc.prompt, null);
        try gpu.encodeAwq(e, aw, .{ .buf = x_buf }, null, .{ .buf = y_aw }, tc.rows, n, tc.prompt, null);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |_| return error.GpuFailed;
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        for (y_bf.slice(u16, tc.rows * n), y_aw.slice(u16, tc.rows * n)) |b, a| {
            const fb = c.fromBf16(b);
            const fa = c.fromBf16(a);
            dot += fa * fb;
            na += fa * fa;
            nb += fb * fb;
        }
        const cos = dot / (@sqrt(na) * @sqrt(nb));
        if (!(cos > 0.999)) return error.QuantRoundTripCosTooLow;
    }
}

pub fn checkAwqSplitAndQuant(a: std.mem.Allocator, dev: mtl.Device, q: mtl.Queue) !void {
    var gpu = lgpu.LinearGpu.init(a, dev);
    defer gpu.deinit();
    try gpu.reserve(128, true);
    for ([_]bool{ false, true }) |prompt| {
        const rows: u32 = if (prompt) 65 else 12;
        try verifyAwqSplitCase(&gpu, q, 10240, 4096, rows, prompt, 8192, 9216);
        try verifyAwqSplitCase(&gpu, q, 12288, 4096, rows, prompt, 8192, 0);
    }
    try checkQuantRoundTrip(&gpu, q, 256, 1024);
    try checkQuantRoundTrip(&gpu, q, 40, 256);
}
