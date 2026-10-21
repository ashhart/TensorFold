//! exl3-pack: v1.1.0's EXL3 operators (exl3.Dense, exl3_experts.Experts) on a real Flash Next EXL3 pack's shapes.
//! Each projection: the error of rows 1 and 5 against an exact float64 product (x · diag(suh) H Wq H diag(svh)),
//! then the min over 5 batches of the mean call time (cudaEvent). The routed experts: one row's ten picks of
//! layer 0 (gate, up, SwiGLU in fp32, down, weighted sum) against float64 the same way.
//! Usage: tf-cuda-test exl3-pack <pack dir> [launches per batch, default 20]
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const exl3fmt = core.exl3_format;
const st = core.safetensors;
const exl3 = cuda.exl3;
const exl3e = cuda.exl3_experts;
const Gpu = check.Gpu;

const shard_count = 7;

const Pack = struct {
    files: [shard_count]st.File = undefined,
    open_n: usize = 0,

    fn open(p: *Pack, gpu: Gpu, dir: []const u8) !void {
        var buf: [512]u8 = undefined;
        for (0..shard_count) |i| {
            const path = try std.fmt.bufPrint(&buf, "{s}/model-{d:0>5}-of-{d:0>5}.safetensors", .{ dir, i + 1, shard_count });
            p.files[i] = try st.File.open(gpu.gpa, gpu.io, path);
            p.open_n += 1;
        }
    }

    fn close(p: *Pack, io: std.Io) void {
        for (p.files[0..p.open_n]) |*f| f.close(io);
    }

    fn get(p: *const Pack, name: []const u8) ?st.Tensor {
        for (p.files[0..p.open_n]) |*f| if (f.get(name)) |t| return t;
        return null;
    }
};

/// One EXL3 matrix copied to aligned host memory: trellis int16 [K/16, N/16, 8 * halves], fp16 suh [K], svh [N].
const Mat = struct {
    trellis: []i16,
    suh: []f16,
    svh: []f16,
    tiles_k: usize,
    tiles_n: usize,
    halves: u16,

    fn k(m: Mat) usize {
        return m.tiles_k * 16;
    }
    fn n(m: Mat) usize {
        return m.tiles_n * 16;
    }
};

fn copyAs(comptime T: type, a: std.mem.Allocator, bytes: []const u8) ![]T {
    const out = try a.alloc(T, bytes.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. out.len * @sizeOf(T)]);
    return out;
}

fn mat(p: *const Pack, a: std.mem.Allocator, base: []const u8) !Mat {
    var buf: [256]u8 = undefined;
    const t = p.get(try std.fmt.bufPrint(&buf, "{s}.trellis", .{base})) orelse return error.MissingTensor;
    const su = p.get(try std.fmt.bufPrint(&buf, "{s}.suh", .{base})) orelse return error.MissingTensor;
    const sv = p.get(try std.fmt.bufPrint(&buf, "{s}.svh", .{base})) orelse return error.MissingTensor;
    if (p.get(try std.fmt.bufPrint(&buf, "{s}.mul1", .{base})) == null) return error.NotMul1;
    if (t.rank != 3 or t.dtype != .i16) return error.UnexpectedTensor;
    return .{
        .trellis = try copyAs(i16, a, t.bytes),
        .suh = try copyAs(f16, a, su.bytes),
        .svh = try copyAs(f16, a, sv.bytes),
        .tiles_k = t.dim(0),
        .tiles_n = t.dim(1),
        .halves = @intCast(t.dim(2) / 8),
    };
}

fn bf16Bits(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

fn bfValue(bits: u16) f64 {
    return @floatCast(@as(f32, @bitCast(@as(u32, bits) << 16)));
}

/// y [rows, N] = x [rows, K] · diag(suh) H Wq H diag(svh), all in float64 (x taken as given, no fp16 step).
fn exact(a: std.mem.Allocator, m: Mat, x: []const f64, rows: usize, y: []f64) !void {
    const k = m.k();
    const n = m.n();
    const wq = try a.alloc(f16, k * n);
    defer a.free(wq);
    exl3fmt.unpack(.{ .halves = m.halves }, .mul1, m.trellis, m.tiles_k, m.tiles_n, wq);
    const xs = try a.alloc(f64, rows * k);
    defer a.free(xs);
    for (0..rows) |r| for (0..k) |i| {
        xs[r * k + i] = x[r * k + i] * @as(f64, @floatCast(m.suh[i]));
    };
    exl3fmt.rotate(xs, rows, k, 1);
    @memset(y, 0);
    for (0..k) |ki| {
        const row_w = wq[ki * n ..][0..n];
        for (0..rows) |r| {
            const xv = xs[r * k + ki];
            const yr = y[r * n ..][0..n];
            for (yr, row_w) |*o, w| o.* += xv * @as(f64, @floatCast(w));
        }
    }
    exl3fmt.rotate(y, rows, n, 1);
    for (0..rows) |r| for (0..n) |c| {
        y[r * n + c] *= @as(f64, @floatCast(m.svh[c]));
    };
}

const Err = struct { max_abs: f64 = 0, peak: f64 = 0, worst_rel: f64 = 0 };

fn compareBf16(got: []const u16, want: []const f64) Err {
    var e: Err = .{};
    for (got, want) |g, w| {
        const d = @abs(bfValue(g) - w);
        e.max_abs = @max(e.max_abs, d);
        e.peak = @max(e.peak, @abs(w));
    }
    e.worst_rel = e.max_abs / @max(e.peak, 1e-30);
    return e;
}

fn compareF32(got: []const f32, want: []const f64) Err {
    var e: Err = .{};
    for (got, want) |g, w| {
        const d = @abs(@as(f64, g) - w);
        e.max_abs = @max(e.max_abs, d);
        e.peak = @max(e.peak, @abs(w));
    }
    e.worst_rel = e.max_abs / @max(e.peak, 1e-30);
    return e;
}

const Bench = struct {
    gpu: Gpu,
    s: cuda.Stream,
    ev0: cuda.Event,
    ev1: cuda.Event,
    reps: usize,
    total_ms: f64 = 0,

    fn time(b: *Bench, ctx: anytype, comptime f: fn (@TypeOf(ctx)) anyerror!void) !f64 {
        for (0..3) |_| try f(ctx);
        try b.s.synchronize();
        var best: f64 = std.math.inf(f64);
        for (0..5) |_| {
            try b.ev0.record(b.s);
            for (0..b.reps) |_| try f(ctx);
            try b.ev1.record(b.s);
            try b.ev1.synchronize();
            const ms = try b.ev0.elapsedMs(b.ev1);
            best = @min(best, @as(f64, ms) / @as(f64, @floatFromInt(b.reps)));
        }
        return best;
    }
};

const Call = struct {
    de: *exl3.Dense,
    s: cuda.Stream,
    w: exl3.Weight,
    x: u64,
    y: u64,
    rows: usize,
    fn go(c: Call) anyerror!void {
        try c.de.decode(c.s, c.w, c.x, .bf16, c.rows, c.y, .bf16);
    }
};

fn randX(a: std.mem.Allocator, rng: std.Random, n: usize) !struct { bits: []u16, vals: []f64 } {
    const bits = try a.alloc(u16, n);
    const vals = try a.alloc(f64, n);
    for (bits, vals) |*b, *v| {
        b.* = bf16Bits((rng.float(f32) - 0.5) * 2.0);
        v.* = bfValue(b.*);
    }
    return .{ .bits = bits, .vals = vals };
}

fn linear(b: *Bench, de: *exl3.Dense, pack: *const Pack, a: std.mem.Allocator, rng: std.Random, name: []const u8, impl: @TypeOf(de.implementation)) !void {
    const m = mat(pack, a, name) catch |e| {
        std.debug.print("{s}: skipped ({t})\n", .{ name, e });
        return;
    };
    const k = m.k();
    const n = m.n();
    const strides = exl3.strides(k, n, m.halves, false);
    var tr = try cuda.DeviceBuffer.fromHost(b.gpu.d, std.mem.sliceAsBytes(m.trellis));
    defer tr.free();
    var su = try cuda.DeviceBuffer.fromHost(b.gpu.d, std.mem.sliceAsBytes(m.suh));
    defer su.free();
    var sv = try cuda.DeviceBuffer.fromHost(b.gpu.d, std.mem.sliceAsBytes(m.svh));
    defer sv.free();
    const w: exl3.Weight = .{ .trellis = tr.ptr, .suh = su.ptr, .svh = sv.ptr, .k = @intCast(k), .n = @intCast(n), .halves = m.halves, .codebook = .mul1, .stride_k = strides.k, .stride_nb = strides.nb };
    if (impl == .tiles) {
        _ = exl3.Dense.candidateWeight(w) catch |e| {
            std.debug.print("{s:<58} tiles: refused ({t})\n", .{ name, e });
            return;
        };
    }
    de.implementation = impl;
    defer de.implementation = .reference;
    for ([_]usize{ 1, 5 }) |rows| {
        const x = try randX(a, rng, rows * k);
        var xd = try cuda.DeviceBuffer.fromHost(b.gpu.d, std.mem.sliceAsBytes(x.bits));
        defer xd.free();
        var yd = try cuda.DeviceBuffer.alloc(b.gpu.d, rows * n * 2);
        defer yd.free();
        try yd.fill8(0xff, b.s.handle);
        const call: Call = .{ .de = de, .s = b.s, .w = w, .x = xd.ptr, .y = yd.ptr, .rows = rows };
        try call.go();
        try b.s.synchronize();
        const got = try a.alloc(u16, rows * n);
        try yd.download(0, std.mem.sliceAsBytes(got));
        // the reference row count: lm_head's 636M products a row, so only its first row
        const ref_rows: usize = if (n > 100_000) 1 else rows;
        const want = try a.alloc(f64, ref_rows * n);
        try exact(a, m, x.vals[0 .. ref_rows * k], ref_rows, want);
        const e = compareBf16(got[0 .. ref_rows * n], want);
        // a second call's bytes equal the first's (deterministic)
        try call.go();
        try b.s.synchronize();
        const again = try a.alloc(u16, rows * n);
        try yd.download(0, std.mem.sliceAsBytes(again));
        const same = std.mem.eql(u16, got, again);
        const ms = try b.time(call, Call.go);
        if (rows == 1) b.total_ms += ms;
        std.debug.print("{s:<58} {s:<9} K={d:<5} N={d:<6} halves={d:<2} rows={d} ms={d:>8.4}  max|err|={e:.3} peak={d:.3} rel={e:.2} repeat={s}\n", .{ name, @tagName(impl), k, n, m.halves, rows, ms, e.max_abs, e.peak, e.worst_rel, if (same) "same" else "DIFF" });
        if (rows == 1) {
            // the same row with fp32 outputs: the accumulation's own error, without bf16's output rounding
            var yf = try cuda.DeviceBuffer.alloc(b.gpu.d, n * 4);
            defer yf.free();
            try de.decode(b.s, w, xd.ptr, .bf16, 1, yf.ptr, .f32);
            try b.s.synchronize();
            const gf = try a.alloc(f32, n);
            try yf.download(0, std.mem.sliceAsBytes(gf));
            const ef = compareF32(gf, want[0..n]);
            std.debug.print("{s:<58} {s:<9} fp32 out rows=1: max|err|={e:.3} peak={d:.3} rel={e:.2}\n", .{ name, @tagName(impl), ef.max_abs, ef.peak, ef.worst_rel });
        }
    }
}

const ExpCall = struct {
    ex: *const exl3e.Experts,
    s: cuda.Stream,
    l: exl3e.Layer,
    x: u64,
    pick: u64,
    wts: u64,
    out: u64,
    sc: exl3e.Scratch,
    fn go(c: ExpCall) anyerror!void {
        try c.ex.run(c.s, c.l, c.x, c.l.dims, .bf16, c.pick, c.wts, 1, c.out, c.sc);
    }
};

/// Ten experts of `layer` as E = 10, one row picking all ten (weights 0.1): fp32 out vs float64. The first `mapped`
/// experts' trellis words sit in mapped page-locked host memory (the kernel reads them over PCIe, no copy).
fn experts(b: *Bench, ex: *const exl3e.Experts, pack: *const Pack, a: std.mem.Allocator, rng: std.Random, layer: usize, act_mode: i32, mapped: usize) !void {
    const count = 10;
    var buf: [160]u8 = undefined;
    var g: [count]Mat = undefined;
    var u: [count]Mat = undefined;
    var dn: [count]Mat = undefined;
    for (0..count) |e| {
        g[e] = try mat(pack, a, try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.mlp.experts.{d}.gate_proj", .{ layer, e }));
        u[e] = try mat(pack, a, try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.mlp.experts.{d}.up_proj", .{ layer, e }));
        dn[e] = try mat(pack, a, try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.mlp.experts.{d}.down_proj", .{ layer, e }));
    }
    const dims = g[0].k();
    const width = g[0].n();
    var bufs: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (bufs.items) |*x| x.free();
        bufs.deinit(b.gpu.gpa);
    }
    const up = struct {
        fn f(gpu: Gpu, list: *std.ArrayList(cuda.DeviceBuffer), bytes: []const u8) !u64 {
            const d = try cuda.DeviceBuffer.fromHost(gpu.d, bytes);
            try list.append(gpu.gpa, d);
            return d.ptr;
        }
    }.f;
    var hosts: std.ArrayList(cuda.HostBuffer) = .empty;
    defer {
        for (hosts.items) |*h| h.free();
        hosts.deinit(b.gpu.gpa);
    }
    const hostUp = struct {
        fn f(gpu: Gpu, list: *std.ArrayList(cuda.HostBuffer), bytes: []const u8) !u64 {
            var h = try cuda.HostBuffer.allocMapped(gpu.d, bytes.len);
            errdefer h.free();
            @memcpy(h.bytes[0..bytes.len], bytes);
            try list.append(gpu.gpa, h);
            return try h.device();
        }
    }.f;
    var gp: [count]u64 = undefined;
    var upp: [count]u64 = undefined;
    var dp: [count]u64 = undefined;
    var gk2: [count]i32 = undefined;
    var uk2: [count]i32 = undefined;
    var dk2: [count]i32 = undefined;
    var suh_g: std.ArrayList(f16) = .empty;
    var suh_u: std.ArrayList(f16) = .empty;
    var svh_g: std.ArrayList(f16) = .empty;
    var svh_u: std.ArrayList(f16) = .empty;
    var suh_d: std.ArrayList(f16) = .empty;
    var svh_d: std.ArrayList(f16) = .empty;
    for (0..count) |e| {
        if (e < mapped) {
            gp[e] = try hostUp(b.gpu, &hosts, std.mem.sliceAsBytes(g[e].trellis));
            upp[e] = try hostUp(b.gpu, &hosts, std.mem.sliceAsBytes(u[e].trellis));
            dp[e] = try hostUp(b.gpu, &hosts, std.mem.sliceAsBytes(dn[e].trellis));
        } else {
            gp[e] = try up(b.gpu, &bufs, std.mem.sliceAsBytes(g[e].trellis));
            upp[e] = try up(b.gpu, &bufs, std.mem.sliceAsBytes(u[e].trellis));
            dp[e] = try up(b.gpu, &bufs, std.mem.sliceAsBytes(dn[e].trellis));
        }
        gk2[e] = g[e].halves;
        uk2[e] = u[e].halves;
        dk2[e] = dn[e].halves;
        try suh_g.appendSlice(a, g[e].suh);
        try suh_u.appendSlice(a, u[e].suh);
        try svh_g.appendSlice(a, g[e].svh);
        try svh_u.appendSlice(a, u[e].svh);
        try suh_d.appendSlice(a, dn[e].suh);
        try svh_d.appendSlice(a, dn[e].svh);
    }
    const l: exl3e.Layer = .{
        .gate = try up(b.gpu, &bufs, std.mem.sliceAsBytes(&gp)),
        .up = try up(b.gpu, &bufs, std.mem.sliceAsBytes(&upp)),
        .down = try up(b.gpu, &bufs, std.mem.sliceAsBytes(&dp)),
        .gate_k2 = try up(b.gpu, &bufs, std.mem.sliceAsBytes(&gk2)),
        .up_k2 = try up(b.gpu, &bufs, std.mem.sliceAsBytes(&uk2)),
        .down_k2 = try up(b.gpu, &bufs, std.mem.sliceAsBytes(&dk2)),
        .suh_g = try up(b.gpu, &bufs, std.mem.sliceAsBytes(suh_g.items)),
        .suh_u = try up(b.gpu, &bufs, std.mem.sliceAsBytes(suh_u.items)),
        .svh_g = try up(b.gpu, &bufs, std.mem.sliceAsBytes(svh_g.items)),
        .svh_u = try up(b.gpu, &bufs, std.mem.sliceAsBytes(svh_u.items)),
        .suh_d = try up(b.gpu, &bufs, std.mem.sliceAsBytes(suh_d.items)),
        .svh_d = try up(b.gpu, &bufs, std.mem.sliceAsBytes(svh_d.items)),
        .experts = count,
        .dims = dims,
        .width = width,
        .codebook = .mul1,
        .act_mode = act_mode,
    };
    const x = try randX(a, rng, dims);
    var pick: [count]i32 = undefined;
    for (&pick, 0..) |*p, i| p.* = @intCast((i * 7) % count);
    var wts: [count]f32 = @splat(0.1);
    const sbytes = try exl3e.scratchBytes(l, 1, count);
    var sbuf = try cuda.DeviceBuffer.alloc(b.gpu.d, sbytes);
    defer sbuf.free();
    var outd = try cuda.DeviceBuffer.alloc(b.gpu.d, dims * 4);
    defer outd.free();
    try outd.fill8(0xff, b.s.handle);
    const call: ExpCall = .{
        .ex = ex,
        .s = b.s,
        .l = l,
        .x = try up(b.gpu, &bufs, std.mem.sliceAsBytes(x.bits)),
        .pick = try up(b.gpu, &bufs, std.mem.sliceAsBytes(&pick)),
        .wts = try up(b.gpu, &bufs, std.mem.sliceAsBytes(&wts)),
        .out = outd.ptr,
        .sc = try exl3e.scratch(sbuf.ptr, l, 1, count),
    };
    try call.go();
    try b.s.synchronize();
    const got = try a.alloc(f32, dims);
    try outd.download(0, std.mem.sliceAsBytes(got));
    // float64: out = sum_e 0.1 * down_e(silu(gate_e(x)) * up_e(x))
    const want = try a.alloc(f64, dims);
    @memset(want, 0);
    const gy = try a.alloc(f64, width);
    const uy = try a.alloc(f64, width);
    const dy = try a.alloc(f64, dims);
    for (pick) |pe| {
        const e: usize = @intCast(pe);
        try exact(a, g[e], x.vals, 1, gy);
        try exact(a, u[e], x.vals, 1, uy);
        for (gy, uy) |*gv, uv| gv.* = gv.* / (1.0 + @exp(-gv.*)) * uv;
        try exact(a, dn[e], gy, 1, dy);
        for (want, dy) |*o, v| o.* += 0.1 * v;
    }
    const err = compareF32(got, want);
    const ms = try b.time(call, ExpCall.go);
    // with a 64 MiB memset between calls (evicts the 32 MiB L2, so mapped host bytes come over PCIe each call),
    // minus the memset alone
    var flush = try cuda.DeviceBuffer.alloc(b.gpu.d, 64 << 20);
    defer flush.free();
    const Flushed = struct {
        c: ExpCall,
        f: u64,
        with_call: bool,
        fn go(fl: @This()) anyerror!void {
            try fl.c.s.d.check(fl.c.s.d.api.cuMemsetD8Async(fl.f, 0, 64 << 20, fl.c.s.handle), "cuMemsetD8Async");
            if (fl.with_call) try fl.c.go();
        }
    };
    const with = try b.time(Flushed{ .c = call, .f = flush.ptr, .with_call = true }, Flushed.go);
    const alone = try b.time(Flushed{ .c = call, .f = flush.ptr, .with_call = false }, Flushed.go);
    std.debug.print("  cold L2 (64 MiB memset between calls, its time subtracted): {d:.4} ms\n", .{with - alone});
    std.debug.print("routed experts L{d} (1 row x 10 picks, act_mode {d}, {d} of 10 from mapped host memory) K={d} I={d} halves={d}/{d}/{d} ms={d:>8.4}  max|err|={e:.3} peak={d:.3} rel={e:.2}\n", .{ layer, act_mode, mapped, dims, width, g[0].halves, u[0].halves, dn[0].halves, ms, err.max_abs, err.peak, err.worst_rel });
    if (mapped > 0) {
        // the copy-engine alternative: the same experts' trellis bytes, pinned host to device, one copy an expert matrix
        var total: usize = 0;
        for (0..mapped) |e| total += (g[e].trellis.len + u[e].trellis.len + dn[e].trellis.len) * 2;
        var src = try cuda.HostBuffer.alloc(b.gpu.d, total);
        defer src.free();
        var dst = try cuda.DeviceBuffer.alloc(b.gpu.d, total);
        defer dst.free();
        const Copy = struct {
            d: *const cuda.Driver,
            s: cuda.Stream,
            src: []u8,
            dst: u64,
            fn go(c: @This()) anyerror!void {
                try c.d.check(c.d.api.cuMemcpyHtoDAsync_v2(c.dst, c.src.ptr, c.src.len, c.s.handle), "cuMemcpyHtoDAsync");
            }
        };
        const c: Copy = .{ .d = b.gpu.d, .s = b.s, .src = src.bytes[0..total], .dst = dst.ptr };
        const cms = try b.time(c, Copy.go);
        std.debug.print("  pinned H2D copy of those {d} experts' trellis ({d} bytes): {d:.4} ms ({d:.1} GB/s)\n", .{ mapped, total, cms, @as(f64, @floatFromInt(total)) / cms / 1e6 });
    }
}

/// Rows=1 timing only, warm and with a 64 MiB memset between calls (the memset's own time subtracted).
fn coldLinear(b: *Bench, de: *exl3.Dense, pack: *const Pack, a: std.mem.Allocator, rng: std.Random, name: []const u8) !void {
    const m = try mat(pack, a, name);
    const k = m.k();
    const n = m.n();
    const strides = exl3.strides(k, n, m.halves, false);
    var tr = try cuda.DeviceBuffer.fromHost(b.gpu.d, std.mem.sliceAsBytes(m.trellis));
    defer tr.free();
    var su = try cuda.DeviceBuffer.fromHost(b.gpu.d, std.mem.sliceAsBytes(m.suh));
    defer su.free();
    var sv = try cuda.DeviceBuffer.fromHost(b.gpu.d, std.mem.sliceAsBytes(m.svh));
    defer sv.free();
    const w: exl3.Weight = .{ .trellis = tr.ptr, .suh = su.ptr, .svh = sv.ptr, .k = @intCast(k), .n = @intCast(n), .halves = m.halves, .codebook = .mul1, .stride_k = strides.k, .stride_nb = strides.nb };
    const x = try randX(a, rng, k);
    var xd = try cuda.DeviceBuffer.fromHost(b.gpu.d, std.mem.sliceAsBytes(x.bits));
    defer xd.free();
    var yd = try cuda.DeviceBuffer.alloc(b.gpu.d, n * 2);
    defer yd.free();
    var flush = try cuda.DeviceBuffer.alloc(b.gpu.d, 64 << 20);
    defer flush.free();
    const call: Call = .{ .de = de, .s = b.s, .w = w, .x = xd.ptr, .y = yd.ptr, .rows = 1 };
    const warm = try b.time(call, Call.go);
    const F = struct {
        c: Call,
        f: u64,
        with_call: bool,
        fn go(fl: @This()) anyerror!void {
            try fl.c.s.d.check(fl.c.s.d.api.cuMemsetD8Async(fl.f, 0, 64 << 20, fl.c.s.handle), "cuMemsetD8Async");
            if (fl.with_call) try fl.c.go();
        }
    };
    const with = try b.time(F{ .c = call, .f = flush.ptr, .with_call = true }, F.go);
    const alone = try b.time(F{ .c = call, .f = flush.ptr, .with_call = false }, F.go);
    const bytes = m.trellis.len * 2;
    std.debug.print("{s:<58} rows=1 bytes={d:>10} warm {d:.4} ms  cold {d:.4} ms ({d:.0} GB/s cold)\n", .{ name, bytes, warm, with - alone, @as(f64, @floatFromInt(bytes)) / (with - alone) / 1e6 });
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const dir = rest[0];
    const reps: usize = if (rest.len > 1) try std.fmt.parseInt(usize, rest[1], 10) else 20;
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var pack: Pack = .{};
    defer pack.close(gpu.io);
    try pack.open(gpu, dir);
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var b: Bench = .{ .gpu = gpu, .s = s, .ev0 = try cuda.Event.init(gpu.d, true), .ev1 = try cuda.Event.init(gpu.d, true), .reps = reps };
    defer b.ev0.deinit();
    defer b.ev1.deinit();
    var de = try exl3.Dense.load(gpu.d);
    defer de.unload();
    var ex = try exl3e.Experts.load(gpu.d);
    defer ex.unload();
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rng = prng.random();

    const names = [_][]const u8{
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.k_proj",
        "model.language_model.layers.3.self_attn.v_proj",
        "model.language_model.layers.3.self_attn.o_proj",
        "model.language_model.layers.3.mlp.shared_expert.gate_proj",
        "model.language_model.layers.3.mlp.shared_expert.down_proj",
        "model.language_model.layers.3.self_attn.indexer.index_qk_proj",
        "model.language_model.layers.0.mlp.experts.0.gate_proj",
        "model.language_model.layers.0.mlp.experts.0.down_proj",
        "lm_head",
    };
    if (rest.len > 2 and std.mem.eql(u8, rest[2], "cold")) {
        for (names) |nm| try coldLinear(&b, &de, &pack, a, rng, nm);
        try experts(&b, &ex, &pack, a, rng, 17, 1, 0);
        return;
    }
    if (rest.len > 2 and std.mem.eql(u8, rest[2], "experts")) {
        for ([_]usize{ 0, 1, 4, 10 }) |mapped| try experts(&b, &ex, &pack, a, rng, 0, 1, mapped);
        return;
    }
    for (names) |nm| try linear(&b, &de, &pack, a, rng, nm, .reference);
    std.debug.print("sum of rows=1 ms over the projections above (reference Dense): {d:.3}\n", .{b.total_ms});
    b.total_ms = 0;
    for (names) |nm| try linear(&b, &de, &pack, a, rng, nm, .tiles);
    try experts(&b, &ex, &pack, a, rng, 0, 1, 0);
    try experts(&b, &ex, &pack, a, rng, 0, 0, 0);
    try experts(&b, &ex, &pack, a, rng, 17, 1, 0);
    check.pass("exl3-pack: v1.1.0 EXL3 operators ran on the pack's shapes (see the error and timing lines)", .{});
}

/// nccl-lat <rank> <id file>: two processes, one GPU each; fp32 all-reduce (sum) latency at a few sizes, cudaEvent
/// timed on rank 0 (min over 5 batches of 100). Rank 0 writes NCCL's unique id to the file, rank 1 reads it.
pub fn ncclLatency(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 2) return error.MissingArgument;
    const rank = try std.fmt.parseInt(c_int, rest[0], 10);
    const id_path = rest[1];
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    if (rank == 0) {
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        const hex = std.fmt.bytesToHex(id.internal, .lower);
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = id_path, .data = &hex });
    } else {
        var hex: [256]u8 = undefined;
        var tries: usize = 0;
        while (true) : (tries += 1) {
            const got = std.Io.Dir.cwd().readFile(gpu.io, id_path, &hex) catch {
                if (tries > 200) return error.NoUniqueId;
                try std.Io.sleep(gpu.io, .fromMilliseconds(50), .awake);
                continue;
            };
            if (got.len == 256) break;
            if (tries > 200) return error.NoUniqueId;
            try std.Io.sleep(gpu.io, .fromMilliseconds(50), .awake);
        }
        _ = try std.fmt.hexToBytes(&id.internal, &hex);
    }
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, 2, id, rank), "ncclCommInitRank");
    defer _ = lib.api.ncclCommDestroy(comm);
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var ev0 = try cuda.Event.init(gpu.d, true);
    defer ev0.deinit();
    var ev1 = try cuda.Event.init(gpu.d, true);
    defer ev1.deinit();
    const sizes = [_]usize{ 2560, 11 * 2560, 4 * 11 * 2560, 16 * 11 * 2560, 512 * 2560, 4096 * 2560 };
    var buf = try cuda.DeviceBuffer.alloc(gpu.d, sizes[sizes.len - 1] * 4);
    defer buf.free();
    try buf.fill8(0, s.handle);
    for (sizes) |count| {
        for (0..10) |_| try lib.check(lib.api.ncclAllReduce(buf.ptr, buf.ptr, count, .f32, .sum, comm, s.handle), "ncclAllReduce");
        try s.synchronize();
        var best: f64 = std.math.inf(f64);
        for (0..5) |_| {
            try ev0.record(s);
            for (0..100) |_| try lib.check(lib.api.ncclAllReduce(buf.ptr, buf.ptr, count, .f32, .sum, comm, s.handle), "ncclAllReduce");
            try ev1.record(s);
            try ev1.synchronize();
            best = @min(best, @as(f64, try ev0.elapsedMs(ev1)) / 100.0);
        }
        if (rank == 0) std.debug.print("RESULT nccl all-reduce f32 {d:>9} values ({d:>9} bytes): {d:.4} ms\n", .{ count, count * 4, best });
    }
}
