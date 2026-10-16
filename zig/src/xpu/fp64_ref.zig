//! int4 GEMMs against FP64 references (XPU_FP64_REF) and teacher-forced top tokens (XPU_FIDELITY), runner only.
const std = @import("std");
const core = @import("core");
const q = @import("qwen27.zig");
const fwd = @import("forward.zig");
const ze = q.ze;
const Dev = q.Dev;
const Acts = q.Acts;
const Proj = q.Proj;
const alloc = q.alloc;
const qmlx = core.quant.mlx;
const QTensor = core.quant.Tensor;
extern "c" fn memfd_create(name: [*:0]const u8, flags: c_uint) c_int;
extern "c" fn pread(fd: c_int, buf: [*]u8, n: usize, off: i64) isize;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;

/// Layers measured: two GDN (0, 1), two attention (31, 63); the head comes as layer LAYERS.
const LAYERS_ON = [_]usize{ 0, 1, 31, 63 };
const SITES = [_][]const u8{ "mixer_in", "mixer_out", "gate_up", "down", "head" };
const PATHS = [_][]const u8{ "decode, 1 row", "decode, 8 rows", "prompt (nGEN)", "fp32 sequential", "fp16(FP64)" };
/// Rows compared: 0 .. 7 (the decode paths), then 16 spread over the prompt.
const NS = 24;

var runner: *q.Runner = undefined;

pub fn arm(r: *q.Runner) void {
    runner = r;
    fwd.gemm_probe = probe;
    std.debug.print("| checkpoint | layer | projection | path | outputs | max abs | max abs / rms | rel L2 | " ++
        "rel L2 vs fp16(FP64) | worst column rel L2 (column) | max ulp | 0 ulp | ≤ 1 ulp |\n" ++
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n", .{});
}

pub fn disarm() void {
    fwd.gemm_probe = null;
}

const Stats = struct {
    n: usize = 0,
    max_abs: f64 = 0,
    sq_err: f64 = 0,
    sq_ref: f64 = 0,
    sq_rn: f64 = 0, // against fp16(FP64): what is left once the output's own rounding is set aside
    max_ulp: u32 = 0,
    ulp0: usize = 0,
    ulp1: usize = 0,
    bad: usize = 0, // non-finite outputs
    col_err: []f64 = &.{}, // by device column: squared errors and squared references
    col_ref: []f64 = &.{},

    fn add(s: *Stats, out: f16, ref: f64, c: usize) void {
        if (!std.math.isFinite(out)) {
            s.bad += 1;
            return;
        }
        const e = @as(f64, out) - ref;
        s.n += 1;
        s.max_abs = @max(s.max_abs, @abs(e));
        s.sq_err += e * e;
        s.sq_ref += ref * ref;
        s.col_err[c] += e * e;
        s.col_ref[c] += ref * ref;
        const rn: f16 = @floatCast(ref);
        const er = @as(f64, out) - @as(f64, rn);
        s.sq_rn += er * er;
        const u = ulps(out, rn);
        s.max_ulp = @max(s.max_ulp, u);
        if (u == 0) s.ulp0 += 1;
        if (u <= 1) s.ulp1 += 1;
    }
};

/// fp16 values in their order on the line: the distance between two of them counts the fp16 steps between them.
fn ordered(h: f16) i32 {
    const b: u16 = @bitCast(h);
    return if (b & 0x8000 != 0) -@as(i32, b & 0x7fff) else @as(i32, b);
}

fn ulps(a: f16, b: f16) u32 {
    return @abs(ordered(a) - ordered(b));
}

/// One stored part of a projection: MLX affine (words [N, K/8]), GPTQ sym (qweight [K/8, N]) or the device layout.
const Part = struct {
    fmt: enum { mlx, gptq, dev },
    w: []const u8,
    s: []const u8,
    b: []const u8 = &.{},
    tables: core.quant.DType = .f16, // MLX: the checkpoint's table dtype
    n: usize,
    k: usize,
    gs: usize = 64,
    col0: usize = 0, // first device column (stacked parts)
    base: usize = 0, // first column in the order of the checkpoint's parts
    ilv: usize = 0, // gate|up: 1 gate, 2 up, interleaved by 16 columns
    nt: usize = 0, // device layout: columns of the stored projection

    fn col(p: *const Part, n: usize) usize {
        return if (p.ilv != 0) (n / 16) * 32 + (p.ilv - 1) * 16 + n % 16 else p.col0 + n;
    }

    fn half(bytes: []const u8, i: usize) f32 {
        return @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little))));
    }

    fn word(bytes: []const u8, i: usize) u32 {
        return std.mem.readInt(u32, bytes[4 * i ..][0..4], .little);
    }

    fn tensor(dtype: core.quant.DType, cols: usize, bytes: []const u8) QTensor {
        return .{ .dtype = dtype, .rank = 2, .shape = .{ 1, cols, 1, 1, 1 }, .bytes = bytes };
    }

    /// Column n as an MLX affine row (GPTQ: bias −8·scale) dequantized by core/quant: FP64 and torch's fp32.
    fn column(p: *const Part, n: usize, cb: *ColBuf, w64: []f64, w32: []f32) !void {
        const kw = p.k / 8;
        const groups = p.k / p.gs;
        var h: qmlx.Host = undefined;
        switch (p.fmt) {
            .mlx => h = .{
                .words = tensor(.i32, kw, p.w[n * kw * 4 ..][0 .. kw * 4]),
                .scales = tensor(p.tables, groups, p.s[n * groups * 2 ..][0 .. groups * 2]),
                .biases = tensor(p.tables, groups, p.b[n * groups * 2 ..][0 .. groups * 2]),
                .bits = 4,
                .group = @intCast(p.gs),
            },
            .gptq, .dev => {
                for (0..kw) |wi| {
                    if (p.fmt == .gptq) {
                        cb.words[wi] = word(p.w, wi * p.n + n);
                        continue;
                    }
                    // device words: nibble 2i in bits 4i, nibble 2i + 1 in bits 16 + 4i (glue_quant.cl repack)
                    const o = word(p.w, wi * p.nt + n);
                    var m: u32 = 0;
                    for (0..4) |i| {
                        const sh: u5 = @intCast(4 * i);
                        m |= ((o >> sh) & 0xF) << (2 * sh);
                        m |= ((o >> (sh + 16)) & 0xF) << (2 * sh + 4);
                    }
                    cb.words[wi] = m;
                }
                for (0..groups) |g| if (p.fmt == .gptq) {
                    cb.s[g] = half(p.s, g * p.n + n);
                    cb.b[g] = -8.0 * cb.s[g];
                } else {
                    cb.s[g] = half(p.s, g * p.nt + n);
                    cb.b[g] = half(p.b, g * p.nt + n) * 256.0 / @as(f32, @floatFromInt(p.gs));
                };
                h = .{
                    .words = tensor(.i32, kw, std.mem.sliceAsBytes(cb.words[0..kw])),
                    .scales = tensor(.f32, groups, std.mem.sliceAsBytes(cb.s[0..groups])),
                    .biases = tensor(.f32, groups, std.mem.sliceAsBytes(cb.b[0..groups])),
                    .bits = 4,
                    .group = @intCast(p.gs),
                };
            },
        }
        try qmlx.reference(h, w64);
        if (w32.len == 0) return;
        var fba = std.heap.FixedBufferAllocator.init(cb.scratch);
        @memcpy(w32, try qmlx.dequant(fba.allocator(), h.words, h.scales, h.biases, p.gs));
    }
};

const ColBuf = struct { words: []u32, s: []f32, b: []f32, scratch: []u8 };

fn with(p: Part, col0: usize, ilv: usize) Part {
    var o = p;
    o.col0 = col0;
    o.base = col0;
    o.ilv = ilv;
    return o;
}

var head_cache: ?[]Part = null;

/// The stored parts of a projection, from the checkpoint (or from the device for a head quantized at load).
fn parts(c: *const q.Checkpoint, w: Proj, layer: usize, site: u32) ![]Part {
    const gptq = q.GS == 128;
    var list: std.ArrayList(Part) = .empty;
    if (site == 4 and gptq) { // GPTQ head: quantized to affine 4/64 at load, no int4 copy in the checkpoint
        if (head_cache) |h| return h;
        const d = &runner.d;
        const words = try alloc.alloc(u8, w.k / 8 * w.n * 4);
        const sc = try alloc.alloc(u8, w.k / w.gs * w.n * 2);
        const bg = try alloc.alloc(u8, w.k / w.gs * w.n * 2);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, words.ptr, w.w, words.len, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, sc.ptr, w.s, sc.len, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, bg.ptr, w.bg, bg.len, null, 0, null));
        try d.sync();
        const p: Part = .{ .fmt = .dev, .w = words, .s = sc, .b = bg, .n = w.n, .k = w.k, .gs = w.gs, .nt = w.n };
        try list.append(alloc, p);
        head_cache = list.items;
        return list.items;
    }
    const attn = layer < q.LAYERS and q.isAttention(layer);
    const names: []const []const u8 = switch (site) {
        0 => if (attn) &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" } else if (gptq)
            &.{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z" }
        else
            &.{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b" },
        1 => if (attn) &.{"self_attn.o_proj"} else &.{"linear_attn.out_proj"},
        2 => &.{ "mlp.gate_proj", "mlp.up_proj" },
        3 => &.{"mlp.down_proj"},
        else => &.{"lm_head"},
    };
    var col0: usize = 0;
    for (names, 0..) |name, i| {
        const pre = if (site == 4) name else try std.fmt.allocPrint(alloc, "model.layers.{d}.{s}", .{ layer, name });
        const ilv: usize = if (site == 2) i + 1 else 0;
        if (gptq) {
            const t = try c.get(try std.fmt.allocPrint(alloc, "{s}.qweight", .{pre}));
            const s = try c.get(try std.fmt.allocPrint(alloc, "{s}.scales", .{pre}));
            const n: usize = @intCast(t.shape[1]);
            const p: Part = .{ .fmt = .gptq, .w = t.bytes, .s = s.bytes, .n = n, .k = w.k, .gs = 128 };
            try list.append(alloc, with(p, col0, ilv));
            col0 += n;
        } else {
            const t = try c.get(try std.fmt.allocPrint(alloc, "{s}.weight", .{pre}));
            const s = try c.get(try std.fmt.allocPrint(alloc, "{s}.scales", .{pre}));
            const b = try c.get(try std.fmt.allocPrint(alloc, "{s}.biases", .{pre}));
            if (!std.mem.eql(u8, s.dtype, b.dtype)) return error.UnexpectedDtype;
            const dt: core.quant.DType = if (std.mem.eql(u8, s.dtype, "BF16")) .bf16 else .f16;
            const n: usize = @intCast(t.shape[0]);
            const p: Part = .{ .fmt = .mlx, .w = t.bytes, .s = s.bytes, .b = b.bytes, .tables = dt, .n = n, .k = w.k };
            var pm = with(p, col0, ilv);
            pm.gs = w.k / @as(usize, @intCast(s.shape[1]));
            try list.append(alloc, pm);
            col0 += n;
        }
    }
    return list.items;
}

/// Host work: columns [from, to) of the parts' concatenation, FP64 (and fp32 sequential) sums over R rows.
const Job = struct {
    parts: []const Part,
    x64: []const f64, // [K, R]
    x32: []const f32, // [K, R], empty: no fp32 sums
    rows: usize,
    ref: []f64, // [R, N] by device column
    seq: []f32,
    nt: usize,
    from: usize = 0,
    to: usize = 0,
    failed: bool = false,

    fn run(j: *Job) void {
        j.work() catch {
            j.failed = true;
        };
    }

    fn work(j: *Job) !void {
        const k = j.parts[0].k;
        const R = j.rows;
        const fp32 = j.x32.len != 0;
        const w64 = try alloc.alloc(f64, k);
        defer alloc.free(w64);
        const w32 = try alloc.alloc(f32, if (fp32) k else 0);
        defer alloc.free(w32);
        const a64 = try alloc.alloc(f64, R);
        defer alloc.free(a64);
        const a32 = try alloc.alloc(f32, R);
        defer alloc.free(a32);
        var cb: ColBuf = .{
            .words = try alloc.alloc(u32, k / 8),
            .s = try alloc.alloc(f32, k / 32),
            .b = try alloc.alloc(f32, k / 32),
            .scratch = try alloc.alloc(u8, k * 4 + 4096),
        };
        defer {
            alloc.free(cb.words);
            alloc.free(cb.s);
            alloc.free(cb.b);
            alloc.free(cb.scratch);
        }
        var base: usize = 0;
        for (j.parts) |*p| {
            defer base += p.n;
            var n = @max(j.from, base);
            while (n < @min(j.to, base + p.n)) : (n += 1) {
                try p.column(n - base, &cb, w64, w32);
                @memset(a64, 0);
                @memset(a32, 0);
                for (0..k) |kk| {
                    const xa = j.x64[kk * R ..][0..R];
                    for (a64, xa) |*acc, xv| acc.* += xv * w64[kk];
                    if (!fp32) continue;
                    const xb = j.x32[kk * R ..][0..R];
                    for (a32, xb) |*acc, xv| acc.* = acc.* + xv * w32[kk]; // k ascending, product then sum, no FMA
                }
                const c = p.col(n - base);
                for (0..R) |s| {
                    j.ref[s * j.nt + c] = a64[s];
                    if (fp32) j.seq[s * j.nt + c] = a32[s];
                }
            }
        }
    }
};

/// The parts' products with R fp16 rows (xh [R, K]) in FP64, and in sequential fp32 when `seq` is not empty.
fn hostGemm(ps: []const Part, xh: []const f16, R: usize, nt: usize, ref: []f64, seq: []f32) !void {
    const K = ps[0].k;
    const x64 = try alloc.alloc(f64, K * R);
    defer alloc.free(x64);
    const x32 = try alloc.alloc(f32, if (seq.len != 0) K * R else 0);
    defer alloc.free(x32);
    for (0..R) |s| for (0..K) |k| {
        x64[k * R + s] = xh[s * K + k];
        if (seq.len != 0) x32[k * R + s] = xh[s * K + k];
    };
    var total: usize = 0;
    for (ps) |p| total += p.n;
    const workers = @min(@max(std.Thread.getCpuCount() catch 1, 1), 64);
    var jobs: [64]Job = undefined;
    var threads: [64]?std.Thread = @splat(null);
    for (0..workers) |i| {
        const from = total * i / workers;
        const to = total * (i + 1) / workers;
        jobs[i] = .{ .parts = ps, .x64 = x64, .x32 = x32, .rows = R, .ref = ref, .seq = seq, .nt = nt };
        jobs[i].from = from;
        jobs[i].to = to;
        threads[i] = std.Thread.spawn(.{}, Job.run, .{&jobs[i]}) catch null;
        if (threads[i] == null) jobs[i].run();
    }
    for (threads[0..workers]) |t| if (t) |th| th.join();
    for (jobs[0..workers]) |j| if (j.failed) return error.ReferenceFailed;
}

fn readRow(d: *Dev, dst: []f16, src: *anyopaque, row: usize, n: usize) !void {
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dst.ptr, q.off(src, row * n * 2), n * 2, null, 0, null));
}

/// The probe: decode GEMMs on 1 and 8 rows, the prompt GEMM on all rows, then the host references on NS rows.
fn probe(d: *Dev, a: *Acts, w: Proj, x: *anyopaque, T: u32, layer: usize, site: u32) !void {
    if (T < NS or (layer < q.LAYERS and std.mem.indexOfScalar(usize, &LAYERS_ON, layer) == null)) return;
    const t0 = q.now();
    try d.sync();
    var rows: [NS]usize = undefined;
    for (&rows, 0..) |*r, i| r.* = if (i < 8) i else 8 + (i - 8) * (T - 8) / 16;
    const K: usize = w.k;
    const N: usize = w.n;
    const xa = try alloc.alloc(f16, T * K);
    defer alloc.free(xa);
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, xa.ptr, x, xa.len * 2, null, 0, null));
    // device outputs: path 0 row 0, path 1 rows 0 .. 7, path 2 the NS rows
    const out = try alloc.alloc(f16, 3 * NS * N);
    defer alloc.free(out);
    const m_prompt: u32 = if (site == 4) 0 else T; // the head never runs as a prompt GEMM
    const scratch = try d.g.deviceAlloc(@max(T, 8) * N * 2);
    defer d.g.free(scratch);
    for ([_]u32{ 1, 8, m_prompt }, 0..) |m, path| {
        if (m == 0) continue;
        try fwd.gemm(d, a.pscr, x, a.ahi, a.alo, w, scratch, m);
        try d.sync();
        const nr: usize = if (m == 1) 1 else if (m == 8) 8 else NS;
        for (0..nr) |i| try readRow(d, out[(path * NS + i) * N ..][0..N], scratch, rows[i], N);
    }
    try d.sync();
    const xh = try alloc.alloc(f16, NS * K);
    defer alloc.free(xh);
    for (rows, 0..) |r, s| @memcpy(xh[s * K ..][0..K], xa[r * K ..][0..K]);
    const ps = try parts(&runner.ckpt, w, layer, site);
    const ref = try alloc.alloc(f64, NS * N);
    defer alloc.free(ref);
    const seq = try alloc.alloc(f32, NS * N);
    defer alloc.free(seq);
    try hostGemm(ps, xh, NS, N, ref, seq);
    // statistics over the stored columns only (the tile padding is left out)
    var st: [5]Stats = @splat(.{});
    for (&st) |*s| {
        s.col_err = try alloc.alloc(f64, N);
        s.col_ref = try alloc.alloc(f64, N);
        @memset(s.col_err, 0);
        @memset(s.col_ref, 0);
    }
    defer for (&st) |*s| {
        alloc.free(s.col_err);
        alloc.free(s.col_ref);
    };
    for (ps) |*p| for (0..p.n) |n| {
        const c = p.col(n);
        for (0..NS) |s| {
            const r = ref[s * N + c];
            if (s == 0) st[0].add(out[c], r, c);
            if (s < 8) st[1].add(out[(NS + s) * N + c], r, c);
            if (m_prompt != 0) st[2].add(out[(2 * NS + s) * N + c], r, c);
            st[3].add(@floatCast(seq[s * N + c]), r, c);
            st[4].add(@floatCast(r), r, c);
        }
    };
    const ck = if (q.GS == 128) "GPTQ 4/128" else "MLX 4/64";
    const lname = if (layer == q.LAYERS) "head" else try std.fmt.allocPrint(alloc, "{d}", .{layer});
    for (st, 0..) |s, i| {
        if (s.n == 0) continue;
        const nf: f64 = @floatFromInt(s.n);
        const rms = @sqrt(s.sq_ref / nf);
        var worst: f64 = 0;
        var wcol: usize = 0;
        for (ps) |*p| for (0..p.n) |n| {
            const c = p.col(n);
            if (s.col_ref[c] == 0) continue;
            const e = @sqrt(s.col_err[c] / s.col_ref[c]);
            if (e > worst) {
                worst = e;
                wcol = p.base + n;
            }
        };
        const rel = @sqrt(s.sq_err / s.sq_ref);
        const rel_rn = @sqrt(s.sq_rn / s.sq_ref);
        const p0 = 100 * @as(f64, @floatFromInt(s.ulp0)) / nf;
        const p1 = 100 * @as(f64, @floatFromInt(s.ulp1)) / nf;
        std.debug.print("| {s} | {s} | {s} | {s} | {d} ", .{ ck, lname, SITES[site], PATHS[i], s.n });
        std.debug.print("| {e:.2} | {e:.2} | {e:.2} | {e:.2} ", .{ s.max_abs, s.max_abs / rms, rel, rel_rn });
        std.debug.print("| {e:.2} ({d}) | {d} | {d:.1} % | {d:.1} % |", .{ worst, wcol, s.max_ulp, p0, p1 });
        if (s.bad != 0) std.debug.print(" {d} non-finite", .{s.bad});
        std.debug.print("\n", .{});
    }
    std.debug.print("fp64 ref: layer {d} {s} (K {d}) in {d:.1} s\n", .{ layer, SITES[site], K, q.now() - t0 });
}

// ---------------------------------------------------------------- teacher-forced fidelity

/// The FP64 forward's GEMMs: each XPU GEMM output replaced by fp16(FP64) of the same int4 weights and input rows.
fn replace(d: *Dev, h: *anyopaque, w: Proj, out: *anyopaque, m: u32) !void {
    const r = runner;
    var layer: usize = q.LAYERS;
    var site: u32 = 4;
    for (&r.model.layers, 0..) |*l, i| {
        for ([_]Proj{ l.mixer_in, l.mixer_out, l.gu, l.down }, 0..) |p, si| if (p.w == w.w) {
            layer = i;
            site = @intCast(si);
        };
    }
    if (site == 4 and w.w != r.model.head.w) return error.UnknownProjection;
    try d.sync();
    const K: usize = w.k;
    const N: usize = w.n;
    const xh = try alloc.alloc(f16, m * K);
    defer alloc.free(xh);
    const o = try alloc.alloc(f16, m * N);
    defer alloc.free(o);
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, xh.ptr, h, xh.len * 2, null, 0, null));
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, o.ptr, out, o.len * 2, null, 0, null));
    try d.sync();
    const ps = try parts(&r.ckpt, w, layer, site);
    const ref = try alloc.alloc(f64, m * N);
    defer alloc.free(ref);
    try hostGemm(ps, xh, m, N, ref, &.{});
    for (ps) |*p| for (0..p.n) |n| {
        const c = p.col(n);
        for (0..m) |s| o[s * N + c] = @floatCast(ref[s * N + c]);
    };
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, out, o.ptr, o.len * 2, null, 0, null));
    try d.sync();
}

/// The prompt's final normed hidden rows (fp16 [T, D]) through XPU_DUMP_H's path, into memory.
fn hidden(r: *q.Runner, ids: []const u32) ![]f16 {
    const fd = memfd_create("xpu_fidelity", 0);
    if (fd < 0) return error.MemfdFailed;
    defer _ = close(fd);
    try r.reset();
    q.dump_h = fd;
    defer q.dump_h = null;
    try r.prompt(ids, true);
    const h = try alloc.alloc(f16, ids.len * q.D);
    const bytes = std.mem.sliceAsBytes(h);
    if (pread(fd, bytes.ptr, bytes.len, 0) != bytes.len) return error.ShortRead;
    return h;
}

const Top = struct { t1: u32 = 0, v1: f64 = -std.math.inf(f64), t2: u32 = 0, v2: f64 = -std.math.inf(f64) };

fn push(t: *Top, i: u32, v: f64) void {
    if (v > t.v1) {
        t.t2 = t.t1;
        t.v2 = t.v1;
        t.t1 = i;
        t.v1 = v;
    } else if (v > t.v2) {
        t.t2 = i;
        t.v2 = v;
    }
}

/// The two top tokens of each row's logits by the FP64 head (same int4 head weights).
fn headTops(h: []const f16, T: usize) ![]Top {
    const head = runner.model.head;
    const ps = try parts(&runner.ckpt, head, q.LAYERS, 4);
    const tops = try alloc.alloc(Top, T);
    @memset(tops, .{});
    const step: usize = 64; // rows at a time: the FP64 logits of a block fit in memory
    var r0: usize = 0;
    while (r0 < T) : (r0 += step) {
        const R: usize = @min(step, T - r0);
        const ref = try alloc.alloc(f64, R * head.n);
        defer alloc.free(ref);
        try hostGemm(ps, h[r0 * q.D ..][0 .. R * q.D], R, head.n, ref, &.{});
        for (0..R) |s| for (0..q.VOCAB) |c| push(&tops[r0 + s], @intCast(c), ref[s * head.n + c]);
    }
    return tops;
}

/// The FP64 forward's top tokens, kept in XPU_FIDELITY_CACHE (a file) when set: they do not depend on the kernels.
fn referenceTops(r: *q.Runner, ids: []const u32) ![]Top {
    const T = ids.len;
    const path = if (std.c.getenv("XPU_FIDELITY_CACHE")) |v| v else null;
    if (path) |p| {
        const fd = open(p, 0);
        if (fd >= 0) {
            defer _ = close(fd);
            const tops = try alloc.alloc(Top, T);
            const bytes = std.mem.sliceAsBytes(tops);
            if (pread(fd, bytes.ptr, bytes.len, 0) == bytes.len) return tops;
        }
    }
    fwd.gemm_hook = replace;
    const h = hidden(r, ids);
    fwd.gemm_hook = null;
    const tops = try headTops(try h, T);
    if (path) |p| {
        const fd = open(p, 0o1101, @as(c_uint, 0o644)); // write, create, truncate
        if (fd >= 0) {
            _ = write(fd, std.mem.sliceAsBytes(tops).ptr, std.mem.sliceAsBytes(tops).len);
            _ = close(fd);
        }
    }
    return tops;
}

/// Top tokens of the prompt path (head in FP64) and the decode path against the FP64-GEMM forward, every position.
pub fn fidelity(r: *q.Runner, ids: []const u32) !void {
    runner = r;
    const T = ids.len;
    if (T > q.CHUNK) return error.PromptLongerThanAChunk;
    const t0 = q.now();
    const ours = try headTops(try hidden(r, ids), T);
    const ref = try referenceTops(r, ids);
    const t1 = q.now();
    const dec = try alloc.alloc(Top, T);
    const logits = try alloc.alloc(f16, q.VOCAB);
    defer alloc.free(logits);
    try r.reset();
    for (0..T) |i| {
        if (i == 0) try r.prompt(ids[0..1], true) else try r.step(ids[i]);
        const api = r.d.g.api;
        const src = r.acts.logits;
        try ze.check(api.zeCommandListAppendMemoryCopy(r.d.list, logits.ptr, src, q.VOCAB * 2, null, 0, null));
        try r.d.sync();
        dec[i] = .{};
        for (logits, 0..) |v, c| push(&dec[i], @intCast(c), v);
        if (dec[i].t1 != r.last) return error.DecodeArgmaxMismatch;
    }
    std.debug.print("fidelity: {d} positions, FP64 forward {d:.0} s, decode {d:.0} s\n", .{ T, t1 - t0, q.now() - t1 });
    var min_margin: f64 = std.math.inf(f64);
    for (ref) |t| min_margin = @min(min_margin, t.v1 - t.v2);
    std.debug.print("fidelity: smallest FP64 margin {e:.2}\n", .{min_margin});
    for ([_][]const Top{ ours, dec }, [_][]const u8{ "prompt", "decode" }) |path, name| {
        var diff: usize = 0;
        for (path, ref, 0..) |a, f, i| if (a.t1 != f.t1) {
            diff += 1;
            std.debug.print("| {s} | {d} | {d} ", .{ name, i, ids[@min(i + 1, T - 1)] });
            std.debug.print("| {d} ({d:.4}) | {d} ({d:.4}) | {e:.2} ", .{ f.t1, f.v1, f.t2, f.v2, f.v1 - f.v2 });
            std.debug.print("| {d} ({d:.4}) | {d} ({d:.4}) | {e:.2} |\n", .{ a.t1, a.v1, a.t2, a.v2, a.v1 - a.v2 });
        };
        std.debug.print("fidelity: {s} path, top-1 differences against FP64: {d} of {d}\n", .{ name, diff, T });
    }
    var pd: usize = 0;
    for (ours, dec) |a, b| pd += @intFromBool(a.t1 != b.t1);
    std.debug.print("fidelity: prompt and decode paths differ at {d} of {d} positions\n", .{ pd, T });
}
