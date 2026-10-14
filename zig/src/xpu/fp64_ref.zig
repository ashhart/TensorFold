//! XPU_FP64_REF=1: the int4 GEMMs on the prompt's real activations against an FP64 host reference, same int4 weights.
const std = @import("std");
const q = @import("qwen27.zig");
const fwd = @import("forward.zig");
const ze = q.ze;
const Dev = q.Dev;
const Acts = q.Acts;
const Proj = q.Proj;
const alloc = q.alloc;

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
    std.debug.print("| checkpoint | layer | projection | path | outputs | max abs | mean abs | rel L2 | " ++
        "rel L2 vs fp16(FP64) | max ulp | 0 ulp | ≤ 1 ulp |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n", .{});
}

pub fn disarm() void {
    fwd.gemm_probe = null;
}

const Stats = struct {
    n: usize = 0,
    max_abs: f64 = 0,
    sum_abs: f64 = 0,
    sq_err: f64 = 0,
    sq_ref: f64 = 0,
    sq_rn: f64 = 0, // against fp16(FP64): what is left once the output's own rounding is set aside
    max_ulp: u32 = 0,
    ulp0: usize = 0,
    ulp1: usize = 0,
    bad: usize = 0, // non-finite outputs

    fn add(s: *Stats, out: f16, ref: f64) void {
        if (!std.math.isFinite(out)) {
            s.bad += 1;
            return;
        }
        const e = @as(f64, out) - ref;
        s.n += 1;
        s.max_abs = @max(s.max_abs, @abs(e));
        s.sum_abs += @abs(e);
        s.sq_err += e * e;
        s.sq_ref += ref * ref;
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
    s_bf16: bool = false,
    n: usize,
    k: usize,
    gs: usize = 64,
    col0: usize = 0, // first device column (stacked parts)
    ilv: usize = 0, // gate|up: 1 gate, 2 up, interleaved by 16 columns
    nt: usize = 0, // device layout: columns of the stored projection

    fn col(p: *const Part, n: usize) usize {
        return if (p.ilv != 0) (n / 16) * 32 + (p.ilv - 1) * 16 + n % 16 else p.col0 + n;
    }

    fn half(bytes: []const u8, i: usize, bf16: bool) f32 {
        const u = std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
        return if (bf16) @bitCast(@as(u32, u) << 16) else @floatCast(@as(f16, @bitCast(u)));
    }

    fn word(bytes: []const u8, i: usize) u32 {
        return std.mem.readInt(u32, bytes[4 * i ..][0..4], .little);
    }

    /// Column n's K weights dequantized in FP64 and in fp32 (q·s + b, a product then a sum).
    fn column(p: *const Part, n: usize, w64: []f64, w32: []f32) void {
        const kw = p.k / 8;
        const groups = p.k / p.gs;
        for (0..kw) |wi| {
            const wd = switch (p.fmt) {
                .mlx => word(p.w, n * kw + wi),
                .gptq => word(p.w, wi * p.n + n),
                .dev => word(p.w, wi * p.nt + n),
            };
            for (0..8) |j| {
                const k = wi * 8 + j;
                const g = k / p.gs;
                // device words: nibble 2p in bits 4p, nibble 2p + 1 in bits 16 + 4p (glue_quant.cl repack)
                const sh: u5 = @intCast(if (p.fmt == .dev) (j % 2) * 16 + (j / 2) * 4 else j * 4);
                const qv: f32 = @floatFromInt((wd >> sh) & 0xF);
                var s: f32 = undefined;
                var b: f32 = undefined;
                switch (p.fmt) {
                    .mlx => {
                        s = half(p.s, n * groups + g, p.s_bf16);
                        b = half(p.b, n * groups + g, p.s_bf16);
                    },
                    .gptq => {
                        s = half(p.s, g * p.n + n, false);
                        b = -8.0 * s;
                    },
                    .dev => {
                        s = half(p.s, g * p.nt + n, false);
                        b = half(p.b, g * p.nt + n, false) / @as(f32, @floatFromInt(p.gs));
                    },
                }
                w64[k] = @as(f64, qv) * @as(f64, s) + @as(f64, b);
                w32[k] = qv * s + b;
            }
        }
    }
};

/// The stored parts of a projection, from the checkpoint (or from the device for a head quantized at load).
fn parts(c: *const q.Checkpoint, w: Proj, layer: usize, site: u32) ![]Part {
    const gptq = q.GS == 128;
    var list: std.ArrayList(Part) = .empty;
    if (site == 4 and gptq) { // GPTQ head: quantized to affine 4/64 at load, no int4 copy in the checkpoint
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
            const n: usize = @intCast(t.shape[0]);
            const bf = std.mem.eql(u8, s.dtype, "BF16");
            const p: Part = .{ .fmt = .mlx, .w = t.bytes, .s = s.bytes, .b = b.bytes, .s_bf16 = bf, .n = n, .k = w.k };
            var pm = with(p, col0, ilv);
            pm.gs = w.k / @as(usize, @intCast(s.shape[1]));
            try list.append(alloc, pm);
            col0 += n;
        }
    }
    return list.items;
}

fn with(p: Part, col0: usize, ilv: usize) Part {
    var o = p;
    o.col0 = col0;
    o.ilv = ilv;
    return o;
}

/// Host work: columns [from, to) of the parts' concatenation, FP64 and sequential fp32 sums over the NS rows.
const Job = struct {
    parts: []const Part,
    x64: []const f64, // [K, NS]
    x32: []const f32,
    ref: []f64, // [NS, N] by device column
    seq: []f32,
    nt: usize,
    from: usize,
    to: usize,

    fn run(j: *const Job) void {
        const k = j.parts[0].k;
        const w64 = alloc.alloc(f64, k) catch return;
        defer alloc.free(w64);
        const w32 = alloc.alloc(f32, k) catch return;
        defer alloc.free(w32);
        var base: usize = 0;
        for (j.parts) |*p| {
            defer base += p.n;
            const lo = @max(j.from, base);
            const hi = @min(j.to, base + p.n);
            var n = lo;
            while (n < hi) : (n += 1) {
                p.column(n - base, w64, w32);
                var a64: [NS]f64 = @splat(0);
                var a32: [NS]f32 = @splat(0);
                for (0..k) |kk| {
                    const xa = j.x64[kk * NS ..][0..NS];
                    const xb = j.x32[kk * NS ..][0..NS];
                    for (0..NS) |s| {
                        a64[s] += xa[s] * w64[kk];
                        a32[s] = a32[s] + xb[s] * w32[kk]; // k ascending, a product then a sum, no FMA
                    }
                }
                const c = p.col(n - base);
                for (0..NS) |s| {
                    j.ref[s * j.nt + c] = a64[s];
                    j.seq[s * j.nt + c] = a32[s];
                }
            }
        }
    }
};

fn readRow(d: *Dev, dst: []f16, src: *anyopaque, row: usize, n: usize) !void {
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dst.ptr, q.off(src, row * n * 2), n * 2, null, 0, null));
}

/// The probe: decode GEMMs on 1 and 8 rows, the prompt GEMM on all rows, then the host reference on NS rows.
fn probe(d: *Dev, a: *Acts, w: Proj, x: *anyopaque, T: u32, layer: usize, site: u32) !void {
    if (T < NS or (layer < q.LAYERS and std.mem.indexOfScalar(usize, &LAYERS_ON, layer) == null)) return;
    const t0 = q.now();
    try d.sync();
    var rows: [NS]usize = undefined;
    for (&rows, 0..) |*r, i| r.* = if (i < 8) i else 8 + (i - 8) * (T - 8) / 16;
    const K: usize = w.k;
    const N: usize = w.n;
    // inputs: the GEMM's own fp16 rows
    const xh = try alloc.alloc(f16, T * K);
    defer alloc.free(xh);
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, xh.ptr, x, xh.len * 2, null, 0, null));
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
    // host reference
    const x64 = try alloc.alloc(f64, K * NS);
    defer alloc.free(x64);
    const x32 = try alloc.alloc(f32, K * NS);
    defer alloc.free(x32);
    for (0..K) |k| for (0..NS) |s| {
        x32[k * NS + s] = xh[rows[s] * K + k];
        x64[k * NS + s] = xh[rows[s] * K + k];
    };
    const ps = try parts(&runner.ckpt, w, layer, site);
    var total: usize = 0;
    for (ps) |p| total += p.n;
    const ref = try alloc.alloc(f64, NS * N);
    defer alloc.free(ref);
    const seq = try alloc.alloc(f32, NS * N);
    defer alloc.free(seq);
    const workers = @min(@max(std.Thread.getCpuCount() catch 1, 1), 64);
    var jobs: [64]Job = undefined;
    var threads: [64]?std.Thread = @splat(null);
    for (0..workers) |i| {
        const from = total * i / workers;
        const to = total * (i + 1) / workers;
        jobs[i] = .{ .parts = ps, .x64 = x64, .x32 = x32, .ref = ref, .seq = seq, .nt = N, .from = from, .to = to };
        threads[i] = std.Thread.spawn(.{}, Job.run, .{&jobs[i]}) catch null;
        if (threads[i] == null) jobs[i].run();
    }
    for (threads[0..workers]) |t| if (t) |th| th.join();
    // statistics over the stored columns only (the tile padding is left out)
    var st: [5]Stats = @splat(.{});
    for (ps) |*p| for (0..p.n) |n| {
        const c = p.col(n);
        for (0..NS) |s| {
            const r = ref[s * N + c];
            if (s == 0) st[0].add(out[c], r);
            if (s < 8) st[1].add(out[(NS + s) * N + c], r);
            if (m_prompt != 0) st[2].add(out[(2 * NS + s) * N + c], r);
            st[3].add(@floatCast(seq[s * N + c]), r);
            st[4].add(@floatCast(r), r);
        }
    };
    if (site == 4 and q.GS == 128) for (ps) |p| {
        alloc.free(p.w);
        alloc.free(p.s);
        alloc.free(p.b);
    };
    const ck = if (q.GS == 128) "GPTQ 4/128" else "MLX 4/64";
    for (st, 0..) |s, i| {
        if (s.n == 0) continue;
        const nf: f64 = @floatFromInt(s.n);
        const lname = if (layer == q.LAYERS) "head" else try std.fmt.allocPrint(alloc, "{d}", .{layer});
        const rel = @sqrt(s.sq_err / s.sq_ref);
        const rel_rn = @sqrt(s.sq_rn / s.sq_ref);
        const p0 = 100 * @as(f64, @floatFromInt(s.ulp0)) / nf;
        const p1 = 100 * @as(f64, @floatFromInt(s.ulp1)) / nf;
        std.debug.print("| {s} | {s} | {s} | {s} | {d} ", .{ ck, lname, SITES[site], PATHS[i], s.n });
        std.debug.print("| {e:.2} | {e:.2} | {e:.2} | {e:.2} ", .{ s.max_abs, s.sum_abs / nf, rel, rel_rn });
        std.debug.print("| {d} | {d:.1} % | {d:.1} % |", .{ s.max_ulp, p0, p1 });
        if (s.bad != 0) std.debug.print(" {d} non-finite", .{s.bad});
        std.debug.print("\n", .{});
    }
    const dt = q.now() - t0;
    std.debug.print("fp64 ref: layer {d} {s} (N {d}, K {d}) in {d:.1} s\n", .{ layer, SITES[site], total, K, dt });
}
