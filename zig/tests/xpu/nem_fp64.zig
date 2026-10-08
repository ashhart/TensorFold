//! FP64 reference per Nemotron-H projection class against the device kernels; --prompt FILE uses captured activations.

const std = @import("std");
const rt = @import("rig.zig");
const stop = @import("xpu").stop;
const ld = @import("xpu").loader;
const cfgm = @import("nemotron_xpu").cfg;
const model = @import("nemotron_xpu").model;
const nw = @import("nemotron_xpu").win;
const cap = @import("nem_fp64_capture.zig");

pub const gpa = std.heap.page_allocator;
const Buf = rt.Buffer;

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);

pub fn run(k: *rt.Kernel, groups: [3]u32, a: anytype) !void {
    inline for (a, 0..) |v, i| {
        const T = @TypeOf(v);
        if (T == Buf) {
            try k.setBuffer(i, v);
        } else if (T == f32) {
            try k.setF32(i, v);
        } else try k.setU32(i, @as(u32, v));
    }
    try k.launch(groups);
}

pub fn bf(u: u16) f64 {
    return @as(f32, @bitCast(@as(u32, u) << 16));
}

fn toBf(x: f64) u16 { // round to nearest even
    const f: f32 = @floatCast(x);
    const b: u32 = @bitCast(f);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// Quantized table copied to the host (the first `rows` rows).
pub const Host = struct {
    w: []u32,
    s: []u16,
    b: []u16,
    in: usize,

    pub fn fetch(r: *rt.Runtime, t: model.Table, rows: usize, in: usize) !Host {
        const h: Host = .{ .w = try gpa.alloc(u32, rows * in / 8), .s = try gpa.alloc(u16, rows * in / 64), .b = try gpa.alloc(u16, rows * in / 64), .in = in };
        try r.download(std.mem.sliceAsBytes(h.w), t.w);
        try r.download(std.mem.sliceAsBytes(h.s), t.s);
        try r.download(std.mem.sliceAsBytes(h.b), t.b);
        try r.sync();
        return h;
    }

    pub fn free(h: Host) void {
        gpa.free(h.w);
        gpa.free(h.s);
        gpa.free(h.b);
    }

    /// sum_i x[i] * (code * scale + bias) of one row in f64 (the dequantized weight is exact in f64).
    fn dot(h: Host, row: usize, x: []const f64) f64 {
        const words = h.in / 8;
        const groups = h.in / 64;
        var y: f64 = 0;
        for (0..groups) |g| {
            var d: f64 = 0;
            var sx: f64 = 0;
            for (0..8) |wj| {
                const pk = h.w[row * words + g * 8 + wj];
                for (0..8) |j| {
                    const v = x[g * 64 + wj * 8 + j];
                    d += v * @as(f64, @floatFromInt((pk >> @as(u5, @intCast(4 * j))) & 15));
                    sx += v;
                }
            }
            y += bf(h.s[row * groups + g]) * d + bf(h.b[row * groups + g]) * sx;
        }
        return y;
    }
};

pub const Ctx = struct {
    r: *rt.Runtime,
    m: *model.Model,
    w: *nw.Win,
    prng: std.Random.DefaultPrng,
    xb: Buf, // input rows (bf16) of the dense classes
    yb: Buf, // output rows
    kb: Buf,
    vb: Buf,
    only: []const u8,
    bad: u32 = 0,
    real: bool = false,
    caps: std.AutoHashMapUnmanaged(u64, std.ArrayList(f64)) = .empty, // (layer * 4 + stage) -> rows; stage 3: final norm
    want: []bool = &.{},
    ptoks: []u32 = &.{},

    /// The input rows of a class: captured (--prompt) or random.
    fn xrows(c: *Ctx, layer: usize, st: ?nw.Stage, n: usize, in: usize, scale: ?[]const f64, mult: f64) ![]f64 {
        if (c.real) {
            const key: u64 = layer * 4 + (if (st) |s| @intFromEnum(s) else 3);
            if (c.caps.get(key)) |rows_| {
                const x = try gpa.alloc(f64, n * in);
                @memcpy(x, rows_.items[0 .. n * in]);
                return x;
            }
        }
        return c.rows(n, in, scale, mult);
    }

    fn gauss(c: *Ctx) f64 {
        const rnd = c.prng.random();
        const u1_ = @max(rnd.float(f64), 1e-300);
        return @sqrt(-2.0 * @log(u1_)) * @cos(2.0 * std.math.pi * rnd.float(f64));
    }

    /// n rows of `in` bf16 values: N(0,1) times scale[i] (null: 1), rounded to bf16; returns the exact f64 values.
    fn rows(c: *Ctx, n: usize, in: usize, scale: ?[]const f64, mult: f64) ![]f64 {
        const x = try gpa.alloc(f64, n * in);
        for (0..n) |r| for (0..in) |i| {
            x[r * in + i] = bf(toBf(c.gauss() * mult * (if (scale) |s| s[i] else 1.0)));
        };
        return x;
    }

    fn upload(c: *Ctx, dst: Buf, x: []const f64) !void {
        const h = try gpa.alloc(u16, x.len);
        defer gpa.free(h);
        for (x, h) |v, *o| o.* = toBf(v);
        try c.r.upload(dst, std.mem.sliceAsBytes(h));
        try c.r.sync();
    }

    fn norm(c: *Ctx, b: Buf, n: usize) ![]f64 {
        const h = try gpa.alloc(u16, n);
        defer gpa.free(h);
        try c.r.download(std.mem.sliceAsBytes(h), b);
        try c.r.sync();
        const o = try gpa.alloc(f64, n);
        for (h, o) |v, *d| d.* = @abs(bf(v));
        return o;
    }

    pub fn dev16(c: *Ctx, b: Buf, count: usize) ![]f64 {
        const h = try gpa.alloc(u16, count);
        defer gpa.free(h);
        try c.r.download(std.mem.sliceAsBytes(h), b);
        try c.r.sync();
        const o = try gpa.alloc(f64, count);
        for (h, o) |v, *d| d.* = bf(v);
        return o;
    }

    fn dev32(c: *Ctx, b: Buf, count: usize) ![]f64 {
        const h = try gpa.alloc(f32, count);
        defer gpa.free(h);
        try c.r.download(std.mem.sliceAsBytes(h), b);
        try c.r.sync();
        const o = try gpa.alloc(f64, count);
        for (h, o) |v, *d| d.* = v;
        return o;
    }
};

/// Error bound (aggregate and worst per-column, relative to the column max) for a bf16-output or fp32-output kernel.
fn bound(bf16_out: bool) f64 {
    return if (bf16_out) 1.0e-2 else 1.0e-4;
}

pub fn compare(c: *Ctx, class: []const u8, layer: usize, path: []const u8, kernel: []const u8, n: usize, cols: usize, stride: usize, dev: []const f64, ref: []const f64, bf16_out: bool) void {
    var se: f64 = 0;
    var sr: f64 = 0;
    var worst_abs: f64 = 0;
    var worst_rel: f64 = 0;
    var worst_ulp: f64 = 0;
    var exact: usize = 0;
    var wcol: usize = 0;
    var gmax: f64 = 0; // largest |y| of the class: columns whose own max is below 1e-6 of it (all-zero weight rows) get that floor in the relative error
    for (ref[0 .. n * cols]) |v| gmax = @max(gmax, @abs(v));
    for (0..cols) |j| {
        var maxd: f64 = 0;
        var maxref: f64 = 0;
        for (0..n) |r| maxref = @max(maxref, @abs(ref[r * cols + j]));
        for (0..n) |r| {
            const rv = ref[r * cols + j];
            const dv = dev[r * stride + j];
            const d = @abs(dv - rv);
            se += d * d;
            sr += rv * rv;
            maxd = @max(maxd, d);
            if (bf16_out and @as(f64, bf(toBf(rv))) == dv) exact += 1;
            // ulps only for elements at least 1% of their column max (cancellation otherwise)
            if (d > 0 and @abs(rv) >= 0.01 * maxref and @abs(rv) > 1e-30) {
                const ulp = @exp2(@floor(@log2(@abs(rv))) - @as(f64, if (bf16_out) 7 else 23));
                worst_ulp = @max(worst_ulp, d / ulp);
            }
        }
        worst_abs = @max(worst_abs, maxd);
        const rel = maxd / @max(maxref, 1e-6 * gmax);
        if (rel > worst_rel) {
            worst_rel = rel;
            wcol = j;
        }
    }
    const agg = @sqrt(se / @max(sr, 1e-300));
    const flag = if (agg > bound(bf16_out) or worst_rel > 4 * bound(bf16_out)) " FAIL" else "";
    if (flag.len > 0) c.bad += 1;
    std.debug.print("{s:<18} L{d:<2} {s:<8} {s:<22} n={d:<3} cols={d:<6} agg {e:>9.2} worst-col abs {e:>9.2} rel {e:>9.2} (col {d}) worst {d:>6.2} ulp(>=1%)  bf16-exact {s}{s}\n", .{
        class, layer, path, kernel, n, cols, agg, worst_abs, worst_rel, wcol, worst_ulp,
        if (bf16_out) std.fmt.allocPrint(gpa, "{d:.1}%", .{100.0 * @as(f64, @floatFromInt(exact)) / @as(f64, @floatFromInt(n * cols))}) catch "?" else "-", flag,
    });
}

const Kind = enum { qmv, qkv_q, qkv_k, qkv_v, aqmv, head };

/// One dense projection `t` ([rows][in]) in the three paths. `ref_cols` leading output columns are checked.
fn dense(c: *Ctx, class: []const u8, layer: usize, st: ?nw.Stage, t: model.Table, aw: ?model.Attn, kind: Kind, scale: ?[]const f64, mult: f64, in: usize, rows: usize, ref_cols: usize) !void {
    if (c.only.len > 0 and std.mem.indexOf(u8, class, c.only) == null) return;
    const r = c.r;
    const m = c.m;
    const nd: usize = 8;
    const np: usize = 64;
    const out_f32 = kind == .head;
    const x = try c.xrows(layer, st, np, in, scale, mult);
    defer gpa.free(x);
    const h = try Host.fetch(r, t, ref_cols, in);
    defer h.free();
    const ref = try gpa.alloc(f64, np * ref_cols);
    defer gpa.free(ref);
    for (0..np) |i| for (0..ref_cols) |j| {
        ref[i * ref_cols + j] = h.dot(j, x[i * in ..][0..in]);
    };
    const esz: usize = if (out_f32) 4 else 2;
    const cfg = m.cfg;
    const kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    const q_dim = cfg.num_attention_heads * cfg.head_dim;
    // decode: row by row through the kernel the model uses
    {
        const dev = try gpa.alloc(f64, nd * ref_cols);
        defer gpa.free(dev);
        var kname: []const u8 = "";
        for (0..nd) |i| {
            try c.upload(c.xb, x[i * in ..][0..in]);
            switch (kind) {
                .qmv => {
                    kname = "qmv4_bf16";
                    try run(&m.k.m_qmv, .{ @as(u32, @intCast(rows)), 1, 1 }, .{ t.w, t.s, t.b, c.xb, c.yb, @as(u32, @intCast(in)) });
                },
                .qkv_q, .qkv_k, .qkv_v => {
                    kname = "qkv4_bf";
                    const a = aw.?;
                    try run(&m.k.a_qkv, .{ @as(u32, @intCast(q_dim / 4 + kv_dim / 2)), 1, 1 }, .{ a.q.w, a.q.s, a.q.b, a.k.w, a.k.s, a.k.b, a.v.w, a.v.s, a.v.b, c.xb, c.yb, c.kb, c.vb, @as(u32, @intCast(in)), @as(u32, @intCast(i * kv_dim)), @as(u32, @intCast(q_dim)), @as(u32, @intCast(kv_dim)) });
                },
                .aqmv => {
                    kname = "qmv4_bf (attn o)";
                    const none: u32 = 0;
                    try run(&m.k.a_qmv, .{ @as(u32, @intCast(rows / 4)), 1, 1 }, .{ t.w, t.s, t.b, c.xb, c.yb, @as(u32, @intCast(in)), none, none, @as(u32, @intCast(rows)) });
                },
                .head => {
                    kname = "head (qmv4_f32)";
                    const none: u32 = 0;
                    try run(&m.k.head, .{ @as(u32, @intCast(rows / 4)), 1, 1 }, .{ t.w, t.s, t.b, c.xb, c.yb, @as(u32, @intCast(in)), none, none, @as(u32, @intCast(rows)) });
                },
            }
            try r.sync();
            const src = switch (kind) {
                .qkv_k => c.kb,
                .qkv_v => c.vb,
                else => c.yb,
            };
            if (kind == .qkv_k or kind == .qkv_v) {
                const all = try c.dev16(src, nd * kv_dim);
                defer gpa.free(all);
                @memcpy(dev[i * ref_cols ..][0..ref_cols], all[i * kv_dim ..][0..ref_cols]);
            } else if (out_f32) {
                const o = try c.dev32(src, ref_cols);
                defer gpa.free(o);
                @memcpy(dev[i * ref_cols ..][0..ref_cols], o);
            } else {
                const o = try c.dev16(src, ref_cols);
                defer gpa.free(o);
                @memcpy(dev[i * ref_cols ..][0..ref_cols], o);
            }
        }
        compare(c, class, layer, "decode", kname, nd, ref_cols, ref_cols, dev, ref[0 .. nd * ref_cols], !out_f32);
    }
    const pf = &c.w.mpf.?;
    // window (8 rows, split-K matrix-engine GEMM) and prefill (64 rows, dense GEMM)
    for ([_]usize{ nd, np }) |n| {
        try c.upload(c.xb, x[0 .. n * in]);
        if (n == nd) try pf.denseS(t.w, t.s, t.b, c.xb, @intCast(n), c.yb, @intCast(in), 0, @intCast(rows), out_f32) else try pf.dense(r, t.w, t.s, t.b, c.xb, @intCast(n), c.yb, @intCast(in), 0, @intCast(rows), out_f32);
        try r.sync();
        const full = if (out_f32) try c.dev32(c.yb, n * rows) else try c.dev16(c.yb, n * rows);
        defer gpa.free(full);
        compare(c, class, layer, if (n == nd) "window" else "prefill", if (n == nd) "denseS (pfgemm_bs)" else "dense (pfgemm_b)", n, ref_cols, rows, full, ref[0 .. n * ref_cols], !out_f32);
        _ = esz;
    }
}

fn relu2(v: f64) f64 {
    const u = @max(v, 0.0);
    return u * u;
}

/// Router, shared expert and routed experts of one MoE layer in the three paths.
fn moe(c: *Ctx, layer: usize, mw: model.Moe, normw: []const f64) !void {
    if (c.only.len > 0 and std.mem.indexOf(u8, "moe", c.only) == null and std.mem.indexOf(u8, c.only, "gate") == null and std.mem.indexOf(u8, c.only, "expert") == null and std.mem.indexOf(u8, c.only, "fc") == null) return;
    const r = c.r;
    const m = c.m;
    const cfg = m.cfg;
    const h: usize = cfg.hidden_size;
    const wd: usize = cfg.moe_intermediate_size;
    const sw: usize = cfg.moe_shared_expert_intermediate_size;
    const E: usize = cfg.n_routed_experts;
    const K: usize = 6;
    const np: usize = 64;
    const nd: usize = 8;
    const x = try c.xrows(layer, .xn, np, h, normw, 1.0);
    defer gpa.free(x);
    const gate = try c.dev16(mw.gate, E * h);
    defer gpa.free(gate);
    const shup = try Host.fetch(r, mw.shup, sw, h);
    defer shup.free();
    const shdn = try Host.fetch(r, mw.shdn, h, sw);
    defer shdn.free();
    const fc1 = try Host.fetch(r, mw.fc1, E * wd, h);
    defer fc1.free();
    const fc2 = try Host.fetch(r, mw.fc2, E * h, wd);
    defer fc2.free();
    const paths = [_]struct { name: []const u8, n: usize }{ .{ .name = "decode", .n = nd }, .{ .name = "window", .n = nd }, .{ .name = "prefill", .n = np } };
    for (paths) |p| {
        const n = p.n;
        const dec = std.mem.eql(u8, p.name, "decode");
        var d_logit = try gpa.alloc(f64, n * E);
        var d_sact = try gpa.alloc(f64, n * sw);
        var d_sy = try gpa.alloc(f64, n * h);
        var d_act = try gpa.alloc(f64, n * K * wd);
        var d_ey = try gpa.alloc(f64, n * K * h);
        var ids = try gpa.alloc(u32, n * K);
        defer for ([_][]f64{ d_logit, d_sact, d_sy, d_act, d_ey }) |s| gpa.free(s);
        defer gpa.free(ids);
        if (dec) {
            const s = m.s;
            const scaling: u32 = @bitCast(cfg.routed_scaling_factor);
            for (0..n) |i| {
                try c.upload(s.xn, x[i * h ..][0..h]);
                try run(&m.k.e_rup, .{ @intCast(E + sw), 1, 1 }, .{ s.xn, mw.gate, s.r_logits, @as(u32, @intCast(h)), @as(u32, @intCast(E)), mw.shup.w, mw.shup.s, mw.shup.b, s.sact });
                try run(&m.k.e_rdown, .{ @intCast(1 + h), 1, 1 }, .{ s.r_logits, mw.bias, s.ids, s.wts, @as(u32, @intCast(E)), @as(u32, K), scaling, mw.shdn.w, mw.shdn.s, mw.shdn.b, s.sact, s.sy, @as(u32, @intCast(sw)) });
                try run(&m.k.e_up, .{ @intCast(wd), K, 1 }, .{ mw.fc1.w, mw.fc1.s, mw.fc1.b, s.xn, s.ids, s.act, @as(u32, @intCast(h)), @as(u32, @intCast(wd)), @as(u32, 0) });
                try run(&m.k.e_down, .{ @intCast(h), K, 1 }, .{ mw.fc2.w, mw.fc2.s, mw.fc2.b, s.act, s.ids, s.ey, @as(u32, @intCast(wd)), @as(u32, @intCast(h)), @as(u32, @intCast(wd)) });
                try r.sync();
                const a = try c.dev16(s.r_logits, E);
                defer gpa.free(a);
                @memcpy(d_logit[i * E ..][0..E], a);
                const b = try c.dev16(s.sact, sw);
                defer gpa.free(b);
                @memcpy(d_sact[i * sw ..][0..sw], b);
                const sy = try c.dev32(s.sy, h);
                defer gpa.free(sy);
                @memcpy(d_sy[i * h ..][0..h], sy);
                const ac = try c.dev16(s.act, K * wd);
                defer gpa.free(ac);
                @memcpy(d_act[i * K * wd ..][0 .. K * wd], ac);
                const ey = try c.dev32(s.ey, K * h);
                defer gpa.free(ey);
                @memcpy(d_ey[i * K * h ..][0 .. K * h], ey);
                try r.download(std.mem.sliceAsBytes(ids[i * K ..][0..K]), s.ids);
                try r.sync();
            }
        } else {
            const w = c.w;
            try c.upload(w.xn, x[0 .. n * h]);
            try w.moe(m, mw, @intCast(n));
            try r.sync();
            const a = try c.dev16(w.r_logits, n * E);
            @memcpy(d_logit, a);
            gpa.free(a);
            const b = try c.dev16(w.sact, n * sw);
            @memcpy(d_sact, b);
            gpa.free(b);
            const sy = try c.dev32(w.sy, n * h);
            @memcpy(d_sy, sy);
            gpa.free(sy);
            const ac = try c.dev16(w.act, n * K * wd);
            @memcpy(d_act, ac);
            gpa.free(ac);
            const ey = try c.dev32(w.ey, n * K * h);
            @memcpy(d_ey, ey);
            gpa.free(ey);
            try r.download(std.mem.sliceAsBytes(ids), w.eids);
            try r.sync();
        }
        // references
        const ref_l = try gpa.alloc(f64, n * E);
        defer gpa.free(ref_l);
        const ref_su = try gpa.alloc(f64, n * sw);
        defer gpa.free(ref_su);
        const ref_sd = try gpa.alloc(f64, n * h);
        defer gpa.free(ref_sd);
        const ref_a = try gpa.alloc(f64, n * K * wd);
        defer gpa.free(ref_a);
        const ref_e = try gpa.alloc(f64, n * K * h);
        defer gpa.free(ref_e);
        for (0..n) |i| {
            const xi = x[i * h ..][0..h];
            for (0..E) |e| {
                var s: f64 = 0;
                for (0..h) |t| s += xi[t] * gate[e * h + t];
                ref_l[i * E + e] = s;
            }
            for (0..sw) |j| ref_su[i * sw + j] = relu2(shup.dot(j, xi));
            for (0..h) |j| ref_sd[i * h + j] = shdn.dot(j, d_sact[i * sw ..][0..sw]);
            for (0..K) |k| {
                const e: usize = ids[i * K + k];
                const slot = i * K + k;
                for (0..wd) |j| ref_a[slot * wd + j] = relu2(fc1.dot(e * wd + j, xi));
                for (0..h) |j| ref_e[slot * h + j] = fc2.dot(e * h + j, d_act[slot * wd ..][0..wd]);
            }
        }
        const kn = if (dec) [_][]const u8{ "router_up_sh", "router_up_sh (relu2)", "route_down_sh", "expert_up_relu2", "expert_down_f32" } else if (n == nd) [_][]const u8{ "router_logits", "denseSM mode 2", "dense GEMM f32", "runExpertsS (pfgemm_moe_bs)", "runExpertsS (pfgemm_moe_bs)" } else [_][]const u8{ "router_logits", "dense GEMM + relu2", "dense GEMM f32", "runExperts (pfgemm_moe_b)", "runExperts (pfgemm_moe_b)" };
        compare(c, "moe.router_gate", layer, p.name, kn[0], n, E, E, d_logit, ref_l, true);
        compare(c, "moe.shared_up", layer, p.name, kn[1], n, sw, sw, d_sact, ref_su, true);
        compare(c, "moe.shared_down", layer, p.name, kn[2], n, h, h, d_sy, ref_sd, false);
        compare(c, "moe.routed_fc1", layer, p.name, kn[3], n * K, wd, wd, d_act, ref_a, true);
        compare(c, "moe.routed_fc2", layer, p.name, kn[4], n * K, h, h, d_ey, ref_e, false);
    }
}

pub fn main(init: std.process.Init) !u8 {
    stop.install();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: xpu-nem_fp64_test CHECKPOINT_DIR [--layers first|last|both] [--only SUBSTRING]\n", .{});
        return 2;
    }
    const dir = args[1];
    var which: []const u8 = "both";
    var only: []const u8 = "";
    var prompt_file: ?[]const u8 = null;
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--layers")) {
            i += 1;
            which = args[i];
        } else if (std.mem.eql(u8, args[i], "--only")) {
            i += 1;
            only = args[i];
        } else if (std.mem.eql(u8, args[i], "--prompt")) {
            i += 1;
            prompt_file = args[i];
        } else return error.BadArgument;
    }
    const cfg_bytes = try ld.readFile(gpa, dir, "config.json");
    const cfg = try cfgm.parse(gpa, cfg_bytes);
    var r = try rt.open();
    defer r.deinit();
    var l = try ld.Loader.init(gpa, &r, dir);
    var ptoks: []u32 = &.{};
    if (prompt_file) |pf| {
        const txt = try ld.readFile(gpa, "/", pf[1..]);
        var ids: std.ArrayList(u32) = .empty;
        var it = std.mem.tokenizeScalar(u8, txt, ',');
        while (it.next()) |tk| try ids.append(gpa, try std.fmt.parseInt(u32, std.mem.trim(u8, tk, " \r\n"), 10));
        ptoks = try ids.toOwnedSlice(gpa);
    }
    var m = try model.Model.load(gpa, &r, &l, cfg.value, @intCast(@max(256, ptoks.len + 16)));
    nw.default_rows = 512;
    var w = try nw.Win.init(gpa, &m);
    const c_ = cfg.value;
    var c: Ctx = .{
        .r = &r,
        .m = &m,
        .w = &w,
        .prng = std.Random.DefaultPrng.init(2024),
        .xb = try r.alloc(64 * 4096 * 2),
        .yb = try r.alloc(64 * @as(usize, c_.vocab_size) * 4),
        .kb = try r.alloc(64 * 256 * 2),
        .vb = try r.alloc(64 * 256 * 2),
        .only = only,
        .real = prompt_file != null,
        .ptoks = ptoks,
    };
    if (c.real) {
        try cap.captureRun(&c);
        std.debug.print("rows: CAPTURED activations of the last 64 tokens of a {d}-token prompt\n", .{ptoks.len});
    } else std.debug.print("rows: random N(0,1) x the layer norm weight\n", .{});
    std.debug.print("FP64 reference per projection class (weights exact 4-bit affine g64); bound agg {e:.0} bf16 / {e:.0} fp32\n", .{ bound(true), bound(false) });
    // first and last layer of each kind
    var first = [_]?usize{ null, null, null };
    var last = [_]?usize{ null, null, null };
    for (m.layers, 0..) |ly, li| {
        const k: usize = @intFromEnum(std.meta.activeTag(ly.mixer));
        if (first[k] == null) first[k] = li;
        last[k] = li;
    }
    for (0..3) |k| {
        const picks = [_]?usize{ first[k], if (std.mem.eql(u8, which, "first")) null else last[k] };
        for (picks, 0..) |pick, pi| {
            const li = pick orelse continue;
            if (pi == 1 and last[k] == first[k]) continue;
            if (pi == 1 and std.mem.eql(u8, which, "first")) continue;
            if (pi == 0 and std.mem.eql(u8, which, "last")) continue;
            const ly = m.layers[li];
            const nrm = try c.norm(ly.norm, c_.hidden_size);
            switch (ly.mixer) {
                .mamba => |mw| {
                    try dense(&c, "mamba.in_proj", li, .xn, mw.in, null, .qmv, nrm, 1.0, c_.hidden_size, c_.projDim(), c_.projDim());
                    const gn = try c.norm(mw.gnorm, c_.xd());
                    try dense(&c, "mamba.out_proj", li, .yn, mw.out, null, .qmv, gn, 1.0, c_.xd(), c_.hidden_size, c_.hidden_size);
                },
                .attention => |aw| {
                    const q_dim = c_.num_attention_heads * c_.head_dim;
                    const kv_dim = c_.num_key_value_heads * c_.head_dim;
                    try dense(&c, "attn.q_proj", li, .xn, aw.q, aw, .qkv_q, nrm, 1.0, c_.hidden_size, q_dim, q_dim);
                    try dense(&c, "attn.k_proj", li, .xn, aw.k, aw, .qkv_k, nrm, 1.0, c_.hidden_size, kv_dim, kv_dim);
                    try dense(&c, "attn.v_proj", li, .xn, aw.v, aw, .qkv_v, nrm, 1.0, c_.hidden_size, kv_dim, kv_dim);
                    try dense(&c, "attn.o_proj", li, .att, aw.o, aw, .aqmv, null, 0.5, q_dim, c_.hidden_size, c_.hidden_size);
                },
                .moe => |mw| try moe(&c, li, mw, nrm),
            }
            try nw.stopCheck(&r);
        }
    }
    {
        const nrm = try c.norm(m.norm_f, c_.hidden_size);
        try dense(&c, "lm_head", 0, null, m.head, null, .head, nrm, 1.0, c_.hidden_size, c_.vocab_size, 8192);
        try cap.embed(&c);
    }
    std.debug.print("{s}\n", .{if (c.bad == 0) "fp64 reference: all classes within bounds" else "fp64 reference: classes OUT OF BOUNDS (FAIL lines)"});
    return if (c.bad == 0) 0 else 1;
}
