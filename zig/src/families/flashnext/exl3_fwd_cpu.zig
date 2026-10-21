//! CPU f64 oracle for the whole Qwen3.8-Flash-Next EXL3 forward, assembled from the per-op oracles
//! (`exl3_hc` preF64/postF64, `exl3_gdn_cpu`, `exl3_attn_cpu`, `exl3_moe` routing, `exl3_cpu` swiglu).
//! It works layer-major: every prompt row goes through layer l before any row goes through layer l + 1, which is
//! exactly the causal order for a teacher-forced pass and lets each layer's weights be decoded once.
//!
//! `QMat` is the EXL3 matrix as the Zig decoder (`exl3_format`) defines it, W = diag(suh) H_K Wq H_N diag(svh), applied to a
//! vector without materialising W: y = ((x * suh) H_K) Wq H_N * svh, with the 128-wide Hadamard blocks done as a fast
//! Walsh-Hadamard transform. `exl3_format.dequantize` (dense 128x128 rotations) is the reference it is tested against.
//! Wq is the unpacked rotated-domain fp16 matrix, kept as f32 (exact), accumulated in f64.
//! No CUDA.

const std = @import("std");
const format = @import("exl3_format.zig");
const hc = @import("exl3_hc.zig");
const gdn_cpu = @import("exl3_gdn_cpu.zig");
const attn_cpu = @import("exl3_attn_cpu.zig");
const moe = @import("exl3_moe.zig");
const cpu = @import("exl3_cpu.zig");

pub const hidden: usize = 2560;
pub const streams: usize = 4;
pub const wide: usize = streams * hidden;
pub const topk: usize = 10;
pub const experts: usize = 512;
pub const inter: usize = 640;

// -- parallel for ------------------------------------------------------------------------------------------

/// Runs `f(ctx, i)` for i in 0..n over up to `max_threads` threads (the caller included). `f` stores its own errors in `ctx`.
pub fn parallelFor(n: usize, max_threads: usize, ctx: anytype, comptime f: anytype) void {
    const Ctx = @TypeOf(ctx);
    const W = struct {
        fn run(next: *std.atomic.Value(usize), count: usize, c: Ctx) void {
            while (true) {
                const i = next.fetchAdd(1, .monotonic);
                if (i >= count) return;
                f(c, i);
            }
        }
    };
    var next = std.atomic.Value(usize).init(0);
    const cpus = std.Thread.getCpuCount() catch 4;
    const want = @min(@min(cpus, max_threads), n);
    var handles: [128]std.Thread = undefined;
    var spawned: usize = 0;
    while (spawned + 1 < want and spawned < handles.len) : (spawned += 1) {
        handles[spawned] = std.Thread.spawn(.{}, W.run, .{ &next, n, ctx }) catch break;
    }
    W.run(&next, n, ctx);
    for (handles[0..spawned]) |h| h.join();
}

// -- QMat --------------------------------------------------------------------------------------------------

/// In-place orthonormal Walsh-Hadamard transform of each 128 block of `x` (Sylvester order, H[i][j] = (-1)^popcount(i&j) / sqrt(128)).
pub fn fwht128(x: []f64) void {
    std.debug.assert(x.len % 128 == 0);
    const inv = 1.0 / @sqrt(128.0);
    var base: usize = 0;
    while (base < x.len) : (base += 128) {
        const b = x[base..][0..128];
        var h: usize = 1;
        while (h < 128) : (h *= 2) {
            var i: usize = 0;
            while (i < 128) : (i += 2 * h) {
                for (i..i + h) |j| {
                    const lo = b[j];
                    const hi = b[j + h];
                    b[j] = lo + hi;
                    b[j + h] = lo - hi;
                }
            }
        }
        for (b) |*v| v.* *= inv;
    }
}

pub const QMat = struct {
    k: usize,
    n: usize,
    wq: []f32,
    suh: []f32,
    svh: []f32,

    /// `threads` parallelises the trellis unpack over tile rows.
    pub fn init(gpa: std.mem.Allocator, halves: u16, trellis: []const i16, tiles_k: usize, tiles_n: usize, suh: []const f16, svh: []const f16, threads: usize) !QMat {
        const k = tiles_k * 16;
        const n = tiles_n * 16;
        if (k % 128 != 0 or n % 128 != 0 or suh.len != k or svh.len != n) return error.Length;
        if (trellis.len != tiles_k * tiles_n * @as(usize, halves) * 8) return error.Length;
        const wq = try gpa.alloc(f32, k * n);
        errdefer gpa.free(wq);
        const su = try gpa.alloc(f32, k);
        errdefer gpa.free(su);
        const sv = try gpa.alloc(f32, n);
        errdefer gpa.free(sv);
        for (su, suh) |*d, s| d.* = @floatCast(s);
        for (sv, svh) |*d, s| d.* = @floatCast(s);
        var job: UnpackJob = .{ .halves = halves, .trellis = trellis, .tiles_n = tiles_n, .wq = wq, .n = n };
        parallelFor(tiles_k, threads, &job, UnpackJob.row);
        return .{ .k = k, .n = n, .wq = wq, .suh = su, .svh = sv };
    }

    pub fn deinit(m: *QMat, gpa: std.mem.Allocator) void {
        gpa.free(m.wq);
        gpa.free(m.suh);
        gpa.free(m.svh);
        m.* = undefined;
    }

    /// y[0..n] = x[0..k] W. `tmp` is k long.
    pub fn apply(m: QMat, x: []const f64, y: []f64, tmp: []f64) void {
        std.debug.assert(x.len == m.k and y.len == m.n and tmp.len == m.k);
        for (tmp, x, m.suh) |*d, v, s| d.* = v * s;
        fwht128(tmp);
        @memset(y, 0);
        for (0..m.k) |i| {
            const u = tmp[i];
            if (u == 0) continue;
            const row = m.wq[i * m.n ..][0..m.n];
            for (y, row) |*o, w| o.* += u * @as(f64, w);
        }
        fwht128(y);
        for (y, m.svh) |*o, s| o.* *= s;
    }

    /// Same as `apply` with the output columns [lo, hi) only (the Hadamard blocks must be whole: lo and hi multiples of 128).
    pub fn applyCols(m: QMat, tmp_h: []const f64, y: []f64, lo: usize, hi: usize) void {
        std.debug.assert(lo % 128 == 0 and hi % 128 == 0 and y.len == hi - lo);
        @memset(y, 0);
        for (0..m.k) |i| {
            const u = tmp_h[i];
            if (u == 0) continue;
            const row = m.wq[i * m.n + lo ..][0 .. hi - lo];
            for (y, row) |*o, w| o.* += u * @as(f64, w);
        }
        fwht128(y);
        for (y, m.svh[lo..hi]) |*o, s| o.* *= s;
    }

    /// The transformed input (x * suh) H_K that `applyCols` consumes.
    pub fn prepare(m: QMat, x: []const f64, out: []f64) void {
        for (out, x, m.suh) |*d, v, s| d.* = v * s;
        fwht128(out);
    }
};

const UnpackJob = struct {
    halves: u16,
    trellis: []const i16,
    tiles_n: usize,
    wq: []f32,
    n: usize,
    fn row(j: *UnpackJob, tk: usize) void {
        var placement: [256]format.Placement = undefined;
        format.tilePlacement(&placement);
        const w: format.Width = .{ .halves = j.halves };
        const words = w.tileWords();
        for (0..j.tiles_n) |tn| {
            const tile = j.trellis[(tk * j.tiles_n + tn) * words ..][0..words];
            var states: [256]u32 = undefined;
            format.tileStates(w, tile, &states);
            for (0..256) |p| {
                const v: f16 = format.codebookValue(.mul1, @intCast(states[p]));
                j.wq[(tk * 16 + placement[p].row) * j.n + tn * 16 + placement[p].col] = @floatCast(v);
            }
        }
    }
};

// -- small ops ---------------------------------------------------------------------------------------------

pub fn silu(x: f64) f64 {
    return x / (1.0 + @exp(-x));
}

pub fn sigmoid(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

/// f64 rows [t][hidden] -> mixed hidden of the hyper-connection (pre), per row. `inject` is [t][4] or null.
pub fn hcPreRows(gpa: std.mem.Allocator, w: hc.Weights, offset: f64, h: []const f64, rows: usize, mixed: []f64, inject: ?[]f64) !void {
    const low = try gpa.alloc(f64, hc.real.lowrank);
    defer gpa.free(low);
    const mix = try gpa.alloc(f64, wide);
    defer gpa.free(mix);
    const normed = try gpa.alloc(f64, wide);
    defer gpa.free(normed);
    for (0..rows) |t| {
        const inj: ?[]f64 = if (inject) |i| i[t * streams ..][0..streams] else null;
        try hc.preF64(hc.real, w, offset, 1e-6, h[t * wide ..][0..wide], mixed[t * hidden ..][0..hidden], inj, low, mix, normed);
    }
}

pub fn hcPostRows(h: []f64, branch: []const f64, inject: []const f64, rows: usize) void {
    for (0..rows) |t| {
        for (0..streams) |s| {
            const g = inject[t * streams + s];
            for (0..hidden) |d| h[t * wide + s * hidden + d] += branch[t * hidden + d] * g;
        }
    }
}

// -- layer weights -----------------------------------------------------------------------------------------

pub const GdnW = struct {
    qkv: QMat,
    z: QMat,
    out: QMat,
    a_w: []const f64, // [48][2560]
    b_w: []const f64,
    params: gdn_cpu.Params,

    pub fn deinit(w: *GdnW, gpa: std.mem.Allocator) void {
        w.qkv.deinit(gpa);
        w.z.deinit(gpa);
        w.out.deinit(gpa);
        gpa.free(w.a_w);
        gpa.free(w.b_w);
        gpa.free(w.params.conv_w);
        gpa.free(w.params.a_log);
        gpa.free(w.params.dt_bias);
        gpa.free(w.params.norm_w);
    }
};

pub const AttnW = struct {
    q: QMat,
    k: QMat,
    v: QMat,
    o: QMat,
    idx: QMat,
    norms: attn_cpu.Norms,

    pub fn deinit(w: *AttnW, gpa: std.mem.Allocator) void {
        w.q.deinit(gpa);
        w.k.deinit(gpa);
        w.v.deinit(gpa);
        w.o.deinit(gpa);
        w.idx.deinit(gpa);
        gpa.free(w.norms.q);
        gpa.free(w.norms.k);
        gpa.free(w.norms.iq);
        gpa.free(w.norms.ik);
    }
};

/// Where the 512 routed experts of one layer come from: decode expert `e` (gate, up, down) into `out`.
pub const ExpertSource = struct {
    ctx: *anyopaque,
    load: *const fn (ctx: *anyopaque, e: usize, out: *[3]QMat) anyerror!void,
};

pub const MoeW = struct {
    gate16: []const f16, // [512][2560]
    sgate16: []const f16, // [2560]
    shared: [3]QMat,
    source: ExpertSource,

    pub fn deinit(w: *MoeW, gpa: std.mem.Allocator) void {
        for (&w.shared) |*m| m.deinit(gpa);
    }
};

pub const Mixer = union(enum) { gdn: *GdnW, attn: *AttnW };

pub const LayerW = struct {
    attn_hc: hc.Weights,
    mlp_hc: hc.Weights,
    mixer: Mixer,
    moe: *MoeW,
};

pub const LayerState = union(enum) { gdn: gdn_cpu.State, attn: attn_cpu.Cache };

/// What a layer pass records for comparison with the device.
pub const Trace = struct {
    attn_branch: []f64, // [rows][2560]
    mlp_branch: []f64,
    picks: []u16, // [rows][10]
    h_mid: []f64, // [rows][10240] streams after the attention block
};

// -- mixers ------------------------------------------------------------------------------------------------

fn roundF16Slice(dst: []f16, src: []const f64) void {
    for (dst, src) |*d, s| d.* = @floatCast(s);
}

const GdnProj = struct {
    w: *const GdnW,
    x: []const f64,
    qkv: []f64,
    z: []f64,
    a: []f64,
    b: []f64,
    gpa: std.mem.Allocator,
    err: ?anyerror = null,

    fn row(j: *GdnProj, t: usize) void {
        const tmp = j.gpa.alloc(f64, hidden) catch |e| {
            j.err = e;
            return;
        };
        defer j.gpa.free(tmp);
        const xr = j.x[t * hidden ..][0..hidden];
        const c = gdn_cpu.qwen.channels();
        const zw = gdn_cpu.qwen.zWidth();
        j.w.qkv.apply(xr, j.qkv[t * c ..][0..c], tmp);
        j.w.z.apply(xr, j.z[t * zw ..][0..zw], tmp);
        gdn_cpu.rowsDot(j.w.a_w, xr, j.a[t * 48 ..][0..48]);
        gdn_cpu.rowsDot(j.w.b_w, xr, j.b[t * 48 ..][0..48]);
    }
};

const OutJob = struct {
    m: *const QMat,
    inp: []const f64,
    in_w: usize,
    out: []f64,
    gpa: std.mem.Allocator,
    err: ?anyerror = null,
    fn row(j: *OutJob, t: usize) void {
        const tmp = j.gpa.alloc(f64, j.m.k) catch |e| {
            j.err = e;
            return;
        };
        defer j.gpa.free(tmp);
        j.m.apply(j.inp[t * j.in_w ..][0..j.in_w], j.out[t * j.m.n ..][0..j.m.n], tmp);
    }
};

pub fn gdnBranch(gpa: std.mem.Allocator, w: *const GdnW, st: gdn_cpu.State, x: []const f64, rows: usize, out: []f64) !void {
    const d = gdn_cpu.qwen;
    const c = d.channels();
    const zw = d.zWidth();
    const qkv = try gpa.alloc(f64, rows * c);
    defer gpa.free(qkv);
    const z = try gpa.alloc(f64, rows * zw);
    defer gpa.free(z);
    const a = try gpa.alloc(f64, rows * 48);
    defer gpa.free(a);
    const b = try gpa.alloc(f64, rows * 48);
    defer gpa.free(b);
    var pj: GdnProj = .{ .w = w, .x = x, .qkv = qkv, .z = z, .a = a, .b = b, .gpa = gpa };
    parallelFor(rows, 48, &pj, GdnProj.row);
    if (pj.err) |e| return e;
    const gated = try gpa.alloc(f64, rows * zw);
    defer gpa.free(gated);
    try gdn_cpu.tokens(gpa, d, w.params, st, rows, qkv, z, a, b, gated);
    var oj: OutJob = .{ .m = &w.out, .inp = gated, .in_w = zw, .out = out, .gpa = gpa };
    parallelFor(rows, 48, &oj, OutJob.row);
    if (oj.err) |e| return e;
}

const AttnProj = struct {
    w: *const AttnW,
    x: []const f64,
    qp: []f16,
    kp: []f16,
    vp: []f16,
    ip: []f16,
    gpa: std.mem.Allocator,
    err: ?anyerror = null,
    const q_width = attn_cpu.q_heads * 2 * attn_cpu.head_dim;
    const kv_width = attn_cpu.kv_heads * attn_cpu.head_dim;

    fn row(j: *AttnProj, t: usize) void {
        const tmp = j.gpa.alloc(f64, hidden) catch |e| {
            j.err = e;
            return;
        };
        defer j.gpa.free(tmp);
        const big = j.gpa.alloc(f64, q_width) catch |e| {
            j.err = e;
            return;
        };
        defer j.gpa.free(big);
        const xr = j.x[t * hidden ..][0..hidden];
        j.w.q.apply(xr, big, tmp);
        roundF16Slice(j.qp[t * q_width ..][0..q_width], big);
        j.w.k.apply(xr, big[0..kv_width], tmp);
        roundF16Slice(j.kp[t * kv_width ..][0..kv_width], big[0..kv_width]);
        j.w.v.apply(xr, big[0..kv_width], tmp);
        roundF16Slice(j.vp[t * kv_width ..][0..kv_width], big[0..kv_width]);
        j.w.idx.apply(xr, big[0..attn_cpu.index_width], tmp);
        roundF16Slice(j.ip[t * attn_cpu.index_width ..][0..attn_cpu.index_width], big[0..attn_cpu.index_width]);
    }
};

/// Attention rows pos0 .. pos0 + rows. The projections are rounded to fp16 where the EXL3 linears emit them.
pub fn attnBranch(gpa: std.mem.Allocator, w: *const AttnW, cache: *attn_cpu.Cache, pos0: usize, x: []const f64, rows: usize, out: []f64) !void {
    const q_width = AttnProj.q_width;
    const kv_width = AttnProj.kv_width;
    const iw = attn_cpu.index_width;
    const qp = try gpa.alloc(f16, rows * q_width);
    defer gpa.free(qp);
    const kp = try gpa.alloc(f16, rows * kv_width);
    defer gpa.free(kp);
    const vp = try gpa.alloc(f16, rows * kv_width);
    defer gpa.free(vp);
    const ip = try gpa.alloc(f16, rows * iw);
    defer gpa.free(ip);
    var pj: AttnProj = .{ .w = w, .x = x, .qp = qp, .kp = kp, .vp = vp, .ip = ip, .gpa = gpa };
    parallelFor(rows, 48, &pj, AttnProj.row);
    if (pj.err) |e| return e;

    const attn_width = attn_cpu.q_heads * attn_cpu.head_dim;
    const o_in = try gpa.alloc(f64, rows * attn_width);
    defer gpa.free(o_in);
    const scores = try gpa.alloc(f64, cache.tokens / attn_cpu.ratio + 2);
    defer gpa.free(scores);
    var oq: [attn_width]f64 = undefined;
    var oiq: [attn_cpu.index_heads * attn_cpu.index_dim]f64 = undefined;
    var info = attn_cpu.RowInfo{};
    for (0..rows) |t| {
        try attn_cpu.row(cache, pos0 + t, .{
            .qp = qp[t * q_width ..][0..q_width],
            .kp = kp[t * kv_width ..][0..kv_width],
            .vp = vp[t * kv_width ..][0..kv_width],
            .ip = ip[t * iw ..][0..iw],
        }, &w.norms, scores, &oq, &oiq, o_in[t * attn_width ..][0..attn_width], &info);
    }
    var oj: OutJob = .{ .m = &w.o, .inp = o_in, .in_w = attn_width, .out = out, .gpa = gpa };
    parallelFor(rows, 48, &oj, OutJob.row);
    if (oj.err) |e| return e;
}

// -- MoE ---------------------------------------------------------------------------------------------------

const Loaded = struct {
    id: usize,
    mats: [3]QMat,
};

const LoadJob = struct {
    src: ExpertSource,
    ids: []const usize,
    slots: []Loaded,
    err: ?anyerror = null,
    fn one(j: *LoadJob, i: usize) void {
        j.slots[i].id = j.ids[i];
        j.src.load(j.src.ctx, j.ids[i], &j.slots[i].mats) catch |e| {
            j.err = e;
        };
    }
};

const MoeRow = struct {
    w: *const MoeW,
    x: []const f64,
    out: []f64,
    picks: []const u16, // [rows][10]
    weights: []const f64, // [rows][10]
    shared_w: []const f64,
    loaded: []const Loaded,
    gpa: std.mem.Allocator,
    err: ?anyerror = null,

    fn find(j: *const MoeRow, id: usize) *const Loaded {
        for (j.loaded) |*l| if (l.id == id) return l;
        unreachable;
    }

    fn row(j: *MoeRow, t: usize) void {
        const buf = j.gpa.alloc(f64, hidden + 4 * inter + 3 * hidden) catch |e| {
            j.err = e;
            return;
        };
        defer j.gpa.free(buf);
        const tmp = buf[0..hidden];
        const g = buf[hidden..][0..inter];
        const u = buf[hidden + inter ..][0..inter];
        const glu = buf[hidden + 2 * inter ..][0..inter];
        const y = buf[hidden + 3 * inter ..][0..hidden];
        const acc = j.out[t * hidden ..][0..hidden];
        @memset(acc, 0);
        const xr = j.x[t * hidden ..][0..hidden];
        for (0..topk) |k| {
            const l = j.find(j.picks[t * topk + k]);
            expertApply(&l.mats, xr, tmp, g, u, glu, y);
            const wk = j.weights[t * topk + k];
            for (acc, y) |*o, v| o.* += wk * v;
        }
        expertApply(&j.w.shared, xr, tmp, g, u, glu, y);
        const sw = j.shared_w[t];
        for (acc, y) |*o, v| o.* += sw * v;
    }
};

fn expertApply(m: *const [3]QMat, x: []const f64, tmp: []f64, g: []f64, u: []f64, glu: []f64, y: []f64) void {
    m[0].apply(x, g, tmp);
    m[1].apply(x, u, tmp);
    for (glu, g, u) |*o, gv, uv| o.* = silu(gv) * uv;
    m[2].apply(glu, y, tmp[0..inter]);
}

/// Routed + shared experts for `rows` rows from the f64 hidden `x` (the f64 router, not the device's). Records the picks.
pub fn moeBranch(gpa: std.mem.Allocator, w: *const MoeW, x: []const f64, rows: usize, out: []f64, picks_out: []u16) !void {
    const picks = picks_out;
    const weights = try gpa.alloc(f64, rows * topk);
    defer gpa.free(weights);
    const shared_w = try gpa.alloc(f64, rows);
    defer gpa.free(shared_w);
    var logits: [experts + 1]f64 = undefined;
    var ids: std.ArrayList(usize) = .empty;
    defer ids.deinit(gpa);
    for (0..rows) |t| {
        try moe.logitsF16(w.gate16, experts, w.sgate16, x[t * hidden ..][0..hidden], &logits);
        var sw: f64 = 0;
        try moe.routeFromLogits(&logits, topk, picks[t * topk ..][0..topk], weights[t * topk ..][0..topk], &sw);
        shared_w[t] = sw;
        for (picks[t * topk ..][0..topk]) |p| {
            var seen = false;
            for (ids.items) |i| seen = seen or i == p;
            if (!seen) try ids.append(gpa, p);
        }
    }
    const loaded = try gpa.alloc(Loaded, ids.items.len);
    defer gpa.free(loaded);
    var lj: LoadJob = .{ .src = w.source, .ids = ids.items, .slots = loaded };
    parallelFor(ids.items.len, 48, &lj, LoadJob.one);
    defer if (lj.err == null) for (loaded) |*l| for (&l.mats) |*m| m.deinit(gpa);
    if (lj.err) |e| return e;
    var mj: MoeRow = .{ .w = w, .x = x, .out = out, .picks = picks, .weights = weights, .shared_w = shared_w, .loaded = loaded, .gpa = gpa };
    parallelFor(rows, 48, &mj, MoeRow.row);
    if (mj.err) |e| return e;
}

// -- one layer ---------------------------------------------------------------------------------------------

/// `h` is [rows][10240] f64 streams, updated in place. `pos0` is the absolute position of row 0.
pub fn layerRows(gpa: std.mem.Allocator, w: LayerW, st: *LayerState, pos0: usize, offset: f64, h: []f64, rows: usize, trace: ?Trace) !void {
    const mixed = try gpa.alloc(f64, rows * hidden);
    defer gpa.free(mixed);
    const inject = try gpa.alloc(f64, rows * streams);
    defer gpa.free(inject);
    const branch = try gpa.alloc(f64, rows * hidden);
    defer gpa.free(branch);

    try hcPreRows(gpa, w.attn_hc, offset, h, rows, mixed, inject);
    switch (w.mixer) {
        .gdn => |g| try gdnBranch(gpa, g, st.gdn, mixed, rows, branch),
        .attn => |a| try attnBranch(gpa, a, &st.attn, pos0, mixed, rows, branch),
    }
    if (trace) |tr| @memcpy(tr.attn_branch[0 .. rows * hidden], branch);
    hcPostRows(h, branch, inject, rows);
    if (trace) |tr| @memcpy(tr.h_mid[0 .. rows * wide], h[0 .. rows * wide]);

    try hcPreRows(gpa, w.mlp_hc, offset, h, rows, mixed, inject);
    const picks = try gpa.alloc(u16, rows * topk);
    defer gpa.free(picks);
    try moeBranch(gpa, w.moe, mixed, rows, branch, picks);
    if (trace) |tr| {
        @memcpy(tr.mlp_branch[0 .. rows * hidden], branch);
        @memcpy(tr.picks[0 .. rows * topk], picks);
    }
    hcPostRows(h, branch, inject, rows);
}

// -- tests -------------------------------------------------------------------------------------------------

test "fwht128 is orthonormal and equals the dense Sylvester matrix" {
    var rng = std.Random.DefaultPrng.init(3);
    var x: [256]f64 = undefined;
    for (&x) |*v| v.* = rng.random().float(f64) * 2 - 1;
    var y = x;
    fwht128(&y);
    for (0..2) |blk| {
        for (0..128) |i| {
            var s: f64 = 0;
            for (0..128) |j| {
                const sign: f64 = if (@popCount(i & j) & 1 == 1) -1 else 1;
                s += sign * x[blk * 128 + j];
            }
            try std.testing.expectApproxEqAbs(s / @sqrt(128.0), y[blk * 128 + i], 1e-12);
        }
    }
    fwht128(&y);
    for (x, y) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
}

test "QMat.apply equals the dense decoder's dequantize then matvec" {
    const gpa = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(17);
    const r = rng.random();
    for ([_]u16{ 4, 6, 10 }) |halves| {
        const tiles_k: usize = 8;
        const tiles_n: usize = 16;
        const k = tiles_k * 16;
        const n = tiles_n * 16;
        const words = @as(usize, halves) * 8;
        const trellis = try gpa.alloc(i16, tiles_k * tiles_n * words);
        defer gpa.free(trellis);
        for (trellis) |*v| v.* = @bitCast(r.int(u16));
        const suh = try gpa.alloc(f16, k);
        defer gpa.free(suh);
        const svh = try gpa.alloc(f16, n);
        defer gpa.free(svh);
        for (suh) |*v| v.* = if (r.boolean()) 1.0 else -1.0;
        for (svh) |*v| v.* = @floatCast(0.5 + r.float(f32));
        const dense = try gpa.alloc(f64, k * n);
        defer gpa.free(dense);
        const scratch = try gpa.alloc(f64, 2 * k * n);
        defer gpa.free(scratch);
        format.dequantize(.{ .halves = halves }, .mul1, trellis, tiles_k, tiles_n, suh, svh, dense, scratch);
        var q = try QMat.init(gpa, halves, trellis, tiles_k, tiles_n, suh, svh, 4);
        defer q.deinit(gpa);
        var x: [k]f64 = undefined;
        for (&x) |*v| v.* = r.floatNorm(f64);
        var want: [n]f64 = undefined;
        for (0..n) |j| {
            var s: f64 = 0;
            for (0..k) |i| s += x[i] * dense[i * n + j];
            want[j] = s;
        }
        var got: [n]f64 = undefined;
        var tmp: [k]f64 = undefined;
        q.apply(&x, &got, &tmp);
        for (want, got) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-9 * (1 + @abs(a)));
        // column slices equal the whole
        var prep: [k]f64 = undefined;
        q.prepare(&x, &prep);
        var part: [128]f64 = undefined;
        q.applyCols(&prep, &part, 128, 256);
        for (part, got[128..256]) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
    }
}

test "parallelFor visits every index once" {
    const Ctx = struct {
        hits: []std.atomic.Value(u8),
        fn f(c: *@This(), i: usize) void {
            _ = c.hits[i].fetchAdd(1, .monotonic);
        }
    };
    var hits: [1000]std.atomic.Value(u8) = undefined;
    for (&hits) |*h| h.* = std.atomic.Value(u8).init(0);
    var c: Ctx = .{ .hits = &hits };
    parallelFor(hits.len, 8, &c, Ctx.f);
    for (&hits) |*h| try std.testing.expectEqual(@as(u8, 1), h.load(.monotonic));
}
