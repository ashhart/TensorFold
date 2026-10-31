//! Flash Next's prompt attention on the M5's tensor units (tf_sattn_nax, tf_attn256_nax), as replay_prompt.zig builds.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;

const D = 256;
const H = 24;
const GQA = 12;
const KVH = H / GQA;
const PW = 13952; // the attention projection's row: each head's 256 queries, then its 256 gates
const BK = 32; // both kernels' key blocks
const BQ = 64; // tf_attn256_nax's query rows a threadgroup
const TOP = tf.flashnext_replay.TOP;
const KW = tf.flashnext_replay.KW;
const scale: f64 = 0.0625; // engine_load.zig's attention_scale: 1 / sqrt(head_dim)
const nan16 = 0x7fc0;
/// Gate 4's keys: the first 1,024 score high (their probabilities exactly 1), every other key scores far below.
const stress_on = 1024;

/// fn_attn.metal's sparse rows (replay_prompt_segments.zig, while the indexer runs) or qmm6_nax_b.metal's causal chunk.
pub const Which = enum { sattn, attn256 };
/// P cut as the tensor units take it; P spread over [0, 1], values in [0, 1); that range without the cut; an edit.
pub const Case = enum { cut, spread, cutless, edit };

fn name(w: Which, k: Case) []const u8 {
    return switch (w) {
        inline else => |x| switch (k) {
            inline else => |y| comptime label(x, y),
        },
    };
}

fn label(comptime w: Which, comptime k: Case) []const u8 {
    const base = "metal flashnext prompt attention " ++ (if (w == .sattn) "tf_sattn_nax" else "tf_attn256_nax");
    return switch (k) {
        .cut => base ++ " 24 heads over 2, head dim 256",
        .spread => base ++ ", P spread over [0, 1]",
        .cutless => "control: " ++ base ++ ", P spread, its range without the operand cut",
        .edit => "control: " ++ base ++ (if (w == .sattn) ", the key past a row's last counted" else ", one key past the row seen"),
    };
}

/// Each kernel's planted control: one source edit gate 3 must catch.
fn edit(w: Which) struct { from: []const u8, to: []const u8 } {
    return switch (w) {
        .sattn => .{ .from = "s[f][e] = j < n ? v * scale2 : masked;", .to = "s[f][e] = j <= n ? v * scale2 : masked;" },
        .attn256 => .{ .from = "(col >= kL || col > row) ? a_masked", .to = "(col >= kL || col > row + 1) ? a_masked" },
    };
}

/// tf_sattn_nax: four simdgroups' 4 mma steps over 64 dims each, then 3 adds; a 32-key block: rescale, 2 P x V steps.
fn sattnForm(n: usize, cut: f64) ar.Form {
    const blocks: f64 = @floatFromInt(n / BK + 2);
    // den: total times keep and one add a block, plus row_fold's 5-level tree of 32 weights; div: fast 1 / total
    return .{ .score = 4 * ar.mma_step + 3, .fast_exp = true, .chunk_exp = false, .p_rel = 0, .p_cut = cut, .num = (2 * ar.mma_step + 1) * blocks, .den = 2 * blocks + 5, .score_round = 4, .div = 6 };
}

/// tf_attn256_nax: 16 mma steps over 256 dims; a 32-key block: rescale, 2 P x V steps, two 4-level folds into total.
fn attnForm(n: usize, cut: f64) ar.Form {
    const blocks: f64 = @floatFromInt(n / BK + 2);
    return .{ .score = 16 * ar.mma_step, .fast_exp = true, .chunk_exp = false, .p_rel = 0, .p_cut = cut, .num = (2 * ar.mma_step + 1) * blocks, .den = 3 * blocks + 4, .score_round = 4, .div = 6 };
}

/// P x V's chain at the most keys a row reads (the range's own counts replace it through `bounds`).
fn depth(s: c.Shape) usize {
    return (2 * ar.mma_step + 1) * ((s.k + s.rows) / BK + 2);
}

// rows: a prompt chunk's rows; n: the head dim; k: the chunk's first position
const sattn_served = [_]c.Shape{ .{ .rows = 64, .n = D, .k = 0 }, .{ .rows = 24, .n = D, .k = 2036 }, .{ .rows = 6, .n = D, .k = 30000 } };
const attn_served = [_]c.Shape{ .{ .rows = 96, .n = D, .k = 0 }, .{ .rows = 40, .n = D, .k = 700 }, .{ .rows = 8, .n = D, .k = 1100 } };

pub fn spec(w: Which, k: Case) c.Spec {
    const keys: c.Tile = .{ .size = BK, .step = 1, .min = 0 };
    return .{
        .name = name(w, k),
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        // sattn: a threadgroup a row, so rows have no edge; attn256: 64-row query tiles
        .tiles = if (w == .sattn) .{ .{}, .{}, keys } else .{ .{ .size = BQ }, .{}, keys },
        // sattn: a chunk from the start, one crossing into selected blocks, one deep in the context; attn256 dense
        .served = if (w == .sattn) &sattn_served else &attn_served,
        .sums = .{ .exact = true },
    };
}

fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// bsig's range: Metal's fast exp (attention_range's 40 + 2 |x| ulps), the add, fast division (2.5 ulps), bf16.
fn sigRange(x: f64) [2]f64 {
    const e = @exp(-x);
    const ee = (40 + 2 * @abs(x)) * 0x1p-23;
    const lo = (1 + e * (1 - ee)) * (1 - 0x1p-24);
    const hi = (1 + e * (1 + ee)) * (1 + 0x1p-24);
    return .{ bf(1 / hi * (1 - 2.5 * 0x1p-23)), bf(1 / lo * (1 + 2.5 * 0x1p-23)) };
}

/// bf16(o) times bsig, rounded: the four corners of the two ranges; bf16 times bf16 is exact in fp32 but below normals.
fn gated(o: [2]f64, s: [2]f64) [2]f64 {
    const p = [4]f64{ o[0] * s[0], o[0] * s[1], o[1] * s[0], o[1] * s[1] };
    const lo = @min(@min(p[0], p[1]), @min(p[2], p[3]));
    const hi = @max(@max(p[0], p[1]), @max(p[2], p[3]));
    return .{ bf(lo - 0x1p-149), bf(hi + 0x1p-149) };
}

fn fillRandom(rng: std.Random, out: []u16, r: f32) void {
    for (out) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * r);
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    which: Which,
    case: Case,
    rows: usize = 0,
    pos: usize = 0,
    cap: usize = 0,
    fill: g.Fill = .random,
    q: []u16 = &.{},
    k: []u16 = &.{},
    v: []u16 = &.{},
    gp: []u16 = &.{},
    ids: []i32 = &.{},
    nk: []i32 = &.{},
    sparse: []i32 = &.{},
    lo: []f64 = &.{},
    hi: []f64 = &.{},
    refs: [8]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// The library as replay_prompt.zig builds it here (frags.source, MLX's options), the control's edit if asked.
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which, k: Case) !Probe {
        const src = try tf.frags.source(device, a, if (w == .sattn) sources.flashnext_attn else sources.flashnext_qmm6);
        defer a.free(src);
        var text = src;
        if (k == .edit) {
            const e = edit(w);
            if (std.mem.count(u8, src, e.from) != 1) return error.ControlEditMissing;
            text = try std.mem.replaceOwned(u8, a, src, e.from, e.to);
        }
        defer if (k == .edit) a.free(text);
        const pipe = try rig_.pipeline(device, text, if (w == .sattn) "tf_sattn_nax" else "tf_attn256_nax");
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .which = w, .case = k };
    }

    fn free(p: *Probe) void {
        for ([_]*[]u16{ &p.q, &p.k, &p.v, &p.gp }) |x| {
            p.a.free(x.*);
            x.* = &.{};
        }
        for ([_]*[]i32{ &p.ids, &p.nk, &p.sparse }) |x| {
            p.a.free(x.*);
            x.* = &.{};
        }
        for ([_]*[]f64{ &p.lo, &p.hi }) |x| {
            p.a.free(x.*);
            x.* = &.{};
        }
    }

    pub fn deinit(p: *Probe) void {
        p.free();
        p.rig.deinit();
        p.pipe.deinit();
    }

    fn at(p: *const Probe, kv: usize, key: usize) usize {
        return (kv * p.cap + key) * D;
    }

    /// Select.meta's rows: dense while a row's complete 4-key blocks are at most TOP, else TOP blocks and the tail.
    fn select(p: *Probe, rng: std.Random) !void {
        const blocks = try p.a.alloc(bool, (p.pos + p.rows) / 4 + 1);
        defer p.a.free(blocks);
        for (0..p.rows) |r| {
            const e = p.pos + r + 1;
            const cb = e / 4;
            const ids = p.ids[r * KW ..][0..KW];
            @memset(ids, @intCast(p.pos + p.rows)); // a dense row never reads its ids: they name a NaN cache row here
            p.sparse[r] = @intFromBool(cb > TOP);
            if (cb <= TOP) {
                p.nk[r] = @intCast(e);
                continue;
            }
            @memset(blocks, false);
            if (p.fill == .stress) @memset(blocks[0..TOP], true) else {
                var kept: usize = 0;
                while (kept < TOP) {
                    const b = rng.uintLessThan(usize, cb);
                    if (blocks[b]) continue;
                    blocks[b] = true;
                    kept += 1;
                }
            }
            // the chosen complete blocks in ascending order, then the tail's keys
            var n: usize = 0;
            for (blocks[0..cb], 0..) |on, b| if (on) for (0..4) |i| {
                ids[n] = @intCast(4 * b + i);
                n += 1;
            };
            for (4 * cb..e) |key| {
                ids[n] = @intCast(key);
                n += 1;
            }
            p.nk[r] = @intCast(n);
        }
    }

    /// Scores q . k = q0 k0 alone, so P = e^(s - max) spreads over about [2^-4, 1]; values in [0, 1) add the cut up.
    fn spread(p: *Probe, rng: std.Random) void {
        for (0..p.rows * H) |r| {
            @memset(p.q[r * D ..][0..D], 0);
            p.q[r * D] = c.toBf16(0.5 + rng.float(f32));
        }
        for (0..KVH) |kv| for (0..p.pos + p.rows) |key| {
            p.k[p.at(kv, key)] = c.toBf16(-32 * rng.float(f32));
            for (p.v[p.at(kv, key)..][0..D]) |*x| x.* = c.toBf16(rng.float(f32));
        };
        @memset(p.gp, c.toBf16(20)); // bsig(20) is 1: the outputs show bf16(o) itself
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != D) return false;
        if (fill == .stress and shape.k < stress_on) return false;
        p.rig.reset();
        p.free();
        p.rows = shape.rows;
        p.pos = shape.k;
        p.fill = fill;
        const keys = p.pos + p.rows;
        // sattn reads only its rows' keys (one NaN row past them); attn256 reads whole 32-key blocks
        p.cap = if (p.which == .sattn) keys + 1 else std.mem.alignForward(usize, keys, BK);
        const pad = if (p.which == .sattn) p.rows else std.mem.alignForward(usize, p.rows, BQ);
        p.q = try p.a.alloc(u16, pad * H * D);
        p.k = try p.a.alloc(u16, KVH * p.cap * D);
        p.v = try p.a.alloc(u16, KVH * p.cap * D);
        p.gp = try p.a.alloc(u16, p.rows * PW);
        p.ids = try p.a.alloc(i32, p.rows * KW);
        p.nk = try p.a.alloc(i32, p.rows);
        p.sparse = try p.a.alloc(i32, p.rows);
        p.lo = try p.a.alloc(f64, p.count());
        p.hi = try p.a.alloc(f64, p.count());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        fillRandom(rng, p.q, 2);
        fillRandom(rng, p.k, 2);
        fillRandom(rng, p.v, 2);
        fillRandom(rng, p.gp, 4);
        switch (fill) {
            .random => if (p.case == .spread or p.case == .cutless) p.spread(rng),
            .nonfinite => {
                // key 0's values, which every row attends; three queries' heads; key 2's key, which rows 2 on see
                for (0..KVH) |kv| for ([_]u16{ nan16, 0x7f80, 0xff80 }, 0..) |x, d| {
                    p.v[p.at(kv, 0) + d] = x;
                };
                for ([_]u16{ nan16, 0x7f80, 0xff80 }, 1..) |x, r| if (r < p.rows) {
                    p.q[(r * H + 5 * r) * D + r] = x;
                };
                p.k[p.at(1, 2) + 7] = nan16;
            },
            .stress => {
                // keys of 3s or -3s: the first 1,024 score high, the rest probability 0; values cancel but for 8 nudged
                @memset(p.q, c.toBf16(3));
                for (0..KVH) |kv| for (0..keys) |key| @memset(p.k[p.at(kv, key)..][0..D], c.toBf16(if (key < stress_on) 3 else -3));
                for (0..KVH) |kv| for (0..stress_on / 2) |key| for (0..D) |e| {
                    const x: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 255));
                    p.v[p.at(kv, key) + e] = c.toBf16(x);
                    p.v[p.at(kv, key + stress_on / 2) + e] = c.toBf16(-x + @as(f32, @floatFromInt(@intFromBool(key < 8))));
                };
                @memset(p.gp, c.toBf16(20));
            },
        }
        if (p.which == .sattn) {
            // cache rows past the chunk hold anything: NaN here, which tf_sattn_nax never reads
            for (0..KVH) |kv| for ([_][]u16{ p.k, p.v }) |x| @memset(x[p.at(kv, keys)..][0..((p.cap - keys) * D)], nan16);
            try p.select(rng);
        } else {
            // query rows past the chunk are read into tiles whose outputs are never written: NaN here
            @memset(p.q[p.rows * H * D ..], nan16);
        }
        try p.bind();
        return true;
    }

    fn bind(p: *Probe) !void {
        const f = [1]f32{@floatCast(scale)};
        const list = if (p.which == .sattn)
            [_][]const u8{ std.mem.sliceAsBytes(p.q), std.mem.sliceAsBytes(p.k), std.mem.sliceAsBytes(p.v), std.mem.sliceAsBytes(p.ids), std.mem.sliceAsBytes(p.nk), std.mem.sliceAsBytes(p.sparse), std.mem.sliceAsBytes(p.gp), std.mem.asBytes(&f) }
        else
            [_][]const u8{ std.mem.sliceAsBytes(p.q), std.mem.sliceAsBytes(p.k), std.mem.sliceAsBytes(p.v), std.mem.sliceAsBytes(p.gp), std.mem.asBytes(&f), "", "", "" };
        for (list, &p.refs) |bytes, *r| if (bytes.len > 0) {
            r.* = try p.rig.input(bytes);
        };
        p.out = try p.rig.output(p.count() * 2);
    }

    /// One launch as replay_prompt_segments.zig encodes it.
    pub fn run(p: *Probe) !void {
        const ps = p.rig.begin();
        ps.e.setPipeline(p.pipe);
        if (p.which == .sattn) {
            for (p.refs[0..8], 0..) |r, i| ps.e.setBuffer(r.buf, r.off, i);
            const cap: i32 = @intCast(p.cap);
            ps.e.setBytes(std.mem.asBytes(&cap), 8);
            ps.e.setBuffer(p.out.buf, p.out.off, 9);
            ps.e.dispatchThreads(mtl.Size.of(p.rows * 128, KVH, 1), mtl.Size.of(128, 1, 1));
        } else {
            for (p.refs[0..4], 0..) |r, i| ps.e.setBuffer(r.buf, r.off, i);
            const ap = [4]i32{ @intCast(p.rows), @intCast(p.pos + p.rows), @intCast(p.pos), @intCast(p.cap) };
            ps.e.setBytes(std.mem.asBytes(&ap), 4);
            ps.e.setBuffer(p.refs[4].buf, p.refs[4].off, 5);
            ps.e.setBuffer(p.out.buf, p.out.off, 6);
            ps.e.dispatchThreads(mtl.Size.of(((p.rows + BQ - 1) / BQ) * 128, H, 1), mtl.Size.of(128, 1, 1));
        }
        try p.rig.finish(ps);
    }

    pub fn count(p: *const Probe) usize {
        return p.rows * H * D;
    }

    /// Row r's keys: a sparse row's ids, else every position up to the row's own.
    fn rowKeys(p: *const Probe, r: usize, out: []usize) []usize {
        const n: usize = if (p.which == .sattn) @intCast(p.nk[r]) else p.pos + r + 1;
        for (out[0..n], 0..) |*x, j| x.* = if (p.which == .sattn and p.sparse[r] != 0) @intCast(p.ids[r * KW + j]) else j;
        return out[0..n];
    }

    /// Each head's float64 output at bf16(bf16(o) * bsig(gate)), and its range: o's with the cut, then the gate's.
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const most = p.pos + p.rows;
        const buf = try p.a.alloc(f64, 2 * most * D + 2 * most + 5 * D);
        defer p.a.free(buf);
        const keys = try p.a.alloc(usize, most);
        defer p.a.free(keys);
        const kf = buf[0 .. most * D];
        const vf = buf[most * D .. 2 * most * D];
        const w = buf[2 * most * D ..][0..most];
        const tau = buf[2 * most * D + most ..][0..most];
        const head = buf[2 * most * D + 2 * most ..];
        const qv = head[0..D];
        const pt = head[D .. 2 * D];
        const hf = head[2 * D .. 3 * D];
        const down = head[3 * D .. 4 * D];
        const up = head[4 * D .. 5 * D];
        const cut: f64 = if (p.case == .cutless) 0 else ar.operand_cut;
        for (0..p.rows) |r| {
            const ks = p.rowKeys(r, keys);
            const n = ks.len;
            const f = if (p.which == .sattn) sattnForm(n, cut) else attnForm(n, cut);
            for (0..KVH) |kv| {
                for (ks, 0..) |key, j| for (0..D) |e| {
                    kf[j * D + e] = c.fromBf16(p.k[p.at(kv, key) + e]);
                    vf[j * D + e] = c.fromBf16(p.v[p.at(kv, key) + e]);
                };
                for (kv * GQA..(kv + 1) * GQA) |h| {
                    for (qv, 0..) |*x, e| x.* = c.fromBf16(p.q[(r * H + h) * D + e]);
                    ar.head(qv, kf, vf, n, D, scale, f, w, tau, pt, hf);
                    ar.cutSides(f, w, tau, vf, n, D, down, up);
                    for (0..D) |e| {
                        const i = (r * H + h) * D + e;
                        const gv: f64 = c.fromBf16(p.gp[r * PW + h * 2 * D + D + e]);
                        want[i] = bf(bf(pt[e]) * bf(1 / (1 + @exp(-gv))));
                        mag[i] = if (std.math.isFinite(want[i])) @abs(want[i]) else 0;
                        const open = p.fill == .stress or !std.math.isFinite(pt[e]) or !std.math.isFinite(hf[e]);
                        const o = ar.ends(pt[e], hf[e], down[e], up[e]);
                        const ends = if (open) [2]f64{ -std.math.floatMax(f64), std.math.floatMax(f64) } else gated(o, sigRange(gv));
                        p.lo[i] = ends[0];
                        p.hi[i] = ends[1];
                    }
                }
            }
        }
    }

    /// Each output's farther end from float64's value; gate 4 allows the rounding alone, a wide range any finite value.
    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        const s = spec(p.which, p.case);
        for (allow, want, p.lo, p.hi) |*l, w, lo, hi| {
            if (p.fill == .stress) {
                l.* = c.boundExact(s, w);
            } else if (!std.math.isFinite(w)) {
                l.* = 0;
            } else l.* = @min(std.math.floatMax(f64), @max(@abs(w - lo), @abs(hi - w)));
        }
    }

    /// Each bf16 output within its range's ends: o's with the cut on its own side, through bf16 and the rounded gate.
    pub fn intervals(p: *Probe, low: []f64, high: []f64) !void {
        @memcpy(low, p.lo);
        @memcpy(high, p.hi);
    }

    /// Gate 4: the outputs if P x V's sum of the 1,024 values ran in `acc` (bsig is 1).
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        var terms: [stress_on]f64 = undefined;
        for (0..p.rows) |r| for (0..H) |h| for (0..D) |e| {
            for (&terms, 0..) |*x, key| x.* = c.fromBf16(p.v[p.at(h / GQA, key) + e]);
            out[(r * H + h) * D + e] = bf(contract.rows.sumIn(acc, order, &terms) / stress_on);
        };
    }
};

const vt = g.hooks(Probe);

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    /// Only on the tensor units: before the M5, nax.h's fragments keep P whole, and these forms are not run there.
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) return;
        for (std.enums.values(Which)) |w| for (std.enums.values(Case)) |k| {
            if (only) |o| if (std.mem.indexOf(u8, name(w, k), o) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, w, k);
            try s.probes.append(s.a, p);
            const control: ?c.Gate = if (k == .cutless or k == .edit) .shapes else null;
            try s.kernels.append(s.a, .{ .spec = spec(w, k), .ctx = p, .vt = &vt, .control = control });
        };
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
