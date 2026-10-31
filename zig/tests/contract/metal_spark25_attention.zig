//! Spark-X2.6-4B's prompt attention (spark25/attention_nax.metal) as metal_model.zig builds it, both layer kinds.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;
const cfg = tf.spark25_metal.config;
const st = tf.spark25_metal.state;
const geo = cfg.spark4b;
const dh: usize = cfg.head_dim;
const nq: usize = geo.query_heads;
const kvh: usize = geo.kv_heads;
const grp: usize = nq / kvh;
const kw: usize = kvh * dh;
const scale: f64 = cfg.scale;
const rt = 32; // query rows a tile
const bkt = dh / 64 * 32; // keys a block: 32 a simdgroup, a simdgroup each 64 output columns
const nan16 = 0x7fc0;

/// The two pipelines metal_model.zig builds: sliding layers (a 512-key window over a ring) and full layers.
pub const Layer = enum { sliding, full };

/// The engine's P with the M5's cut; P spread over [0, 1], values in [0, 1); that without the cut; a bad edit.
const Case = enum { plain, spread, cutless, edit };

fn window(l: Layer) usize {
    return if (l == .sliding) cfg.window else 0;
}

fn ring(l: Layer) usize {
    return if (l == .sliding) st.sliding_ring else 0;
}

/// metal_model.zig's header for a layer kind.
fn header(a: std.mem.Allocator, l: Layer) ![]u8 {
    return std.fmt.allocPrint(a, "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define KVH {d}\n#define WINDOW {d}\n#define RING {d}\n#define SCALE {e}f\n", .{ cfg.head_dim, geo.query_heads, geo.query_heads / geo.kv_heads, geo.kv_heads, window(l), ring(l), cfg.scale });
}

/// Each layer's planted control: sliding sees one key past its window, full one key past its row.
fn edit(l: Layer) [2][]const u8 {
    return switch (l) {
        .sliding => .{ "col < row - WINDOW + 1", "col < row - WINDOW" },
        .full => .{ "const bool out = col > row ||", "const bool out = col > row + 1 ||" },
    };
}

/// The source as metal_model.zig builds it on this GPU (header, then frags.source's text), with a control's edit.
fn built(a: std.mem.Allocator, device: mtl.Device, l: Layer, case: Case) ![]u8 {
    const head = try header(a, l);
    defer a.free(head);
    const body = try tf.frags.source(device, a, sources.spark25_attention_nax);
    defer a.free(body);
    const src = try std.mem.concat(a, u8, &.{ head, body });
    if (case != .edit) return src;
    defer a.free(src);
    const e = edit(l);
    if (std.mem.count(u8, src, e[0]) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, src, e[0], e[1]);
}

/// Gate 5's texts for `spark25 attention <layer> ` launches, nax.h inlined (fragments before the M5).
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, launch: []const u8) !struct { here: []const u8, pre_m5: []const u8 } {
    const l: Layer = if (std.mem.indexOf(u8, launch, " sliding ") != null) .sliding else if (std.mem.indexOf(u8, launch, " full ") != null) .full else return error.UnknownLaunch;
    const head = try header(a, l);
    const body = try std.mem.replaceOwned(u8, a, sources.spark25_attention_nax, "#include \"../nax.h\"", sources.nax);
    const pre = try std.mem.concat(a, u8, &.{ head, "#define TF_SIMD_FRAGS 1\n", body });
    return .{ .here = if (device.tensorUnits()) try std.mem.concat(a, u8, &.{ head, body }) else pre, .pre_m5 = pre };
}

pub fn name(l: Layer, case: Case) []const u8 {
    return switch (l) {
        inline else => |x| switch (case) {
            inline else => |k| comptime label(x, k),
        },
    };
}

fn label(comptime l: Layer, comptime k: Case) []const u8 {
    const base = "metal spark25 prompt attention " ++ @tagName(l);
    return switch (k) {
        .plain => base,
        .spread => base ++ ", P spread over [0, 1]",
        .cutless => "control: " ++ base ++ ", P spread, its range without the operand cut",
        .edit => "control: " ++ base ++ (if (l == .sliding) ", a key past the window" else ", a key past the row"),
    };
}

/// The score's mma chain; the bounds hook below replaces the generic bound it would set.
fn depth(_: c.Shape) usize {
    return dh / 16 * ar.mma_step;
}

/// Rows: a prompt chunk's rows (the engine takes this path past 16); n: the head dim; k: the chunk's first position.
fn spec(l: Layer, case: Case) c.Spec {
    const plain_shapes: []const c.Shape = switch (l) {
        // a chunk mid-block, the shortest from the start, one deep in a wrapped ring, a whole chunk after a wrap
        .sliding => &.{ .{ .rows = 33, .n = dh, .k = 777 }, .{ .rows = 17, .n = dh, .k = 0 }, .{ .rows = 300, .n = dh, .k = 100003 }, .{ .rows = st.prompt_rows, .n = dh, .k = 2600 } },
        // a chunk mid-block, the shortest from the start, a whole chunk from the start, one deep in the context
        .full => &.{ .{ .rows = 33, .n = dh, .k = 300 }, .{ .rows = 17, .n = dh, .k = 0 }, .{ .rows = st.prompt_rows, .n = dh, .k = 0 }, .{ .rows = 17, .n = dh, .k = 32700 } },
    };
    const spread_shapes: []const c.Shape = switch (l) {
        .sliding => &.{ .{ .rows = 64, .n = dh, .k = 700 }, .{ .rows = 33, .n = dh, .k = 0 } },
        .full => &.{ .{ .rows = 64, .n = dh, .k = 1100 }, .{ .rows = 33, .n = dh, .k = 0 } },
    };
    const spread = case == .spread or case == .cutless;
    return .{
        .name = name(l, case),
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .tiles = if (spread) .{ .{}, .{}, .{} } else .{ .{ .size = rt, .min = st.window_rows + 1 }, .{}, .{ .size = bkt, .min = 0 } },
        .served = if (spread) spread_shapes else plain_shapes,
        .sums = .{ .exact = true },
        .known = if (case == .plain or case == .spread) masked_key else null,
    };
}

/// Gate 1's plant at key p0 + 5, which rows 0 to 4 do not see but their tile reads, makes their outputs NaN.
const masked_key: c.Known = .{ .gate = .nonfinite, .finding = "a non-finite value at a key masked for a row of its tile reaches that row as NaN (P = 0 times Inf inside the P x V mma)" };

/// sp_prompt_attention over n keys, its counts from the source; `cut` the fp32 P's operand cut into P x V.
fn form(n: usize, cut: f64) ar.Form {
    const blocks: f64 = @floatFromInt(n / bkt + 2); // 128-key blocks a row's tile visits, a window starting mid-block
    return .{
        .score = @as(f64, @floatFromInt(dh / 16)) * ar.mma_step, // DH / 16 mma steps a score, from zero each block
        .score_round = 4, // the fp32 SCALE * log2e and its product
        .fast_exp = true,
        // fast::exp2's own 3 + 1.2 |x|; each block's keep, exp2(top - top_new), adds 3, and their arguments sum to |x|
        .exp_ulps = .{ 3 + 3 * blocks, 1.2 + 1.2 + 0.5 },
        .chunk_exp = false,
        .p_rel = 0, // P is fp32 in threadgroup memory
        .p_cut = cut,
        .num = (bkt / 16 * ar.mma_step + 1) * blocks, // a block's 8 P x V steps on the running acc, then its keep
        .den = 3 * blocks + 7, // a block's keep and two fold adds, a fold's 4-level tree; 3 adds over the simdgroups
        .div = 2, // 1 / total, then the product
        .nan_max = true,
    };
}

/// Gate 4's keys: values repeat every 512 positions, a half and its negation but for 8 nudged keys a column.
const period = 512;
/// Gate 4's values are multiples of 64 up to 255 * 64: a 32-lane order's lane sums pass fp16's largest, all stay exact.
const step: f32 = 64;
/// Full layers' high keys (every other key scores far below); a sliding row's window is all high.
const stress_on = 1024;

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    layer: Layer,
    case: Case,
    cut: f64,
    rows: usize = 0,
    p0: usize = 0,
    slots: usize = 0,
    fill: g.Fill = .random,
    q: []u16 = &.{},
    k: []u16 = &.{},
    v: []u16 = &.{},
    point: []f64 = &.{},
    half: []f64 = &.{},
    down: []f64 = &.{},
    up: []f64 = &.{},
    refs: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, l: Layer, case: Case) !Probe {
        const src = try built(a, device, l, case);
        defer a.free(src);
        const cut: f64 = if (case == .cutless) 0 else ar.operand_cut; // the cutless control states P whole
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, src, "sp_prompt_attention"), .layer = l, .case = case, .cut = cut };
    }

    fn free(p: *Probe) void {
        for ([_][]u16{ p.q, p.k, p.v }) |x| p.a.free(x);
        for ([_][]f64{ p.point, p.half, p.down, p.up }) |x| p.a.free(x);
        p.q = &.{};
        p.k = &.{};
        p.v = &.{};
        p.point = &.{};
        p.half = &.{};
        p.down = &.{};
        p.up = &.{};
    }

    fn deinit(p: *Probe) void {
        p.free();
        p.rig.deinit();
        p.pipe.deinit();
    }

    fn kernel(p: *Probe) g.Kernel {
        const control: ?g.Gate = if (p.case == .cutless or p.case == .edit) .shapes else null;
        return .{ .spec = spec(p.layer, p.case), .ctx = p, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Probe);

    /// Position x's cache slot: the ring's, or x itself.
    fn slot(p: *const Probe, x: usize) usize {
        return if (p.layer == .sliding) x % st.sliding_ring else x;
    }

    /// The first key position row `pos` sees.
    fn first(p: *const Probe, pos: usize) usize {
        return if (p.layer == .sliding) (pos + 1) -| cfg.window else 0;
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > st.prompt_rows or shape.n != dh) return false;
        if (fill == .stress and shape.k < @as(usize, if (p.layer == .sliding) cfg.window else stress_on)) return false;
        p.rig.reset();
        p.free();
        p.rows = shape.rows;
        p.p0 = shape.k;
        p.fill = fill;
        const end = p.p0 + p.rows; // positions written once the chunk's keys are in
        // the ring, or whole blocks: a block scores all its 128 keys, past the layer it reads the engine's values
        p.slots = if (p.layer == .sliding) st.sliding_ring else std.mem.alignForward(usize, end, bkt);
        const pad = std.mem.alignForward(usize, p.rows, rt); // a tile reads all its 32 query rows
        p.q = try p.a.alloc(u16, pad * nq * dh);
        p.k = try p.a.alloc(u16, p.slots * kw);
        p.v = try p.a.alloc(u16, p.slots * kw);
        for ([_]*[]f64{ &p.point, &p.half, &p.down, &p.up }) |x| x.* = try p.a.alloc(f64, p.count());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        fillRandom(rng, p.q, 3);
        fillRandom(rng, p.k, 3);
        fillRandom(rng, p.v, 2);
        switch (fill) {
            .random => if (p.case == .spread or p.case == .cutless) p.spread(rng),
            .nonfinite => {
                // every row sees key p0, rows 2 on key p0 + 2; rows 0 to 4 not key p0 + 5, which their tile reads
                for ([_]usize{ 3, 1 * dh, 2 * dh + 5 }, [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |at, x, r| if (r < p.rows) {
                    p.q[(r * nq + r * grp + 1) * dh + at % dh] = c.toBf16(x);
                };
                p.v[p.slot(p.p0) * kw + 0 * dh + 5] = c.toBf16(std.math.inf(f32));
                p.v[p.slot(p.p0) * kw + 1 * dh + 6] = c.toBf16(std.math.nan(f32));
                if (p.rows > 5) p.v[p.slot(p.p0 + 5) * kw + 3 * dh + 7] = c.toBf16(std.math.inf(f32));
                p.k[p.slot(p.p0 + 2) * kw + 2 * dh] = c.toBf16(std.math.nan(f32));
            },
            .stress => try p.stress(rng),
        }
        // rows past the chunk and slots no position has reached hold anything: NaN here, which no output may see
        @memset(p.q[p.rows * nq * dh ..], nan16);
        if (end < p.slots) {
            @memset(p.k[end * kw ..], nan16);
            @memset(p.v[end * kw ..], nan16);
        }
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.q));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.k));
        p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.v));
        p.refs[3] = try p.rig.input(std.mem.asBytes(&[2]i32{ @intCast(p.rows), @intCast(p.p0) }));
        p.out = try p.rig.output(p.count() * 2);
        return true;
    }

    /// Scores q . k = q0 k0 alone, so P = 2^(s - max) spreads over about [2^-9, 1]; values in [0, 1) add the bias up.
    fn spread(p: *Probe, rng: std.Random) void {
        for (0..p.rows * nq) |r| {
            @memset(p.q[r * dh ..][0..dh], 0);
            p.q[r * dh] = c.toBf16(0.5 + rng.float(f32));
        }
        for (0..p.slots * kvh) |r| {
            p.k[r * dh] = c.toBf16(-64 * rng.float(f32));
            for (p.v[r * dh ..][0..dh]) |*x| x.* = c.toBf16(rng.float(f32));
        }
    }

    /// Equal high scores (P exactly 1) over a sliding window or a full layer's first 1,024 keys, the rest P = 0.
    fn stress(p: *Probe, rng: std.Random) !void {
        @memset(p.q, c.toBf16(3));
        const base = try p.a.alloc(f32, period / 2 * kw);
        defer p.a.free(base);
        for (base) |*x| x.* = step * @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, 255)));
        // a slot holds the latest position at it, so its value follows its position modulo 512 (the ring is 5 periods)
        for (0..p.slots) |s| {
            const high = p.layer == .sliding or s < stress_on;
            @memset(p.k[s * kw ..][0..kw], c.toBf16(if (high) 3 else -3));
            const j = s % period;
            for (0..kw) |e| p.v[s * kw + e] = c.toBf16(if (j < period / 2) base[j * kw + e] else -base[(j - period / 2) * kw + e] + step * @as(f32, @floatFromInt(@intFromBool(j - period / 2 < 8))));
        }
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        for (p.refs[0..3], 0..) |r, i| pass.e.setBuffer(r.buf, r.off, i);
        pass.e.setBuffer(p.out.buf, p.out.off, 3);
        pass.e.setBuffer(p.refs[3].buf, p.refs[3].off, 4);
        // as metal_forward.zig: (rows / 32, query heads) groups, 32 threads a 64-column simdgroup
        pass.e.dispatchGroups(mtl.Size.of((p.rows + rt - 1) / rt, nq, 1), mtl.Size.of(dh / 64 * 32, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return p.rows * nq * dh;
    }

    /// Each row's heads over the keys it sees (a window's or every one up to it): float64's points and ranges.
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const lo = p.first(p.p0);
        const most = p.p0 + p.rows - lo;
        const buf = try p.a.alloc(f64, 2 * most * dh + 2 * most + 2 * dh);
        defer p.a.free(buf);
        const ks = buf[0 .. most * dh];
        const vs = buf[most * dh .. 2 * most * dh];
        const w = buf[2 * most * dh ..][0..most];
        const tau = buf[2 * most * dh + most ..][0..most];
        const qv = buf[buf.len - 2 * dh ..][0..dh];
        const hf = buf[buf.len - dh ..];
        for (0..kvh) |kv| {
            for (0..most) |j| for (0..dh) |e| {
                const at = p.slot(lo + j) * kw + kv * dh + e;
                ks[j * dh + e] = c.fromBf16(p.k[at]);
                vs[j * dh + e] = c.fromBf16(p.v[at]);
            };
            for (0..p.rows) |r| {
                const pos = p.p0 + r;
                const from = p.first(pos) - lo;
                const n = pos + 1 - p.first(pos);
                const f = form(n, p.cut);
                const kr = ks[from * dh ..][0 .. n * dh];
                const vr = vs[from * dh ..][0 .. n * dh];
                for (kv * grp..(kv + 1) * grp) |hq| {
                    for (qv, 0..) |*x, e| x.* = c.fromBf16(p.q[(r * nq + hq) * dh + e]);
                    const o = (r * nq + hq) * dh;
                    ar.head(qv, kr, vr, n, dh, scale, f, w, tau, want[o..][0..dh], hf);
                    @memcpy(p.half[o..][0..dh], hf);
                    ar.cutSides(f, w, tau, vr, n, dh, p.down[o..][0..dh], p.up[o..][0..dh]);
                }
            }
        }
        @memcpy(mag, p.half);
        @memcpy(p.point, want);
    }

    /// The range's half width and the cut's wider side plus bf16's rounding; gate 4 allows the rounding alone.
    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, p.half, p.down, p.up) |*l, w, h, d, u| {
            if (p.fill == .stress) {
                l.* = c.boundExact(spec(p.layer, p.case), w);
            } else if (!std.math.isFinite(w)) {
                l.* = 0;
            } else if (!std.math.isFinite(h)) {
                l.* = std.math.floatMax(f64); // scores too far apart for the range's first order: any finite value
            } else l.* = h + @max(d, u) + 0x1p-8 * (@abs(w) + h + @max(d, u)) + 0x1p-133;
        }
    }

    /// Each bf16 output within its range's ends as rounded, the cut on its own side; gate 4 and wide ranges are open.
    pub fn intervals(p: *Probe, low: []f64, high: []f64) !void {
        for (low, high, p.point, p.half, p.down, p.up) |*l, *hi, pt, h, d, u| {
            const open = p.fill == .stress or !std.math.isFinite(pt) or !std.math.isFinite(h);
            const e = if (open) [2]f64{ -std.math.floatMax(f64), std.math.floatMax(f64) } else ar.ends(pt, h, d, u);
            l.* = e[0];
            hi.* = e[1];
        }
    }

    /// Gate 4: the outputs if P x V's sum of the high keys' values ran in `acc`, in position order.
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        var terms: [stress_on]f64 = undefined;
        for (0..p.rows) |r| for (0..nq) |hq| for (0..dh) |e| {
            const col = (hq / grp) * dh + e;
            const pos = p.p0 + r;
            const keys = if (p.layer == .sliding) terms[0..cfg.window] else terms[0..stress_on];
            const from = if (p.layer == .sliding) p.first(pos) else 0;
            for (keys, 0..) |*x, j| x.* = c.fromBf16(p.v[p.slot(from + j) * kw + col]);
            const sum = contract.rows.sumIn(acc, order, keys) / @as(f64, @floatFromInt(keys.len));
            out[(r * nq + hq) * dh + e] = c.fromBf16(c.toBf16(@floatCast(sum)));
        };
    }
};

/// Random bf16 values in [-r, r).
fn fillRandom(rng: std.Random, out: []u16, r: f32) void {
    for (out) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * r);
}

/// Both layers' pipelines with their controls, on the tensor units only: before the M5 the path is not qualified.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) return;
        for (std.enums.values(Layer)) |l| for (std.enums.values(Case)) |k| {
            if (only) |o| if (std.mem.indexOf(u8, name(l, k), o) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, l, k);
            try s.probes.append(s.a, p);
            try s.kernels.append(s.a, p.kernel());
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
