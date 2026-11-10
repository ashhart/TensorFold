//! One prompt kernel's contract test: its inputs for each gate's fill, and its launch as prompt_kernels.zig binds it.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("metal_h3_image_prompt_ref.zig");
const want = @import("metal_h3_image_prompt_want.zig");

const c = contract.spec;
const g = contract.gates;
pub const Mode = ref.Mode;

pub const Kind = enum { gemm, norm, heads, attention, glu, add, scatter };

/// A registration: its kernel, its rounding mode and layout, where it is served, and any finding it carries.
pub const Form = struct {
    kind: Kind,
    /// The name's suffix past the function: the mode, the layout or the planted finding.
    label: []const u8 = "",
    mode: Mode = .qwen,
    eps: f32 = 0,
    /// gemm: condition_proj's bias.
    bias: bool = false,
    /// attention: each row over earlier rows (the text model).
    causal: bool = false,
    /// heads, attention, glu: the refiner's fused rows (q, k, v of a head together; gate | value).
    fused: bool = false,
    /// attention: the non-finite plant is an infinite value at a row's last key, alone.
    last_key: bool = false,
    served: []const c.Shape,
    known: ?c.Known = null,
};

pub fn function(k: Kind) []const u8 {
    return switch (k) {
        .gemm => "hp_gemm",
        .norm => "hp_norm",
        .heads => "hp_heads",
        .attention => "hp_attention",
        .glu => "hp_glu",
        .add => "hp_add",
        .scatter => "hp_scatter_add",
    };
}

/// The text model's key heads for `heads` query heads: 64 over 8, so a group of 8 where it divides, else one.
pub fn kvHeads(f: Form, heads: usize) usize {
    if (f.fused) return heads;
    return if (heads % 8 == 0) heads / 8 else 1;
}

pub const nan16: u16 = 0x7fc0;
pub const inf16: u16 = 0x7f80;
pub const ninf16: u16 = 0xff80;

/// The rig and the host copies of one prepared call; `reset` (the harness's release) frees both.
pub const Bench = struct {
    rig: rig_.Rig,
    arena: std.heap.ArenaAllocator,

    pub fn outputBytes(b: *Bench) ![]const u8 {
        return b.rig.outputBytes();
    }
    pub fn inputBytes(b: *Bench) ![]const u8 {
        return b.rig.inputBytes();
    }
    pub fn intact(b: *Bench) !bool {
        return b.rig.intact();
    }
    pub fn overlap(b: *Bench) bool {
        return b.rig.overlap();
    }
    pub fn reset(b: *Bench) void {
        b.rig.reset();
        _ = b.arena.reset(.free_all);
    }
};

const Fresh = struct { ref: rig_.Ref, bytes: []const u8 };

pub const Case = struct {
    a: std.mem.Allocator,
    rig: Bench,
    pipe: mtl.Pipeline,
    form: Form,
    spec: c.Spec,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    stress: bool = false,
    /// Host bf16 inputs in binding order (gemm x w b; norm x w; heads q k wq wk; attention q k v; glu g v; x y).
    in: [4][]u16 = @splat(&.{}),
    /// hp_heads' cos and sin [rows, 64].
    tab: [2][]f32 = @splat(&.{}),
    idx: []u32 = &.{},
    refs: [8]rig_.Ref = undefined,
    /// In-place outputs, their starting bytes copied in before every launch.
    fresh: [2]?Fresh = @splat(null),
    /// Each output's range ends and allowance from the last reference (null: the spec's own bound).
    lo: []f64 = &.{},
    hi: []f64 = &.{},
    allow: ?[]f64 = null,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, pipe: mtl.Pipeline, f: Form, spec: c.Spec, control: ?g.Gate) Case {
        return .{ .a = a, .rig = .{ .rig = .{ .a = a, .device = device, .queue = queue }, .arena = std.heap.ArenaAllocator.init(a) }, .pipe = pipe, .form = f, .spec = spec, .control = control };
    }

    pub fn deinit(t: *Case) void {
        t.rig.rig.deinit();
        t.rig.arena.deinit();
    }

    pub fn kernel(t: *Case) g.Kernel {
        return .{ .spec = t.spec, .ctx = t, .vt = &vt, .control = t.control };
    }

    const vt = g.hooks(Case);

    pub fn alloc(t: *Case, comptime T: type, n: usize) ![]T {
        return t.rig.arena.allocator().alloc(T, n);
    }

    fn input(t: *Case, at: usize, data: anytype) !void {
        t.refs[at] = try t.rig.rig.input(std.mem.sliceAsBytes(data));
    }

    /// An output the kernel reads and writes, holding `data` before every launch.
    fn inPlace(t: *Case, at: usize, slot: usize, data: []const u16) !void {
        const bytes = std.mem.sliceAsBytes(data);
        t.refs[at] = try t.rig.rig.output(bytes.len);
        t.fresh[slot] = .{ .ref = t.refs[at], .bytes = bytes };
    }

    fn output(t: *Case, at: usize, len: usize) !void {
        t.refs[at] = try t.rig.rig.output(len);
    }

    pub fn prepare(t: *Case, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        t.rig.reset();
        t.shape = s;
        t.stress = fill == .stress;
        t.in = @splat(&.{});
        t.tab = @splat(&.{});
        t.idx = &.{};
        t.fresh = @splat(null);
        t.lo = &.{};
        t.hi = &.{};
        t.allow = null;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        return switch (t.form.kind) {
            .gemm => t.gemm(s, fill, rng),
            .norm => t.norm(s, fill, rng),
            .heads => t.heads(s, fill, rng),
            .attention => t.attention(s, fill, rng),
            .glu => t.glu(s, fill, rng),
            .add => t.add(s, fill, rng),
            .scatter => t.scatter(s, fill, rng),
        };
    }

    pub fn run(t: *Case) !void {
        for (t.fresh) |f| if (f) |x| @memcpy(t.rig.rig.bytes(x.ref, x.bytes.len), x.bytes);
        const p = t.rig.rig.begin();
        t.launch(p.e);
        try t.rig.rig.finish(p);
    }

    pub fn count(t: *Case) usize {
        const s = t.shape;
        return switch (t.form.kind) {
            .gemm, .glu, .add => s.rows * s.n,
            .norm => s.rows * s.k,
            .heads => s.rows * (if (t.form.fused) 3 * s.n else s.n + kvHeads(t.form, s.n)) * 128,
            .attention => s.k * s.rows * 128,
            .scatter => scatterRows(s.rows) * s.n,
        };
    }

    pub fn reference(t: *Case, w: []f64, mag: []f64) !void {
        return want.reference(t, w, mag);
    }

    pub fn bounds(t: *Case, _: []const f64, _: []const f64, allow: []f64) !void {
        if (t.allow) |x| @memcpy(allow, x);
    }

    pub fn intervals(t: *Case, low: []f64, high: []f64) !void {
        if (t.lo.len == low.len) {
            @memcpy(low, t.lo);
            @memcpy(high, t.hi);
            return;
        }
        @memset(low, -std.math.floatMax(f64));
        @memset(high, std.math.floatMax(f64));
    }

    pub fn simulate(t: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        return want.simulate(t, .{ .acc = acc, .order = order }, out);
    }

    fn gemm(t: *Case, s: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        const M = s.rows;
        const N = s.n;
        const K = s.k;
        if (M == 0 or N == 0 or K == 0 or N % 64 != 0 or K % 32 != 0) return false;
        const x = try t.alloc(u16, M * K);
        const w = try t.alloc(u16, N * K);
        const b = try t.alloc(u16, if (t.form.bias) N else 0);
        if (fill == .stress) {
            mirror(rng, x, w, M, N, K);
            @memset(b, 0);
        } else {
            uniform(rng, x, -1, 1);
            uniform(rng, w, -1, 1);
            uniform(rng, b, -0.5, 0.5);
        }
        if (fill == .nonfinite) {
            plant(x, K, M);
            w[(N - 1) * K + 5] = nan16;
            if (b.len > 7) b[7] = nan16;
        }
        t.in = .{ x, w, b, &.{} };
        try t.input(0, x);
        try t.input(1, w);
        if (t.form.bias) try t.input(2, b) else t.refs[2] = t.refs[1];
        try t.output(3, M * N * 2);
        return true;
    }

    fn norm(t: *Case, s: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        const W = s.k;
        if (s.rows == 0 or W == 0) return false;
        const x = try t.alloc(u16, s.rows * W);
        const w = try t.alloc(u16, W);
        if (fill == .stress) ints(rng, x, 32) else uniform(rng, x, -2, 2);
        uniform(rng, w, 0.25, 2);
        if (fill == .nonfinite) {
            plant(x, W, s.rows);
            w[W / 5] = nan16;
        }
        t.in = .{ x, w, &.{}, &.{} };
        try t.input(0, x);
        try t.input(1, w);
        try t.output(2, x.len * 2);
        return true;
    }

    fn heads(t: *Case, s: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        const T = s.rows;
        const H = s.n;
        if (T == 0 or H == 0 or s.k != 128) return false;
        const kv = kvHeads(t.form, H);
        const q = try t.alloc(u16, T * (if (t.form.fused) 3 * H else H) * 128);
        const k = try t.alloc(u16, if (t.form.fused) 0 else T * kv * 128);
        const wq = try t.alloc(u16, 128);
        const wk = try t.alloc(u16, 128);
        for ([_][]u16{ q, k }) |x| if (fill == .stress) ints(rng, x, 64) else uniform(rng, x, -2, 2);
        uniform(rng, wq, 0.25, 2);
        uniform(rng, wk, 0.25, 2);
        const rot = !t.form.fused;
        for (&t.tab) |*x| x.* = try t.alloc(f32, if (rot) T * 64 else 0);
        for (t.tab[0], t.tab[1]) |*cs, *sn| {
            const angle = rng.float(f64) * 2 * std.math.pi;
            cs.* = @floatCast(ref.bf(@cos(angle)));
            sn.* = @floatCast(ref.bf(@sin(angle)));
        }
        if (fill == .nonfinite and T >= 5) {
            // q: NaN, +Inf, -Inf, then both infinities in one head; k: a NaN; the rest of the q k v row as it was
            const qr = if (t.form.fused) 3 * H * 128 else H * 128;
            const qh: usize = if (t.form.fused) 384 else 128;
            const places = [_][3]usize{ .{ 0, 0, 3 }, .{ 1, 1 % H, 70 }, .{ 2, 0, 127 }, .{ 3, 0, 0 }, .{ 3, 0, 64 } };
            for (places, [_]u16{ nan16, inf16, ninf16, inf16, ninf16 }) |p, v| q[p[0] * qr + p[1] * qh + p[2]] = v;
            if (t.form.fused) {
                q[4 * qr + 128 + 9] = nan16;
                q[4 * qr + 256 + 1] = nan16;
            } else {
                k[4 * kv * 128 + 9] = nan16;
                t.tab[0][(T - 3) * 64 + 10] = std.math.nan(f32);
                t.tab[1][(T - 2) * 64 + 20] = std.math.inf(f32);
            }
        }
        t.in = .{ q, k, wq, wk };
        try t.inPlace(0, 0, q);
        if (t.form.fused) t.refs[1] = .{ .buf = t.refs[0].buf, .off = t.refs[0].off + 128 * 2 } else try t.inPlace(1, 1, k);
        try t.input(2, wq);
        try t.input(3, wk);
        if (rot) {
            try t.input(4, t.tab[0]);
            try t.input(5, t.tab[1]);
        } else {
            t.refs[4] = t.refs[2];
            t.refs[5] = t.refs[2];
        }
        return true;
    }

    fn attention(t: *Case, s: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        const H = s.rows;
        const T = s.k;
        if (H == 0 or T == 0 or s.n != 128) return false;
        const kv = kvHeads(t.form, H);
        const q = try t.alloc(u16, T * (if (t.form.fused) 3 * H else H) * 128);
        const k = try t.alloc(u16, if (t.form.fused) 0 else T * kv * 128);
        const v = try t.alloc(u16, if (t.form.fused) 0 else T * kv * 128);
        t.in = .{ q, k, v, &.{} };
        if (fill == .stress) try want.stressAttention(t, T, kv, rng) else for ([_][]u16{ q, k, v }) |x| uniform(rng, x, -1, 1);
        if (fill == .nonfinite) want.plantAttention(t, T, H, kv);
        try t.input(0, q);
        if (t.form.fused) {
            t.refs[1] = .{ .buf = t.refs[0].buf, .off = t.refs[0].off + 128 * 2 };
            t.refs[2] = .{ .buf = t.refs[0].buf, .off = t.refs[0].off + 256 * 2 };
        } else {
            try t.input(1, k);
            try t.input(2, v);
        }
        try t.output(3, T * H * 128 * 2);
        return true;
    }

    fn glu(t: *Case, s: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        const T = s.rows;
        const W = s.n;
        if (T == 0 or W == 0) return false;
        const gt = try t.alloc(u16, T * W * (if (t.form.fused) @as(usize, 2) else 1));
        const v = try t.alloc(u16, if (t.form.fused) 0 else T * W);
        const vs = if (t.form.fused) gt else v;
        const gr = if (t.form.fused) 2 * W else W;
        const vo = if (t.form.fused) W else 0;
        for (0..T) |r| {
            uniform(rng, gt[r * gr ..][0..W], -8, 8);
            uniform(rng, vs[r * gr + vo ..][0..W], -2, 2);
        }
        if (fill == .nonfinite and T >= 7 and W >= 4) {
            for ([_][2]usize{ .{ 0, W / 3 }, .{ 1, W / 2 }, .{ 2, W - 1 }, .{ 3, 0 }, .{ 3, W - 1 } }, [_]u16{ nan16, inf16, ninf16, inf16, ninf16 }) |p, x| gt[p[0] * gr + p[1]] = x;
            for ([_][2]usize{ .{ 4, 1 }, .{ 5, 2 }, .{ 6, 3 } }, [_]u16{ nan16, inf16, ninf16 }) |p, x| vs[p[0] * gr + vo + p[1]] = x;
        }
        t.in = .{ gt, v, &.{}, &.{} };
        try t.input(0, gt);
        if (t.form.fused) t.refs[1] = .{ .buf = t.refs[0].buf, .off = t.refs[0].off + W * 2 } else try t.input(1, v);
        try t.output(2, T * W * 2);
        return true;
    }

    fn add(t: *Case, s: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        const n = s.rows * s.n;
        if (n == 0) return false;
        const x = try t.alloc(u16, n);
        const y = try t.alloc(u16, n);
        uniform(rng, x, -4, 4);
        uniform(rng, y, -4, 4);
        if (fill == .nonfinite) {
            plant(x, s.n, s.rows);
            if (s.rows > 5) {
                y[4 * s.n + 1] = nan16;
                y[5 * s.n + 2] = inf16;
            }
        }
        t.in = .{ x, y, &.{}, &.{} };
        try t.inPlace(0, 0, x);
        try t.input(1, y);
        return true;
    }

    fn scatter(t: *Case, s: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        const P = s.rows;
        const W = s.n;
        if (P == 0 or W == 0) return false;
        const T = scatterRows(P);
        const x = try t.alloc(u16, T * W);
        const y = try t.alloc(u16, P * W);
        uniform(rng, x, -4, 4);
        uniform(rng, y, -4, 4);
        // the pads: P rising rows of T, after a label's rows and in runs, as a presentation places them
        const rows = try t.alloc(u32, T);
        for (rows, 0..) |*r, i| r.* = @intCast(i);
        rng.shuffle(u32, rows[3..]);
        t.idx = rows[3..][0..P];
        std.mem.sort(u32, t.idx, {}, std.sort.asc(u32));
        if (fill == .nonfinite) {
            plant(y, W, P);
            if (P > 5) x[@as(usize, t.idx[4]) * W + 1] = nan16;
            x[1 * W + 2] = inf16;
        }
        t.in = .{ x, y, &.{}, &.{} };
        try t.inPlace(0, 0, x);
        try t.input(1, y);
        try t.input(2, t.idx);
        return true;
    }

    fn bind(t: *Case, e: mtl.ComputeEncoder, at: usize) void {
        e.setBuffer(t.refs[at].buf, t.refs[at].off, at);
    }

    fn u(v: usize) u32 {
        return @intCast(v);
    }

    /// The dispatch of the prompt_kernels.zig function each kind mirrors: same buffers, constants, grid and groups.
    fn launch(t: *Case, e: mtl.ComputeEncoder) void {
        const s = t.shape;
        const f = t.form;
        e.setPipeline(t.pipe);
        switch (f.kind) {
            .gemm => {
                // Kernels.gemm
                for (0..4) |i| t.bind(e, i);
                e.setValue(extern struct { M: u32, N: u32, K: u32, bias: u32 }{ .M = u(s.rows), .N = u(s.n), .K = u(s.k), .bias = @intFromBool(f.bias) }, 4);
                e.dispatchGroups(mtl.Size.of(s.n / 64, (s.rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
            },
            .norm => {
                // Kernels.norm
                for (0..3) |i| t.bind(e, i);
                e.setValue(extern struct { width: u32, mode: u32, eps: f32 }{ .width = u(s.k), .mode = @backingInt(f.mode), .eps = f.eps }, 3);
                e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(256, 1, 1));
            },
            .heads => {
                // Kernels.heads
                const kv = kvHeads(f, s.n);
                const row: usize = if (f.fused) 3 * s.n * 128 else s.n * 128;
                for (0..6) |i| t.bind(e, i);
                e.setValue(extern struct { heads: u32, kv_heads: u32, q_row: u32, k_row: u32, head: u32, mode: u32, rotary: u32, eps: f32 }{
                    .heads = u(s.n),
                    .kv_heads = u(kv),
                    .q_row = u(row),
                    .k_row = u(if (f.fused) row else kv * 128),
                    .head = if (f.fused) 384 else 128,
                    .mode = @backingInt(f.mode),
                    .rotary = @intFromBool(!f.fused),
                    .eps = f.eps,
                }, 6);
                e.dispatchGroups(mtl.Size.of(s.rows, (s.n + kv + 7) / 8, 1), mtl.Size.of(256, 1, 1));
            },
            .attention => {
                // Kernels.attention
                const kv = kvHeads(f, s.rows);
                const row: usize = if (f.fused) 3 * s.rows * 128 else s.rows * 128;
                const kr: usize = if (f.fused) row else kv * 128;
                for (0..4) |i| t.bind(e, i);
                e.setValue(extern struct { rows: u32, heads: u32, kv_heads: u32, q_row: u32, k_row: u32, v_row: u32, head: u32, causal: u32, scale: f32 }{
                    .rows = u(s.k),
                    .heads = u(s.rows),
                    .kv_heads = u(kv),
                    .q_row = u(row),
                    .k_row = u(kr),
                    .v_row = u(kr),
                    .head = if (f.fused) 384 else 128,
                    .causal = @intFromBool(f.causal),
                    .scale = want.scale,
                }, 4);
                e.dispatchGroups(mtl.Size.of(s.k, (s.rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
            },
            .glu => {
                // Kernels.glu
                const r: usize = if (f.fused) 2 * s.n else s.n;
                for (0..3) |i| t.bind(e, i);
                e.setValue(extern struct { width: u32, g_row: u32, v_row: u32, mode: u32 }{ .width = u(s.n), .g_row = u(r), .v_row = u(r), .mode = @backingInt(f.mode) }, 3);
                e.dispatchThreads(mtl.Size.of(s.n, s.rows, 1), mtl.Size.of(256, 1, 1));
            },
            .add => {
                // Kernels.add
                for (0..2) |i| t.bind(e, i);
                e.setValue(u(s.rows * s.n), 2);
                e.dispatchThreads(mtl.Size.of(s.rows * s.n, 1, 1), mtl.Size.of(256, 1, 1));
            },
            .scatter => {
                // Kernels.scatterAdd
                for (0..3) |i| t.bind(e, i);
                e.setValue(u(s.n), 3);
                e.dispatchThreads(mtl.Size.of(s.n, s.rows, 1), mtl.Size.of(256, 1, 1));
            },
        }
    }
};

/// The presentation rows around `pads` image pads: labels, vision marks and the prompt's own text.
pub fn scatterRows(pads: usize) usize {
    return pads + 37;
}

pub fn uniform(rng: std.Random, out: []u16, lo: f32, hi: f32) void {
    for (out) |*v| v.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
}

/// Integers in [-bound, bound].
pub fn ints(rng: std.Random, out: []u16, bound: u32) void {
    for (out) |*v| v.* = c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 2 * bound + 1))) - @as(f32, @floatFromInt(bound)));
}

/// NaN, +Inf and -Inf in rows 0 to 2 of [rows, width], and both infinities in row 3.
pub fn plant(x: []u16, width: usize, rows: usize) void {
    const at = [_][2]usize{ .{ 0, width / 3 }, .{ 1, width / 2 }, .{ 2, width - 1 }, .{ 3, 0 }, .{ 3, width - 1 } };
    for (at, [_]u16{ nan16, inf16, ninf16, inf16, ninf16 }) |p, v| if (p[0] < rows) {
        x[p[0] * width + p[1]] = v;
    };
}

/// Gate 4's gemm rows: integers to 16, the second half cancelling the first but for 4 ones; every output is 4.
fn mirror(rng: std.Random, x: []u16, w: []u16, M: usize, N: usize, K: usize) void {
    // partials pass fp16's range and each 32nd-term chain 2,048, all below 2^24 in any order
    const h = K / 2;
    for (0..M) |m| for (0..h) |i| {
        const v: f32 = if (i < 4) 1 else @floatFromInt(1 + rng.uintLessThan(u32, 16));
        x[m * K + i] = c.toBf16(v);
        x[m * K + h + i] = c.toBf16(v);
    };
    for (0..N) |n| for (0..h) |i| {
        const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 16));
        w[n * K + i] = c.toBf16(v);
        w[n * K + h + i] = c.toBf16(if (i < 4) 1 - v else -v);
    };
}
