//! nemo_audio.cu under the contract (nemo-audio-contract): the audio encoders' kernels, with controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const na = @import("nemotron").nemo_audio.kernels;
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows_ = core.contract.rows;
const ar = core.attention_range;
const bfr = core.bf16_round;

const u: f64 = 0x1p-24;
const half_bf16: f64 = 0x1p-8;
const floor: f64 = 0x1p-133;
const eps: f32 = 1e-5;

pub const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

/// The ASR encoder's 600-row windows (10 heads of 128, RoPE on 64) and the Sortformer's (to 828 rows, 8 heads of 64).
const asr = .{ .heads = 10, .d = 128, .rot = 64 };
const sf = .{ .heads = 8, .d = 64, .rot = 64 };

pub fn val(h: u16) f64 {
    return bfr.value(h);
}

fn gamma(n: f64) f64 {
    return n * u / (1 - n * u);
}

pub fn randomBf16(rng: std.Random, out: []u16, lo: f32, hi: f32) void {
    for (out) |*v| v.* = bfr.bits(lo + rng.float(f32) * (hi - lo));
}

pub const planted = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };

// --- tf_na_qk_rope: q and k LayerNormed (one bf16 rounding, fp32 inside), then bf16(bf16(x c) + bf16(rot(x) s)) ----

pub const Rope = struct {
    rig: rig_.Rig,
    k: *const na.Kernels,
    name: []const u8,
    heads: usize,
    rot: usize,
    norm: bool,
    ctl: ?cuda.Function = null, // the one-pass variance control: tf_na_qk_rope's arguments
    control: ?g.Gate = null,
    arena: ?std.heap.ArenaAllocator = null,
    t: usize = 0,
    d: usize = 0,
    r: usize = 0,
    x: []u16 = &.{},
    w: [2][]u16 = .{ &.{}, &.{} },
    b: [2][]u16 = .{ &.{}, &.{} },
    cs: []u16 = &.{},
    sn: []u16 = &.{},
    /// Each output's allowance, from reference().
    allow: []f64 = &.{},
    dev: struct { qkv: u64 = 0, w: [2]u64 = .{ 0, 0 }, b: [2]u64 = .{ 0, 0 }, cs: u64 = 0, sn: u64 = 0 } = .{},

    fn depth(s: c.Shape) usize {
        return (s.k + 31) / 32 + 5;
    }

    pub fn spec(t: *const Rope) c.Spec {
        return .{
            .name = t.name,
            .accum = if (t.norm) .f32 else .none,
            .depth = if (t.norm) depth else noSums,
            .ops = 12,
            .mids = 3,
            .out = .bf16,
            .in_place = true,
            .tiles = .{ .{}, .{}, .{ .size = 32, .step = 2, .min = 2 } },
            .served = if (t.heads == asr.heads)
                &.{ .{ .rows = 600, .n = asr.heads * asr.d, .k = asr.d }, .{ .rows = 137, .n = asr.heads * asr.d, .k = asr.d } }
            else
                &.{ .{ .rows = 828, .n = sf.heads * sf.d, .k = sf.d }, .{ .rows = 268, .n = sf.heads * sf.d, .k = sf.d } },
            // the outputs round to bf16 at three points after the sums: the bound is RoPE's, not an exact sum's
            .sums = .{ .bf16 = 0, .exact = false, .why = "q and k round to bf16 after their LayerNorm, before RoPE: a bf16 sum's error shows only past that rounding" },
        };
    }

    pub fn deinit(t: *Rope) void {
        if (t.arena) |*a| a.deinit();
        t.rig.deinit();
    }

    const vt = g.hooks(Rope);
    pub fn kernel(t: *Rope) g.Kernel {
        return .{ .spec = t.spec(), .ctx = t, .vt = &vt, .control = t.control };
    }

    /// qkv rows offset by 0, 32 or 256 over noise; stress: pairs v, -v of integers to 256, whose squares overflow fp16.
    pub fn prepare(t: *Rope, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.k > na.max_dim or shape.k % 2 != 0) return false;
        t.rig.reset();
        if (t.arena) |*a| a.deinit();
        t.arena = std.heap.ArenaAllocator.init(t.rig.gpu.gpa);
        const a = t.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.t = shape.rows;
        t.d = shape.k;
        t.r = @min(t.rot, t.d) & ~@as(usize, 1);
        const d = t.d;
        const hd = t.heads * d;
        t.x = try a.alloc(u16, t.t * 3 * hd);
        for (0..t.t * 3 * t.heads) |v| {
            const row = t.x[v * d ..][0..d];
            if (fill == .stress) {
                for (0..d / 2) |j| {
                    const m: f32 = if (j == 0) 256 else @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 513))) - 256);
                    row[2 * j] = bfr.bits(m);
                    row[2 * j + 1] = bfr.bits(-m);
                }
                continue;
            }
            const offset: f32 = switch (v % 3) {
                0 => 0,
                1 => 32,
                else => 256,
            };
            for (row, 0..) |*y, i| y.* = bfr.bits(offset + (rng.float(f32) * 2 - 1) * (if (i % 16 == 5) @as(f32, 10) else 1));
        }
        if (fill == .nonfinite) for (planted, 0..) |p, i| {
            // a q, a k and a v head, each one value
            const at = ((i % t.t) * 3 + i) * hd + (i % t.heads) * d + (i * 7) % d;
            t.x[at] = bfr.bits(p);
        };
        for (0..2) |j| {
            t.w[j] = try a.alloc(u16, d);
            t.b[j] = try a.alloc(u16, d);
            randomBf16(rng, t.w[j], -0.5, 2.5);
            randomBf16(rng, t.b[j], -1, 1);
        }
        t.cs = try a.alloc(u16, t.t * t.r);
        t.sn = try a.alloc(u16, t.t * t.r);
        randomBf16(rng, t.cs, -1, 1);
        randomBf16(rng, t.sn, -1, 1);
        t.dev.qkv = try t.rig.output(t.x.len * 2);
        if (t.norm) for (0..2) |j| {
            t.dev.w[j] = try t.rig.input(std.mem.sliceAsBytes(t.w[j]));
            t.dev.b[j] = try t.rig.input(std.mem.sliceAsBytes(t.b[j]));
        };
        t.dev.cs = try t.rig.input(std.mem.sliceAsBytes(t.cs));
        t.dev.sn = try t.rig.input(std.mem.sliceAsBytes(t.sn));
        return true;
    }

    /// qkv goes in fresh each run (the launch writes it in place).
    pub fn run(t: *Rope) !void {
        const s = t.rig.s;
        const buf: cuda.DeviceBuffer = .{ .d = t.rig.gpu.d, .ptr = t.dev.qkv, .len = t.x.len * 2 };
        try buf.upload(0, std.mem.sliceAsBytes(t.x));
        const norm: ?[4]u64 = if (t.norm) .{ t.dev.w[0], t.dev.b[0], t.dev.w[1], t.dev.b[1] } else null;
        if (t.ctl) |f| {
            var args: cuda.Args = .{};
            args.add(t.dev.qkv);
            for (norm.?) |p| args.add(p);
            args.add(t.dev.cs);
            args.add(t.dev.sn);
            for ([_]usize{ t.heads, t.d, t.r }) |n| args.add(@as(c_int, @intCast(n)));
            args.add(eps);
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(t.t * t.heads * 2) }, .block = .{ .x = 32 } }, s, &args);
        } else try t.k.qkRope(s, t.dev.qkv, norm, t.dev.cs, t.dev.sn, t.t, t.heads, t.d, t.r, eps);
        try s.synchronize();
    }

    pub fn count(t: *Rope) usize {
        return t.x.len;
    }

    /// Every output in float64 with each sum as `sum` gives it; `allow` gets each one's LayerNorm and RoPE bound.
    fn outputs(t: *Rope, out: []f64, allow: ?[]f64, sum: anytype) !void {
        const a = t.arena.?.allocator();
        const d = t.d;
        const y = try a.alloc(f64, d);
        const e = try a.alloc(f64, d);
        const tmp = try a.alloc(f64, d);
        for (0..t.t) |row| for (0..3) |which| for (0..t.heads) |h| {
            const at = ((row * 3 + which) * t.heads + h) * d;
            const x = t.x[at..][0..d];
            if (which == 2) { // v: as it was
                for (x, 0..) |v, i| {
                    out[at + i] = val(v);
                    if (allow) |l| l[at + i] = 0;
                }
                continue;
            }
            if (t.norm) {
                var abs: f64 = 0;
                for (tmp, x) |*v, bits| {
                    v.* = val(bits);
                    abs += @abs(v.*);
                }
                const n: f64 = @floatFromInt(d);
                const mean = sum.of(tmp) / n;
                for (tmp, x) |*v, bits| v.* = (val(bits) - mean) * (val(bits) - mean);
                const rstd = 1 / @sqrt(sum.of(tmp) / n + eps);
                for (0..d) |i| {
                    const xv = val(x[i]);
                    const term = (xv - mean) * rstd * val(t.w[which][i]);
                    y[i] = term + val(t.b[which][i]);
                    const mag = (2 * @abs(xv - mean) + abs / n) * rstd * @abs(val(t.w[which][i]));
                    const fp = gamma(@floatFromInt(depth(.{ .rows = 0, .n = 0, .k = d }))) * mag + 12 * u * (@abs(term) + @abs(val(t.b[which][i])));
                    e[i] = fp + half_bf16 * (@abs(y[i]) + fp); // and its bf16 rounding
                }
            } else for (0..d) |i| {
                y[i] = val(x[i]);
                e[i] = 0;
            }
            const hr = t.r / 2;
            for (0..d) |i| {
                if (i >= t.r) {
                    out[at + i] = y[i];
                    if (allow) |l| l[at + i] = e[i] + floor;
                    continue;
                }
                const cv = val(t.cs[row * t.r + i]);
                const sv = val(t.sn[row * t.r + i]);
                const j = if (i < hr) i + hr else i - hr;
                const rv = if (i < hr) -y[j] else y[j];
                const p = y[i] * cv;
                const q = rv * sv;
                out[at + i] = p + q;
                if (allow) |l| {
                    const work = @abs(cv) * e[i] + @abs(sv) * e[j];
                    const rounded = work + half_bf16 * (@abs(p) + @abs(q) + work) * (1 + u);
                    l[at + i] = rounded + half_bf16 * (@abs(p + q) + rounded) + floor;
                }
            }
        };
    }

    const Exact = struct {
        fn of(_: Exact, v: []const f64) f64 {
            var s: f64 = 0;
            for (v) |x| s += x;
            return s;
        }
    };
    const Sim = struct {
        acc: c.Accum,
        order: g.Order,
        fn of(s: Sim, v: []const f64) f64 {
            return rows_.sumIn(s.acc, s.order, v);
        }
    };

    pub fn reference(t: *Rope, want: []f64, mag: []f64) !void {
        t.allow = try t.arena.?.allocator().alloc(f64, want.len);
        try t.outputs(want, t.allow, Exact{});
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn bounds(t: *Rope, _: []const f64, _: []const f64, allow: []f64) !void {
        @memcpy(allow, t.allow);
    }

    pub fn simulate(t: *Rope, acc: c.Accum, order: g.Order, out: []f64) !void {
        try t.outputs(out, null, Sim{ .acc = acc, .order = order });
    }
};

pub fn noSums(_: c.Shape) usize {
    return 0;
}

// --- tf_na_attention: softmax(q k^T scale) over keys [0, len) in fp32, P as bf16, P V summed in fp32 ---------------

pub const Attention = struct {
    rig: rig_.Rig,
    k: *const na.Kernels,
    name: []const u8,
    heads: usize,
    /// The control: one key short of the window.
    short: bool = false,
    control: ?g.Gate = null,
    arena: ?std.heap.ArenaAllocator = null,
    t: usize = 0,
    len: usize = 0,
    d: usize = 0,
    qkv: []u16 = &.{},
    half: []f64 = &.{},
    in: u64 = 0,
    out: u64 = 0,

    /// The P V chain over the keys, or a score's over d.
    fn depth(s: c.Shape) usize {
        return @max(s.n + 1, s.k + 1);
    }

    pub fn spec(t: *const Attention) c.Spec {
        const why = "P rounds to bf16 before P V: its 2^-8 of each term outweighs what a bf16 sum of the values loses";
        return .{
            .name = t.name,
            .accum = .f32,
            .depth = depth,
            .ops = 4,
            .mids = 1,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 128 }, .{} },
            .served = if (t.heads == asr.heads)
                &.{ .{ .rows = 600, .n = 600, .k = asr.d }, .{ .rows = 600, .n = 137, .k = asr.d } }
            else
                &.{ .{ .rows = 828, .n = 828, .k = sf.d }, .{ .rows = 828, .n = 791, .k = sf.d }, .{ .rows = 268, .n = 240, .k = sf.d } },
            .sums = .{ .bf16 = 0, .exact = false, .why = why },
        };
    }

    pub fn deinit(t: *Attention) void {
        if (t.arena) |*a| a.deinit();
        t.rig.deinit();
    }

    const vt = g.hooks(Attention);
    pub fn kernel(t: *Attention) g.Kernel {
        return .{ .spec = t.spec(), .ctx = t, .vt = &vt, .control = t.control };
    }

    /// qkv [t, 3, heads, d] in [-1, 1); stress: q zero (every weight equal), each head's values mirrored over the keys.
    pub fn prepare(t: *Attention, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        // rows: the window's (a gate's few queries still read every key), n: its valid keys
        const rows = @max(shape.rows, shape.n);
        if (shape.rows == 0 or shape.n == 0 or rows > na.max_keys or shape.k > na.max_dim) return false;
        t.rig.reset();
        if (t.arena) |*a| a.deinit();
        t.arena = std.heap.ArenaAllocator.init(t.rig.gpu.gpa);
        const a = t.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.t = rows;
        t.len = shape.n;
        t.d = shape.k;
        const hd = t.heads * t.d;
        t.qkv = try a.alloc(u16, t.t * 3 * hd);
        randomBf16(rng, t.qkv, -1, 1);
        if (fill == .stress) {
            for (0..t.t) |r| @memset(t.qkv[r * 3 * hd ..][0..hd], 0);
            const col = try a.alloc(f32, t.len);
            for (0..hd) |e| {
                mirrored(rng, col);
                for (col, 0..) |v, j| t.qkv[(j * 3 + 2) * hd + e] = bfr.bits(v);
            }
        }
        if (fill == .nonfinite) {
            // a key of head 0 (every query of that head), a value of head 1, a query of the last head
            t.qkv[(0 * 3 + 1) * hd + 3] = bfr.bits(planted[0]);
            if (t.heads > 1) t.qkv[(@min(t.len, t.t) / 2 * 3 + 2) * hd + t.d + 5] = bfr.bits(planted[1]);
            t.qkv[((t.t - 1) * 3 + 0) * hd + (t.heads - 1) * t.d] = bfr.bits(planted[2]);
        }
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.qkv));
        t.out = try t.rig.output(t.t * hd * 2);
        return true;
    }

    fn scale(t: *const Attention) f32 {
        return @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(t.d))));
    }

    pub fn run(t: *Attention) !void {
        try t.k.attend(t.rig.s, t.in, t.out, t.t, t.heads, t.d, if (t.short) t.len - 1 else t.len, t.scale());
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Attention) usize {
        return t.t * t.heads * t.d;
    }

    /// The kernel's arithmetic in attention_range's form: fma scores, expf, P = bf16(e / l), one fma chain for P V.
    fn form(t: *const Attention) ar.Form {
        const n: f64 = @floatFromInt(t.len);
        return .{ .score = @floatFromInt(t.d + 1), .fast_exp = false, .chunk_exp = false, .p_rel = half_bf16 + u, .num = n + 1, .den = @floor(n / 128) + 11 };
    }

    pub fn reference(t: *Attention, want: []f64, mag: []f64) !void {
        const a = t.arena.?.allocator();
        const d = t.d;
        const hd = t.heads * d;
        const n = t.len;
        t.half = try a.alloc(f64, want.len);
        const q = try a.alloc(f64, d);
        const kk = try a.alloc(f64, n * d);
        const vv = try a.alloc(f64, n * d);
        const w = try a.alloc(f64, n);
        const tau = try a.alloc(f64, n);
        for (0..t.heads) |h| {
            for (0..n) |j| for (0..d) |e| {
                kk[j * d + e] = val(t.qkv[(j * 3 + 1) * hd + h * d + e]);
                vv[j * d + e] = val(t.qkv[(j * 3 + 2) * hd + h * d + e]);
            };
            for (0..t.t) |r| {
                for (q, 0..) |*x, e| x.* = val(t.qkv[r * 3 * hd + h * d + e]);
                const at = (r * t.heads + h) * d;
                ar.head(q, kk, vv, n, d, t.scale(), t.form(), w, tau, want[at..][0..d], t.half[at..][0..d]);
            }
        }
        for (mag, want) |*m, x| m.* = @abs(x);
    }

    /// The range's half width, then the output's own rounding at its far end.
    pub fn bounds(t: *Attention, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, t.half) |*l, w, h| l.* = h + half_bf16 * (@abs(w) + h) + floor;
    }

    /// Gate 4: every weight is 1, so an output is its value column's sum over the keys, over their count.
    pub fn simulate(t: *Attention, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.arena.?.allocator();
        const hd = t.heads * t.d;
        const terms = try a.alloc(f64, t.len);
        for (0..hd) |e| {
            for (terms, 0..) |*x, j| x.* = val(t.qkv[(j * 3 + 2) * hd + e]);
            const o = rows_.sumIn(acc, order, terms) / @as(f64, @floatFromInt(t.len));
            for (0..t.t) |r| out[r * hd + e] = o;
        }
    }
};

/// Gate 4's values: odd integers, every other one times 64 (bf16-exact), the second half cancelling the first.
fn mirrored(rng: std.Random, out: []f32) void {
    const h = out.len / 2;
    for (0..h) |i| {
        const v: f32 = @floatFromInt((1 + 2 * rng.uintLessThan(u32, 128)) * @as(u32, if (i % 2 == 1) 64 else 1));
        out[i] = v;
        out[h + i] = (if (i < 8) @as(f32, 1) else 0) - v;
    }
    if (out.len % 2 == 1) out[out.len - 1] = 0;
}

// --- the Sortformer head's glue: im2col3, relu and zero_rows, each exact -------------------------------------------

pub const Glue = @import("contract_nemo_audio_glue.zig").Glue;

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: na.Kernels,
    ctl: cuda.Module,
    ropes: [3]Rope = undefined,
    attns: [3]Attention = undefined,
    glue: [6]Glue = undefined,

    /// The set, its kernels appended to `list` (each control after the kernel it breaks).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try na.Kernels.load(gpu.d);
        errdefer k.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k, .ctl = ctl };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const kp = &set.k;
        set.ropes = .{
            .{ .rig = rig, .k = kp, .name = "cuda nemo_audio tf_na_qk_rope, ASR encoder (q/k LayerNorm, RoPE on 64 of 128)", .heads = asr.heads, .rot = asr.rot, .norm = true },
            .{ .rig = rig, .k = kp, .name = "control: nemo_audio q/k LayerNorm with the variance in one pass (E[x^2] - mean^2)", .heads = asr.heads, .rot = asr.rot, .norm = true, .ctl = try ctl.function("tf_ctl_na_qk_onepass"), .control = .shapes },
            .{ .rig = rig, .k = kp, .name = "cuda nemo_audio tf_na_qk_rope, Sortformer encoder (RoPE on 64 of 64)", .heads = sf.heads, .rot = sf.rot, .norm = false },
        };
        set.attns = .{
            .{ .rig = rig, .k = kp, .name = "cuda nemo_audio tf_na_attention, ASR encoder (10 heads of 128)", .heads = asr.heads },
            .{ .rig = rig, .k = kp, .name = "control: nemo_audio tf_na_attention one key short of the window", .heads = asr.heads, .short = true, .control = .shapes },
            .{ .rig = rig, .k = kp, .name = "cuda nemo_audio tf_na_attention, Sortformer encoder (8 heads of 64)", .heads = sf.heads },
        };
        set.glue = .{
            .{ .rig = rig, .k = kp, .name = "cuda nemo_audio tf_na_im2col3 (the subpixel conv's taps)", .op = .im2col3 },
            .{ .rig = rig, .k = kp, .name = "control: nemo_audio tf_na_im2col3 reading a padding row", .op = .im2col3, .late = true, .control = .shapes },
            .{ .rig = rig, .k = kp, .name = "cuda nemo_audio tf_na_zero_rows (padding rows)", .op = .zero_rows },
            .{ .rig = rig, .k = kp, .name = "control: nemo_audio tf_na_zero_rows keeping a padding row", .op = .zero_rows, .late = true, .control = .shapes },
            .{ .rig = rig, .k = kp, .name = "cuda nemo_audio tf_na_relu", .op = .relu },
            .{ .rig = rig, .k = kp, .name = "control: nemo_audio relu by fmaxf, which drops a NaN", .op = .relu, .ctl = try ctl.function("tf_ctl_na_relu_fmax"), .control = .nonfinite },
        };
        for (&set.ropes) |*x| try list.append(a, x.kernel());
        for (&set.attns) |*x| try list.append(a, x.kernel());
        for (&set.glue) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.ropes) |*x| x.deinit();
        for (&set.attns) |*x| x.deinit();
        for (&set.glue) |*x| x.deinit();
        set.ctl.unload();
        set.k.unload();
        set.gpa.destroy(set);
    }
};

/// nemo_audio.cu's gates 1-4 and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x6b63, &t); // kernel-contract's seed: the same inputs as the full run
    std.debug.print("SUMMARY nemo_audio: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "nemo-audio-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
