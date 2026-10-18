//! qwen27's attention (attention_mpp, attention_prompt) under the contract on Metal: whole calls against float64.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;
const q = tf.qwen27;
const attn = q.attention;
const planning = q.attention_plan;
const src = q.runtime_sources;

const D = 256;
const u = 0x1p-24;
const scale = 0.0625;

/// The head counts the engine qualifies (Geometry.check): the 27B's 24 query heads over 4, the 35B-A3B's 16 over 2.
pub const Heads = enum { q27b, q35b };

pub fn geometry(h: Heads, shared: bool) planning.Geometry {
    return switch (h) {
        .q27b => .{ .heads = 24, .kv_heads = 4, .shared_prefix = shared },
        .q35b => .{ .heads = 16, .kv_heads = 2, .shared_prefix = shared },
    };
}

pub fn label(h: Heads) []const u8 {
    return if (h == .q27b) "27B" else "35B-A3B";
}

/// Which head counts a registered launch compiles: its label names the model.
pub fn named(name: []const u8) Heads {
    return if (std.mem.indexOf(u8, name, "35B") != null) .q35b else .q27b;
}

/// A planted control's one edit to the shipped source; `from` must occur once.
pub const Edit = struct { from: []const u8, to: []const u8 };

/// The source as decode_round.zig builds it (before M5: tail rewritten, no prompt kernel; `prompt` adds it for gate 5).
pub fn source(a: std.mem.Allocator, h: Heads, tensor: bool, prompt: bool, edit: ?Edit) ![]u8 {
    var buf: [64]u8 = undefined;
    const head = try geometry(h, tensor).defines(&buf);
    const body = if (tensor) try a.dupe(u8, src.glue ++ src.attention_mpp ++ src.attention_io ++ src.attention_prompt) else try q.simd_attention.rewrite(a, src.glue ++ src.attention_mpp ++ src.attention_io);
    defer a.free(body);
    const text = try std.mem.concat(a, u8, &.{ head, body, if (prompt and !tensor) src.attention_prompt else "" });
    const e = edit orelse return text;
    defer a.free(text);
    if (std.mem.count(u8, text, e.from) != 1) return error.ControlEdit;
    return std.mem.replaceOwned(u8, a, text, e.from, e.to);
}

/// decode: a round's trees and chains as decode_round.zig plans them here; prompt: a chain through the prompt kernel.
pub const Which = enum { decode, prompt };

/// Gate 3's planted controls: a row seeing a key past its own, a merge losing later tail chunks, a lost prompt edge.
pub const Control = enum { none, late_key, lost_chunk, lost_edge };

fn editOf(k: Control) ?Edit {
    return switch (k) {
        .none => null,
        .late_key => .{ .from = "  const int nmax = P + depth + 1;", .to = "  const int nmax = P + depth + 2;" },
        .lost_chunk => .{ .from = "for (int c = 0; c < NCBS; c++) {", .to = "for (int c = 0; c < min(NCBS, 1); c++) {" },
        .lost_edge => .{ .from = "  if (kt < kend) q27_prompt_step<true>", .to = "  if (false && kt < kend) q27_prompt_step<true>" },
    };
}

pub const Reg = struct { heads: Heads, which: Which, control: Control = .none };

/// The error is no single sum's, so `mag` carries the call's whole first-order budget in units of one fp32 rounding.
fn oneRounding(_: c.Shape) usize {
    return 1;
}

/// Shapes as rows of the round, query heads and the first stream's cached keys; past 16 rows, streams of 16.
pub fn served(comptime h: Heads, comptime w: Which) []const c.Shape {
    const n: usize = comptime geometry(h, false).heads;
    // even starts fork; 2 and 3 tiles at 64 and 65 take prefix_2 and _3 as trees and chains; 2046 and 510 split a tree
    const two: usize = 3;
    const three: usize = if (h == .q27b) 6 else 5;
    return comptime switch (w) {
        .decode => &.{ .{ .rows = 1, .n = n, .k = 2 }, .{ .rows = two, .n = n, .k = 64 }, .{ .rows = three, .n = n, .k = 64 }, .{ .rows = two, .n = n, .k = 65 }, .{ .rows = three, .n = n, .k = 65 }, .{ .rows = 16, .n = n, .k = 2046 }, .{ .rows = 16, .n = n, .k = 2047 }, .{ .rows = 128, .n = n, .k = 510 }, .{ .rows = 128, .n = n, .k = 511 }, .{ .rows = 1, .n = n, .k = 8191 } },
        .prompt => &.{ .{ .rows = 16, .n = n, .k = 0 }, .{ .rows = 128, .n = n, .k = 0 }, .{ .rows = 128, .n = n, .k = 1920 }, .{ .rows = 16, .n = n, .k = 8176 } },
    };
}

fn nameOf(comptime r: Reg) []const u8 {
    const what = comptime if (r.which == .decode) "decode rows" else "prompt rows";
    return switch (r.control) {
        .none => std.fmt.comptimePrint("metal qwen27 attention {s} {s}", .{ comptime label(r.heads), what }),
        .late_key => "control: qwen27 attention decode rows seeing one key past their own",
        .lost_chunk => "control: qwen27 attention merge dropping a round's tail chunks after the first",
        .lost_edge => "control: qwen27 attention prompt rows losing the bounded last step",
    };
}

/// fp32 scores and sums, half probabilities into P V, the gate's bf16 sigmoid; `mids`: bf16(o) and bf16(sigmoid).
pub fn spec(r: Reg) c.Spec {
    return switch (r.heads) {
        inline else => |h| switch (r.which) {
            inline else => |w| switch (r.control) {
                inline else => |k| .{ .name = nameOf(.{ .heads = h, .which = w, .control = k }), .accum = .f32, .depth = oneRounding, .ops = 8, .mids = 2, .out = .bf16, .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 512 } }, .served = served(h, w) },
            },
        },
    };
}

fn bf(x: u16) f64 {
    return c.fromBf16(x);
}

/// The engine's plan, scratch and binding for one round.
pub const State = struct {
    plan: attn.Plan,
    scratch: attn.Scratch,
    binding: attn.Binding,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, r: *const Round, gm: planning.Geometry, caches: []const attn.Cache) !State {
        var ws: [planning.max_streams]planning.Window = undefined;
        var plan = try attn.Plan.init(a, r.windows(&ws), gm);
        errdefer plan.deinit();
        const limits = attn.Limits{ .rows = @intCast(r.rows), .streams = @intCast(r.streams), .keys = @intCast(r.cap), .shared_prefix = gm.shared_prefix, .geometry = gm };
        var scratch = try attn.Scratch.init(device, limits);
        errdefer scratch.deinit();
        var binding = try attn.Binding.init(device, limits);
        errdefer binding.deinit();
        try scratch.upload(plan);
        try binding.upload(plan, caches);
        return .{ .plan = plan, .scratch = scratch, .binding = binding };
    }

    pub fn deinit(s: *State) void {
        s.binding.deinit();
        s.scratch.deinit();
        s.plan.deinit();
    }
};

/// A round's windows: up to 16 rows each (the prompt's one window), 97 keys apart; at even starts 3 rows or more fork.
pub const Round = struct {
    rows: usize = 0,
    streams: usize = 0,
    cap: usize = 0,
    counts: [planning.max_streams]usize = @splat(0),
    firsts: [planning.max_streams]usize = @splat(0),
    starts: [planning.max_streams]usize = @splat(0),
    parents: [planning.max_rows]i32 = @splat(-1),

    pub fn init(shape: c.Shape, one: bool, trees: bool, rng: std.Random) Round {
        var r = Round{ .rows = shape.rows, .streams = if (one) 1 else (shape.rows + 15) / 16 };
        var first: usize = 0;
        var end: usize = 0;
        for (0..r.streams) |s| {
            const n = if (one) shape.rows else @min(16, shape.rows - first);
            r.counts[s] = n;
            r.firsts[s] = first;
            r.starts[s] = shape.k + 97 * s;
            const tree = trees and r.starts[s] % 2 == 0 and n >= 3;
            for (0..n) |i| r.parents[first + i] = if (i == 0) -1 else if (tree and (i <= 2 or rng.uintLessThan(u32, 3) != 0)) @intCast(if (i <= 2) 0 else rng.uintLessThan(usize, i)) else @intCast(i - 1);
            first += n;
            end = @max(end, r.starts[s] + n);
        }
        r.cap = end + 64;
        return r;
    }

    pub fn windows(r: *const Round, ws: *[planning.max_streams]planning.Window) []const planning.Window {
        for (0..r.streams) |s| ws[s] = .{ .parents = r.parents[r.firsts[s]..][0..r.counts[s]], .start = @intCast(r.starts[s]), .capacity = @intCast(r.cap) };
        return ws[0..r.streams];
    }

    /// Element 0 of key head `hk`'s cache row `row` in stream `s`'s caches, `kv_heads` heads each.
    pub fn at(r: *const Round, kv_heads: usize, s: usize, hk: usize, row: usize) usize {
        return ((s * kv_heads + hk) * r.cap + row) * D;
    }
};

/// What gate 3's decode shapes would launch on tensor units, from the engine's plans alone (no M5 dispatch).
pub const Reach = struct {
    /// [chains, trees][tiles - 1]: the prefix launch Attention.treeRows picks, min(tiles, 4).
    prefix: [2][4]bool = @splat(@splat(false)),
    /// A tree whose own keys span two tail chunks, so the merge takes more than one.
    split_tree: bool = false,
};

pub fn reach(a: std.mem.Allocator, h: Heads) !Reach {
    var seen = Reach{};
    const list = try c.shapes(a, spec(.{ .heads = h, .which = .decode }));
    defer a.free(list);
    var prng = std.Random.DefaultPrng.init(1);
    for (list) |shape| {
        const r = Round.init(shape, false, true, prng.random());
        var ws: [planning.max_streams]planning.Window = undefined;
        var plan = try attn.Plan.init(a, r.windows(&ws), geometry(h, true));
        defer plan.deinit();
        if (plan.prefix_chunks > 0) seen.prefix[@intFromBool(!plan.fused)][@min(plan.tiles, 4) - 1] = true;
        for (0..r.streams) |s| {
            const ps = r.parents[r.firsts[s]..][0..r.counts[s]];
            const chain = for (ps, 0..) |p, i| {
                if (i > 0 and p != @as(i32, @intCast(i)) - 1) break false;
            } else true;
            if (!chain and plan.meta[planning.global_words + s * planning.stream_words + 5] > 1) seen.split_tree = true;
        }
    }
    return seen;
}

pub fn ref(r: rig_.Ref) q.projection.Ref {
    return .{ .buffer = r.buf, .offset = r.off };
}

pub const Attn = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    reg: Reg,
    op: attn.Attention,
    tensor: bool,
    st: ?State = null,
    rd: Round = .{},
    q_in: []u16 = &.{},
    gate_in: []u16 = &.{},
    k_in: []u16 = &.{},
    v_in: []u16 = &.{},
    kc: []u16 = &.{},
    vc: []u16 = &.{},
    refs: [5]rig_.Ref = undefined,
    caches: [planning.max_streams]attn.Cache = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, r: Reg) !Attn {
        const tensor = device.tensorUnits();
        if (!tensor) try q.simd_attention.check(device, queue);
        const text = try source(a, r.heads, tensor, false, editOf(r.control));
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var op = try attn.Attention.init(device, lib, tensor);
        op.geometry = geometry(r.heads, tensor);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .reg = r, .op = op, .tensor = tensor };
    }

    pub fn deinit(t: *Attn) void {
        t.release();
        t.op.deinit();
        t.rig.deinit();
    }

    fn release(t: *Attn) void {
        if (t.st) |*s| s.deinit();
        t.st = null;
        for ([_]*[]u16{ &t.q_in, &t.gate_in, &t.k_in, &t.v_in, &t.kc, &t.vc }) |p| {
            t.a.free(p.*);
            p.* = &.{};
        }
    }

    pub fn kernel(t: *Attn) g.Kernel {
        return .{ .spec = spec(t.reg), .ctx = t, .vt = &vt, .control = if (t.reg.control == .none) null else .shapes };
    }

    const vt = g.hooks(Attn);

    fn geo(t: *const Attn) planning.Geometry {
        return geometry(t.reg.heads, t.tensor);
    }

    fn at(t: *const Attn, s: usize, hk: usize, row: usize) usize {
        return t.rd.at(t.geo().kv_heads, s, hk, row);
    }

    pub fn prepare(t: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const gm = t.geo();
        const heads: usize = gm.heads;
        const hk: usize = gm.kv_heads;
        if (shape.n != heads or shape.rows == 0 or shape.rows > planning.max_rows) return false;
        if (t.reg.which == .prompt and !t.tensor) return false;
        t.release();
        t.rig.reset();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.rd = Round.init(shape, t.reg.which == .prompt, t.reg.which == .decode, rng);
        const r = &t.rd;
        t.q_in = try t.a.alloc(u16, r.rows * heads * D);
        t.gate_in = try t.a.alloc(u16, r.rows * heads * 2 * D);
        t.k_in = try t.a.alloc(u16, r.rows * hk * D);
        t.v_in = try t.a.alloc(u16, r.rows * hk * D);
        t.kc = try t.a.alloc(u16, r.streams * hk * r.cap * D);
        t.vc = try t.a.alloc(u16, t.kc.len);
        for ([_][]u16{ t.q_in, t.gate_in, t.kc, t.vc }) |xs| for (xs) |*x| {
            x.* = c.toBf16(rng.float(f32) * 4 - 2);
        };
        if (fill == .stress) t.stress(rng);
        if (fill == .nonfinite) t.plant();
        // the round's keys as they sit in the caches: append writes the same bits (io's append check proves where)
        for (0..r.streams) |s| for (0..r.counts[s]) |i| for (0..hk) |h| {
            const row = ((r.firsts[s] + i) * hk + h) * D;
            @memcpy(t.k_in[row..][0..D], t.kc[t.at(s, h, r.starts[s] + i)..][0..D]);
            @memcpy(t.v_in[row..][0..D], t.vc[t.at(s, h, r.starts[s] + i)..][0..D]);
        };
        for ([_][]u16{ t.q_in, t.k_in, t.v_in, t.gate_in }, 0..) |xs, i| t.refs[i] = try t.rig.input(std.mem.sliceAsBytes(xs));
        const per = hk * r.cap * D;
        for (0..r.streams) |s| {
            const keys = try t.rig.input(std.mem.sliceAsBytes(t.kc[s * per ..][0..per]));
            const values = try t.rig.input(std.mem.sliceAsBytes(t.vc[s * per ..][0..per]));
            t.caches[s] = .{ .keys = ref(keys), .values = ref(values), .capacity = @intCast(r.cap) };
        }
        t.refs[4] = try t.rig.output(r.rows * heads * D * 2);
        t.st = try State.init(t.a, t.rig.device, r, gm, t.caches[0..r.streams]);
        return true;
    }

    /// Gate 1: key 0 zero with +Inf in a value, -Inf in round row 1's value, NaN in row 2's key, a query and a gate.
    fn plant(t: *Attn) void {
        const gm = t.geo();
        const heads: usize = gm.heads;
        const r = &t.rd;
        const s0 = r.starts[0];
        @memset(t.kc[t.at(0, 0, 0)..][0..D], 0);
        t.vc[t.at(0, 0, 0) + 3] = c.toBf16(std.math.inf(f32));
        if (r.counts[0] > 1) t.vc[t.at(0, 0, s0 + 1) + 3] = c.toBf16(-std.math.inf(f32));
        if (r.counts[0] > 2) t.kc[t.at(0, gm.kv_heads - 1, s0 + 2) + 10] = c.toBf16(std.math.nan(f32));
        if (r.rows > 3) t.q_in[(3 * heads + 1) * D] = c.toBf16(std.math.nan(f32));
        if (r.rows > 4) t.gate_in[4 * heads * 2 * D + D + 7] = c.toBf16(std.math.nan(f32));
    }

    /// Gate 4: exact fp32 sums past fp16's range; key 0 scores 32 (value 1), the rest 0 (1,024); scores alone separate.
    fn stress(t: *Attn, rng: std.Random) void {
        // queries 32 and 33, keys 64 m (m in 33-60) mirrored in reverse: no bf16 order cancels, every sum stays < 2^24
        const gm = t.geo();
        for (0..t.rd.streams) |s| for (0..gm.kv_heads) |h| for (0..t.rd.cap) |row| {
            const k = t.kc[t.at(s, h, row)..][0..D];
            for (0..D / 2) |d| {
                const m: f32 = @floatFromInt(33 + rng.uintLessThan(u32, 28));
                k[d] = c.toBf16(64 * m);
                k[D - 1 - d] = c.toBf16(-64 * m + (if (row == 0 and d == 0) @as(f32, 16) else 0));
            }
            @memset(t.vc[t.at(s, h, row)..][0..D], c.toBf16(if (row == 0) 1 else 1024));
        };
        for (0..t.q_in.len / D) |i| for (t.q_in[i * D ..][0..D], 0..) |*x, d| {
            x.* = c.toBf16(@floatFromInt(32 + @min(d, D - 1 - d) % 2));
        };
    }

    pub fn run(t: *Attn) !void {
        const s = &t.st.?;
        const p = t.rig.begin();
        t.op.encode(p.e, &s.scratch, &s.binding, s.plan, ref(t.refs[0]), ref(t.refs[1]), ref(t.refs[2]), ref(t.refs[4]), t.caches[0..t.rd.streams], ref(t.refs[3]), t.reg.which == .prompt) catch |err| {
            p.e.end();
            p.pool.pop();
            return err;
        };
        try t.rig.finish(p);
    }

    pub fn count(t: *Attn) usize {
        return t.rd.rows * t.geo().heads * D;
    }

    /// Float64 gated attention over the cached keys and each row's ancestors; with `acc`, every sum summed in it.
    fn outputs(t: *Attn, acc: ?c.Accum, order: g.Order, want: []f64, mag: []f64) !void {
        const gm = t.geo();
        const heads: usize = gm.heads;
        const group: usize = gm.group();
        const r = &t.rd;
        const keys = try t.a.alloc(usize, r.cap);
        defer t.a.free(keys);
        const sc = try t.a.alloc(f64, r.cap);
        defer t.a.free(sc);
        const p = try t.a.alloc(f64, r.cap);
        defer t.a.free(p);
        const pv = try t.a.alloc(f64, r.cap);
        defer t.a.free(pv);
        var terms: [D]f64 = undefined;
        var o: [D]f64 = undefined;
        var b: [D]f64 = undefined;
        var f: [D]f64 = undefined;
        var dev: [D]f64 = undefined;
        var xs: [D]f64 = undefined;
        for (0..r.streams) |s| for (0..r.counts[s]) |i| {
            const node = r.firsts[s] + i;
            var chain: [planning.max_rows]usize = undefined;
            var depth: usize = 0;
            var x: i32 = @intCast(i);
            while (x >= 0) : (x = r.parents[r.firsts[s] + @as(usize, @intCast(x))]) {
                chain[depth] = @intCast(x);
                depth += 1;
            }
            const start = r.starts[s];
            for (0..start) |j| keys[j] = j;
            for (0..depth) |j| keys[start + j] = start + chain[depth - 1 - j];
            const n = start + depth;
            for (0..heads) |h| {
                const hk = h / group;
                const qv = t.q_in[(node * heads + h) * D ..][0..D];
                var top = -std.math.inf(f64);
                var eps: f64 = 0;
                for (0..n) |j| {
                    const kv = t.kc[t.at(s, hk, keys[j])..][0..D];
                    var dot: f64 = 0;
                    var abs: f64 = 0;
                    for (&terms, qv, kv) |*e, a, k| {
                        e.* = bf(a) * bf(k);
                        dot += e.*;
                        abs += @abs(e.*);
                    }
                    if (acc) |ac| dot = rows_.sumIn(ac, order, &terms);
                    sc[j] = scale * dot;
                    if (!std.math.isNan(sc[j])) top = @max(top, sc[j]);
                    eps = @max(eps, c.gamma(D + 1, u) * scale * abs + u * @abs(sc[j]));
                }
                var l: f64 = 0;
                for (0..n) |j| {
                    p[j] = @exp(sc[j] - top);
                    l += p[j];
                }
                @memset(&o, 0);
                @memset(&b, 0);
                @memset(&f, 0);
                for (0..n) |j| for (t.vc[t.at(s, hk, keys[j])..][0..D], 0..) |v16, d| {
                    const v = bf(v16);
                    o[d] += p[j] * v;
                    b[d] += p[j] * @abs(v);
                    f[d] += @abs(v);
                };
                if (acc) |ac| {
                    l = rows_.sumIn(ac, order, p[0..n]);
                    for (&o, 0..) |*e, d| {
                        for (pv[0..n], keys[0..n], p[0..n]) |*y, key, w| y.* = w * bf(t.vc[t.at(s, hk, key) + d]);
                        e.* = rows_.sumIn(ac, order, pv[0..n]);
                    }
                }
                for (&o) |*e| e.* /= l;
                @memset(&dev, 0);
                @memset(&xs, 0);
                for (0..n) |j| for (t.vc[t.at(s, hk, keys[j])..][0..D], 0..) |v16, d| {
                    const w = p[j] * @abs(bf(v16) - o[d]);
                    dev[d] += w;
                    xs[d] += w * (4 + 5 * @abs(sc[j] - top));
                };
                // scores' error 2 eps |v - o|, fast exp, fp32 sums of P V and of P, half P (and its floor), the divide
                const gn = c.gamma(@floatFromInt(2 * n + 64), u);
                for (0..D) |d| {
                    const e = (2 * eps * dev[d] + u * xs[d]) / l + gn * (b[d] / l + @abs(o[d])) + 0x1p-11 * b[d] / l + 0x1p-25 * f[d] / l + 4 * u * @abs(o[d]);
                    const sig = 1 / (1 + @exp(-bf(t.gate_in[(node * heads + h) * 2 * D + D + d])));
                    want[(node * heads + h) * D + d] = o[d] * sig;
                    mag[(node * heads + h) * D + d] = e * sig / c.gamma(1, u);
                }
            }
        };
    }

    pub fn reference(t: *Attn, want: []f64, mag: []f64) !void {
        return t.outputs(null, .sequential, want, mag);
    }

    pub fn simulate(t: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        return t.outputs(acc, order, out, mag);
    }
};
