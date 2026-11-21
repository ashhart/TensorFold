//! Router (Ops.route) vs float64 Inkling2TopKRouter: picks rank by score + bias, NaN highest, ties to the lower index.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const u = @import("inkling2_util.zig");
const math = @import("inkling2_math_norm.zig");
const rig_ = @import("rig.zig");
const c = u.c;
const g = u.g;
const ops = u.ops;

pub const Kind = enum { route, route_split };

const topk = 8;
const scale: f32 = 2.5;
/// bf16 words before and after each tensor (16 bytes); an fp32 tensor has half as many.
const lead = 8;

fn split(k: Kind) bool {
    return k == .route_split;
}

pub fn name(k: Kind) []const u8 {
    return switch (k) {
        .route => "metal inkling2 router",
        .route_split => "metal inkling2 router, rows split over two calls",
    };
}

fn noSums(_: c.Shape) usize {
    return 0;
}

const rule = "picks rank by score + bias, NaN above every number, ties to the lower expert index, in rank order; a NaN pick makes the weights NaN";

// rows, experts, hidden. A split needs two rows at least.
const served_shared = [_]c.Shape{
    .{ .rows = 17, .n = 128, .k = 2304 },
    .{ .rows = 128, .n = 128, .k = 2304 },
    .{ .rows = 6, .n = 96, .k = 128 },
    .{ .rows = 16, .n = 384, .k = 4608 },
    .{ .rows = 40, .n = 96, .k = 128 },
    .{ .rows = 33, .n = 384, .k = 1024 },
    .{ .rows = 17, .n = 100, .k = 128 },
    .{ .rows = 2, .n = 384, .k = 4608 },
    .{ .rows = 31, .n = 96, .k = 128 },
    .{ .rows = 15, .n = 96, .k = 128 },
    .{ .rows = 32, .n = 384, .k = 128 },
    .{ .rows = 5, .n = 512, .k = 128 },
    .{ .rows = 3, .n = 384, .k = 4608 },
};
const served_single = served_shared ++ [_]c.Shape{.{ .rows = 1, .n = 128, .k = 2304 }};

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    o: *const ops.Ops,
    kind: Kind,
    label: []const u8,
    /// rows, n = experts, k = hidden.
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    x: []u16 = &.{},
    w: []u16 = &.{},
    bias: []f32 = &.{},
    joined: []u8 = &.{},
    msg: [200]u8 = undefined,
    r_x: rig_.Ref = undefined,
    r_w: rig_.Ref = undefined,
    r_bias: rig_.Ref = undefined,
    out_ids: rig_.Ref = undefined,
    out_wt: rig_.Ref = undefined,
    alt_ids: rig_.Ref = undefined,
    alt_wt: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, o: *const ops.Ops, kind: Kind, label: []const u8) Probe {
        return .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .o = o, .kind = kind, .label = label };
    }

    pub fn deinit(p: *Probe) void {
        p.arena.deinit();
        p.rig.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe) g.Kernel {
        const served: []const c.Shape = if (split(p.kind)) &served_shared else &served_single;
        return .{ .ctx = p, .vt = &vt, .spec = .{
            .name = p.label,
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .nan = .{ .stated = rule },
            .replaces = if (split(p.kind)) "the same rows in one call" else null,
            .tiles = .{ .{ .size = 16 }, .{}, .{} },
            .served = served,
        } };
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const rows = shape.rows;
        const E = shape.n;
        const H = shape.k;
        if (rows == 0 or H == 0 or E < 12 or E > ops.max_experts or (split(p.kind) and rows < 2)) return false;
        const a = u.reset(&p.arena, &p.rig);
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const x = try a.alloc(u16, rows * H);
        u.fillDyadic(rng, x, 8, 8);
        const w = try a.alloc(u16, E * H);
        const denom: f32 = if (H > 1024) 64 else 16;
        u.fillDyadic(rng, w, 2, denom);
        const bias = try a.alloc(f32, E);
        for (bias) |*b| b.* = c.fromBf16(u.dyadic(rng, 64, 256));
        // Tiers: a tied pair on top, four distinct experts, then a tied triple of one lane cut by the top-k.
        var used: [ops.max_experts]bool = @splat(false);
        const pair = [2]usize{ 3, E - 5 };
        const tri = if (E > 65) [3]usize{ 1, 33, 65 } else [3]usize{ 1, 2, 4 };
        for (pair) |idx| used[idx] = true;
        for (tri) |idx| used[idx] = true;
        var mid: [topk - 4]usize = undefined;
        var found: usize = 0;
        var at: usize = 6;
        while (found < mid.len) {
            at = (at + 11) % E;
            if (used[at]) continue;
            used[at] = true;
            mid[found] = at;
            found += 1;
        }
        if (fill == .nonfinite and H >= 4) {
            // Columns where almost every expert is non-zero: an infinity there gives NaN logits only to the zeros.
            var zero: [3]usize = undefined;
            var nz: usize = 0;
            var idx: usize = 10;
            while (nz < 3 and idx < E) : (idx += 1) {
                if (used[idx]) continue;
                zero[nz] = idx;
                nz += 1;
            }
            const cols = [2]usize{ H / 2, H - 1 };
            for (0..E) |ex| for (cols, 0..) |col, ci| {
                var is_zero = false;
                for (zero[0..nz], 0..) |ze, zi| if (ze == ex and (ci == 0 or zi < 2)) {
                    is_zero = true;
                };
                const at_word = &w[ex * H + col];
                if (is_zero) {
                    at_word.* = c.toBf16(0);
                } else if (u.val(at_word.*) == 0) {
                    at_word.* = c.toBf16(1 / denom);
                }
            };
            const planted = [_][3]usize{ .{ 0, H / 3, u.nan_bits }, .{ 1, H / 2, u.pinf_bits }, .{ 2, H - 1, u.ninf_bits }, .{ 3, 0, u.pinf_bits }, .{ 3, H - 1, u.ninf_bits } };
            for (planted) |q| if (q[0] < rows) {
                x[q[0] * H + q[1]] = @intCast(q[2]);
            };
        }
        @memcpy(w[pair[1] * H ..][0..H], w[pair[0] * H ..][0..H]);
        for (tri[1..]) |idx| @memcpy(w[idx * H ..][0..H], w[tri[0] * H ..][0..H]);
        for (pair) |idx| bias[idx] = 8;
        for (mid) |idx| bias[idx] = 6;
        for (tri) |idx| bias[idx] = 4;
        p.x = x;
        p.w = w;
        p.bias = bias;
        const nan_w = try a.alloc(u16, lead + E * H + H + lead);
        @memset(nan_w, u.nan_bits);
        @memcpy(nan_w[lead..][0 .. E * H], w);
        const nan_b = try a.alloc(f32, lead / 2 + E + 8);
        @memset(nan_b, std.math.nan(f32));
        @memcpy(nan_b[lead / 2 ..][0..E], bias);
        const nan_x = try a.alloc(u16, lead + rows * H + lead);
        @memset(nan_x, u.nan_bits);
        @memcpy(nan_x[lead..][0 .. rows * H], x);
        p.r_x = try p.rig.input(std.mem.sliceAsBytes(nan_x));
        p.r_w = try p.rig.input(std.mem.sliceAsBytes(nan_w));
        p.r_bias = try p.rig.input(std.mem.sliceAsBytes(nan_b));
        p.out_ids = try p.rig.output(rows * topk * 4);
        p.out_wt = try p.rig.output(rows * topk * 4);
        p.alt_ids = try p.rig.scratch(rows * topk * 4);
        p.alt_wt = try p.rig.scratch(rows * topk * 4);
        p.joined = try a.alloc(u8, rows * topk * 8);
        return true;
    }

    fn piece(p: *Probe, e: mtl.ComputeEncoder, ids: rig_.Ref, wt: rig_.Ref, first: usize, n: usize) !void {
        const H = p.shape.k;
        try p.o.route(e, .{
            .x = .{ .buffer = p.r_x.buf, .offset = p.r_x.off + (lead + first * H) * 2 },
            .w = .{ .buffer = p.r_w.buf, .offset = p.r_w.off + lead * 2 },
            .bias = .{ .buffer = p.r_bias.buf, .offset = p.r_bias.off + lead * 2 },
            .ids = .{ .buffer = ids.buf, .offset = ids.off + first * topk * 4 },
            .weights = .{ .buffer = wt.buf, .offset = wt.off + first * topk * 4 },
            .rows = @intCast(n),
            .hidden = @intCast(H),
            .experts = @intCast(p.shape.n),
            .topk = topk,
            .scale = scale,
        });
    }

    fn compute(p: *Probe, ids: rig_.Ref, wt: rig_.Ref, cut: bool) !void {
        const rows = p.shape.rows;
        const pass = p.rig.begin();
        // 15 rows first when there are more than 15: the first call is a matvec, the second (and the whole) a tile.
        const m = if (!cut) rows else if (rows > 15) 15 else rows / 2;
        try p.piece(pass.e, ids, wt, 0, m);
        if (cut) try p.piece(pass.e, ids, wt, m, rows - m);
        try p.rig.finish(pass);
    }

    pub fn run(p: *Probe) !void {
        try p.compute(p.out_ids, p.out_wt, split(p.kind));
    }

    /// The whole rows in one call, as the ids' bytes then the weights' (the outputs' order).
    pub fn replaced(p: *Probe) ![]const u8 {
        try p.compute(p.alt_ids, p.alt_wt, false);
        const n = p.shape.rows * topk * 4;
        @memcpy(p.joined[0..n], p.rig.bytes(p.alt_ids, n));
        @memcpy(p.joined[n .. 2 * n], p.rig.bytes(p.alt_wt, n));
        return p.joined;
    }

    pub fn count(p: *Probe) usize {
        return 2 * p.shape.rows * topk;
    }

    /// The rule's picks (as ids) then their weights, in float64; the gates use `check`, which accepts near ties.
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const a = p.arena.allocator();
        const rows = p.shape.rows;
        const E = p.shape.n;
        const H = p.shape.k;
        const x = try u.decode(a, p.x);
        const w = try u.decode(a, p.w);
        const score = try a.alloc(f64, E);
        const biased = try a.alloc(f64, E);
        const order = try a.alloc(u32, E);
        var ids: [topk]u32 = undefined;
        var wt: [topk]f64 = undefined;
        for (0..rows) |r| {
            for (0..E) |e| {
                score[e] = math.sigmoid(math.dot(x[r * H ..][0..H], w[e * H ..][0..H]));
                biased[e] = score[e] + p.bias[e];
            }
            math.rank(biased, order, &ids);
            math.routeWeights(score, &ids, scale, &wt);
            for (0..topk) |j| {
                want[r * topk + j] = @floatFromInt(ids[j]);
                want[rows * topk + r * topk + j] = wt[j];
            }
        }
        for (mag, want) |*m, v| m.* = @abs(v);
    }

    fn tolerance(b: f64) f64 {
        return 0x1p-20 * (1 + @abs(b));
    }

    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const a = p.arena.allocator();
        const rows = p.shape.rows;
        const E = p.shape.n;
        const H = p.shape.k;
        const ids_b = p.rig.bytes(p.out_ids, rows * topk * 4);
        const wt_b = p.rig.bytes(p.out_wt, rows * topk * 4);
        const x = try u.decode(a, p.x);
        const w = try u.decode(a, p.w);
        const score = try a.alloc(f64, E);
        const biased = try a.alloc(f64, E);
        const taken = try a.alloc(bool, E);
        var picks: [topk]u32 = undefined;
        var want: [topk]f64 = undefined;
        for (0..rows) |r| {
            for (0..E) |e| {
                score[e] = math.sigmoid(math.dot(x[r * H ..][0..H], w[e * H ..][0..H]));
                biased[e] = score[e] + p.bias[e];
            }
            @memset(taken, false);
            for (0..topk) |j| {
                const got = std.mem.readInt(u32, ids_b[(r * topk + j) * 4 ..][0..4], .little);
                if (got >= E or taken[got]) return try std.fmt.bufPrint(&p.msg, "row {d} rank {d}: id {d} is out of range or picked twice", .{ r, j, got });
                // The rule's pick among the experts the kernel has not taken yet.
                var best: u32 = 0;
                var have = false;
                for (0..E) |e| {
                    if (taken[e]) continue;
                    if (!have or math.before(biased, @intCast(e), best)) {
                        best = @intCast(e);
                        have = true;
                    }
                }
                if (got != best) {
                    const bv = biased[best];
                    const gv = biased[got];
                    if (std.math.isNan(bv) or std.math.isNan(gv) or gv == bv or bv - gv > tolerance(bv))
                        return try std.fmt.bufPrint(&p.msg, "row {d} rank {d}: picked expert {d} (biased {e:.6}) where the rule picks {d} ({e:.6})", .{ r, j, got, gv, best, bv });
                }
                taken[got] = true;
                picks[j] = got;
            }
            math.routeWeights(score, &picks, scale, &want);
            for (0..topk) |j| {
                const got = c.read(.f32, wt_b, r * topk + j);
                if (c.classOf(got) != c.classOf(want[j]))
                    return try std.fmt.bufPrint(&p.msg, "row {d} rank {d}: weight {e:.6} where float64 gives {e:.6}", .{ r, j, got, want[j] });
                if (std.math.isFinite(want[j]) and @abs(got - want[j]) > 64 * 0x1p-24 * @abs(want[j]) + 1e-30)
                    return try std.fmt.bufPrint(&p.msg, "row {d} rank {d}: weight {e:.6} is past its bound of float64's {e:.6}", .{ r, j, got, want[j] });
            }
        }
        return null;
    }
};

/// One planted fault in the router source; its `gate` must go red on it.
pub const Mutation = struct { kind: Kind, gate: c.Gate, label: []const u8, from: []const u8, to: []const u8 };

pub const mutations = [_]Mutation{
    .{ .kind = .route, .gate = .shapes, .label = "control: inkling2 router, ties to the higher expert index", .from = "const uint winner = simd_min(best == top ? at : 0xFFFFFFFFu);", .to = "const uint winner = simd_max(best == top ? at : 0u);" },
    .{ .kind = .route, .gate = .shapes, .label = "control: inkling2 router, ties inside a lane to the higher expert index", .from = "if (key[t] > best) {", .to = "if (key[t] >= best) {" },
    .{ .kind = .route, .gate = .nonfinite, .label = "control: inkling2 router, NaN ranked lowest", .from = "if (isnan(v)) return 0xFFFFFFFFu;", .to = "if (isnan(v)) return 1u;" },
    .{ .kind = .route, .gate = .shapes, .label = "control: inkling2 router, the correction bias not added to the picking scores", .from = "key[t] = i2_order_key(score[t] + bias[e]);", .to = "key[t] = i2_order_key(score[t]);" },
    .{ .kind = .route, .gate = .shapes, .label = "control: inkling2 router, route scale not applied", .from = "(picked[r] / (sum + 1e-20f)) * a.scale", .to = "(picked[r] / (sum + 1e-20f))" },
};

pub fn controlSource(a: std.mem.Allocator, m: Mutation) ![]u8 {
    return u.mutate(a, sources.inkling2_ops, m.from, m.to);
}

pub fn controlKernel(p: *Probe, gate: c.Gate) g.Kernel {
    var k = p.kernel();
    k.control = gate;
    return k;
}
