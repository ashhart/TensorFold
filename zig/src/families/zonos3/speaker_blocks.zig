//! The speaker encoder's layers in fp32, each spread over the host's cores.
const std = @import("std");

const A = std.mem.Allocator;

pub const experts = 4;
const heads = 4;

/// The maps' memory, reused from layer to layer: fresh pages for every map would cost more than the layers do.
pub const Pool = struct {
    gpa: A,
    /// A request's small allocations, freed with the pool.
    arena: std.heap.ArenaAllocator,
    all: std.ArrayList([]f32) = .empty,
    idle: std.ArrayList([]f32) = .empty,

    pub fn init(gpa: A) Pool {
        return .{ .gpa = gpa, .arena = .init(gpa) };
    }

    pub fn deinit(p: *Pool) void {
        for (p.all.items) |buf| p.gpa.free(buf);
        p.all.deinit(p.gpa);
        p.idle.deinit(p.gpa);
        p.arena.deinit();
    }

    fn take(p: *Pool, n: usize) ![]f32 {
        var best: ?usize = null;
        for (p.idle.items, 0..) |buf, i| {
            if (buf.len >= n and (best == null or buf.len < p.idle.items[best.?].len)) best = i;
        }
        if (best) |i| return p.idle.swapRemove(i)[0..n];
        try p.all.ensureUnusedCapacity(p.gpa, 1);
        try p.idle.ensureTotalCapacity(p.gpa, p.all.items.len + 1);
        const buf = try p.gpa.alloc(f32, n);
        p.all.appendAssumeCapacity(buf);
        return buf;
    }

    /// A map no layer reads again: its memory is the next map's.
    pub fn give(p: *Pool, m: Map) void {
        for (p.all.items) |buf| if (buf.ptr == m.data.ptr) return p.idle.appendAssumeCapacity(buf);
    }
};

/// Feature maps [c][f][t]; a sequence is a map of one bin.
pub const Map = struct {
    c: usize,
    f: usize,
    t: usize,
    data: []f32,

    pub fn init(p: *Pool, c: usize, f: usize, t: usize) !Map {
        return .{ .c = c, .f = f, .t = t, .data = try p.take(c * f * t) };
    }

    pub fn row(m: Map, ci: usize, fi: usize) []f32 {
        return m.data[(ci * m.f + fi) * m.t ..][0..m.t];
    }
};

/// y = x * scale + shift, a channel each.
pub const Affine = struct { scale: []f32, shift: []f32 };
/// y [n] = W [n][k] x (+ b).
pub const Dense = struct { w: []f32, b: ?[]f32, n: usize, k: usize };

/// A check's view of the stages: rows [rows][width] under the reference dump's names.
pub const Taps = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, name: []const u8, values: []const f32, rows: usize, width: usize) anyerror!void,

    pub fn map(tp: Taps, name: []const u8, x: Map, a: *Pool) !void {
        const flat = try binMajor(a, x);
        defer a.give(flat);
        try tp.f(tp.ctx, name, flat.data, flat.c, flat.t);
    }
};

/// A layer's work split over the cores: `work(ctx, lo, hi)` once a share of [0, n).
fn spread(n: usize, ctx: anytype, comptime work: fn (@TypeOf(ctx), usize, usize) void) void {
    const cores = @min(16, std.Thread.getCpuCount() catch 1, @max(1, n));
    if (cores <= 1) return work(ctx, 0, n);
    var threads: [16]?std.Thread = @splat(null);
    const share = (n + cores - 1) / cores;
    for (0..cores) |i| {
        const lo = @min(n, i * share);
        const hi = @min(n, lo + share);
        if (i + 1 == cores or lo == hi) {
            if (lo < hi) work(ctx, lo, hi);
            continue;
        }
        threads[i] = std.Thread.spawn(.{}, work, .{ ctx, lo, hi }) catch null;
        if (threads[i] == null) work(ctx, lo, hi);
    }
    for (threads) |th| if (th) |t| t.join();
}

/// A 3 x 3 convolution, a zero row and column around the map: weights [cout][cin / groups][3][3], strides (f, t).
pub fn conv3(a: *Pool, x: Map, w: []const f32, bias: ?[]const f32, cout: usize, groups: usize, stride: [2]usize) !Map {
    const y = try Map.init(a, cout, (x.f - 1) / stride[0] + 1, (x.t - 1) / stride[1] + 1);
    const Ctx = struct {
        x: Map,
        y: Map,
        w: []const f32,
        bias: ?[]const f32,
        groups: usize,
        stride: [2]usize,

        fn run(c: @This(), lo: usize, hi: usize) void {
            const per = c.x.c / c.groups; // inputs a group
            const outs = c.y.c / c.groups;
            for (lo..hi) |co| {
                const first = (co / outs) * per;
                for (0..c.y.f) |fo| {
                    const out = c.y.row(co, fo);
                    @memset(out, if (c.bias) |b| b[co] else 0);
                    for (0..per) |g| for (0..3) |kf| {
                        const fi = fo * c.stride[0] + kf;
                        if (fi < 1 or fi > c.x.f) continue;
                        const in = c.x.row(first + g, fi - 1);
                        const taps = c.w[((co * per + g) * 3 + kf) * 3 ..][0..3];
                        const st = c.stride[1];
                        // frame to of the output reads frames to * st - 1 .. to * st + 1
                        const last = out.len - 1;
                        const inner_hi = if (last * st + 1 < in.len) out.len else last;
                        if (inner_hi < 2) {
                            // one frame: the taps around it alone
                        } else if (st == 1) {
                            for (out[1..inner_hi], in[0 .. inner_hi - 1], in[1..inner_hi], in[2 .. inner_hi + 1]) |*o, p, q, r| o.* += taps[0] * p + taps[1] * q + taps[2] * r;
                        } else for (1..inner_hi) |to| {
                            out[to] += taps[0] * in[to * st - 1] + taps[1] * in[to * st] + taps[2] * in[to * st + 1];
                        }
                        out[0] += taps[1] * in[0] + (if (in.len > 1) taps[2] * in[1] else 0);
                        if (inner_hi == last and last > 0) out[last] += taps[0] * in[last * st - 1] + taps[1] * in[last * st];
                    };
                }
            }
        }
    };
    spread(cout, Ctx{ .x = x, .y = y, .w = w, .bias = bias, .groups = groups, .stride = stride }, Ctx.run);
    return y;
}

/// Every stride-th bin and frame of a map: what a strided 1 x 1 convolution reads.
pub fn strided(a: *Pool, x: Map, stride: [2]usize) !Map {
    if (stride[0] == 1 and stride[1] == 1) return x;
    const y = try Map.init(a, x.c, (x.f - 1) / stride[0] + 1, (x.t - 1) / stride[1] + 1);
    for (0..y.c) |ci| for (0..y.f) |fi| {
        for (y.row(ci, fi), 0..) |*o, j| o.* = x.row(ci, fi * stride[0])[j * stride[1]];
    };
    return y;
}

/// A 1 x 1 convolution: every position's channels through `d`.
pub fn dense(a: *Pool, x: Map, d: Dense) !Map {
    return denseNorm(a, x, d, null, false);
}

/// As dense, each output channel then through `post`'s scale and shift and (`relu`) its positive part.
pub fn denseNorm(a: *Pool, x: Map, d: Dense, post: ?Affine, relu: bool) !Map {
    std.debug.assert(d.k == x.c);
    const y = try Map.init(a, d.n, x.f, x.t);
    // the weights by input channel, each output block's eight side by side
    const wt = try a.arena.allocator().alloc(f32, d.k * d.n);
    for (0..d.n) |co| for (0..d.k) |ci| {
        wt[ci * d.n + co] = d.w[co * d.k + ci];
    };
    const Ctx = struct {
        x: Map,
        y: Map,
        d: Dense,
        wt: []const f32,
        post: ?Affine,
        relu: bool,
        const lanes = 16;
        const V = @Vector(lanes, f32);

        fn run(c: @This(), lo: usize, hi: usize) void {
            const plane = c.x.f * c.x.t;
            const k = c.d.k;
            const n = c.d.n;
            // sixteen positions of every input channel, gathered: rows of one cache line
            const in = std.heap.c_allocator.alignedAlloc(f32, .@"64", k * lanes) catch return;
            defer std.heap.c_allocator.free(in);
            var p = lo * lanes;
            while (p < @min(plane, hi * lanes)) : (p += lanes) {
                const have = @min(lanes, plane - p);
                if (have < lanes) @memset(in, 0);
                for (0..k) |ci| @memcpy(in[ci * lanes ..][0..have], c.x.data[ci * plane + p ..][0..have]);
                var co: usize = 0;
                while (co < n) : (co += 8) {
                    const outs = @min(8, n - co);
                    var acc: [8]V = undefined;
                    for (acc[0..outs], 0..) |*v, j| v.* = @splat(if (c.d.b) |b| b[co + j] else 0);
                    for (0..k) |ci| {
                        const v: V = in[ci * lanes ..][0..lanes].*;
                        const w = c.wt[ci * n + co ..];
                        inline for (0..8) |j| {
                            if (j < outs) acc[j] = @mulAdd(V, @splat(w[j]), v, acc[j]);
                        }
                    }
                    for (acc[0..outs], 0..) |v, j| {
                        var out = v;
                        if (c.post) |norm| out = out * @as(V, @splat(norm.scale[co + j])) + @as(V, @splat(norm.shift[co + j]));
                        if (c.relu) out = @max(out, @as(V, @splat(0)));
                        const flat: [lanes]f32 = out;
                        @memcpy(c.y.data[(co + j) * plane + p ..][0..have], flat[0..have]);
                    }
                }
            }
        }
    };
    spread((x.f * x.t + Ctx.lanes - 1) / Ctx.lanes, Ctx{ .x = x, .y = y, .d = d, .wt = wt, .post = post, .relu = relu }, Ctx.run);
    return y;
}

/// In place: a channel's scale and shift, then (`relu`) the positive part.
pub fn affine(x: Map, n: Affine, relu: bool) void {
    const Ctx = struct {
        x: Map,
        n: Affine,
        relu: bool,

        fn run(c: @This(), lo: usize, hi: usize) void {
            const plane = c.x.f * c.x.t;
            for (lo..hi) |ci| for (c.x.data[ci * plane ..][0..plane]) |*v| {
                const y = v.* * c.n.scale[ci] + c.n.shift[ci];
                v.* = if (c.relu) @max(y, 0) else y;
            };
        }
    };
    spread(x.c, Ctx{ .x = x, .n = n, .relu = relu }, Ctx.run);
}

/// A map as its sequence [f * c][t], bin-major: row f * c + ci is channel ci of bin f.
pub fn binMajor(a: *Pool, x: Map) !Map {
    const y = try Map.init(a, x.f * x.c, 1, x.t);
    for (0..x.f) |fi| for (0..x.c) |ci| @memcpy(y.row(fi * x.c + ci, 0), x.row(ci, fi));
    return y;
}

/// The block's LayerNorm over the channels of each frame: (x - mean) / sqrt(var + 1e-6) * weight + bias.
pub fn channelNorm(x: Map, n: Affine) void {
    for (0..x.t) |j| {
        var mean: f32 = 0;
        for (0..x.c) |ci| mean += x.data[ci * x.t + j];
        mean /= @floatFromInt(x.c);
        var variance: f32 = 0;
        for (0..x.c) |ci| variance += (x.data[ci * x.t + j] - mean) * (x.data[ci * x.t + j] - mean);
        const inv = 1.0 / @sqrt(variance / @as(f32, @floatFromInt(x.c)) + 1e-6);
        for (0..x.c) |ci| x.data[ci * x.t + j] = n.scale[ci] * ((x.data[ci * x.t + j] - mean) * inv) + n.shift[ci];
    }
}

/// A depthwise convolution of `k` taps around each frame of a sequence, zeros outside it.
pub fn depthwise(a: *Pool, x: Map, w: []const f32, bias: []const f32, k: usize) !Map {
    const y = try Map.init(a, x.c, 1, x.t);
    const half = k / 2;
    for (0..x.c) |ci| {
        const in = x.row(ci, 0);
        const taps = w[ci * k ..][0..k];
        for (y.row(ci, 0), 0..) |*o, j| {
            var acc: f32 = bias[ci];
            const lo = if (j < half) half - j else 0;
            const hi = @min(k, in.len + half - j);
            for (taps[lo..hi], in[j + lo - half ..][0 .. hi - lo]) |tap, v| acc += tap * v;
            o.* = acc;
        }
    }
    return y;
}

extern "c" fn erff(x: f32) f32;

/// nn.GELU: x * Phi(x).
pub fn gelu(x: f32) f32 {
    return 0.5 * x * (1.0 + erff(x * std.math.sqrt1_2));
}

/// NewGELUActivation: the tanh form.
fn geluTanh(x: f32) f32 {
    return 0.5 * x * (1.0 + std.math.tanh(@sqrt(2.0 / std.math.pi) * (x + 0.044715 * (x * x * x))));
}

pub const Layer = struct { q: Dense, k: Dense, v: Dense, o: Dense, norm1: Affine, gate: Dense, experts: Dense, down: Dense, norm2: Affine };

fn apply(d: Dense, x: []const f32, y: []f32) void {
    for (y, 0..) |*o, n| {
        var acc: f32 = if (d.b) |b| b[n] else 0;
        for (d.w[n * d.k ..][0..d.k], x) |w, v| acc += w * v;
        o.* = acc;
    }
}

fn layerNorm(x: []f32, n: Affine) void {
    var mean: f32 = 0;
    for (x) |v| mean += v;
    mean /= @floatFromInt(x.len);
    var variance: f32 = 0;
    for (x) |v| variance += (v - mean) * (v - mean);
    const inv = 1.0 / @sqrt(variance / @as(f32, @floatFromInt(x.len)) + 1e-6);
    for (x, n.scale, n.shift) |*v, sc, sh| v.* = (v.* - mean) * inv * sc + sh;
}

/// TransformerEncoderLayer over a sequence's frames, in place: attention, a norm, the soft expert mixture, a norm.
pub fn transformer(a: *Pool, l: *const Layer, s: Map) !void {
    const d = s.c;
    const t = s.t;
    const hd = d / heads;
    // frames as rows
    const x = try a.arena.allocator().alloc(f32, t * d);
    for (0..d) |ci| for (s.row(ci, 0), 0..) |v, j| {
        x[j * d + ci] = v;
    };
    const q = try a.arena.allocator().alloc(f32, t * d);
    const k = try a.arena.allocator().alloc(f32, t * d);
    const v = try a.arena.allocator().alloc(f32, t * d);
    const scaling = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    for (0..t) |j| {
        apply(l.q, x[j * d ..][0..d], q[j * d ..][0..d]);
        for (q[j * d ..][0..d]) |*e| e.* *= scaling;
        apply(l.k, x[j * d ..][0..d], k[j * d ..][0..d]);
        apply(l.v, x[j * d ..][0..d], v[j * d ..][0..d]);
    }
    const Ctx = struct {
        l: *const Layer,
        x: []f32,
        q: []const f32,
        k: []const f32,
        v: []const f32,
        t: usize,
        d: usize,
        gpa: A,

        fn run(c: @This(), lo: usize, hi: usize) void {
            const dd = c.d;
            const h = dd / heads;
            const p = c.gpa.alloc(f32, c.t) catch return;
            defer c.gpa.free(p);
            var att: [256]f32 = undefined;
            var mid: [256]f32 = undefined;
            var wide: [experts * 256]f32 = undefined;
            for (lo..hi) |j| {
                for (0..heads) |head| {
                    const qh = c.q[j * dd + head * h ..][0..h];
                    var top: f32 = -std.math.inf(f32);
                    for (p, 0..) |*score, r| {
                        var acc: f32 = 0;
                        for (qh, c.k[r * dd + head * h ..][0..h]) |x1, x2| acc += x1 * x2;
                        score.* = acc;
                        top = @max(top, acc);
                    }
                    var sum: f32 = 0;
                    for (p) |*score| {
                        score.* = @exp(score.* - top);
                        sum += score.*;
                    }
                    const out = att[head * h ..][0..h];
                    @memset(out, 0);
                    for (p, 0..) |score, r| {
                        const weight = score / sum;
                        for (out, c.v[r * dd + head * h ..][0..h]) |*o, x2| o.* += weight * x2;
                    }
                }
                const row = c.x[j * dd ..][0..dd];
                apply(c.l.o, att[0..dd], mid[0..dd]);
                for (row, mid[0..dd]) |*o, m| o.* += m;
                layerNorm(row, c.l.norm1);
                // TokenSMoE: every expert's GELU output weighed by the gate's softmax, then the outer GELU
                var gate: [experts]f32 = undefined;
                apply(c.l.gate, row, &gate);
                var top: f32 = -std.math.inf(f32);
                for (gate) |g| top = @max(top, g);
                var sum: f32 = 0;
                for (&gate) |*g| {
                    g.* = @exp(g.* - top);
                    sum += g.*;
                }
                apply(c.l.experts, row, wide[0 .. experts * dd]);
                @memset(mid[0..dd], 0);
                for (0..experts) |e| for (mid[0..dd], wide[e * dd ..][0..dd]) |*o, m| {
                    o.* += geluTanh(m) * (gate[e] / sum);
                };
                for (mid[0..dd]) |*m| m.* = geluTanh(m.*);
                apply(c.l.down, mid[0..dd], att[0..dd]);
                for (row, att[0..dd]) |*o, m| o.* += m;
                layerNorm(row, c.l.norm2);
            }
        }
    };
    if (d > 256) return error.SpeakerLayerTooWide;
    // each row's attention reads the layer's input, so the rows are written to a copy
    const out = try a.arena.allocator().dupe(f32, x);
    spread(t, Ctx{ .l = l, .x = out, .q = q, .k = k, .v = v, .t = t, .d = d, .gpa = std.heap.c_allocator }, Ctx.run);
    for (0..d) |ci| for (s.row(ci, 0), 0..) |*o, j| {
        o.* = out[j * d + ci];
    };
}

test "conv3: a centre tap of one copies the map; the edges read zeros; a stride keeps every other frame" {
    const t = std.testing;
    var pool: Pool = .init(t.allocator);
    defer pool.deinit();
    const x = try Map.init(&pool, 2, 3, 5);
    for (x.data, 0..) |*v, i| v.* = @floatFromInt(i + 1);
    // two channels, depthwise: channel 0 copies, channel 1 sums the frame before and after
    var w: [2 * 9]f32 = @splat(0);
    w[4] = 1;
    w[9 + 3] = 1;
    w[9 + 5] = 1;
    const y = try conv3(&pool, x, &w, null, 2, 2, .{ 1, 1 });
    try t.expectEqualSlices(f32, x.row(0, 1), y.row(0, 1));
    try t.expectEqual(x.row(1, 2)[1], y.row(1, 2)[0]);
    try t.expectEqual(x.row(1, 2)[1] + x.row(1, 2)[3], y.row(1, 2)[2]);
    try t.expectEqual(x.row(1, 2)[3], y.row(1, 2)[4]);
    const half = try conv3(&pool, x, &w, null, 2, 2, .{ 2, 2 });
    try t.expectEqual(@as(usize, 2), half.f);
    try t.expectEqual(@as(usize, 3), half.t);
    try t.expectEqual(x.row(0, 2)[4], half.row(0, 1)[2]);
}

test "denseNorm: every position's channels through the matrix, the bias, the norm and the positive part" {
    const t = std.testing;
    var pool: Pool = .init(t.allocator);
    defer pool.deinit();
    const x = try Map.init(&pool, 3, 1, 37);
    for (x.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) - 5;
    var w = [_]f32{ 1, 2, 3, -1, 0, 1 };
    var b = [_]f32{ 0.5, -0.5 };
    var scale = [_]f32{ 2, 1 };
    var shift = [_]f32{ 0, 1 };
    const y = try denseNorm(&pool, x, .{ .w = &w, .b = &b, .n = 2, .k = 3 }, .{ .scale = &scale, .shift = &shift }, true);
    for (0..37) |j| {
        const in = [3]f32{ x.row(0, 0)[j], x.row(1, 0)[j], x.row(2, 0)[j] };
        try t.expectApproxEqAbs(@max((in[0] + 2 * in[1] + 3 * in[2] + 0.5) * 2, 0), y.row(0, 0)[j], 1e-5);
        try t.expectApproxEqAbs(@max((in[2] - in[0] - 0.5) + 1, 0), y.row(1, 0)[j], 1e-5);
    }
    // a map given back is the next map's memory
    pool.give(y);
    const again = try Map.init(&pool, 2, 1, 37);
    try t.expectEqual(y.data.ptr, again.data.ptr);
    try t.expectApproxEqAbs(@as(f32, 0.8413447), gelu(1.0), 1e-6);
}
