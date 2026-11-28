//! Full GLM-5.3's indexer on the GLM-5.3-Flash kernels under the contract: pools of 1 and 64 roped dims, with controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const index = @import("contract_glm_index.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows_ = core.contract.rows;
const sparse = glm.sparse;

const dk = 128;
const rope = 64; // full GLM-5.3's roped index dims
const theta = 8.0e6; // its rope_theta
const eps: f32 = 1e-6;
const k_stride = dk + 32; // the keys sit in a wider row, as the block passes them

/// A lane's chain of 4 products or adds, then 5 butterflies.
fn depth(_: c.Shape) usize {
    return 10;
}

fn spec(name: []const u8) c.Spec {
    return .{ .name = name, .accum = .f32, .depth = depth, .out = .bf16, .served = &.{ .{ .rows = 8, .n = dk, .k = dk }, .{ .rows = 64, .n = dk, .k = dk } } };
}

/// sparse._index_write: each row's key LayerNormed, its first 64 dims roped at its position, at row pos + r of ik.
pub const IndexWrite = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    name: []const u8,
    fault: index.Fault = .none,
    arena: ?std.heap.ArenaAllocator = null,
    rows: usize = 0,
    at: usize = 0,
    x: []const u16 = &.{},
    w: []const u16 = &.{},
    b: []const u16 = &.{},
    freqs: [32]f32 = @splat(0),
    extra: []f64 = &.{},
    dev: [7]u64 = undefined, // kr, gr, lnw, lnb, ik, ig, pos
    inv: u64 = 0,

    pub fn deinit(self: *IndexWrite) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(IndexWrite);

    pub fn kernel(self: *IndexWrite) g.Kernel {
        return .{ .spec = spec(self.name), .ctx = self, .vt = &vt, .control = if (self.fault != .none) .shapes else null };
    }

    /// Rows offset by 0, 16 or 128 over noise in [-1, 1); stress: 128 plus integer pairs, an exact rope (inv 0).
    pub fn prepare(self: *IndexWrite, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != dk or shape.n != dk or shape.rows == 0) return false;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const x = try a.alloc(u16, shape.rows * k_stride);
        const w = try a.alloc(u16, dk);
        const b = try a.alloc(u16, dk);
        const gr = try a.alloc(f32, shape.rows * dk);
        for (w) |*v| v.* = c.toBf16(rng.float(f32) * 3 - 0.5);
        for (b) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (gr) |*v| v.* = rng.float(f32) * 4 - 2;
        for (0..shape.rows) |r| {
            const row = x[r * k_stride ..][0..k_stride];
            if (fill == .stress) { // 128 + v, 128 - v: every sum and square exact in fp32, the mean 128
                for (0..k_stride / 2) |j| {
                    const v: f32 = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 129))) - 64);
                    row[2 * j] = c.toBf16(128 + v);
                    row[2 * j + 1] = c.toBf16(128 - v);
                }
                continue;
            }
            const offset: f32 = switch (r % 3) {
                0 => 0,
                1 => 16,
                else => 128,
            };
            for (row) |*v| v.* = c.toBf16(offset + rng.float(f32) * 2 - 1);
        }
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, usize, f32 }{ .{ 0, 3, std.math.nan(f32) }, .{ 1, 70, std.math.inf(f32) }, .{ 2, dk - 1, -std.math.inf(f32) } };
            for (planted) |p| if (p[0] < shape.rows) {
                x[p[0] * k_stride + p[1]] = c.toBf16(p[2]);
            };
        }
        self.rows = shape.rows;
        self.at = if (shape.rows <= 8) 70_000 else 300_000;
        self.x = x;
        self.w = w;
        self.b = b;
        self.freqs = if (fill == .stress) @splat(0) else cuda.mla.ropeFreqs(theta);
        self.extra = try a.alloc(f64, shape.rows * dk);
        const p: i32 = @intCast(self.at);
        const out = try self.rig.output(shape.rows * dk * 2);
        const ig = try self.rig.scratch(shape.rows * dk * 2);
        const back = self.at * dk * 2; // the kernel writes row pos + r: the buffers' start less pos rows
        self.dev = .{ try self.rig.input(std.mem.sliceAsBytes(x)), try self.rig.input(std.mem.sliceAsBytes(gr)), try self.rig.input(std.mem.sliceAsBytes(w)), try self.rig.input(std.mem.sliceAsBytes(b)), out - back, ig - back, try self.rig.input(std.mem.asBytes(&p)) };
        self.inv = try self.rig.input(std.mem.sliceAsBytes(&self.freqs));
        return true;
    }

    pub fn run(self: *IndexWrite) !void {
        var a: cuda.Args = .{};
        const d = self.dev;
        const roped: c_int = if (self.fault == .unroped) 0 else rope;
        inline for (.{ d[0], @as(i64, k_stride), d[1], d[2], d[3], d[4], d[5], d[6], eps, roped, self.inv }) |v| a.add(v);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.rows) }, .block = .{ .x = 32 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *IndexWrite) usize {
        return self.rows * dk;
    }

    /// Rows in float64 with each sum as `sum` gives it, then the pairs below 64 turned at pos + r.
    fn outputs(self: *IndexWrite, out: []f64, mag: ?[]f64, sum: anytype) !void {
        var v: [dk]f64 = undefined;
        var t: [dk]f64 = undefined;
        var y: [dk]f64 = undefined;
        var m: [dk]f64 = undefined;
        var e: [dk]f64 = undefined;
        for (0..self.rows) |r| {
            var abs: f64 = 0;
            for (&v, self.x[r * k_stride ..][0..dk]) |*z, h| {
                z.* = c.fromBf16(h);
                abs += @abs(z.*);
            }
            const mean = sum.of(&v) / dk;
            for (&t, v) |*z, x| z.* = (x - mean) * (x - mean);
            const sd = @sqrt(@as(f64, eps) + sum.of(&t) / dk);
            for (0..dk) |i| {
                const wi: f64 = c.fromBf16(self.w[i]);
                const bi: f64 = c.fromBf16(self.b[i]);
                const term = (v[i] - mean) / sd * wi;
                y[i] = term + bi;
                // the sums' error reaches an output through x - mean (the mean's) and sd (the squares')
                m[i] = (2 * @abs(v[i] - mean) + abs / dk) / sd * @abs(wi);
                // the divisions (2 ulp each), sqrt.approx, the eps add and the fma after the sums
                e[i] = 12 * 0x1p-24 * (@abs(term) + @abs(bi));
            }
            const pos: f32 = @floatFromInt(self.at + r);
            for (0..rope / 2) |i| {
                const angle: f64 = self.freqs[i] * pos; // the fp32 product, as the kernel forms it
                const co = @cos(angle);
                const si = @sin(angle);
                const y0 = y[2 * i];
                const y1 = y[2 * i + 1];
                y[2 * i] = y0 * co - y1 * si;
                y[2 * i + 1] = y1 * co + y0 * si;
                // an output takes both inputs' errors, then sincosf's 2 ulp, two products and the add
                const both = e[2 * i] + e[2 * i + 1] + 8 * 0x1p-24 * (@abs(y0) + @abs(y1));
                const ms = m[2 * i] + m[2 * i + 1];
                e[2 * i] = both;
                e[2 * i + 1] = both;
                m[2 * i] = ms;
                m[2 * i + 1] = ms;
            }
            @memcpy(out[r * dk ..][0..dk], &y);
            if (mag) |mg| @memcpy(mg[r * dk ..][0..dk], &m);
            for (self.extra[r * dk ..][0..dk], e) |*z, x| z.* = if (std.math.isFinite(x)) x else 0;
        }
    }

    const Exact = struct {
        fn of(_: Exact, t: []const f64) f64 {
            var s: f64 = 0;
            for (t) |x| s += x;
            return s;
        }
    };

    const Sim = struct {
        acc: c.Accum,
        order: g.Order,
        fn of(s: Sim, t: []const f64) f64 {
            return rows_.sumIn(s.acc, s.order, t);
        }
    };

    pub fn reference(self: *IndexWrite, want: []f64, mag: []f64) !void {
        try self.outputs(want, mag, Exact{});
        for (mag) |*x| if (!std.math.isFinite(x.*)) {
            x.* = 0;
        };
    }

    pub fn bounds(self: *IndexWrite, _: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, self.extra) |*l, x| l.* += x * (1 + 0x1p-7);
    }

    pub fn simulate(self: *IndexWrite, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.outputs(out, null, Sim{ .acc = acc, .order = order });
    }
};

/// Full GLM-5.3's cases and their controls, on the heap so the kernels' contexts keep their address.
pub const Full = struct {
    idx: cuda.Module,
    ops: cuda.Module,
    scores: index.Scores,
    fast: index.Scores,
    tokens: index.Tokens,
    write: IndexWrite,
    unroped: index.Scores,
    pool_of_4: index.Scores,
    late: index.Tokens,
    write_unroped: IndexWrite,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, fns: glm.glue.Kernels, kernels: *std.ArrayList(g.Kernel)) !*Full {
        const self = try a.create(Full);
        errdefer a.destroy(self);
        var idx = try cuda.Module.load(gpu.d, cuda.kernels.glm_index);
        errdefer idx.unload();
        var ops = try cuda.Module.load(gpu.d, cuda.kernels.glm_ops);
        errdefer ops.unload();
        const r: rig_.Rig = .{ .gpu = gpu, .s = s };
        const fast = try idx.function("tf_glm_index_scores");
        const toks = try ops.function("tf_glm_dsa_tokens");
        self.* = .{
            .idx = idx,
            .ops = ops,
            .scores = .{ .rig = r, .f = fns.scores, .pool = 1, .rope = rope },
            .fast = .{ .rig = r, .f = fns.scores, .fast = fast, .pool = 1, .rope = rope, .label = "cuda glm tf_glm_index_scores, pools of 1 and 64 roped query dims" },
            .tokens = .{ .rig = r, .f = toks, .pool = 1 },
            .write = .{ .rig = r, .f = fns.index_write, .name = "cuda glm tf_glm_index_write, 64 roped key dims" },
            .unroped = .{ .rig = r, .f = fns.scores, .pool = 1, .rope = rope, .fault = .unroped, .label = "control: GLM scores leave full GLM-5.3's queries unroped" },
            .pool_of_4 = .{ .rig = r, .f = fns.scores, .pool = 1, .rope = rope, .fault = .pool_of_4, .label = "control: GLM scores read full GLM-5.3's keys as pools of 4" },
            .late = .{ .rig = r, .f = toks, .pool = 1, .fault = .late, .label = "control: GLM token lists of pools of 1 one position late" },
            .write_unroped = .{ .rig = r, .f = fns.index_write, .name = "control: GLM index write leaves full GLM-5.3's keys unroped", .fault = .unroped },
        };
        inline for (.{ &self.scores, &self.fast, &self.tokens, &self.write, &self.unroped, &self.pool_of_4, &self.late, &self.write_unroped }) |x| try kernels.append(a, x.kernel());
        return self;
    }

    pub fn destroy(self: *Full, a: std.mem.Allocator) void {
        inline for (.{ &self.scores, &self.fast, &self.tokens, &self.write, &self.unroped, &self.pool_of_4, &self.late, &self.write_unroped }) |x| x.deinit();
        self.ops.unload();
        self.idx.unload();
        a.destroy(self);
    }
};
