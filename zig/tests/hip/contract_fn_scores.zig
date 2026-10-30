//! Flash Next's index block scores under the contract on HIP: the quad form.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

/// Flash Next's index heads: 4 of 128 a row.
const heads = 4;
const dims = 128;
/// Rows past `top` complete blocks score them; every row here completes every block, so every score is written.
const top = 0;

pub const Which = enum { quad };

const relu_nan = "relu is fmaxf(x, 0): a NaN head dot scores 0 where float64 keeps the NaN";

/// A lane's 4 products and their chain, the wave's xor tree, then the heads' relus summed.
fn laneDepth(_: c.Shape) usize {
    return 1 + dims / 32 + 5 + heads;
}

const names = [1][2][]const u8{
    .{ "hip flashnext tf_fn_idx_scores_quad (fp16 keys)", "hip flashnext tf_fn_idx_scores_quad (bf16 keys)" },
};

/// A prompt slice's rows (16 and up).
const quad_served = [_]c.Shape{
    .{ .rows = 32, .n = 1100, .k = dims },
    .{ .rows = 64, .n = 300, .k = dims },
    .{ .rows = 16, .n = 2048, .k = dims },
};

/// rows x blocks scores of shape.k = 128 dims; the quad form must equal tf_fn_idx_scores bit for bit.
pub fn spec(w: Which, k: kit.Kind) c.Spec {
    var s: c.Spec = .{
        .name = names[@backingInt(w)][@intFromBool(k == .bf16)],
        .accum = .f32,
        .depth = laneDepth,
        .ops = 2,
        .out = .f32,
        .served = &quad_served,
        .known = .{ .gate = .nonfinite, .finding = relu_nan },
    };
    switch (w) {
        .quad => {
            s.replaces = "tf_fn_idx_scores";
            s.tiles = .{ .{ .size = 32 }, .{ .size = 64 }, .{} };
        },
    }
    return s;
}

/// q [rows][4][128] fp32 and pooled keys [blocks][128] of the activation kind.
pub const Scores = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    which: Which,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    q: []f64 = &.{},
    keys: []f64 = &.{},
    dq: u64 = 0,
    dk: u64 = 0,
    dc: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, w: Which) Scores {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = kit.act(), .which = w, .arena = .init(a) };
    }

    pub fn deinit(self: *Scores) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Scores) g.Kernel {
        return .{ .spec = spec(self.which, self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Scores);

    /// q in [-1, 1) fp32, keys in [-1, 1) of the kind; every row all blocks.
    pub fn prepare(self: *Scores, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != dims) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.q = try a.alloc(f64, shape.rows * heads * dims);
        self.keys = try a.alloc(f64, shape.n * dims);
        for (self.q) |*v| v.* = @as(f32, @floatCast(rng.float(f64) * 2 - 1));
        for (self.keys) |*v| v.* = kit.round(self.k, rng.float(f64) * 2 - 1);
        if (fill == .stress) self.stress(rng);
        if (fill == .nonfinite) self.plant();
        const kb = try a.alloc(u8, self.keys.len * kit.size(self.k));
        for (self.keys, 0..) |v, i| kit.put(self.k, kb, i, v);
        const complete = try a.alloc(i32, shape.rows);
        @memset(complete, @intCast(shape.n));
        self.dq = try self.rig.input(try kit.f32Bytes(a, self.q));
        self.dk = try self.rig.input(kb);
        self.dc = try self.rig.input(std.mem.sliceAsBytes(complete));
        self.out = try self.rig.output(shape.rows * shape.n * 4);
        self.base = try self.rig.scratch(shape.rows * shape.n * 4);
        return true;
    }

    /// Row 0 a NaN, row 1 a +Inf, row 2 both infinities (one head), a block's key -Inf.
    fn plant(self: *Scores) void {
        const s = self.shape;
        const inf = std.math.inf(f64);
        self.q[3] = std.math.nan(f64);
        if (s.rows > 1) self.q[(1 * heads + 1) * dims + 7] = inf;
        if (s.rows > 2) {
            self.q[(2 * heads + 2) * dims + 4] = inf;
            self.q[(2 * heads + 2) * dims + 5] = -inf;
        }
        self.keys[(s.n / 2) * dims + 9] = -inf;
    }

    /// Gate 4: integers exact in every kind and in bf16; q's second half undoes its first, keys one less for 8 dims.
    fn stress(self: *Scores, rng: std.Random) void {
        const h = dims / 2;
        for (0..self.shape.n) |b| for (0..h) |d| {
            const v: f64 = @floatFromInt(2 + rng.uintLessThan(u32, 127));
            self.keys[b * dims + d] = v;
            self.keys[b * dims + h + d] = v - @as(f64, if (d < 8) 1 else 0);
        };
        for (0..self.shape.rows * heads) |rh| for (0..h) |d| {
            const v: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 128));
            self.q[rh * dims + d] = v;
            self.q[rh * dims + h + d] = -v;
        };
    }

    pub fn run(self: *Scores) !void {
        try self.launch(self.which, self.out);
        try self.rig.s.synchronize();
    }

    /// One launch as idxScores makes it, the form chosen here rather than by the row count.
    fn launch(self: *Scores, w: ?Which, out: u64) !void {
        const s = self.shape;
        const n: c_int = @intCast(s.rows);
        const blocks: c_int = @intCast(s.n);
        const stride: i64 = @intCast(s.n);
        const kind = @backingInt(self.k);
        const t: c_int = top;
        const st = self.rig.s;
        const f = self.l.fnx;
        var a: hip.launch.Args = .{};
        const bx = (s.n + 63) / 64;
        const which = w orelse {
            // tf_fn_idx_scores: a wave a block, 8 rows a block
            inline for (.{ self.dq, self.dk, self.dc, out, kind, n, @as(c_int, heads), @as(c_int, dims), blocks, stride, t }) |v|
                try a.add(v);
            const grid: hip.launch.Dim3 = .{ .x = @intCast((s.n + 7) / 8), .y = @intCast((s.rows + 7) / 8) };
            return hip.launch.launch(f.idx_scores, .{ .grid = grid, .block = .{ .x = 256 } }, st, &a);
        };
        switch (which) {
            .quad => {
                inline for (.{ self.dq, self.dk, self.dc, out, kind, n, blocks, stride, t }) |v| try a.add(v);
                const grid: hip.launch.Dim3 = .{ .x = @intCast(bx), .y = @intCast((s.rows + 31) / 32) };
                try hip.launch.launch(f.idx_scores_quad, .{ .grid = grid, .block = .{ .x = 128 } }, st, &a);
            },
        }
    }

    /// tf_fn_idx_scores on the same inputs.
    pub fn replaced(self: *Scores) ![]const u8 {
        try self.launch(null, self.base);
        return self.rig.bytes(self.base, self.shape.rows * self.shape.n * 4);
    }

    pub fn count(self: *Scores) usize {
        return self.shape.rows * self.shape.n;
    }

    fn reluNan(x: f64) f64 {
        return if (std.math.isNan(x)) x else @max(x, 0);
    }

    /// Row r's score of block b: the heads' relu(q . key) summed over sqrt(128), in float64; mag its terms' sum.
    pub fn reference(self: *Scores, want: []f64, mag: []f64) !void {
        const s = self.shape;
        const norm = @sqrt(@as(f64, dims));
        for (0..s.rows) |r| for (0..s.n) |b| {
            var total: f64 = 0;
            var abs: f64 = 0;
            for (0..heads) |h| {
                var dot: f64 = 0;
                for (0..dims) |d| {
                    const qv = self.q[(r * heads + h) * dims + d];
                    const kv = self.keys[b * dims + d];
                    dot += qv * kv;
                    abs += @abs(qv * kv);
                }
                total += reluNan(dot);
            }
            want[r * s.n + b] = total / norm;
            mag[r * s.n + b] = abs / norm;
        };
    }

    /// Gate 4's scores: each head's dot in the accumulator and order, its relu, the heads' sum, over sqrt(128).
    pub fn simulate(self: *Scores, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = self.shape;
        const norm = @sqrt(@as(f64, dims));
        var t: [dims]f64 = undefined;
        var per: [heads]f64 = undefined;
        for (0..s.rows) |r| for (0..s.n) |b| {
            for (&per, 0..) |*p, h| {
                for (&t, 0..) |*x, d| x.* = self.q[(r * heads + h) * dims + d] * self.keys[b * dims + d];
                p.* = reluNan(rows.sumIn(acc, order, &t));
            }
            out[r * s.n + b] = rows.sumIn(acc, .sequential, &per) / norm;
        };
    }
};
