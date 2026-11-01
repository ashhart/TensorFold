//! Flash Next's sparse attention on the lanes kernel under the contract on HIP: rows against an fp64 reference.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const dims = 256;
const ratio = 4;
const parts = 16;
const stride = 512; // ids a row: at most 120 blocks and a tail of three tokens
const scale: f32 = 1.0 / 16.0;

/// Fp32 sums over a row's keys and the parts merge.
fn depth(s: c.Shape) usize {
    return s.k + 2 * parts;
}

const names = [2][]const u8{
    "hip tf_fn_sparse_lanes12_f16 and tf_fn_sparse_merge",
    "hip tf_fn_sparse_lanes12_bf16 and tf_fn_sparse_merge",
};
const rule = "a NaN a row did not select never reaches it; a selected value reaches its dim, a selected key the head";
const why = "the scalar kernel's fp32 sums in its order: a row's keys in 32-key tiles, a dim at a time";

pub fn spec(k: kit.Kind) c.Spec {
    return .{
        .name = names[@intFromBool(k == .bf16)],
        .accum = .f32,
        .depth = depth,
        .ops = 6, // exp, the running rescale, the parts merge and the final divide
        .out = .f32,
        .nan = .{ .stated = rule },
        .tiles = .{ .{ .size = 4, .min = 2 }, .{}, .{} },
        // a full slice of 12 heads a kv head, a tail slice of a rank's 6, and a deep cache
        .served = &.{
            .{ .rows = 64, .n = 24, .k = 1024 },
            .{ .rows = 17, .n = 6, .k = 1024 },
            .{ .rows = 16, .n = 12, .k = 4096 },
        },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = why, .exact = false },
    };
}

/// A slice of rows over a kv head's cache of the kind, in the format tf_fn_idx_select writes: row 0 dense.
pub const SparseLanes = struct {
    rig: rig_.Rig,
    l: hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    heads: usize = 0,
    group: usize = 0,
    q: []f32 = &.{},
    kc: []f64 = &.{},
    vc: []f64 = &.{},
    ids: []i32 = &.{},
    counts: []i32 = &.{},
    sparse: []i32 = &.{},
    ends: []i32 = &.{},
    dev: [9]u64 = @splat(0),
    out: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) SparseLanes {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l.*, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *SparseLanes) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *SparseLanes) g.Kernel {
        return .{ .spec = spec(self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(SparseLanes);

    fn rows(self: *const SparseLanes) usize {
        return self.shape.rows;
    }

    /// Causal ends march forward, neighbours share most blocks; rows 1..7 select five blocks, the rest 120.
    fn select(self: *SparseLanes, a: std.mem.Allocator) !void {
        const n = self.rows();
        const cap = self.shape.k;
        self.ids = try a.alloc(i32, n * stride);
        @memset(self.ids, 0);
        self.counts = try a.alloc(i32, n);
        self.sparse = try a.alloc(i32, n);
        self.ends = try a.alloc(i32, n);
        for (0..n) |r| {
            const end = cap - n * 12 + r * 12 + r % 4; // not a multiple of ratio on three rows in four
            self.ends[r] = @intCast(end);
            const nb = end / ratio;
            if (r == 0) {
                self.sparse[r] = 0;
                self.counts[r] = @intCast(end);
                continue;
            }
            self.sparse[r] = 1;
            const want = @min(if (r < 8) @as(usize, 5) else 120, nb - 8);
            const start = (r * 3) % (nb - want);
            var w: usize = 0;
            for (0..want * ratio) |i| {
                self.ids[r * stride + w] = @intCast(start * ratio + i);
                w += 1;
            }
            for (nb * ratio..end) |t| {
                self.ids[r * stride + w] = @intCast(t);
                w += 1;
            }
            self.counts[r] = @intCast(w);
        }
    }

    /// Random: values in [-1, 1) exact in the kind. Stress: q = 0 and integer values, so every weight is 1.
    pub fn prepare(self: *SparseLanes, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const group: usize = if (shape.n % 12 == 0) 12 else 6;
        if (shape.rows < 2 or shape.n % group != 0 or shape.k < 12 * shape.rows + 64 or shape.k > 20000) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        self.fill = fill;
        self.heads = shape.n;
        self.group = group;
        const kv = shape.n / group;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const stress = fill == .stress;
        self.q = try a.alloc(f32, shape.rows * shape.n * dims);
        for (self.q) |*x| x.* = if (stress) 0 else @floatCast(rng.float(f64) * 2 - 1);
        const cache = kv * shape.k * dims;
        self.kc, const kb = try kit.values(a, rng, self.k, cache, -1, 1);
        self.vc, const vb = try kit.values(a, rng, self.k, cache, -1, 1);
        if (stress) for (self.vc, 0..) |*x, i| {
            x.* = @floatFromInt(rng.intRangeAtMost(i32, -8, 8));
            kit.put(self.k, vb, i, x.*);
        };
        try self.select(a);
        if (fill == .nonfinite) try self.plant(kb, vb);
        const ids = try self.rig.input(std.mem.sliceAsBytes(self.ids));
        const cnt = try self.rig.input(std.mem.sliceAsBytes(self.counts));
        const sparse = try self.rig.input(std.mem.sliceAsBytes(self.sparse));
        const ends = try self.rig.input(std.mem.sliceAsBytes(self.ends));
        const po = try self.rig.scratch(shape.rows * shape.n * parts * dims * 4);
        const pm = try self.rig.scratch(shape.rows * shape.n * parts * 2 * 4);
        const dq = try self.rig.input(std.mem.sliceAsBytes(self.q));
        self.dev = .{ dq, try self.rig.input(kb), try self.rig.input(vb), ids, cnt, sparse, ends, po, pm };
        self.out = try self.rig.output(shape.rows * shape.n * dims * 4);
        return true;
    }

    /// NaN in kv head 0: a key in block 12, a value in block 16 and a value in row 1's tail, past row 0's end.
    fn plant(self: *SparseLanes, kb: []u8, vb: []u8) !void {
        const nan = std.math.nan(f64);
        const bad_key = 12 * ratio * dims;
        const val = 16 * ratio * dims + 1;
        kit.put(self.k, kb, bad_key, nan);
        kit.put(self.k, vb, val, nan);
        self.kc[bad_key] = nan;
        self.vc[val] = nan;
        const tail: usize = @intCast(@divTrunc(self.ends[1], ratio) * ratio);
        kit.put(self.k, vb, tail * dims + 5, nan);
        self.vc[tail * dims + 5] = nan;
    }

    pub fn run(self: *SparseLanes) !void {
        const s = self.shape;
        const d = self.dev;
        try self.l.fn_sparse_attention(
            @ptrFromInt(d[0]),
            @ptrFromInt(d[1]),
            @ptrFromInt(d[2]),
            @backingInt(self.k),
            @ptrFromInt(d[3]),
            stride,
            @ptrFromInt(d[4]),
            @ptrFromInt(d[5]),
            @ptrFromInt(d[7]),
            @ptrFromInt(d[8]),
            @ptrFromInt(self.out),
            @intCast(s.rows),
            @intCast(s.n),
            @intCast(s.n / self.group),
            dims,
            @intCast(s.k),
            scale,
            parts,
            self.rig.s.handle,
        );
        try self.rig.s.synchronize();
    }

    pub fn count(self: *SparseLanes) usize {
        return self.shape.rows * self.shape.n * dims;
    }

    /// The row's selected keys below its end: its listed tokens, or all of a dense row's.
    fn key(self: *const SparseLanes, r: usize, j: usize) ?usize {
        const t: usize = if (self.sparse[r] != 0) @intCast(self.ids[r * stride + j]) else j;
        return if (t < self.ends[r]) t else null;
    }

    /// fp64 attention over the selected keys only; a NaN score zeroes the head, as the merge does.
    pub fn reference(self: *SparseLanes, want: []f64, mag: []f64) !void {
        const a = self.arena.allocator();
        const w = try a.alloc(f64, @intCast(std.mem.max(i32, self.counts)));
        defer a.free(w);
        for (0..self.rows()) |r| for (0..self.heads) |h| {
            const kvh = h / self.group;
            const n: usize = @intCast(self.counts[r]);
            var top = -std.math.inf(f64);
            var nan = false;
            for (0..n) |j| {
                w[j] = std.math.nan(f64);
                const t = self.key(r, j) orelse continue;
                var s: f64 = 0;
                for (0..dims) |d| {
                    const kv: f64 = self.kc[(kvh * self.shape.k + t) * dims + d];
                    const p = @as(f64, self.q[(r * self.heads + h) * dims + d]) * scale * kv;
                    s += p;
                }
                w[j] = s;
                nan = nan or std.math.isNan(s);
                top = @max(top, s);
            }
            var total: f64 = 0;
            for (0..n) |j| {
                if (self.key(r, j) == null) {
                    w[j] = 0;
                    continue;
                }
                w[j] = if (nan) 0 else @exp(w[j] - top);
                total += w[j];
            }
            for (0..dims) |d| {
                var acc: f64 = 0;
                var m: f64 = 0;
                const i = (r * self.heads + h) * dims + d;
                if (nan) {
                    want[i] = 0;
                    mag[i] = 0;
                    continue;
                }
                for (0..n) |j| {
                    const t = self.key(r, j) orelse continue;
                    const v = self.vc[(kvh * self.shape.k + t) * dims + d];
                    acc += w[j] * v;
                    m += w[j] * @abs(v);
                }
                want[i] = acc / total;
                mag[i] = m / total;
            }
        };
    }

    /// Every sum is exact when every weight is 1 and the values are integers, in any order and either accumulator.
    pub fn simulate(self: *SparseLanes, _: c.Accum, _: g.Order, out: []f64) !void {
        const mag = try self.arena.allocator().alloc(f64, out.len);
        try self.reference(out, mag);
    }
};
