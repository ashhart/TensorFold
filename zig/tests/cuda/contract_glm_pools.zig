//! GLM's prompt-chunk select under the contract on CUDA: tf_glm_index_pools, bit-equal to tf_glm_select_rows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const sparse = glm.sparse;

fn noSums(_: c.Shape) usize {
    return 0;
}

const rule = "tf_glm_select_rows' bytes, and the host's picks: order keys descending, -0 as +0 and any NaN above +inf " ++
    "(the GPU's s + 0 gives its one positive NaN), ties to the lower pool, the K picks in pool order";

/// A prompt block's rows over buckets that take the whole-row path (1k), the candidates (4k to 64k) and the fallback.
pub fn spec() c.Spec {
    const k = sparse.topk_pools;
    const served = &[_]c.Shape{ .{ .rows = 64, .n = 4096, .k = k }, .{ .rows = 16, .n = 65536, .k = k } };
    return .{ .name = "cuda glm select rows tf_glm_index_pools", .accum = .none, .depth = noSums, .out = .raw, .replaces = "tf_glm_select_rows", .nan = .{ .stated = rule }, .tiles = .{ .{}, .{ .size = 8192, .min = k }, .{} }, .served = served };
}

/// sparse._order_key as the GPU computes it: s + 0 turns -0 into +0 and every NaN into one positive NaN.
fn orderKey(s: f32) u32 {
    const b: i32 = if (std.math.isNan(s)) std.math.maxInt(i32) else @bitCast(s + 0.0);
    return @bitCast(b ^ ((b >> 31) | std.math.minInt(i32)));
}

fn before(keys: []const u32, x: u32, y: u32) bool {
    return keys[x] > keys[y] or (keys[x] == keys[y] and x < y);
}

/// Each row's K pools of an n-pool bucket (shape: rows x n x K); tf_glm_select_rows' bytes are its baseline.
pub const Pools = struct {
    rig: rig_.Rig,
    fast: cuda.Function,
    narrow: cuda.Function,
    shape: c.Shape = undefined,
    host: []f32 = &.{},
    at: usize = 0,
    in: u64 = 0,
    pos: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Pools) void {
        self.rig.gpu.gpa.free(self.host);
        self.rig.deinit();
    }

    pub fn kernel(self: *Pools) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Pools);

    fn visible(self: *const Pools, r: usize) usize {
        return (self.at + r + 1) / sparse.pool;
    }

    fn outBytes(self: *const Pools) usize {
        return self.shape.rows * self.shape.k * 8;
    }

    /// Quarter steps in [-512, 512) with -0 among the zeros, -inf past the visible pools; non-finite: rows 0-3 plants.
    pub fn prepare(self: *Pools, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != sparse.topk_pools or shape.n < shape.k or shape.rows == 0) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.host);
        self.host = &.{};
        self.shape = shape;
        // a bucket's three quarters visible; past an edge's n every pool is (a limit of n), at 512 fewer than K are
        self.at = if (std.math.isPowerOfTwo(shape.n)) 3 * shape.n else 4 * shape.n;
        self.host = try a.alloc(f32, shape.rows * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..shape.rows) |r| {
            const vis = self.visible(r);
            for (self.host[r * shape.n ..][0..shape.n], 0..) |*v, p| {
                const level: f32 = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 4096))) - 2048);
                v.* = if (p >= vis) -std.math.inf(f32) else if (level == 0 and rng.boolean()) -0.0 else level * 0.25;
            }
        }
        if (fill == .nonfinite) {
            const neg_nan: f32 = @bitCast(@as(u32, 0xffc00000));
            const put = [_]f32{ std.math.nan(f32), neg_nan, std.math.inf(f32), -std.math.inf(f32) };
            for (put, 0..) |v, r| if (r < shape.rows) {
                const step = @min(self.visible(r), shape.n) / 8;
                for (0..8) |e| self.host[r * shape.n + e * step + r] = v;
            };
        }
        const p: i32 = @intCast(self.at);
        self.in = try self.rig.input(std.mem.sliceAsBytes(self.host));
        self.pos = try self.rig.input(std.mem.asBytes(&p));
        self.out = try self.rig.output(self.outBytes());
        self.base = try self.rig.scratch(self.outBytes());
        return true;
    }

    pub fn run(self: *Pools) !void {
        const x = self.shape;
        const t = glm.index.tile(x.n);
        var a: cuda.Args = .{};
        inline for (.{ self.in, self.out, @as(i64, @intCast(x.n)), self.pos, @as(c_int, @intCast(x.k)), @as(c_int, @intCast(t.cap)), @as(c_int, @intCast(t.target)), @as(u64, 0) }) |v| a.add(v);
        try cuda.launch.launch(self.fast, .{ .grid = .{ .x = @intCast(x.rows) }, .block = .{ .x = glm.index.pool_threads } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    /// tf_glm_select_rows on the same scores, a 128-thread block a row as cuda_sparse launches it.
    pub fn replaced(self: *Pools) ![]const u8 {
        const x = self.shape;
        var a: cuda.Args = .{};
        inline for (.{ self.in, self.base, @as(i64, @intCast(x.n)), self.pos, @as(c_int, @intCast(x.k)) }) |v| a.add(v);
        try cuda.launch.launch(self.narrow, .{ .grid = .{ .x = @intCast(x.rows) }, .block = .{ .x = 128 } }, self.rig.s, &a);
        return self.rig.bytes(self.base, self.outBytes());
    }

    pub fn count(self: *Pools) usize {
        return self.outBytes();
    }

    pub fn reference(_: *Pools, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// tf_glm_select_rows' bytes, then every row's picks against the host's rule over its first max(visible, K) pools.
    pub fn check(self: *Pools, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const x = self.shape;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        if (std.mem.indexOfDiff(u8, got, try self.replaced())) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from tf_glm_select_rows'", .{ at, got.len });
        const keys = try a.alloc(u32, x.n);
        defer a.free(keys);
        const idx = try a.alloc(u32, x.n);
        defer a.free(idx);
        for (0..x.rows) |r| {
            const limit = @min(x.n, @max(self.visible(r), x.k));
            for (keys[0..limit], self.host[r * x.n ..][0..limit]) |*k, s| k.* = orderKey(s);
            for (idx[0..limit], 0..) |*v, p| v.* = @intCast(p);
            std.mem.sort(u32, idx[0..limit], keys[0..limit], before);
            const picks = idx[0..x.k];
            std.mem.sort(u32, picks, {}, std.sort.asc(u32));
            for (picks, 0..) |want, i| {
                const have = std.mem.readInt(i64, got[(r * x.k + i) * 8 ..][0..8], .little);
                if (have != want) return try std.fmt.bufPrint(&self.msg, "row {d} pick {d}: pool {d}, the host's rule gives {d}", .{ r, i, have, want });
            }
        }
        return null;
    }
};

/// The prompt indexer's two kernels, registered together so the runner keeps one entry for them.
pub const Fast = struct {
    mod: cuda.Module,
    scores: @import("contract_glm_index.zig").Scores,
    pools: Pools,

    /// On the heap, so the kernels' contexts keep their address; registered into `kernels`.
    pub fn create(a: std.mem.Allocator, gpu: @import("check.zig").Gpu, s: cuda.Stream, fns: glm.glue.Kernels, kernels: *std.ArrayList(g.Kernel)) !*Fast {
        const idx = @import("contract_glm_index.zig");
        const self = try a.create(Fast);
        errdefer a.destroy(self);
        var m = try cuda.Module.load(gpu.d, cuda.kernels.glm_index);
        errdefer m.unload();
        self.* = .{ .mod = m, .scores = idx.Scores{ .rig = .{ .gpu = gpu, .s = s }, .f = fns.scores, .fast = try m.function("tf_glm_index_scores") }, .pools = .{ .rig = .{ .gpu = gpu, .s = s }, .fast = try m.function("tf_glm_index_pools"), .narrow = fns.select_rows } };
        try kernels.append(a, self.scores.kernel());
        try kernels.append(a, self.pools.kernel());
        return self;
    }

    pub fn destroy(self: *Fast, a: std.mem.Allocator) void {
        self.pools.deinit();
        self.scores.deinit();
        self.mod.unload();
        a.destroy(self);
    }
};
