//! GLM's KV page kernels under the contract on CUDA: the latent write and the pooled keys onto pages out of order.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

/// Rows a page holds, a latent row's elements, an index key's, and the pools a page holds.
pub const rows_a_page = 512;
pub const lw = 512;
pub const dk = 128;
const pools_a_page = rows_a_page / 4;

/// A cache's part of a page in bytes: its 512 latent rows, then their 128 pooled keys.
pub const part_bytes = rows_a_page * lw * 2 + pools_a_page * dk * 2;
/// The tests' pages hold two caches' parts and test the second, so a cache's rows sit past its page's start.
pub const page_bytes = 2 * part_bytes;

/// Page ids for `n` pages: `n` of 0 .. n + 2 in shuffled order, so the table is out of order and skips some.
pub fn shuffled(a: std.mem.Allocator, rng: std.Random, n: usize) ![]i32 {
    const all = try a.alloc(i32, n + 3);
    defer a.free(all);
    for (all, 0..) |*v, i| v.* = @intCast(i);
    rng.shuffle(i32, all);
    return a.dupe(i32, all[0..n]);
}

/// The first row a call writes: a decode window across a page edge, a prompt chunk across four.
pub fn firstRow(rows: usize) usize {
    return if (rows <= 64) 2 * rows_a_page - rows / 2 else 300;
}

/// `len` bytes from device `src` to device `dst`, queued on the rig's stream.
fn copy(r: *rig_.Rig, dst: u64, src: u64, len: usize) !void {
    try (cuda.DeviceBuffer{ .d = r.gpu.d, .ptr = dst, .len = len }).copyFrom(0, src, len, r.s.handle);
}

fn noSums(_: c.Shape) usize {
    return 0;
}

/// A pool's four products summed in order.
fn fourTerms(_: c.Shape) usize {
    return 4;
}

/// A decode window's rows and a prompt chunk's: shape rows x the row's elements (k: the pool's rows for the keys).
pub fn spec(comptime which: enum { lat_write, pool_keys }) c.Spec {
    return switch (which) {
        .lat_write => .{ .name = "cuda glm tf_glm_lat_write on pages out of order", .accum = .none, .depth = noSums, .out = .bf16, .served = &.{ .{ .rows = 8, .n = lw, .k = 1 }, .{ .rows = 2048, .n = lw, .k = 1 } } },
        .pool_keys => .{ .name = "cuda glm tf_glm_pool_keys on pages out of order", .accum = .f32, .depth = fourTerms, .out = .bf16, .served = &.{ .{ .rows = 8, .n = dk, .k = 4 }, .{ .rows = 2048, .n = dk, .k = 4 } }, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "a pool's four bf16 products of softmax weights: no input makes the sum exact in fp32" } },
    };
}

/// Gate 1's values in `x` [rows][w]: NaN, +inf and -inf in the first rows, both infinities in row 3.
fn plant(x: []u16, rows: usize, w: usize) void {
    const at = [_][2]usize{ .{ 0, w / 3 }, .{ 1, w / 2 }, .{ 2, w - 1 }, .{ 3, 0 }, .{ 3, w - 1 } };
    const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
    for (at, put) |p, v| if (p[0] < rows) {
        x[p[0] * w + p[1]] = c.toBf16(v);
    };
}

/// latent._lat_write: `rows` latent rows from row `at` on through an out-of-order table, gathered back in order.
pub const LatWrite = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    shape: c.Shape = undefined,
    lat: []u16 = &.{},
    ids: []i32 = &.{},
    at: usize = 0,
    in: u64 = 0,
    pos: u64 = 0,
    pt: u64 = 0,
    arena: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *LatWrite) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *LatWrite) void {
        const a = self.rig.gpu.gpa;
        a.free(self.lat);
        a.free(self.ids);
        self.lat = &.{};
        self.ids = &.{};
    }

    pub fn kernel(self: *LatWrite) g.Kernel {
        return .{ .spec = spec(.lat_write), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(LatWrite);

    pub fn prepare(self: *LatWrite, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != lw or shape.rows == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.at = firstRow(shape.rows);
        self.lat = try a.alloc(u16, shape.rows * lw);
        for (self.lat) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        if (fill == .nonfinite) plant(self.lat, shape.rows, lw);
        const pages = (self.at + shape.rows + rows_a_page - 1) / rows_a_page;
        self.ids = try shuffled(a, rng, pages);
        const p: i32 = @intCast(self.at);
        self.in = try self.rig.input(std.mem.sliceAsBytes(self.lat));
        self.pos = try self.rig.input(std.mem.asBytes(&p));
        self.pt = try self.rig.input(std.mem.sliceAsBytes(self.ids));
        self.arena = try self.rig.scratch((pages + 3) * page_bytes);
        self.out = try self.rig.output(shape.rows * lw * 2);
        return true;
    }

    pub fn run(self: *LatWrite) !void {
        var a: cuda.Args = .{};
        inline for (.{ self.in, @as(i64, lw), self.arena + part_bytes, self.pos, @as(i64, lw), self.pt, @as(i64, page_bytes) }) |v| a.add(v);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows) }, .block = .{ .x = 128 } }, self.rig.s, &a);
        var r: usize = 0;
        while (r < self.shape.rows) { // the rows back from their pages, a page's run at a time
            const row = self.at + r;
            const n = @min(self.shape.rows - r, rows_a_page - row % rows_a_page);
            const page: u64 = @intCast(self.ids[row / rows_a_page]);
            try copy(&self.rig, self.out + r * lw * 2, self.arena + part_bytes + page * page_bytes + (row % rows_a_page) * lw * 2, n * lw * 2);
            r += n;
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *LatWrite) usize {
        return self.shape.rows * lw;
    }

    /// The rows as given: a write rounds nothing.
    pub fn reference(self: *LatWrite, want: []f64, mag: []f64) !void {
        for (want, mag, self.lat) |*w, *m, v| {
            w.* = c.fromBf16(v);
            m.* = if (std.math.isFinite(w.*)) @abs(w.*) else 0;
        }
    }
};

/// The float64 maximum that keeps a NaN, as the float64 formula's own arithmetic does.
pub fn maxNan(x: f64, y: f64) f64 {
    return if (std.math.isNan(x) or std.math.isNan(y)) std.math.nan(f64) else @max(x, y);
}

/// `v` as the accumulator `acc` holds it.
pub fn held(acc: c.Accum, v: f64) f64 {
    return switch (acc) {
        .f32 => @as(f32, @floatCast(v)),
        .f16 => @as(f16, @floatCast(v)),
        .bf16 => c.fromBf16(c.toBf16(@floatCast(v))),
        .none => v,
    };
}

/// Terms summed in `acc` in `order`: in turn, in halves, or every other term and then the rest.
pub fn sumIn(acc: c.Accum, order: g.Order, t: []const f64) f64 {
    switch (order) {
        .sequential => {
            var s = t[0];
            for (t[1..]) |v| s = held(acc, s + v);
            return s;
        },
        .pairwise => return if (t.len == 1) t[0] else held(acc, sumIn(acc, .pairwise, t[0 .. t.len / 2]) + sumIn(acc, .pairwise, t[t.len / 2 ..])),
        .strided => {
            if (t.len == 1) return t[0];
            var s = [2]f64{ t[0], t[1] };
            for (t[2..], 2..) |v, i| s[i % 2] = held(acc, s[i % 2] + v);
            return held(acc, s[0] + s[1]);
        },
    }
}

/// Four terms summed in `acc` in `order`: in turn, in pairs, or the odd terms then the even.
fn sum4(acc: c.Accum, order: g.Order, t: [4]f64) f64 {
    return switch (order) {
        .sequential => held(acc, held(acc, held(acc, t[0] + t[1]) + t[2]) + t[3]),
        .pairwise => held(acc, held(acc, t[0] + t[1]) + held(acc, t[2] + t[3])),
        .strided => held(acc, held(acc, t[0] + t[2]) + held(acc, t[1] + t[3])),
    };
}

/// sparse._pool_keys: each complete pool's 4 rows softmaxed by their gates (+ ape), summed onto pages out of order.
pub const PoolKeys = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    shape: c.Shape = undefined,
    ik: []u16 = &.{},
    ig: []u16 = &.{},
    ape: []u16 = &.{},
    ids: []i32 = &.{},
    at: usize = 0,
    first: usize = 0, // the first complete pool
    pools: usize = 0,
    in: [3]u64 = @splat(0),
    pos: u64 = 0,
    pt: u64 = 0,
    arena: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *PoolKeys) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *PoolKeys) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ "ik", "ig", "ape" }) |name| {
            a.free(@field(self, name));
            @field(self, name) = &.{};
        }
        a.free(self.ids);
        self.ids = &.{};
    }

    pub fn kernel(self: *PoolKeys) g.Kernel {
        return .{ .spec = spec(.pool_keys), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(PoolKeys);

    /// Keys and gates in [-2, 2); non-finite: NaN, +inf and -inf gates in the first pools, an infinite key in a fourth.
    pub fn prepare(self: *PoolKeys, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != dk or shape.k != 4 or shape.rows < 4) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.at = firstRow(shape.rows);
        self.first = self.at / 4;
        self.pools = (self.at + shape.rows) / 4 - self.first;
        const n = self.at + shape.rows;
        self.ik = try a.alloc(u16, n * dk);
        self.ig = try a.alloc(u16, n * dk);
        self.ape = try a.alloc(u16, 4 * dk);
        for (self.ik) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (self.ig) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (self.ape) |*v| v.* = c.toBf16(rng.float(f32) - 0.5);
        if (fill == .nonfinite) {
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (put, 0..) |v, i| if (i < self.pools) {
                self.ig[(4 * (self.first + i) + 1) * dk + 7 * i] = c.toBf16(v);
            };
            if (self.pools > 3) self.ik[(4 * (self.first + 3) + 2) * dk + 5] = c.toBf16(std.math.inf(f32));
        }
        const pages = (n + rows_a_page - 1) / rows_a_page;
        self.ids = try shuffled(a, rng, pages);
        const p: i32 = @intCast(self.at);
        inline for (.{ self.ik, self.ig, self.ape }, 0..) |x, i| self.in[i] = try self.rig.input(std.mem.sliceAsBytes(x));
        self.pos = try self.rig.input(std.mem.asBytes(&p));
        self.pt = try self.rig.input(std.mem.sliceAsBytes(self.ids));
        self.arena = try self.rig.scratch((pages + 3) * page_bytes);
        self.out = try self.rig.output(self.pools * dk * 2);
        return true;
    }

    pub fn run(self: *PoolKeys) !void {
        var a: cuda.Args = .{};
        const pk = self.arena + part_bytes + rows_a_page * lw * 2; // the cache's pooled keys in page 0
        inline for (.{ self.in[0], self.in[1], self.in[2], pk, self.pos, @as(i64, @intCast(self.shape.rows)), self.pt, @as(i64, page_bytes) }) |v| a.add(v);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows / 4 + 2) }, .block = .{ .x = 32 } }, self.rig.s, &a);
        var i: usize = 0;
        while (i < self.pools) { // the pools back from their pages, a page's run at a time
            const p = self.first + i;
            const n = @min(self.pools - i, pools_a_page - p % pools_a_page);
            const page: u64 = @intCast(self.ids[p / pools_a_page]);
            try copy(&self.rig, self.out + i * dk * 2, pk + page * page_bytes + (p % pools_a_page) * dk * 2, n * dk * 2);
            i += n;
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *PoolKeys) usize {
        return self.pools * dk;
    }

    /// Pool p's weights and products as the kernel forms them, summed in `acc`; null acc: float64 throughout.
    fn pool(self: *const PoolKeys, p: usize, d: usize, acc: ?c.Accum, order: g.Order, mag: *f64) f64 {
        var l: [4]f64 = undefined;
        for (&l, 0..) |*x, q| x.* = c.fromBf16(self.ig[(4 * p + q) * dk + d]) + c.fromBf16(self.ape[q * dk + d]);
        const m = maxNan(maxNan(l[0], l[1]), maxNan(l[2], l[3]));
        var e: [4]f64 = undefined;
        for (&e, l) |*x, v| x.* = @exp(v - m);
        const s = if (acc) |k| sum4(k, order, e) else e[0] + e[1] + e[2] + e[3];
        var t: [4]f64 = undefined;
        mag.* = 0;
        for (&t, e, 0..) |*x, w, q| {
            const key = c.fromBf16(self.ik[(4 * p + q) * dk + d]);
            x.* = if (acc != null) c.fromBf16(c.toBf16(@floatCast(c.fromBf16(c.toBf16(@floatCast(w / s))) * key))) else w / s * key;
            mag.* += if (std.math.isFinite(x.*)) @abs(w / s * key) else 0;
        }
        return if (acc) |k| sum4(k, order, t) else t[0] + t[1] + t[2] + t[3];
    }

    pub fn reference(self: *PoolKeys, want: []f64, mag: []f64) !void {
        for (0..self.pools) |i| for (0..dk) |d| {
            want[i * dk + d] = self.pool(self.first + i, d, null, .sequential, &mag[i * dk + d]);
        };
    }

    /// Each product rounds its weight and itself to bf16 (2^-8 each of its size); the sum and exponent add a little.
    pub fn bounds(_: *PoolKeys, want: []const f64, mag: []const f64, allow: []f64) !void {
        for (allow, want, mag) |*l, w, m| l.* = m * (0x1p-7 + 0x1p-14) + @abs(w) * 0x1p-8 * (1 + 0x1p-6) + 0x1p-133;
    }

    pub fn simulate(self: *PoolKeys, acc: c.Accum, order: g.Order, out: []f64) !void {
        var mag: f64 = 0;
        for (0..self.pools) |i| for (0..dk) |d| {
            out[i * dk + d] = c.fromBf16(c.toBf16(@floatCast(self.pool(self.first + i, d, acc, order, &mag))));
        };
    }
};
