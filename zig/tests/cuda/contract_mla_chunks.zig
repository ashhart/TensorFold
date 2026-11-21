//! The mla image's chunk tiles under the contract: tf_mla_chunks_r16, and the split form with tf_mla_merge_part.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const pages = @import("contract_glm_pages.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const lw = 512; // the latent: the values
const kw = 576; // q and key rows: the latent, then the 64 roped columns
const chunk = 512;
const att_scale: f32 = 0.07216878364870322; // 1 / sqrt(192), as fp32
const soft_why = "online softmax over 512-key chunks, then the chunk merge: no input makes its sums exact in fp32";

fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

/// A key's 576 products, then a chunk's 512 keys and the merge's chunks.
fn attnDepth(s: c.Shape) usize {
    return kw + chunk + s.k / chunk + 1;
}

const chunk_shapes = [_]c.Shape{ .{ .rows = 8, .n = 32 * lw, .k = 2 * chunk }, .{ .rows = 4, .n = 64 * lw, .k = 2 * chunk }, .{ .rows = 64, .n = 32 * lw, .k = chunk } };

/// A planted fault, each a control red at gate 3.
pub const Fault = enum {
    none,
    next_head, // tf_mla_chunks_r16: each head's queries are the next head's
    no_stride, // tf_mla_chunks_split_r16: a rank's page z read as chunk z, not z * world + rank
    part_short, // tf_mla_merge_part: one chunk short of a rank's chunks
};

fn name(split: bool, fault: Fault) []const u8 {
    return switch (fault) {
        .none => if (split) "cuda mla tf_mla_chunks_split_r16 and tf_mla_merge_part (a split cache's two halves), with tf_glm_merge" else "cuda mla tf_mla_chunks_r16 (576-wide q and keys, dense over a linear cache), with tf_glm_merge",
        .next_head => "control: mla tf_mla_chunks_r16 reads each head's queries from the next head",
        .no_stride => "control: mla tf_mla_chunks_split_r16 reads a rank's page z as chunk z",
        .part_short => "control: mla tf_mla_merge_part merges one chunk short of the rank's",
    };
}

/// tf_mla_chunks_r16 as cuda.mla.Mla.chunks calls it: rows x heads x 576 queries, keys 0..pos + r of a linear cache.
pub const Chunks = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    merge: cuda.Function,
    fault: Fault = .none,
    shape: c.Shape = undefined,
    q: []u16 = &.{},
    cache: []u16 = &.{},
    tol: []f64 = &.{},
    at: usize = 0,
    nch: usize = 0,
    heads: usize = 0,
    in: [3]u64 = @splat(0), // q, cache, the merge's counts (unread when dense)
    pos: u64 = 0,
    parts: [3]u64 = @splat(0),
    out: u64 = 0,
    split: ?[2]cuda.Function = null, // tf_mla_chunks_split_r16 and tf_mla_merge_part: the split form (splitRun)
    halves: [2]u64 = @splat(0), // the split form's caches: each rank's chunks 2 j + rank as page j
    got: [3]u64 = @splat(0), // the split form's [2] merged partials: o, m, l

    pub fn deinit(self: *Chunks) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Chunks) void {
        const a = self.rig.gpu.gpa;
        a.free(self.q);
        a.free(self.cache);
        a.free(self.tol);
        self.q = &.{};
        self.cache = &.{};
        self.tol = &.{};
    }

    pub fn kernel(self: *Chunks) g.Kernel {
        const s: c.Spec = .{ .name = name(self.split != null, self.fault), .accum = .f32, .depth = attnDepth, .ops = 8, .out = .bf16, .served = &chunk_shapes, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = soft_why } };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.fault != .none) .shapes else null };
    }

    const vt = g.hooks(Chunks);

    /// Queries and keys in [-1, 1); row r at position k - rows / 2 + r, so the rows straddle a chunk edge.
    pub fn prepare(self: *Chunks, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n % lw != 0 or shape.rows == 0 or shape.k == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.heads = shape.n / lw;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.at = shape.k - @min(shape.k, shape.rows / 2);
        const span = self.at + shape.rows;
        self.nch = (span + chunk - 1) / chunk;
        self.q = try a.alloc(u16, shape.rows * self.heads * kw);
        self.cache = try a.alloc(u16, span * kw);
        for (self.q) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (self.cache) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (fill == .nonfinite) {
            self.q[3] = c.toBf16(std.math.nan(f32));
            self.q[kw + 540] = c.toBf16(std.math.nan(f32)); // head 1, a roped column
            self.cache[9] = c.toBf16(std.math.inf(f32));
            self.cache[2 * kw + 530] = c.toBf16(-std.math.inf(f32)); // key 2, a roped column: scores only
        }
        const p: i32 = @intCast(self.at);
        const cnt: i32 = 0;
        if (self.fault == .next_head) {
            const moved = try a.alloc(u16, self.q.len);
            defer a.free(moved);
            @memset(moved, 0);
            @memcpy(moved[0 .. moved.len - kw], self.q[kw..]);
            self.in[0] = try self.rig.input(std.mem.sliceAsBytes(moved));
        } else self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.q));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.cache));
        self.in[2] = try self.rig.input(std.mem.asBytes(&cnt));
        self.pos = try self.rig.input(std.mem.asBytes(&p));
        const n = self.nch * shape.rows * self.heads;
        self.parts = .{ try self.rig.scratch(n * lw * 4), try self.rig.scratch(n * 4), try self.rig.scratch(n * 4) };
        self.out = try self.rig.output(shape.rows * self.heads * lw * 2);
        if (self.split != null) try splitPrepare(self);
        return true;
    }

    pub fn run(self: *Chunks) !void {
        if (self.split) |f| return splitRun(self, f[0], f[1]);
        const rows: i64 = @intCast(self.shape.rows);
        const heads: i64 = @intCast(self.heads);
        const smem = cuda.mla.chunksSmem(cuda.mla.heads_tile);
        var a: cuda.Args = .{};
        inline for (.{ self.in[0], self.in[1], self.pos, @as(u64, 0), @as(u64, 0), self.parts[0], self.parts[1], self.parts[2], rows, heads, @as(i64, 0), att_scale, @as(c_int, 0), @as(u64, 0), @as(i64, 0) }) |v| a.add(v);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast((self.heads + 15) / 16), .z = @intCast(self.nch) }, .block = .{ .x = 256 }, .shared = smem }, self.rig.s, &a);
        var m: cuda.Args = .{};
        inline for (.{ self.parts[0], self.parts[1], self.parts[2], self.out, self.in[2], rows, heads, @as(c_int, @intCast(self.nch)), @as(c_int, 0) }) |v| m.add(v);
        try cuda.launch.launch(self.merge, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.heads) }, .block = .{ .x = 128 } }, self.rig.s, &m);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Chunks) usize {
        return self.shape.rows * self.heads * lw;
    }

    /// Row r, head h: softmax(q . k scale) . v over keys 0..at + r, sums in `acc` (null: float64), into `o`.
    fn attend(self: *Chunks, r: usize, h: usize, acc: ?c.Accum, order: g.Order, o: []f64, tol: ?[]f64) !void {
        const a = self.rig.gpu.gpa;
        const n = self.at + r + 1;
        const s = try a.alloc(f64, 3 * n);
        defer a.free(s);
        const ds = s[n .. 2 * n];
        const p = s[2 * n ..];
        var terms: [kw]f64 = undefined;
        var most = -std.math.inf(f64);
        const sc: f64 = att_scale;
        for (0..n) |i| {
            var abs: f64 = 0;
            for (&terms, 0..) |*x, d| {
                x.* = bf(self.q[(r * self.heads + h) * kw + d]) * bf(self.cache[i * kw + d]);
                abs += @abs(x.*);
            }
            s[i] = (if (acc) |k| pages.sumIn(k, order, &terms) else pages.sumIn(.none, .sequential, &terms)) * sc;
            ds[i] = abs * sc * (kw + 8) * 0x1p-24;
            most = pages.maxNan(most, s[i]);
        }
        for (p, s[0..n]) |*x, v| x.* = @exp(v - most);
        const l = if (acc) |k| pages.sumIn(k, order, p) else pages.sumIn(.none, .sequential, p);
        const col = try a.alloc(f64, n);
        defer a.free(col);
        for (o, 0..) |*od, d| {
            var spread: f64 = 0;
            for (col, p, 0..) |*x, w, i| {
                const v = bf(self.cache[i * kw + d]);
                x.* = w * v;
                spread += w * @abs(v) * (0x1p-8 + ds[i]);
            }
            od.* = (if (acc) |k| pages.sumIn(k, order, col) else pages.sumIn(.none, .sequential, col)) / l;
            if (tol) |t| t[d] = @abs(od.*) * 0x1p-8 + spread / l + 1e-30;
        }
    }

    /// float64 o = sum p v / sum p, p = exp(s - max); bound: bf16 out and p (2^-8 each), 584 2^-24 scale sum |q k|.
    pub fn reference(self: *Chunks, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        a.free(self.tol);
        self.tol = try a.alloc(f64, want.len);
        for (0..self.shape.rows) |r| for (0..self.heads) |h| {
            const at = (r * self.heads + h) * lw;
            try self.attend(r, h, null, .sequential, want[at..][0..lw], self.tol[at..][0..lw]);
        };
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn bounds(self: *Chunks, _: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, self.tol) |*l, t| l.* = if (std.math.isFinite(t)) t else std.math.floatMax(f64);
    }

    pub fn simulate(self: *Chunks, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..self.shape.rows) |r| for (0..self.heads) |h| {
            const at = (r * self.heads + h) * lw;
            try self.attend(r, h, acc, order, out[at..][0..lw], null);
            for (out[at..][0..lw]) |*v| v.* = bf(c.toBf16(@floatCast(v.*)));
        };
    }
};

/// Chunks' split form: each half of a split cache by tf_mla_chunks_split_r16 and tf_mla_merge_part, as served.
fn splitPrepare(self: *Chunks) !void {
    const a = self.rig.gpu.gpa;
    const span = self.at + self.shape.rows;
    for (0..2) |w| {
        const n_pages = (self.nch + 1 - w) / 2 + 1;
        const host = try a.alloc(u16, n_pages * chunk * kw);
        defer a.free(host);
        @memset(host, 0);
        var j: usize = 0;
        while (2 * j + w < self.nch) : (j += 1) {
            const first = (2 * j + w) * chunk;
            const n: usize = @min(chunk, span - first); // @min with the comptime 512 is a u10: n * kw would wrap
            @memcpy(host[j * chunk * kw ..][0 .. n * kw], self.cache[first * kw ..][0 .. n * kw]);
        }
        self.halves[w] = try self.rig.input(std.mem.sliceAsBytes(host));
    }
    const n = self.shape.rows * self.heads;
    self.got = .{ try self.rig.scratch(2 * n * lw * 4), try self.rig.scratch(2 * n * 4), try self.rig.scratch(2 * n * 4) };
}

fn splitRun(self: *Chunks, split: cuda.Function, part: cuda.Function) !void {
    const rows: i64 = @intCast(self.shape.rows);
    const heads: i64 = @intCast(self.heads);
    const n: u64 = @intCast(self.shape.rows * self.heads);
    const smem = cuda.mla.chunksSmem(cuda.mla.heads_tile);
    const world: c_int = if (self.fault == .no_stride) 1 else 2;
    for (0..2) |w| {
        const nloc = (self.nch + 1 - w) / 2;
        if (nloc > 0) {
            var a: cuda.Args = .{};
            const rank: c_int = if (self.fault == .no_stride) 0 else @intCast(w);
            inline for (.{ self.in[0], self.halves[w], self.pos, self.parts[0], self.parts[1], self.parts[2], rows, heads, att_scale, world, rank }) |v| a.add(v);
            try cuda.launch.launch(split, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast((self.heads + 15) / 16), .z = @intCast(nloc) }, .block = .{ .x = 256 }, .shared = smem }, self.rig.s, &a);
        }
        const merged: usize = if (self.fault == .part_short) nloc -| 1 else nloc;
        var m: cuda.Args = .{};
        inline for (.{ self.parts[0], self.parts[1], self.parts[2], self.got[0] + w * n * lw * 4, self.got[1] + w * n * 4, self.got[2] + w * n * 4, rows, heads, @as(c_int, @intCast(merged)) }) |v| m.add(v);
        try cuda.launch.launch(part, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.heads) }, .block = .{ .x = 128 } }, self.rig.s, &m);
    }
    var m: cuda.Args = .{};
    inline for (.{ self.got[0], self.got[1], self.got[2], self.out, self.in[2], rows, heads, @as(c_int, 2), @as(c_int, 0) }) |v| m.add(v);
    try cuda.launch.launch(self.merge, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.heads) }, .block = .{ .x = 128 } }, self.rig.s, &m);
    try self.rig.s.synchronize();
}

/// The two registrations and their three controls, in this order.
pub const forms = [_]struct { split: bool, fault: Fault }{ .{ .split = false, .fault = .none }, .{ .split = true, .fault = .none }, .{ .split = false, .fault = .next_head }, .{ .split = true, .fault = .no_stride }, .{ .split = true, .fault = .part_short } };
