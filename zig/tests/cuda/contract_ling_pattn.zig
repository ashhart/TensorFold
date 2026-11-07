//! Ling-3.1-flash's expanded MLA prompt attention (tf_ling_pattn): 192-wide q.k, 128-wide v, fp32 carry by blocks.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const pages = @import("contract_glm_pages.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const qk = 192; // q.k dims a head: 128 expanded nope, then the cache's 64 roped columns
const dv = 128;
const kw = 576; // a cache row: the latent, then the 64 roped columns
const bn = 64; // keys a tile
const block_keys = 512; // keys a launch: the fixture's spans take two to four, carried
const att_scale: f32 = 0.07216878364870322; // 1 / sqrt(192), as fp32
const smem: u32 = 2 * 64 * (24 + 16) * 16;
const soft_why = "online softmax over 64-key tiles carried across key blocks: no input makes its sums exact in fp32";

fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

/// A key's 192 products, a tile's 64 keys, then the tiles in turn.
fn depth(s: c.Shape) usize {
    return qk + bn + s.k / bn + 1;
}

const shapes = [_]c.Shape{ .{ .rows = 64, .n = 4 * dv, .k = 1024 }, .{ .rows = 130, .n = 2 * dv, .k = 700 }, .{ .rows = 8, .n = 8 * dv, .k = 2048 } };

fn spec() c.Spec {
    return .{ .name = "cuda ling ling_glue tf_ling_pattn (expanded MLA prompt rows: 192-wide q.k, 128-wide v, 512-key blocks carried)", .accum = .f32, .depth = depth, .ops = 8, .out = .bf16, .served = &shapes, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = soft_why } };
}

/// tf_ling_pattn as block_mla_prompt.attend calls it: rows at positions at + r over keys 0..at + r, block by block.
pub const Pattn = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    shape: c.Shape = undefined,
    q: []u16 = &.{},
    kv: []u16 = &.{},
    cache: []u16 = &.{},
    tol: []f64 = &.{},
    at: usize = 0,
    heads: usize = 0,
    in: [3]u64 = @splat(0), // q, kv, cache
    carry: [3]u64 = @splat(0),
    out: u64 = 0,

    pub fn deinit(self: *Pattn) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Pattn) void {
        const a = self.rig.gpu.gpa;
        a.free(self.q);
        a.free(self.kv);
        a.free(self.cache);
        a.free(self.tol);
        self.q = &.{};
        self.kv = &.{};
        self.cache = &.{};
        self.tol = &.{};
    }

    pub fn kernel(self: *Pattn) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Pattn);

    /// Queries, keys and values in [-1, 1); row r at position k - rows + r, so the last block holds the rows' own keys.
    pub fn prepare(self: *Pattn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n % dv != 0 or shape.rows == 0 or shape.k < shape.rows) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.heads = shape.n / dv;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.at = shape.k - shape.rows;
        const span = shape.k;
        self.q = try a.alloc(u16, shape.rows * self.heads * qk);
        self.kv = try a.alloc(u16, span * self.heads * 2 * dv);
        self.cache = try a.alloc(u16, span * kw);
        for (self.q) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (self.kv) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (self.cache) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (fill == .nonfinite) {
            self.q[3] = c.toBf16(std.math.nan(f32));
            self.q[qk + 150] = c.toBf16(std.math.nan(f32)); // head 1, a roped column
            self.kv[9] = c.toBf16(std.math.inf(f32)); // key 0, head 0: an expanded key column
            self.kv[2 * self.heads * 2 * dv + dv + 5] = c.toBf16(std.math.inf(f32)); // key 2, head 0: a value
            self.cache[3 * kw + 530] = c.toBf16(-std.math.inf(f32)); // key 3, a roped column: scores only
        }
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.q));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.kv));
        self.in[2] = try self.rig.input(std.mem.sliceAsBytes(self.cache));
        const n = shape.rows * self.heads;
        self.carry = .{ try self.rig.scratch(n * dv * 4), try self.rig.scratch(n * 4), try self.rig.scratch(n * 4) };
        self.out = try self.rig.output(n * dv * 2);
        return true;
    }

    pub fn run(self: *Pattn) !void {
        const span = self.shape.k;
        var k0: usize = 0;
        while (k0 < span) : (k0 += block_keys) {
            const n = @min(block_keys, span - k0);
            var a: cuda.Args = .{};
            const kv = self.in[1] + k0 * self.heads * 2 * dv * 2;
            inline for (.{ self.in[0], kv, self.in[2] + k0 * kw * 2, self.carry[0], self.carry[1], self.carry[2], self.out }) |v| a.add(v);
            inline for (.{ self.shape.rows, self.heads, n, k0, self.at }) |v| a.add(@as(c_int, @intCast(v)));
            a.add(att_scale);
            a.add(@as(c_int, @intFromBool(k0 == 0)));
            a.add(@as(c_int, @intFromBool(k0 + block_keys >= span)));
            try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((self.shape.rows + 127) / 128), .y = @intCast(self.heads) }, .block = .{ .x = 256 }, .shared = smem }, self.rig.s, &a);
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Pattn) usize {
        return self.shape.rows * self.heads * dv;
    }

    fn key(self: *const Pattn, i: usize, h: usize, d: usize) f64 {
        if (d < dv) return bf(self.kv[(i * self.heads + h) * 2 * dv + d]);
        return bf(self.cache[i * kw + 512 + d - dv]);
    }

    fn value(self: *const Pattn, i: usize, h: usize, d: usize) f64 {
        return bf(self.kv[(i * self.heads + h) * 2 * dv + dv + d]);
    }

    /// Row r, head h: softmax(q . k scale) . v over keys 0..at + r, sums in `acc` (null: float64), into `o`.
    fn attend(self: *Pattn, r: usize, h: usize, acc: ?c.Accum, order: g.Order, o: []f64, tol: ?[]f64) !void {
        const a = self.rig.gpu.gpa;
        const n = self.at + r + 1;
        const s = try a.alloc(f64, 3 * n);
        defer a.free(s);
        const ds = s[n .. 2 * n];
        const p = s[2 * n ..];
        var terms: [qk]f64 = undefined;
        var most = -std.math.inf(f64);
        const sc: f64 = att_scale;
        for (0..n) |i| {
            var abs: f64 = 0;
            for (&terms, 0..) |*x, d| {
                x.* = bf(self.q[(r * self.heads + h) * qk + d]) * self.key(i, h, d);
                abs += @abs(x.*);
            }
            s[i] = (if (acc) |k| pages.sumIn(k, order, &terms) else pages.sumIn(.none, .sequential, &terms)) * sc;
            ds[i] = abs * sc * (qk + 8) * 0x1p-24;
            most = pages.maxNan(most, s[i]);
        }
        for (p, s[0..n]) |*x, v| x.* = @exp(v - most);
        const l = if (acc) |k| pages.sumIn(k, order, p) else pages.sumIn(.none, .sequential, p);
        const col = try a.alloc(f64, n);
        defer a.free(col);
        for (o, 0..) |*od, d| {
            var spread: f64 = 0;
            for (col, p, 0..) |*x, w, i| {
                const v = self.value(i, h, d);
                x.* = w * v;
                spread += w * @abs(v) * (0x1p-8 + ds[i]);
            }
            od.* = (if (acc) |k| pages.sumIn(k, order, col) else pages.sumIn(.none, .sequential, col)) / l;
            if (tol) |t| t[d] = @abs(od.*) * 0x1p-8 + spread / l + 1e-30;
        }
    }

    /// float64 o = sum p v / sum p, p = exp(s - max); bound: bf16 out and p (2^-8 each), 200 2^-24 scale sum |q k|.
    pub fn reference(self: *Pattn, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        a.free(self.tol);
        self.tol = try a.alloc(f64, want.len);
        for (0..self.shape.rows) |r| for (0..self.heads) |h| {
            const at = (r * self.heads + h) * dv;
            try self.attend(r, h, null, .sequential, want[at..][0..dv], self.tol[at..][0..dv]);
        };
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn bounds(self: *Pattn, _: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, self.tol) |*l, t| l.* = if (std.math.isFinite(t)) t else std.math.floatMax(f64);
    }

    pub fn simulate(self: *Pattn, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..self.shape.rows) |r| for (0..self.heads) |h| {
            const at = (r * self.heads + h) * dv;
            try self.attend(r, h, acc, order, out[at..][0..dv], null);
            for (out[at..][0..dv]) |*v| v.* = bf(c.toBf16(@floatCast(v.*)));
        };
    }
};

/// The rig for tf_ling_pattn on the ling_glue image (`lg`), registered into `list`.
pub fn init(gpu: check.Gpu, s: cuda.Stream, lg: cuda.Module) !Pattn {
    const f = try lg.function("tf_ling_pattn");
    try f.allowDynamicShared(smem);
    return .{ .rig = .{ .gpu = gpu, .s = s }, .f = f };
}
