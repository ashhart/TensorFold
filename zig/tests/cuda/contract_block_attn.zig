//! block_attention.cu (#746) under the contract on CUDA: DFlash block rows over their windowed context, head by head.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const block = cuda.block_attention;

const D = 128;
const h = 32;
const hk = 8;
const window = 2048;

/// The P x V chain of a row: its window's keys and the block's, a 32-key tile at a time.
fn keyChain(s: c.Shape) usize {
    return @min(s.k, window + 1) + 32;
}

/// Shape: streams (rows), query heads (n) and stream 0's context keys (k); each stream's block of L rows.
pub fn spec(causal: bool) c.Spec {
    // Qwen3.8-27B's DFlash2 drafter: blocks of 16 over 32 heads (8 key heads) and a 2,048-key window
    return .{
        .name = if (causal) "cuda block attention tf_block_attention, causal blocks of 7" else "cuda block attention tf_block_attention (DFlash2 blocks of 16)",
        .accum = .f32,
        .depth = keyChain,
        .ops = 1,
        .out = .bf16,
        .tiles = .{ .{}, .{}, .{ .size = 32 } },
        .served = &.{ .{ .rows = 1, .n = h, .k = 2100 }, .{ .rows = 4, .n = h, .k = 700 } },
        .sums = .{ .why = attn.why },
    };
}

pub const Block = struct {
    rig: rig_.Rig,
    ba: *const block.BlockAttention,
    causal: bool,
    r: ?attn.Ref = null,
    ns: usize = 0,
    lens: std.ArrayList(i32) = .empty,
    ctx: std.ArrayList([]u16) = .empty,
    q: std.ArrayList(u16) = .empty,
    kb: std.ArrayList(u16) = .empty,
    vb: std.ArrayList(u16) = .empty,
    dev: [5]u64 = @splat(0),
    out: u64 = 0,

    fn len(self: *const Block) usize {
        return if (self.causal) 7 else 16;
    }

    fn scale() f32 {
        return @floatCast(1 / @sqrt(@as(f64, D)));
    }

    pub fn deinit(self: *Block) void {
        const a = self.rig.gpu.gpa;
        self.drop();
        if (self.r) |*r| r.deinit();
        inline for (.{ "lens", "ctx", "q", "kb", "vb" }) |f| @field(self, f).deinit(a);
        self.rig.deinit();
    }

    fn drop(self: *Block) void {
        for (self.ctx.items) |x| self.rig.gpu.gpa.free(x);
        self.ctx.clearRetainingCapacity();
    }

    pub fn kernel(self: *Block) g.Kernel {
        return .{ .spec = spec(self.causal), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Block);

    pub fn prepare(self: *Block, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        self.drop();
        if (shape.n != h or shape.rows == 0 or shape.k == 0) return false;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const L = self.len();
        self.ns = shape.rows;
        const rows = self.ns * L;
        try self.lens.resize(a, self.ns);
        for (self.lens.items, 0..) |*l, j| l.* = @intCast(@max(1, shape.k - j * shape.k / self.ns));
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(a, D, @min(shape.k, window + 1) + L);
        for (self.lens.items) |l| {
            const x = try a.alloc(u16, 2 * hk * @as(usize, @intCast(l)) * D); // K [hk, len, D], then V
            try self.ctx.append(a, x);
            for (x) |*e| e.* = attn.normal(rng);
        }
        try self.q.resize(a, h * rows * D);
        try self.kb.resize(a, hk * rows * D);
        try self.vb.resize(a, hk * rows * D);
        for ([_][]u16{ self.q.items, self.kb.items, self.vb.items }) |x| for (x) |*e| {
            e.* = attn.normal(rng);
        };
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => try self.stress(rng),
        }
        const table = try a.alloc(i64, 2 * self.ns);
        defer a.free(table);
        for (self.ctx.items, 0..) |x, j| {
            const base = try self.rig.input(std.mem.sliceAsBytes(x));
            table[2 * j] = @intCast(base);
            table[2 * j + 1] = @intCast(base + x.len); // V follows K: half the u16s, all the bytes
        }
        for ([_][]const u8{ std.mem.sliceAsBytes(self.q.items), std.mem.sliceAsBytes(self.kb.items), std.mem.sliceAsBytes(self.vb.items), std.mem.sliceAsBytes(table), std.mem.sliceAsBytes(self.lens.items) }, &self.dev) |bytes, *p| p.* = try self.rig.input(bytes);
        self.out = try self.rig.output(rows * h * D * 2);
        return true;
    }

    fn ctxLen(self: *const Block, j: usize) usize {
        return @intCast(self.lens.items[j]);
    }

    /// NaN in a query, ±inf in values (context, block, a window's first key), both in one column, NaN in a key.
    fn plant(self: *Block) void {
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        const rows = self.ns * self.len();
        const s0 = self.ctxLen(0);
        const v0 = self.ctx.items[0][hk * s0 * D ..];
        self.q.items[3] = nan; // head 0, stream 0's row 0
        v0[(0 * s0 + s0 - 1) * D + 7] = inf;
        self.vb.items[(1 * rows + 2) * D + 9] = ninf; // key head 1, stream 0's row 2
        if (s0 >= 3) {
            v0[(2 * s0 + s0 - 2) * D + 11] = inf;
            v0[(2 * s0 + s0 - 3) * D + 11] = ninf;
        }
        if (s0 > window) v0[(4 * s0 + s0 - window) * D + 13] = inf; // a key block row 0 alone sees
        self.ctx.items[self.ns - 1][(3 * self.ctxLen(self.ns - 1) + self.ctxLen(self.ns - 1) - 1) * D] = nan;
        // +inf values row 0 sees at a float64 weight of 0 (0 x inf: NaN), and at weights below float32's least (inf)
        const k0 = self.ctx.items[0][0 .. hk * s0 * D];
        self.reach(k0, v0, 6, 24, null, 19); // the block's first key
        if (s0 > window) self.reach(k0, v0, 5, 20, s0 - window + 8, 17); // the window's edge tile, rows 0 to 8 see it
        self.under(k0, v0);
        if (self.ns > 1) self.ownMax(1, 7, 23);
    }

    /// Stream j, key head kh: block key 0 all 8, other keys -8 with +inf at e; row 0 queries 2 (inf) and 8 (NaN).
    fn ownMax(self: *Block, j: usize, kh: usize, e: usize) void {
        const L = self.len();
        const rows = self.ns * L;
        const s = self.ctxLen(j);
        const x = self.ctx.items[j];
        const inf = c.toBf16(std.math.inf(f32));
        const G = h / hk;
        @memset(self.q.items[((kh * G) * rows + j * L) * D ..][0..D], c.toBf16(2));
        @memset(self.q.items[((kh * G + 1) * rows + j * L) * D ..][0..D], c.toBf16(8));
        for (0..s) |k| {
            @memset(x[(kh * s + k) * D ..][0..D], c.toBf16(-8));
            x[hk * s * D + (kh * s + k) * D + e] = inf;
        }
        for (0..L) |col| {
            @memset(self.kb.items[(kh * rows + j * L + col) * D ..][0..D], c.toBf16(if (col == 0) 8 else -8));
            if (col != 0) self.vb.items[(kh * rows + j * L + col) * D + e] = inf;
        }
    }

    /// Head `head`'s row 0: query all 8, key s0 - 1 all 8, key `at` (null: block key 0) all -8 with +inf at e.
    fn reach(self: *Block, k0: []u16, v0: []u16, kh: usize, head: usize, at: ?usize, e: usize) void {
        const rows = self.ns * self.len();
        const s0 = self.ctxLen(0);
        @memset(self.q.items[(head * rows) * D ..][0..D], c.toBf16(8));
        @memset(k0[(kh * s0 + s0 - 1) * D ..][0..D], c.toBf16(8));
        const far = if (at) |k| k0[(kh * s0 + k) * D ..][0..D] else self.kb.items[(kh * rows) * D ..][0..D];
        @memset(far, c.toBf16(-8));
        if (at) |k| v0[(kh * s0 + k) * D + e] = c.toBf16(std.math.inf(f32)) else self.vb.items[(kh * rows) * D + e] = c.toBf16(std.math.inf(f32));
    }

    /// Key head 7, stream 0: head 28's row 0 query all 2, key s0 - 1 all 8, every other key all -8 with +inf at 21.
    fn under(self: *Block, k0: []u16, v0: []u16) void {
        const rows = self.ns * self.len();
        const s0 = self.ctxLen(0);
        const kh = 7;
        @memset(self.q.items[(28 * rows) * D ..][0..D], c.toBf16(2));
        for (0..s0 + self.len()) |k| {
            const key = if (k < s0) k0[(kh * s0 + k) * D ..][0..D] else self.kb.items[(kh * rows + k - s0) * D ..][0..D];
            const val = if (k < s0) v0[(kh * s0 + k) * D ..][0..D] else self.vb.items[(kh * rows + k - s0) * D ..][0..D];
            @memset(key, c.toBf16(if (k == s0 - 1) 8 else -8));
            if (k != s0 - 1) val[21] = c.toBf16(std.math.inf(f32));
        }
    }

    /// Gate 4: every key alike (each p is 1); values mirrored over the context keys every row sees, else 0.
    fn stress(self: *Block, rng: std.Random) !void {
        const a = self.rig.gpu.gpa;
        const key = try a.alloc(u16, D);
        defer a.free(key);
        for (key) |*e| e.* = attn.normal(rng);
        const L = self.len();
        @memset(self.vb.items, 0);
        for (0..self.kb.items.len / D) |i| @memcpy(self.kb.items[i * D ..][0..D], key);
        for (self.ctx.items, 0..) |x, j| {
            const s = self.ctxLen(j);
            for (0..hk * s) |i| @memcpy(x[i * D ..][0..D], key);
            const v = x[hk * s * D ..];
            @memset(v, 0);
            const lo = @min(s, (s + L) -| window);
            const col = try a.alloc(f32, s - lo);
            defer a.free(col);
            for (0..hk) |kh| for (0..D) |e| {
                attn.mirrored(rng, col);
                for (col, lo..) |y, k| v[(kh * s + k) * D + e] = c.toBf16(y);
            };
        }
    }

    pub fn run(self: *Block) !void {
        const L = self.len();
        try self.ba.run(self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.dev[3], self.dev[4], self.out, .{ .h = h, .hk = hk, .rows = self.ns * L, .block = L, .window = window, .scale = scale(), .causal = self.causal });
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Block) usize {
        return self.ns * self.len() * h * D;
    }

    /// Output head i (block row i / h, head i % h): context keys k with len + r - k <= window, then the block's.
    pub fn load(self: *Block, i: usize, hd: *attn.Head) !void {
        const L = self.len();
        const rows = self.ns * L;
        const row = i / h;
        const head = i % h;
        const kh = head / (h / hk);
        const j = row / L;
        const r = row % L;
        const s = self.ctxLen(j);
        const x = self.ctx.items[j];
        for (hd.q, 0..) |*y, e| y.* = c.fromBf16(self.q.items[(head * rows + row) * D + e]);
        var n: usize = 0;
        for (0..s) |k| if (s + r - k <= window) {
            for (0..D) |e| {
                hd.k[n * D + e] = c.fromBf16(x[(kh * s + k) * D + e]);
                hd.v[n * D + e] = c.fromBf16(x[hk * s * D + (kh * s + k) * D + e]);
            }
            n += 1;
        };
        for (0..L) |col| if (!self.causal or col <= r) {
            for (0..D) |e| {
                hd.k[n * D + e] = c.fromBf16(self.kb.items[(kh * rows + j * L + col) * D + e]);
                hd.v[n * D + e] = c.fromBf16(self.vb.items[(kh * rows + j * L + col) * D + e]);
            }
            n += 1;
        };
        hd.n = n;
    }

    fn heads(self: *Block) usize {
        return self.ns * self.len() * h;
    }

    pub fn check(self: *Block, at: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.heads(), scale(), try self.rig.outputBytes(), at);
    }

    pub fn reference(self: *Block, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.heads(), scale(), want, mag);
    }

    pub fn simulate(self: *Block, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.heads(), scale(), acc, order, out);
    }
};
