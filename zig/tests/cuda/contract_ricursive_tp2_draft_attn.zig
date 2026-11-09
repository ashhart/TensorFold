//! Ripple's drafter block attention and its chunk merge (ricursive_tp2_draft.cu) against float64's softmax range.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");
const rows = @import("contract_ricursive_tp2_draft_rows.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;
const Function = cuda.Function;
const block = rows.block;

/// A warp's 64-key chain, the 4 warps, the 65 chunks merged, and the warp sum's 5 shuffles.
fn attnDepth(_: c.Shape) usize {
    return 64 + 4 + 65 + 5;
}

/// attn's 256-key partials (ring [first, position), then the block's 16 rows) merged: bf16 softmax(q k / sqrt(128)) v.
pub const Attn = struct {
    attn: Function,
    merge: Function,
    window: usize, // 4096: the sliding layers; 16384: the full layer's reach
    ctl: enum { none, attn, merge } = .none, // controls: ring keys one row on; the running sum never rescaled
    shape: c.Shape = undefined, // rows: streams; n: query heads; k: ring keys behind each stream's block
    q: []u16 = &.{},
    kb: []u16 = &.{},
    vb: []u16 = &.{},
    ring: []u16 = &.{}, // each stream's two layers of K then V rows, a stream after another
    segs: []rows.Seg = &.{},
    dev: [5]u64 = @splat(0),
    parts: [3]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};
    const heads = 40;
    const kv = 4;
    const width = kv * 128;
    const rr = 16384 + block; // the served ring: the widest window and a block
    const chunks = (rr + 255) / 256;
    const layer = 1;
    const per = 4 * rr * width; // a stream's ring words
    const sliding = [_]c.Shape{ .{ .rows = 1, .n = heads, .k = 300 }, .{ .rows = 4, .n = heads, .k = 4100 } };
    const full = [_]c.Shape{ .{ .rows = 1, .n = heads, .k = 300 }, .{ .rows = 2, .n = heads, .k = 1000 }, .{ .rows = 1, .n = heads, .k = 16384 } };

    pub fn spec(self: *const Attn) c.Spec {
        const name = switch (self.ctl) {
            .attn => "control: cuda ricursive_tp2_draft tf_rtp2d_attn reading each ring key one row on",
            .merge => "control: cuda ricursive_tp2_draft tf_rtp2d_merge with the running sum never rescaled",
            .none => if (self.window < 16384) "cuda ricursive_tp2_draft tf_rtp2d_attn + tf_rtp2d_merge (a sliding layer: 4,096 keys)" else "cuda ricursive_tp2_draft tf_rtp2d_attn + tf_rtp2d_merge (the full layer: 16,384 keys)",
        };
        const served: []const c.Shape = if (self.window < 16384) &sliding else &full;
        return .{ .name = name, .accum = .f32, .depth = attnDepth, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 256, .step = 1, .min = 0 } }, .served = served };
    }

    pub fn control(self: *const Attn) ?g.Gate {
        return if (self.ctl != .none) .shapes else null;
    }

    pub fn free(self: *Attn, a: std.mem.Allocator) void {
        for ([_]*[]u16{ &self.q, &self.kb, &self.vb, &self.ring }) |p| {
            a.free(p.*);
            p.* = &.{};
        }
        a.free(self.segs);
        self.segs = &.{};
    }

    fn ints(rng: std.Random, v: []u16) void {
        for (v) |*x| x.* = c.toBf16(ref.int(rng, -256, 256));
    }

    /// Gate 4's inputs: q = 0 (every weight exp(0) = 1), values whole numbers to 256: fp32 sums exact in any order.
    pub fn prepare(self: *Attn, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != heads) return false;
        self.free(a);
        self.shape = shape;
        const ns = shape.rows;
        const b = ns * block;
        self.q = try ref.bfs(a, rng, b * heads * 128, -1, 1);
        self.kb = try ref.bfs(a, rng, b * width, -1, 1);
        self.vb = try ref.bfs(a, rng, b * width, -1, 1);
        self.ring = try a.alloc(u16, ns * per);
        @memset(self.ring, 0);
        for (0..ns) |i| for (self.ring[i * per + 2 * layer * rr * width ..][0 .. 2 * rr * width]) |*x| {
            x.* = c.toBf16(-1 + 2 * rng.float(f32));
        };
        if (fill == .stress) {
            @memset(self.q, 0);
            ints(rng, self.vb);
            for (0..ns) |i| ints(rng, self.ring[i * per + (2 * layer + 1) * rr * width ..][0 .. rr * width]);
        }
        if (fill == .nonfinite) {
            self.q[(1 * heads + 2) * 128 + 3] = c.toBf16(ref.nan);
            self.kb[(2 * kv + 0) * 128 + 9] = c.toBf16(ref.inf);
            self.vb[(5 * kv + 3) * 128 + 11] = c.toBf16(-ref.inf);
        }
        self.segs = try a.alloc(rows.Seg, ns);
        self.dev[0] = try ref.input(rig, self.q);
        self.dev[1] = try ref.input(rig, self.kb);
        self.dev[2] = try ref.input(rig, self.vb);
        self.dev[3] = try ref.input(rig, self.ring);
        for (self.segs, 0..) |*sg, i| {
            const begin: i64 = rng.uintLessThan(u32, 40000);
            sg.* = .{ .ring = self.dev[3] + i * per * 2, .begin = begin, .position = begin + @as(i64, @intCast(shape.k)) };
        }
        self.dev[4] = try ref.input(rig, self.segs);
        const slots = b * heads * chunks;
        self.parts = .{ try rig.scratch(slots * 4), try rig.scratch(slots * 4), try rig.scratch(slots * 128 * 4) };
        self.out = try rig.output(b * heads * 128 * 2);
        return true;
    }

    pub fn run(self: *Attn, rig: *rig_.Rig) !void {
        const b = self.shape.rows * block;
        const d = self.dev;
        const p = self.parts;
        const dims = [_]i32{ block, heads, kv, rr, @intCast(self.window), layer };
        try rows.go(self.attn, rig.s, .{ b, heads, chunks }, 128, 0, .{ d[0], d[1], d[2], d[4], dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], p[0], p[1], p[2] });
        try rows.go(self.merge, rig.s, .{ b, heads, 1 }, 32, 0, .{ p[0], p[1], p[2], @as(i32, heads), @as(i32, chunks), self.out });
    }

    pub fn count(self: *const Attn) usize {
        return self.shape.rows * block * heads * 128;
    }

    /// Key e's K (or V) row of block row m in stream i: the ring's at (first + e) % rr, then the block's rows.
    fn key(self: *const Attn, i: usize, first: i64, rk: usize, e: usize, h: usize, v: bool) []const u16 {
        const kvh = h / (heads / kv);
        if (e >= rk) return (if (v) self.vb else self.kb)[((i * block + e - rk) * kv + kvh) * 128 ..][0..128];
        const row: usize = @intCast(@mod(first + @as(i64, @intCast(e)), rr));
        const base = i * per + (2 * layer + @as(usize, @intFromBool(v))) * rr * width;
        return self.ring[base + (row * kv + kvh) * 128 ..][0..128];
    }

    /// Weights exp(s - M) within (1 + d)^(3 + chunks) (p, warp, chunk and each rescale's __expf) and exp(score range).
    pub fn solve(self: *const Attn, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const most = self.shape.k + block;
        const buf = try a.alloc(Iv, 130 * most);
        defer a.free(buf);
        const sc = buf[0..most];
        const w = buf[most .. 2 * most];
        const t = buf[2 * most ..]; // [128][most]: each column's terms
        const scratch = try a.alloc(f64, most);
        defer a.free(scratch);
        var terms: [128]Iv = undefined;
        const scale = Iv.of(128).rsqrt();
        for (self.segs, 0..) |sg, i| for (0..block) |j| {
            const m = i * block + j;
            const first = std.math.clamp(sg.position + @as(i64, @intCast(j)) - @as(i64, @intCast(self.window)) + 1, sg.begin, sg.position);
            const rk: usize = @intCast(sg.position - first);
            const nk = rk + block;
            for (0..heads) |h| {
                const q = self.q[(m * heads + h) * 128 ..][0..128];
                var top = -std.math.inf(f64);
                var low = std.math.inf(f64);
                var rho: f64 = 0;
                for (sc[0..nk], 0..) |*x, e| {
                    for (&terms, q, self.key(i, first, rk, e, h, false)) |*tm, qv, kv_| tm.* = Iv.of(ref.bf(qv) * ref.bf(kv_));
                    x.* = s.sum(&terms, 128, scratch).mul(scale);
                    if (x.m > top) top = x.m;
                    if (x.m < low) low = x.m;
                    rho = @max(rho, x.m - x.lo, x.hi - x.m);
                }
                const range = top - low + 2 * rho;
                const d = if (range < 0x1p-60) 0 else (2 + @floor(1.2 * range)) * 0x1p-23 + (range + 3) * 0x1p-24;
                const dw = std.math.pow(f64, 1 + d, 3 + chunks) - 1;
                for (w[0..nk], sc[0..nk]) |*x, y| {
                    const p = @exp(y.m - top);
                    const de = (1 + dw) * @exp(@max(y.m - y.lo, y.hi - y.m)) - 1;
                    x.* = .{ .m = p, .lo = p * (1 - de), .hi = p * (1 + de) };
                }
                const den = s.sum(w[0..nk], attnDepth(self.shape), scratch);
                for (w[0..nk], 0..) |y, e| for (self.key(i, first, rk, e, h, true), 0..) |vh, dd| {
                    const v = ref.bf(vh);
                    t[dd * most + e] = .{ .m = y.m * v, .lo = @min(y.lo * v, y.hi * v), .hi = @max(y.lo * v, y.hi * v) };
                };
                for (0..128) |dd| out[(m * heads + h) * 128 + dd] = s.sum(t[dd * most ..][0..nk], attnDepth(self.shape), scratch).div(den).bf();
            }
        };
    }
};
