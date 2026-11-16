//! The glm_glue image's absorb_q and expand_v (256-wide heads on the 512-wide latent) under the contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const lw = 512; // the latent width
const d = 256; // a q head's nope columns and a v head's width
const rows_rb = 64; // cuda_latent.prompt_rows and prompt_rb: rows from 64 on go 64 a block

pub const Kind = enum { absorb, expand };

/// next_head: a control red at gate 3, each head launched with the next head's weights.
pub const Entry = struct { name: []const u8, kind: Kind, next_head: bool = false };

pub const entries = [_]Entry{
    .{ .name = "cuda glm_glue tf_glm_absorb_q (q . wk into the 512-wide latent)", .kind = .absorb },
    .{ .name = "cuda glm_glue tf_glm_expand_v (latent . wv into 256-wide v heads)", .kind = .expand },
    .{ .name = "control: glm_glue tf_glm_absorb_q reads each head's wk from the next head", .kind = .absorb, .next_head = true },
    .{ .name = "control: glm_glue tf_glm_expand_v reads each head's wv from the next head", .kind = .expand, .next_head = true },
};

/// absorb: an 8-step fma chain over rep, 3 tree levels over k0's low bits, 2 across the 4 warps (both row paths).
fn depthAbsorb(_: c.Shape) usize {
    return 8 + 3 + 2;
}

/// expand: an 8-step fma chain, 5 lane levels (the prompt path's 3-level tree and 2 shuffles), then the two halves.
fn depthExpand(_: c.Shape) usize {
    return 8 + 5 + 1;
}

// decode windows (rows 1, 8, 32: main's loops) and prompt blocks (64, 70, 200: the lane-free sums), TP=2's 32 heads
const absorb_shapes = [_]c.Shape{ .{ .rows = 1, .n = 32, .k = d }, .{ .rows = 8, .n = 32, .k = d }, .{ .rows = 32, .n = 16, .k = d }, .{ .rows = 64, .n = 32, .k = d }, .{ .rows = 70, .n = 16, .k = d }, .{ .rows = 200, .n = 16, .k = d } };
const expand_shapes = [_]c.Shape{ .{ .rows = 1, .n = 32, .k = lw }, .{ .rows = 8, .n = 32, .k = lw }, .{ .rows = 32, .n = 16, .k = lw }, .{ .rows = 64, .n = 32, .k = lw }, .{ .rows = 70, .n = 16, .k = lw }, .{ .rows = 200, .n = 16, .k = lw } };

pub fn spec(e: Entry) c.Spec {
    const no: c.Tile = .{};
    const rows_edge: c.Tile = .{ .size = rows_rb };
    return switch (e.kind) {
        .absorb => .{ .name = e.name, .accum = .f32, .depth = depthAbsorb, .out = .bf16, .tiles = .{ rows_edge, no, no }, .served = &absorb_shapes },
        .expand => .{ .name = e.name, .accum = .f32, .depth = depthExpand, .out = .bf16, .tiles = .{ rows_edge, no, no }, .served = &expand_shapes },
    };
}

fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

/// One GLM latent projection call: x rows per head (k wide), w per head, out per head.
pub const Proj = struct {
    rig: rig_.Rig,
    entry: Entry,
    f: cuda.Function,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u16 = &.{},
    xd: u64 = 0,
    wd: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Proj) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Proj) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.w);
        self.x = &.{};
        self.w = &.{};
    }

    pub fn kernel(self: *Proj) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.next_head) .shapes else null };
    }

    const vt = g.hooks(Proj);

    /// Output elements a (row, head): absorb the 512 latent columns, expand the 256 v columns.
    fn width(self: *const Proj) usize {
        return if (self.entry.kind == .absorb) lw else d;
    }

    pub fn count(self: *const Proj) usize {
        return self.shape.rows * self.shape.n * self.width();
    }

    pub fn prepare(self: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        if (shape.k != (if (self.entry.kind == .absorb) @as(usize, d) else lw)) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const heads = shape.n;
        const k = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try a.alloc(u16, shape.rows * heads * k);
        for (self.x) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
        self.w = try a.alloc(u16, heads * d * lw); // wk [heads][d][lw], wv [heads][d][lw]
        for (self.w) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.125);
        switch (fill) {
            .random => {},
            .nonfinite => {
                // a row each: NaN, +Inf, both infinities, in one head's reduced inputs
                const planted = [_]struct { usize, usize, f32 }{ .{ 0, k / 3, std.math.nan(f32) }, .{ 1, k / 2, std.math.inf(f32) }, .{ 2, 0, std.math.inf(f32) }, .{ 2, k - 1, -std.math.inf(f32) } };
                for (planted) |p| if (p[0] < shape.rows) {
                    self.x[(p[0] * heads + 1) * k + p[1]] = c.toBf16(p[2]);
                };
            },
            .stress => self.stress(rng),
        }
        self.xd = try self.rig.input(std.mem.sliceAsBytes(self.x));
        if (self.entry.next_head) {
            const moved = try a.alloc(u16, self.w.len);
            defer a.free(moved);
            const per = self.w.len / heads;
            for (0..heads) |hd| @memcpy(moved[hd * per ..][0..per], self.w[(hd + 1) % heads * per ..][0..per]);
            self.wd = try self.rig.input(std.mem.sliceAsBytes(moved));
        } else self.wd = try self.rig.input(std.mem.sliceAsBytes(self.w));
        self.out = try self.rig.output(self.count() * 2);
        return true;
    }

    /// Gate 4, pocket's scheme: weights 1|2 mirrored by halves of k, inputs mirrored with a small remainder.
    fn stress(self: *Proj, rng: std.Random) void {
        const k = self.shape.k;
        const h = k / 2;
        const heads = self.shape.n;
        for (0..heads) |hd| for (0..h) |i| for (0..self.width()) |o| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 2));
            self.w[wAt(self.entry.kind, hd, i, o)] = c.toBf16(v);
            self.w[wAt(self.entry.kind, hd, h + i, o)] = c.toBf16(v);
        };
        // x = 512 x odd (terms past fp16's range), the second half -x, the first eight of it -x + 512: small sums
        const odd: u32 = if (self.entry.kind == .absorb) 32 else 16; // every partial sum under 2^24
        for (0..self.shape.rows) |r| for (0..heads) |hd| for (0..h) |i| {
            const v: f32 = @floatFromInt(512 * (1 + 2 * rng.uintLessThan(u32, odd)));
            const at = (r * heads + hd) * k;
            self.x[at + i] = c.toBf16(v);
            self.x[at + h + i] = c.toBf16(if (i < 8) 512 - v else -v);
        };
    }

    /// Weight of reduced index i and output column o in head hd: absorb wk [hd*d + i][o], expand wv [hd*d + o][i].
    fn wAt(kind: Kind, hd: usize, i: usize, o: usize) usize {
        return if (kind == .absorb) (hd * d + i) * lw + o else (hd * d + o) * lw + i;
    }

    /// As cuda_latent.rowBlocks launches it: rb = 64 from 64 rows on, else every row in one block.
    pub fn run(self: *Proj) !void {
        const s = self.shape;
        const rb = if (s.rows >= rows_rb) rows_rb else s.rows;
        var a: cuda.Args = .{};
        inline for (.{ self.xd, self.wd, self.out, @as(i64, @intCast(s.rows)), @as(i64, @intCast(s.n)), @as(i64, @intCast(rb)) }) |v| a.add(v);
        const cols: u32 = if (self.entry.kind == .absorb) lw / 32 else d / 16;
        const z: u32 = @intCast((s.rows + rb - 1) / rb);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(s.n), .y = cols, .z = z }, .block = .{ .x = 128 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    /// The products output o sums, in order.
    fn terms(self: *const Proj, o: usize, t: []f64) []f64 {
        const col = o % self.width();
        const rh = o / self.width();
        const hd = rh % self.shape.n;
        const k = self.shape.k;
        for (t[0..k], 0..) |*v, i| v.* = bf(self.x[rh * k + i]) * bf(self.w[wAt(self.entry.kind, hd, i, col)]);
        return t[0..k];
    }

    pub fn reference(self: *Proj, want: []f64, mag: []f64) !void {
        var t: [lw]f64 = undefined;
        for (want, mag, 0..) |*w, *m, o| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (self.terms(o, &t)) |v| {
                sum += v;
                abs += @abs(v);
            }
            w.* = sum;
            m.* = abs;
        }
    }

    pub fn simulate(self: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        var t: [lw]f64 = undefined;
        for (out, 0..) |*v, o| v.* = contract.rows.sumIn(acc, order, self.terms(o, &t));
    }
};
