//! The mla image's absorb_q and expand_v (128-wide heads) under the contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const lw = 512; // the latent width
const qs = 192; // a q head: 128 nope columns, then 64 rope
const qaw = 576; // an absorbed q row: the latent, then the roped columns (left as they were)
const rows_rb = 64; // cuda_latent.prompt_rows and prompt_rb: rows from 64 on go 64 a block

pub const Kind = enum { absorb, expand };

/// next_head: a control red at gate 3, each head launched with the next head's weights.
pub const Entry = struct { name: []const u8, kind: Kind, next_head: bool = false };

pub const entries = [_]Entry{
    .{ .name = "cuda mla tf_mla_absorb_q_128 (q_nope . wk into the latent)", .kind = .absorb },
    .{ .name = "cuda mla tf_mla_expand_v_128 (latent . wv into 128-wide v heads)", .kind = .expand },
    .{ .name = "control: mla tf_mla_absorb_q_128 reads each head's wk from the next head", .kind = .absorb, .next_head = true },
    .{ .name = "control: mla tf_mla_expand_v_128 reads each head's wv from the next head", .kind = .expand, .next_head = true },
};

/// absorb: four fma steps a thread, three butterfly steps, four warps' partials, two more steps.
fn depthAbsorb(_: c.Shape) usize {
    return 12;
}

/// expand: eight fma steps a thread, the warp's five steps, then the two warps.
fn depthExpand(_: c.Shape) usize {
    return 16;
}

const absorb_shapes = [_]c.Shape{ .{ .rows = 1, .n = 32, .k = 128 }, .{ .rows = 8, .n = 64, .k = 128 }, .{ .rows = 70, .n = 32, .k = 128 } };
const expand_shapes = [_]c.Shape{ .{ .rows = 1, .n = 32, .k = lw }, .{ .rows = 8, .n = 64, .k = lw }, .{ .rows = 70, .n = 32, .k = lw } };

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

/// One MLA projection call: x rows per head, w per head, out per head.
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

    /// Output elements a (row, head): absorb writes 512 of each 576-wide row (the rest stays 0), expand 128.
    fn width(self: *const Proj) usize {
        return if (self.entry.kind == .absorb) qaw else 128;
    }

    fn xWidth(self: *const Proj) usize {
        return if (self.entry.kind == .absorb) qs else lw;
    }

    pub fn count(self: *const Proj) usize {
        return self.shape.rows * self.shape.n * self.width();
    }

    pub fn prepare(self: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        if (shape.k != (if (self.entry.kind == .absorb) @as(usize, 128) else lw)) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const heads = shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try a.alloc(u16, shape.rows * heads * self.xWidth());
        for (self.x) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
        const wn = heads * 128 * lw; // wk [heads][128][lw], wv [heads][128][lw]
        self.w = try a.alloc(u16, wn);
        for (self.w) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.125);
        switch (fill) {
            .random => {},
            .nonfinite => {
                // a row each: NaN, +Inf, both infinities, in one head's reduced inputs
                const k = shape.k;
                const planted = [_]struct { usize, usize, f32 }{ .{ 0, k / 3, std.math.nan(f32) }, .{ 1, k / 2, std.math.inf(f32) }, .{ 2, 0, std.math.inf(f32) }, .{ 2, k - 1, -std.math.inf(f32) } };
                for (planted) |p| if (p[0] < shape.rows) {
                    self.x[(p[0] * heads + 1) * self.xWidth() + p[1]] = c.toBf16(p[2]);
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
        for (0..heads) |hd| for (0..h) |i| for (0..128) |o| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 2));
            self.w[self.wAt(hd, i, o)] = c.toBf16(v);
            self.w[self.wAt(hd, h + i, o)] = c.toBf16(v);
        };
        if (self.entry.kind == .absorb) for (0..heads) |hd| for (0..h) |i| for (128..lw) |o| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 2));
            self.w[self.wAt(hd, i, o)] = c.toBf16(v);
            self.w[self.wAt(hd, h + i, o)] = c.toBf16(v);
        };
        // x = 512 x odd (terms past fp16's range), the second half -x, the first eight of it -x + 512: small sums
        const odd: u32 = if (self.entry.kind == .absorb) 64 else 16; // every partial sum under 2^24
        for (0..self.shape.rows) |r| for (0..heads) |hd| for (0..h) |i| {
            const v: f32 = @floatFromInt(512 * (1 + 2 * rng.uintLessThan(u32, odd)));
            const at = (r * heads + hd) * self.xWidth();
            self.x[at + i] = c.toBf16(v);
            self.x[at + h + i] = c.toBf16(if (i < 8) 512 - v else -v);
        };
    }

    /// Weight of reduced index i and output column o in head hd: absorb wk [hd*128 + i][o], expand wv [hd*128 + o][i].
    fn wAt(self: *const Proj, hd: usize, i: usize, o: usize) usize {
        return if (self.entry.kind == .absorb) (hd * 128 + i) * lw + o else (hd * 128 + o) * lw + i;
    }

    pub fn run(self: *Proj) !void {
        const s = self.shape;
        const rb = if (s.rows >= rows_rb) rows_rb else s.rows;
        try self.rig.zero(self.out, self.count() * 2); // absorb leaves the roped columns as they were: 0
        var a: cuda.Args = .{};
        inline for (.{ self.xd, self.wd, self.out, @as(i64, @intCast(s.rows)), @as(i64, @intCast(s.n)), @as(i64, @intCast(rb)) }) |v| a.add(v);
        if (self.entry.kind == .absorb) {
            a.add(@as(i64, qs));
            a.add(@as(i64, qaw));
        }
        const cols: u32 = if (self.entry.kind == .absorb) lw / 32 else 128 / 16;
        const z: u32 = @intCast((s.rows + rb - 1) / rb);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(s.n), .y = cols, .z = z }, .block = .{ .x = 128 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    /// The products output o sums, in order; null where the launch leaves it as it was (0).
    fn terms(self: *const Proj, o: usize, t: []f64) ?[]f64 {
        const heads = self.shape.n;
        const col = o % self.width();
        const rh = o / self.width();
        const hd = rh % heads;
        if (self.entry.kind == .absorb and col >= lw) return null;
        const k = self.shape.k;
        const xa = rh * self.xWidth();
        for (t[0..k], 0..) |*v, i| v.* = bf(self.x[xa + i]) * bf(self.w[self.wAt(hd, i, col)]);
        return t[0..k];
    }

    pub fn reference(self: *Proj, want: []f64, mag: []f64) !void {
        var t: [lw]f64 = undefined;
        for (want, mag, 0..) |*w, *m, o| {
            const ts = self.terms(o, &t) orelse {
                w.* = 0;
                m.* = 0;
                continue;
            };
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (ts) |v| {
                sum += v;
                abs += @abs(v);
            }
            w.* = sum;
            m.* = abs;
        }
    }

    pub fn simulate(self: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        var t: [lw]f64 = undefined;
        for (out, 0..) |*v, o| {
            const ts = self.terms(o, &t) orelse {
                v.* = 0;
                continue;
            };
            v.* = contract.rows.sumIn(acc, order, ts);
        }
    }
};
