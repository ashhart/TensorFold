//! Ripple's DFlash2 drafter prep on CUDA under the contract: its entries, its streams' plan and the Prep kernel.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rc = @import("ricursive_cuda");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig"); // the family-neutral Gated wrapper and fills

const c = core.contract.spec;
const g = core.contract.gates;
const df = rc.dflash2_kernels;
const Iv = ref.Iv;
const Sums = ref.Sums;
const Function = cuda.Function;

const heads = 80;
const kv_heads = 8;
const D = 128;
const kvw = kv_heads * D;
const eps = @import("contract_ricursive_dflash2.zig").eps;
const theta: f32 = 10000;
const sliding_ring = 4224;
const global_ring = 16512; // layer 4's, NoPE
const most_pos = 262144; // positions stay below it

/// Block rows (q, k and v of 80 query heads) or context rows (k and v only); a sliding layer's ring or the global one.
pub const Mode = enum { block, context };
pub const Layer = enum { sliding, global };

/// A prep control: one rounding point dropped (the output truncated for `out`), or one edit.
pub const PrepCtl = enum { norm_row, norm_out, cos, sin, rot_c, rot_s, out, nope_ropes, mask };

pub const PrepEntry = struct {
    name: []const u8,
    mode: Mode,
    layer: Layer,
    ctl: ?PrepCtl = null,

    /// The mask control holds the whole ring: its wrong slots stay in the buffer.
    fn full(e: PrepEntry) bool {
        return if (e.ctl) |x| x == .mask else false;
    }
};

/// NoPE stores the normed head as it is, so the rotation's points show only on sliding layers.
pub const prep_entries = [_]PrepEntry{
    .{ .name = "cuda ricursive_dflash2 tf_rcd_prep block, sliding (RoPE, 4,224-slot rings)", .mode = .block, .layer = .sliding },
    .{ .name = "cuda ricursive_dflash2 tf_rcd_prep block, global (NoPE, 16,512-slot rings)", .mode = .block, .layer = .global },
    .{ .name = "cuda ricursive_dflash2 tf_rcd_prep context, sliding (k and v only, RoPE)", .mode = .context, .layer = .sliding },
    .{ .name = "cuda ricursive_dflash2 tf_rcd_prep context, global (k and v only, NoPE)", .mode = .context, .layer = .global },
    .{ .name = "control: ricursive_dflash2 tf_rcd_prep drops bf16(x r)", .mode = .block, .layer = .sliding, .ctl = .norm_row },
    .{ .name = "control: ricursive_dflash2 tf_rcd_prep drops bf16(w bf16(x r))", .mode = .block, .layer = .sliding, .ctl = .norm_out },
    .{ .name = "control: ricursive_dflash2 tf_rcd_prep drops bf16(cos)", .mode = .block, .layer = .sliding, .ctl = .cos },
    .{ .name = "control: ricursive_dflash2 tf_rcd_prep drops bf16(sin)", .mode = .block, .layer = .sliding, .ctl = .sin },
    .{ .name = "control: ricursive_dflash2 tf_rcd_prep drops bf16(n c)", .mode = .block, .layer = .sliding, .ctl = .rot_c },
    .{ .name = "control: ricursive_dflash2 tf_rcd_prep drops bf16(rot s)", .mode = .block, .layer = .sliding, .ctl = .rot_s },
    .{ .name = "control: ricursive_dflash2 tf_rcd_prep truncates its output, not rounding it", .mode = .block, .layer = .sliding, .ctl = .out },
    .{ .name = "control: ricursive_dflash2 tf_rcd_prep rotates the NoPE layer", .mode = .block, .layer = .global, .ctl = .nope_ropes },
    .{ .name = "control: ricursive_dflash2 tf_rcd_prep wraps its ring by a power-of-two mask", .mode = .block, .layer = .sliding, .ctl = .mask },
};

fn ringOf(l: Layer) usize {
    return if (l == .sliding) sliding_ring else global_ring;
}

/// Rows x streams x stream 0's first position: wraps of the ring, a row on its last slot, deep and last positions.
fn blockServed(comptime ring: usize) [5]c.Shape {
    return .{
        .{ .rows = 16, .n = 1, .k = ring - 3 }, // one stream's block across its ring's wrap
        .{ .rows = 1, .n = 1, .k = ring - 1 }, // a row on the ring's last slot
        .{ .rows = 128, .n = 8, .k = ring - 3 }, // 8 streams' blocks across their wraps, each a ring further on
        .{ .rows = 48, .n = 3, .k = 100_007 },
        .{ .rows = 32, .n = 2, .k = most_pos / ring * ring - 3 }, // the last wrap below 262,144, a stream a ring below it
    };
}

fn contextServed(comptime ring: usize) [4]c.Shape {
    return .{
        .{ .rows = 2048, .n = 1, .k = ring - 3 }, // a prompt chunk's context across the wrap
        .{ .rows = 5, .n = 1, .k = most_pos / ring * ring - 3 },
        .{ .rows = 333, .n = 3, .k = 1000 }, // uneven streams: 111 rows each
        .{ .rows = 64, .n = 4, .k = ring - 3 },
    };
}

const block_sliding = blockServed(sliding_ring);
const block_global = blockServed(global_ring);
const context_sliding = contextServed(sliding_ring);
const context_global = contextServed(global_ring);

/// A lane's 4 fma steps, the warp's 5 shuffles.
fn prepDepth(_: c.Shape) usize {
    return 4 + 5;
}

pub fn prepSpec(e: PrepEntry) c.Spec {
    const served: []const c.Shape = switch (e.mode) {
        .block => if (e.layer == .sliding) &block_sliding else &block_global,
        .context => if (e.layer == .sliding) &context_sliding else &context_global,
    };
    return .{ .name = e.name, .accum = .f32, .depth = prepDepth, .out = .bf16, .served = served, .sums = .{} };
}

/// A call's streams: n row runs, stream s k + s rings on (k - s near the end), rings `spacing` apart in one buffer.
pub const Plan = struct {
    rows: usize,
    streams: usize,
    k: usize,
    ring: usize,
    up: bool,
    spacing: usize,
    first: usize,
    held: usize,

    /// Null when the streams' slots would overlap or a position reach 262,144; `full`: hold the whole ring.
    pub fn of(shape: c.Shape, ring: usize, full: bool) ?Plan {
        if (shape.rows == 0 or shape.n == 0) return null;
        const streams = @min(shape.n, shape.rows);
        const most = (shape.rows + streams - 1) / streams;
        const spacing = std.mem.alignForward(usize, most, 32);
        if (spacing * (streams - 1) + most >= ring) return null;
        const reach = (streams - 1) * ring;
        const up = shape.k + reach + most <= most_pos;
        if (!up and (shape.k < reach or shape.k + most > most_pos)) return null;
        const wraps = shape.k % ring + most > ring;
        const whole = wraps or full;
        return .{
            .rows = shape.rows,
            .streams = streams,
            .k = shape.k,
            .ring = ring,
            .up = up,
            .spacing = spacing,
            .first = if (whole) 0 else shape.k % ring,
            .held = spacing * (streams - 1) + if (whole) ring else most,
        };
    }

    pub fn len(p: Plan, s: usize) usize {
        return p.rows / p.streams + @intFromBool(s < p.rows % p.streams);
    }

    pub fn begin(p: Plan, s: usize) usize {
        return s * (p.rows / p.streams) + @min(s, p.rows % p.streams);
    }

    pub fn pos(p: Plan, s: usize) usize {
        return if (p.up) p.k + s * p.ring else p.k - s * p.ring;
    }

    /// Row j of stream s: its index among the buffer's held slots.
    pub fn slot(p: Plan, s: usize, j: usize) usize {
        return s * p.spacing + (p.pos(s) + j) % p.ring - p.first;
    }
};

/// tf_rcd_prep: q [rows][80][128] (block rows), then the K and V buffers' held slots (unreached ones zero).
pub const Prep = struct {
    f: Function,
    entry: PrepEntry,
    shape: c.Shape = undefined,
    plan: Plan = undefined,
    at: df.Layout = undefined,
    inv: [D / 2]f32 = undefined,
    x: []u16 = &.{},
    qw: []u16 = &.{},
    kw: []u16 = &.{},
    segs: []df.Segment = &.{},
    bufs: [2]u64 = @splat(0), // the K and V buffers as allocated
    p: df.Prep = undefined,

    pub const sums = {};

    pub fn spec(self: *const Prep) c.Spec {
        return prepSpec(self.entry);
    }

    pub fn control(self: *const Prep) ?g.Gate {
        return if (self.entry.ctl != null) .shapes else null;
    }

    pub fn free(self: *Prep, a: std.mem.Allocator) void {
        for ([_]*[]u16{ &self.x, &self.qw, &self.kw }) |x| {
            a.free(x.*);
            x.* = &.{};
        }
        a.free(self.segs);
        self.segs = &.{};
    }

    /// Heads in [-3, 3) (one in 9 a tenth the size), weights in [-0.5, 0.5); stress: integers in [-32, 32].
    pub fn prepare(self: *Prep, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        const ring = ringOf(self.entry.layer);
        const plan = Plan.of(shape, ring, self.entry.full()) orelse return false;
        self.free(a);
        self.shape = shape;
        self.plan = plan;
        self.at = if (self.entry.mode == .block) df.Layout.block(heads, kv_heads) else df.Layout.context(kv_heads);
        self.inv = df.inverseFrequencies(theta);
        const rows = shape.rows;
        const at = self.at;
        self.x = try a.alloc(u16, rows * at.stride);
        for (self.x, 0..) |*e, i| {
            e.* = c.toBf16(if (fill == .stress) ref.int(rng, -32, 32) else (rng.float(f32) * 6 - 3) * (if (i % at.stride / D % 9 == 4) @as(f32, 0.1) else 1));
        }
        if (fill == .nonfinite and rows >= 4) {
            const lead = if (at.heads > 0) at.q_off else at.k_off;
            self.x[lead + 3] = c.toBf16(ref.nan); // row 0's first head: its every output NaN
            self.x[at.stride + at.k_off + D + 9] = c.toBf16(ref.inf); // row 1's second k head: r = 0, so dim 9 (and 73) NaN
            self.x[2 * at.stride + at.k_off + 70] = c.toBf16(-ref.inf); // row 2's first k head, in the second half
            self.x[3 * at.stride + at.v_off + 2 * D + 5] = c.toBf16(ref.inf); // row 3's third v head: copied as it is
        }
        self.qw = try ref.bfs(a, rng, D, -0.5, 0.5);
        self.kw = try ref.bfs(a, rng, D, -0.5, 0.5);
        const q = if (at.heads > 0) try rig.output(rows * at.heads * D * 2) else 0; // gathered in this order: q, k, v
        const held = plan.held * kvw * 2;
        self.bufs = .{ try rig.output(held), try rig.output(held) };
        self.segs = try a.alloc(df.Segment, plan.streams);
        for (self.segs, 0..) |*sg, s| {
            const off = (s * plan.spacing) * kvw * 2;
            const back = plan.first * kvw * 2; // the rings' slot 0, so slot `first` lands on the buffer's first row
            sg.* = .{ .k = self.bufs[0] + off - back, .v = self.bufs[1] + off - back, .begin = @intCast(plan.begin(s)), .pos = @intCast(plan.pos(s)) };
        }
        try df.checkSegments(self.segs, rows, ring);
        self.p = .{
            .x = try ref.input(rig, self.x),
            .qw = try ref.input(rig, self.qw),
            .kw = try ref.input(rig, self.kw),
            .inv = try ref.input(rig, self.inv[0..]),
            .segs = try ref.input(rig, self.segs),
            .streams = plan.streams,
            .q = q,
            .rows = rows,
            .kv_heads = kv_heads,
            .at = at,
            .ring = ring,
            .rope = self.entry.layer == .sliding,
            .eps = eps,
        };
        return true;
    }

    pub fn run(self: *Prep, rig: *rig_.Rig) !void {
        for (self.bufs) |o| try rig.zero(o, self.plan.held * kvw * 2); // slots no row reaches: zero, as in the reference
        try df.prepBy(self.f, rig.s, self.p);
    }

    pub fn count(self: *const Prep) usize {
        return self.shape.rows * self.at.heads * D + 2 * self.plan.held * kvw;
    }

    /// One head's 128 outputs: bf16(w bf16(x r)), then (rope) bf16(bf16(n c) + bf16(rot s)) at the row's bf16 cos, sin.
    fn head(s: Sums, x: []const u16, w: []const u16, rope: bool, cs: []const Iv, sn: []const Iv, out: []Iv, scratch: []f64) void {
        var sq: [D]Iv = undefined;
        var n: [D]Iv = undefined;
        for (&sq, x) |*q, v| q.* = Iv.of(ref.bf(v) * ref.bf(v)); // the fma's product, exact
        const r = s.sum(&sq, prepDepth(undefined), scratch).div(Iv.of(D)).add(Iv.of(eps)).rsqrt();
        for (&n, x, w) |*o, v, wv| o.* = Iv.of(ref.bf(wv)).mul(Iv.of(ref.bf(v)).mul(r).bf()).bf();
        for (out, 0..) |*o, d| {
            if (!rope) {
                o.* = n[d];
                continue;
            }
            const rot = if (d < D / 2) n[d + D / 2].neg() else n[d - D / 2];
            o.* = n[d].mul(cs[d % (D / 2)]).bf().add(rot.mul(sn[d % (D / 2)]).bf()).bf();
        }
    }

    /// Every rounding point kept and the ring wrapped by its modulus, whatever the entry: a control must leave these.
    pub fn solve(self: *const Prep, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const plan = self.plan;
        const at = self.at;
        const qn = self.shape.rows * at.heads * D;
        const ks = out[qn..][0 .. plan.held * kvw];
        const vs = out[qn + plan.held * kvw ..];
        @memset(ks, Iv.of(0));
        @memset(vs, Iv.of(0));
        const rope = self.entry.layer == .sliding;
        var scratch: [D]f64 = undefined;
        var cs: [D / 2]Iv = undefined;
        var sn: [D / 2]Iv = undefined;
        for (0..plan.streams) |st| for (0..plan.len(st)) |j| {
            const r = plan.begin(st) + j;
            const position: f32 = @floatFromInt(plan.pos(st) + j);
            for (&cs, &sn, self.inv) |*co, *si, f| {
                const angle = Iv.of(position * f); // the kernel's fp32 angle, exactly
                co.* = angle.trig(false).bf();
                si.* = angle.trig(true).bf();
            }
            const t = plan.slot(st, j);
            const xr = self.x[r * at.stride ..][0..at.stride];
            for (0..at.heads) |h| head(s, xr[at.q_off + h * D ..][0..D], self.qw, rope, &cs, &sn, out[(r * at.heads + h) * D ..][0..D], &scratch);
            for (0..kv_heads) |h| head(s, xr[at.k_off + h * D ..][0..D], self.kw, rope, &cs, &sn, ks[t * kvw + h * D ..][0..D], &scratch);
            for (vs[t * kvw ..][0..kvw], xr[at.v_off..][0..kvw]) |*o, v| o.* = Iv.of(ref.bf(v));
        };
    }
};

test "dflash2 prep: every served shape and the gates' small forms plan, stream slots apart and positions in range" {
    const t = std.testing;
    for (prep_entries) |e| {
        const spec = prepSpec(e);
        const ring = ringOf(e.layer);
        var shapes: std.ArrayList(c.Shape) = .empty;
        defer shapes.deinit(t.allocator);
        try shapes.appendSlice(t.allocator, spec.served);
        for ([_][2]usize{ .{ 5, 8 }, .{ 1, 64 } }) |lim| {
            var x = spec.served[0];
            x.rows = std.math.clamp(x.rows, lim[0], lim[1]);
            try shapes.append(t.allocator, x);
        }
        var deep = spec.served[0];
        for (spec.served) |x| if (x.k > deep.k) {
            deep = x;
        };
        deep.rows = @min(deep.rows, 4);
        try shapes.append(t.allocator, deep);
        for (shapes.items) |shape| {
            const p = Plan.of(shape, ring, e.full()) orelse return error.TestUnexpectedResult;
            const seen = try t.allocator.alloc(bool, p.held);
            defer t.allocator.free(seen);
            @memset(seen, false);
            var rows: usize = 0;
            for (0..p.streams) |s| {
                try t.expectEqual(rows, p.begin(s));
                for (0..p.len(s)) |j| {
                    try t.expect(p.pos(s) + j < most_pos);
                    const at = p.slot(s, j);
                    try t.expect(at < p.held and !seen[at]);
                    seen[at] = true;
                }
                rows += p.len(s);
            }
            try t.expectEqual(shape.rows, rows);
            if (e.full()) try t.expectEqual(@as(usize, 0), p.first);
        }
    }
}
