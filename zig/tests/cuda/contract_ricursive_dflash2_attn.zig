//! Ripple's DFlash2 draft attention and top-32 on CUDA under the contract, with controls, and a call's time.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const da = @import("ricursive_cuda").dflash2_attn;
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const bfr = core.bf16_round;
const sumIn = core.contract.rows.sumIn;
const Iv = ref.Iv;
const Sums = ref.Sums;
const unit = ref.unit;

const D = da.head_dim;
const heads = da.heads;
const kv_heads = da.kv_heads;
const group = heads / kv_heads;
const block = da.block;
/// Outputs a block row: every head's 128 values (the shape's n).
const width = heads * D;

fn bf32(h: u16) f32 {
    return c.fromBf16(h);
}

// --- draft attention ------------------------------------------------------------------------------------------------

/// A layer kind: sliding (window 4,096, ring 4,224) or global NoPE (16,384, ring 16,512).
pub const Form = enum { sliding, global };

fn spanOf(f: Form) usize {
    return switch (f) {
        .sliding => da.sliding_span,
        .global => da.global_span,
    };
}

/// A registration: the served launches, or a control with one of them replaced by a planted fault of the image.
pub const AttnEntry = struct { name: []const u8, form: Form, part: [:0]const u8 = "tf_rcd_attn_part", merge: [:0]const u8 = "tf_rcd_attn_merge" };

pub const attn_entries = [_]AttnEntry{
    .{ .name = "ricursive dflash2 draft attention, sliding window (4,096 keys, ring 4,224)", .form = .sliding },
    .{ .name = "ricursive dflash2 draft attention, global window (16,384 keys, ring 16,512)", .form = .global },
    .{ .name = "control: ricursive dflash2 attention masks the block causally", .form = .sliding, .part = "tf_rcd_ctl_attn_causal" },
    .{ .name = "control: ricursive dflash2 attention's window starts one key low", .form = .sliding, .part = "tf_rcd_ctl_attn_floor" },
    .{ .name = "control: ricursive dflash2 attention leaves the block's own rows out", .form = .sliding, .part = "tf_rcd_ctl_attn_no_block" },
    .{ .name = "control: ricursive dflash2 attention's merge truncates to bf16, not rounding it", .form = .sliding, .merge = "tf_rcd_ctl_attn_merge_trunc" },
};

fn isControl(e: AttnEntry) bool {
    return !std.mem.eql(u8, e.part, "tf_rcd_attn_part") or !std.mem.eql(u8, e.merge, "tf_rcd_attn_merge");
}

/// Rows: streams; k: stream 0's end (anchor k - 16). 8 streams past a wrapped ring; 1: first block, window edges, wrap.
const sliding_served = [_]c.Shape{
    .{ .rows = 8, .n = width, .k = 20016 }, .{ .rows = 1, .n = width, .k = 16 },   .{ .rows = 1, .n = width, .k = 1000 },
    .{ .rows = 1, .n = width, .k = 4112 },  .{ .rows = 1, .n = width, .k = 9999 }, .{ .rows = 8, .n = width, .k = 5555 },
};
const global_served = [_]c.Shape{ .{ .rows = 8, .n = width, .k = 20016 }, .{ .rows = 1, .n = width, .k = 50016 }, .{ .rows = 1, .n = width, .k = 16 } };

/// A score's dot: a product and 3 fmas a lane, then 5 butterfly adds (10, with room).
fn attnDepth(_: c.Shape) usize {
    return 10 + 128 + da.chunksFor(da.global_span);
}

fn attnSpec(e: AttnEntry) c.Spec {
    return .{ .name = e.name, .accum = .f32, .depth = attnDepth, .out = .bf16, .served = switch (e.form) {
        .sliding => &sliding_served,
        .global => &global_served,
    }, .sums = .{} };
}

/// Stream i's block at anchor(0) >> i plus 37 i; odd streams see from a floor above their window's start.
fn anchorOf(k: usize, i: usize) usize {
    return ((k - block) >> @intCast(i)) + 37 * i;
}

fn fromOf(anchor: usize, span: usize, i: usize) usize {
    if (i % 2 == 0) return 0;
    return if (anchor > span / 2) anchor - span / 2 else anchor / 2;
}

/// Row r's keys: max(p + 1 - span, from) to anchor + 16, p = anchor + r.
fn keysOf(s: da.Segment, r: usize) struct { lo: usize, hi: usize } {
    const lo = @max(@as(i64, s.anchor) + @as(i64, @intCast(r)) + 1 - s.span, s.from);
    return .{ .lo = @intCast(@max(lo, 0)), .hi = @intCast(s.anchor + block) };
}

/// One layer's call: each stream's ring filled whole (a key read outside its window changes the answer), segs, q, out.
pub const Attn = struct {
    entry: AttnEntry,
    part: cuda.Function,
    merge: cuda.Function,
    shape: c.Shape = undefined,
    span: usize = 0,
    ring_rows: usize = 0,
    segs: []da.Segment = &.{},
    q: []u16 = &.{},
    rings: []u16 = &.{}, // stream s's ring at s ring_len: K then V
    call: da.Attn = undefined,

    pub const sums = {};

    pub fn spec(self: *const Attn) c.Spec {
        return attnSpec(self.entry);
    }

    pub fn control(self: *const Attn) ?g.Gate {
        return if (isControl(self.entry)) .shapes else null;
    }

    pub fn free(self: *Attn, a: std.mem.Allocator) void {
        a.free(self.segs);
        self.segs = &.{};
        a.free(self.q);
        self.q = &.{};
        a.free(self.rings);
        self.rings = &.{};
    }

    fn ringLen(self: *const Attn) usize {
        return 2 * self.ring_rows * kv_heads * D;
    }

    fn ring(self: *const Attn, s: usize) []const u16 {
        return self.rings[s * self.ringLen() ..][0..self.ringLen()];
    }

    pub fn prepare(self: *Attn, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.rows > 64 or shape.n != width or shape.k < block) return false;
        self.free(a);
        self.shape = shape;
        self.span = spanOf(self.entry.form);
        self.ring_rows = da.ringRows(self.span);
        const streams = shape.rows;
        self.segs = try a.alloc(da.Segment, streams);
        for (self.segs, 0..) |*s, i| {
            const anchor = anchorOf(shape.k, i);
            s.* = .{ .ring = 0, .anchor = @intCast(anchor), .from = @intCast(fromOf(anchor, self.span, i)), .span = @intCast(self.span), .rows = @intCast(self.ring_rows) };
        }
        self.q = try a.alloc(u16, streams * block * width);
        self.rings = try a.alloc(u16, streams * self.ringLen());
        for (self.q) |*x| x.* = c.toBf16(rng.floatNorm(f32));
        const half = self.ring_rows * kv_heads * D; // K, then V
        for (0..streams) |s| {
            const r = self.rings[s * self.ringLen() ..][0..self.ringLen()];
            if (fill == .stress) {
                // every key the same (each kv head its own vector): every weight exactly 1, the sums integers
                const kv: [kv_heads * D]u16 = blk: {
                    var v: [kv_heads * D]u16 = undefined;
                    for (&v) |*x| x.* = c.toBf16(rng.floatNorm(f32));
                    break :blk v;
                };
                for (0..self.ring_rows) |row| @memcpy(r[row * kv_heads * D ..][0 .. kv_heads * D], &kv);
                stressValues(r[half..], self.segs[s], self.ring_rows, rng);
            } else {
                for (r[0..half]) |*x| x.* = c.toBf16(rng.floatNorm(f32));
                // V: half the dims about 1 (outputs well away from 0), half about 0
                for (r[half..], 0..) |*x, i| x.* = c.toBf16(if (i % D < D / 2) 1 + 0.25 * rng.floatNorm(f32) else rng.floatNorm(f32));
            }
        }
        if (fill == .nonfinite and streams >= 3) {
            const at = struct {
                fn key(self_: *Attn, s: usize, pos: usize, kv: usize, d: usize, v: bool) *u16 {
                    const row = pos % self_.ring_rows;
                    const off = (if (v) self_.ring_rows * kv_heads * D else 0) + (row * kv_heads + kv) * D + d;
                    return &self_.rings[s * self_.ringLen() + off];
                }
            };
            // stream 0: a NaN key in the block (heads 0-9 NaN)
            at.key(self, 0, @as(usize, @intCast(self.segs[0].anchor)) + 3, 0, 11, false).* = c.toBf16(ref.nan);
            // stream 1: a -inf value (heads 30-39, dim 7, -inf)
            at.key(self, 1, @as(usize, @intCast(self.segs[1].anchor)) + 5, 3, 7, true).* = c.toBf16(-ref.inf);
            // stream 2: an infinite q (row 4, head 25: every score infinite, NaN)
            self.q[((2 * block + 4) * heads + 25) * D + 40] = c.toBf16(ref.inf);
        }
        for (self.segs, 0..) |*s, i| s.ring = try ref.input(rig, self.ring(i));
        const qd = try ref.input(rig, self.q);
        const segs = try ref.input(rig, self.segs);
        const out = try rig.output(streams * block * width * 2);
        self.call = .{ .q = qd, .segs = segs, .out = out, .pm = 0, .pl = 0, .po = 0, .streams = streams, .span = self.span };
        const sc = da.scratch(self.call);
        self.call.pm = try rig.scratch(sc.max);
        self.call.pl = try rig.scratch(sc.max);
        self.call.po = try rig.scratch(sc.values);
        return true;
    }

    /// Gate 4's values: odd integers to 255, then negated (+1 on 8): small exact sums, big partials, all exact in fp32.
    fn stressValues(v: []u16, s: da.Segment, ring_rows: usize, rng: std.Random) void {
        @memset(v, 0);
        const lo = keysOf(s, 0).lo;
        const n = keysOf(s, 0).hi - lo;
        for (0..kv_heads * D) |e| for (0..n) |j| {
            const at = ((lo + j) % ring_rows) * kv_heads * D + e;
            if (j < n / 2) {
                v[at] = c.toBf16(@floatFromInt(2 * rng.uintLessThan(u32, 128) + 1));
            } else if (j < 2 * (n / 2)) {
                const mirror = ((lo + j - n / 2) % ring_rows) * kv_heads * D + e;
                v[at] = c.toBf16((if (j - n / 2 < 8) @as(f32, 1) else 0) - bf32(v[mirror]));
            } else v[at] = c.toBf16(1);
        };
    }

    pub fn run(self: *Attn, rig: *rig_.Rig) !void {
        try da.attentionVia(self.part, self.merge, rig.s, self.call);
    }

    pub fn count(self: *Attn) usize {
        return self.shape.rows * block * width;
    }

    /// Each (stream, row, head): float64 scores of q scaled in fp32, softmax at the NaN-skipping max; ranges below.
    pub fn solve(self: *Attn, a: std.mem.Allocator, how: Sums, out: []Iv) !void {
        const most = self.span + block;
        const w = try a.alloc(f64, most);
        defer a.free(w);
        const slab = try a.alloc(f64, if (how.sim != null) D * most else 0);
        defer a.free(slab);
        const nb: f64 = @floatFromInt(da.chunksFor(self.span));
        const R = self.ring_rows;
        const vbase = R * kv_heads * D;
        for (self.segs, 0..) |seg, s| {
            const ring_ = self.ring(s);
            for (0..block) |r| {
                const keys = keysOf(seg, r);
                const nk = keys.hi - keys.lo;
                const row_out = out[(s * block + r) * width ..][0..width];
                // simulated (gate 4's stress, weights 1): each output the mean of its values summed in the accumulator
                if (how.sim) |sim| {
                    // the stress keys are one vector: a group's heads share their weights (all 1), so their outputs
                    for (0..kv_heads) |kv| {
                        for (keys.lo..keys.hi, 0..) |p, j| for (ring_[vbase + ((p % R) * kv_heads + kv) * D ..][0..D], 0..) |v, d| {
                            slab[d * most + j] = bf32(v);
                        };
                        for (0..D) |d| {
                            const m = sumIn(sim.acc, sim.order, slab[d * most ..][0..nk]) / @as(f64, @floatFromInt(nk));
                            for (0..group) |hh| row_out[(kv * group + hh) * D + d] = Iv.of(m);
                        }
                    }
                    continue;
                }
                for (0..heads) |h| {
                    const kv = h / group;
                    var qs: [D]f64 = undefined;
                    for (&qs, self.q[((s * block + r) * heads + h) * D ..][0..D]) |*x, b| x.* = da.scale * bf32(b);
                    var top: f64 = -std.math.inf(f64);
                    var low: f64 = std.math.inf(f64);
                    var amax: f64 = 0;
                    for (keys.lo..keys.hi, 0..) |p, j| {
                        const k = ring_[((p % R) * kv_heads + kv) * D ..][0..D];
                        var dot: f64 = 0;
                        var abs: f64 = 0;
                        for (qs, k) |x, kb| {
                            const t = x * bf32(kb);
                            dot += t;
                            abs += @abs(t);
                        }
                        w[j] = dot;
                        if (dot > top) top = dot; // fmaxf: a NaN score is never the max
                        if (dot < low) low = dot;
                        if (abs > amax) amax = abs;
                    }
                    var W: f64 = 0;
                    for (w[0..nk]) |*x| {
                        x.* = @exp(x.* - top);
                        W += x.*;
                    }
                    var num: [D]f64 = @splat(0);
                    var mag: [D]f64 = @splat(0);
                    var vmax: [D]f64 = @splat(0);
                    for (keys.lo..keys.hi, 0..) |p, j| for (ring_[vbase + ((p % R) * kv_heads + kv) * D ..][0..D], 0..) |vb, d| {
                        const v: f64 = bf32(vb);
                        num[d] += w[j] * v;
                        mag[d] += w[j] * @abs(v);
                        vmax[d] = @max(vmax[d], @abs(v));
                    };
                    // E: the scores' error, the dot's 10 roundings of its terms' size
                    const E = c.gamma(10, unit) * amax + 0x1p-50 * amax;
                    // eps: an exp's argument and 2 ulps, 33 deep (own exp, chunk rescales, merge factor)
                    const eps = @exp(unit * (top - low + 2 * E)) * (1 + 4 * unit) - 1;
                    // products 34 deep, 128 and merge sums, division: |N/W - m| <= 2 delta mag / (1 - delta)
                    const delta = @exp(E) * std.math.pow(f64, 1 + eps, 33) * std.math.pow(f64, 1 + unit, 34) * (1 + c.gamma(128, unit)) * (1 + c.gamma(nb, unit)) - 1;
                    for (row_out[h * D ..][0..D], num, mag, vmax) |*x, nm, mg, vm| {
                        const m = nm / W;
                        if (!std.math.isFinite(m)) {
                            x.* = Iv.of(m);
                            continue;
                        }
                        const dev = if (how.exact) 0 else 2 * delta * (mg / W) / (1 - delta);
                        const tiny = @as(f64, @floatFromInt(nk)) * 0x1p-140 * (1 + vm) / W;
                        const B = dev + unit * (@abs(m) + dev) + tiny;
                        x.* = .{ .m = m, .lo = bfr.bf16(m - B), .hi = bfr.bf16(m + B) };
                    }
                }
            }
        }
    }
};

// --- top-32 ---------------------------------------------------------------------------------------------------------

pub const TopEntry = struct { name: []const u8, part: [:0]const u8 = "tf_rcd_top32_part", merge: [:0]const u8 = "tf_rcd_top32_merge" };

pub const top_entries = [_]TopEntry{
    .{ .name = "ricursive dflash2 top-32 over the vocabulary (49 chunks, then their bests)" },
    .{ .name = "control: ricursive dflash2 top-32 orders ties by the higher id", .part = "tf_rcd_ctl_top32_part_ties", .merge = "tf_rcd_ctl_top32_merge_ties" },
    .{ .name = "control: ricursive dflash2 top-32's merge drops chunk 0's bests", .merge = "tf_rcd_ctl_top32_merge_drop" },
};

/// Rows: the draft rows (15 a stream's block, up to 128); n: the vocabulary.
const top_served = [_]c.Shape{ .{ .rows = 15, .n = da.vocab, .k = 1 }, .{ .rows = 1, .n = da.vocab, .k = 1 }, .{ .rows = 16, .n = da.vocab, .k = 1 }, .{ .rows = 128, .n = da.vocab, .k = 1 } };

const P = struct { v: f32, id: u32 };

fn first(_: void, a: P, b: P) bool {
    return a.v > b.v or (a.v == b.v and a.id < b.id);
}

/// A 2,048-entry chunk in rc_df_topk's order: a host sort when every value is finite, else its network replayed.
fn sortChunk(buf: []P) void {
    for (buf) |x| if (!std.math.isFinite(x.v) and x.id != 0xffffffff) {
        var k: usize = 2;
        while (k <= da.top_chunk) : (k <<= 1) {
            var j = k >> 1;
            while (j > 0) : (j >>= 1) for (0..da.top_chunk) |i| {
                const l = i ^ j;
                if (l <= i) continue;
                const swap = if (i & k == 0) first({}, buf[l], buf[i]) else first({}, buf[i], buf[l]);
                if (swap) std.mem.swap(P, &buf[i], &buf[l]);
            };
        }
        return;
    };
    std.sort.pdq(P, buf, {}, first);
}

pub const Top = struct {
    entry: TopEntry,
    part: cuda.Function,
    merge: cuda.Function,
    shape: c.Shape = undefined,
    l: []u16 = &.{},
    call: da.Top32 = undefined,

    pub fn spec(self: *const Top) c.Spec {
        return .{
            .name = self.entry.name,
            .accum = .none,
            .depth = ref.noSums,
            .out = .raw,
            .nan = .{ .stated = "finite rows: each chunk's 32 largest, then the chunks' bests' 32 (value desc, id asc) as a host sort gives them; with NaN or infinities: rc_df_topk's sorting network (NaN compares false), replayed bit for bit" },
            .tiles = .{ .{}, .{ .size = da.top_chunk, .step = 1, .min = 1 }, .{} },
            .served = &top_served,
        };
    }

    pub fn control(self: *const Top) ?g.Gate {
        const served = std.mem.eql(u8, self.entry.part, "tf_rcd_top32_part") and std.mem.eql(u8, self.entry.merge, "tf_rcd_top32_merge");
        return if (served) null else .shapes;
    }

    pub fn free(self: *Top, a: std.mem.Allocator) void {
        a.free(self.l);
        self.l = &.{};
    }

    /// Even rows normal times 4; odd rows 512 levels of 1/16 (ties everywhere, ids must break them).
    pub fn prepare(self: *Top, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n > da.top_chunk * da.top_chunk / da.top) return false;
        self.free(a);
        self.shape = shape;
        self.l = try a.alloc(u16, shape.rows * shape.n);
        for (self.l, 0..) |*x, i| x.* = c.toBf16(if (i / shape.n % 2 == 0) 4 * rng.floatNorm(f32) else @as(f32, @floatFromInt(rng.uintLessThan(u32, 512))) / 16 - 16);
        if (fill == .nonfinite) for ([_]f32{ ref.nan, ref.inf, -ref.inf, ref.nan }, 0..) |x, k| {
            self.l[(k % shape.rows) * shape.n + (k * 37) % shape.n] = c.toBf16(x);
        };
        const logits = try ref.input(rig, self.l);
        const vals = try rig.output(shape.rows * da.top * 4);
        const ids = try rig.output(shape.rows * da.top * 4);
        self.call = .{ .logits = logits, .stride = shape.n, .n = shape.n, .rows = shape.rows, .vals = vals, .ids = ids, .part_vals = 0, .part_ids = 0 };
        self.call.part_vals = try rig.scratch(da.top32Scratch(self.call));
        self.call.part_ids = try rig.scratch(da.top32Scratch(self.call));
        return true;
    }

    pub fn run(self: *Top, rig: *rig_.Rig) !void {
        try da.top32Via(self.part, self.merge, rig.s, self.call);
    }

    pub fn count(self: *Top) usize {
        return 2 * self.shape.rows * da.top * 4;
    }

    pub fn check(self: *Top, a: std.mem.Allocator, got: []const u8, msg: *[200]u8) !?[]const u8 {
        const n = self.shape.n;
        const chunks = self.call.chunks();
        const buf = try a.alloc(P, da.top_chunk);
        defer a.free(buf);
        const bests = try a.alloc(P, da.top_chunk);
        defer a.free(bests);
        const half = self.shape.rows * da.top * 4;
        for (0..self.shape.rows) |r| {
            for (bests) |*x| x.* = .{ .v = -std.math.inf(f32), .id = 0xffffffff };
            for (0..chunks) |ch| {
                for (buf, 0..) |*x, i| {
                    const at = ch * da.top_chunk + i;
                    x.* = if (at < n) .{ .v = bf32(self.l[r * n + at]), .id = @intCast(at) } else .{ .v = -std.math.inf(f32), .id = 0xffffffff };
                }
                sortChunk(buf);
                @memcpy(bests[ch * da.top ..][0..da.top], buf[0..da.top]);
            }
            sortChunk(bests);
            for (bests[0..da.top], 0..) |want, i| {
                const gv = ref.readF32(got, r * da.top + i);
                const gi: u32 = @bitCast(ref.readI32(got[half..], r * da.top + i));
                const same = @as(u32, @bitCast(want.v)) == @as(u32, @bitCast(gv)) or (std.math.isNan(want.v) and std.math.isNan(gv));
                if (!same or want.id != gi) return try std.fmt.bufPrint(msg, "row {d} place {d}: ({d}, {d}) where the host has ({d}, {d})", .{ r, i, gv, gi, want.v, want.id });
            }
        }
        return null;
    }
};

// --- the set, the runner and the timings ---------------------------------------------------------------------------

/// The registrations, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: da.Kernels,
    attn: [attn_entries.len]ref.Gated(Attn) = undefined,
    tops: [top_entries.len]ref.Checked(Top) = undefined,

    /// The set, its kernels appended to `list` (the controls after the kernel they break).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        set.* = .{ .gpa = gpu.gpa, .k = try da.Kernels.load(gpu.d) };
        errdefer set.k.unload();
        const mod = set.k.mod;
        for (&set.attn, attn_entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .k = .{ .entry = e, .part = try da.partFunction(mod, e.part), .merge = try mod.function(e.merge) } };
        for (&set.tops, top_entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .k = .{ .entry = e, .part = try mod.function(e.part), .merge = try mod.function(e.merge) } };
        for (&set.attn) |*x| try list.append(a, x.kernel());
        for (&set.tops) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.tops) |*x| x.deinit();
        for (&set.attn) |*x| x.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};

/// One call (both launches) at full rings, windows whole and wrapped: mean [total, part] ms of 20 calls after 3.
fn timeAttn(gpu: check.Gpu, s: cuda.Stream, k: *const da.Kernels, span: usize, streams: usize) ![2]f64 {
    const ring_rows = da.ringRows(span);
    const ring_bytes = 2 * ring_rows * kv_heads * D * 2;
    var bufs: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpu.gpa);
    }
    var segs: [8]da.Segment = undefined;
    const host = try gpu.gpa.alloc(u16, ring_bytes / 2);
    defer gpu.gpa.free(host);
    var x: u32 = 0x9e3779b9;
    for (host) |*v| {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        v.* = c.toBf16(@as(f32, @floatFromInt(x >> 8)) / 0x1p23 - 1);
    }
    for (segs[0..streams], 0..) |*sg, i| {
        try bufs.append(gpu.gpa, try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host)));
        sg.* = .{ .ring = bufs.items[i].ptr, .anchor = @intCast(100000 + 1000 * i), .from = 0, .span = @intCast(span), .rows = @intCast(ring_rows) };
    }
    const q = try gpu.gpa.alloc(u16, streams * block * width);
    defer gpu.gpa.free(q);
    @memcpy(q, host[0..q.len]);
    try bufs.append(gpu.gpa, try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q)));
    try bufs.append(gpu.gpa, try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(segs[0..streams])));
    var call: da.Attn = .{ .q = bufs.items[streams].ptr, .segs = bufs.items[streams + 1].ptr, .out = 0, .pm = 0, .pl = 0, .po = 0, .streams = streams, .span = span };
    const sc = da.scratch(call);
    for ([_]usize{ streams * block * width * 2, sc.max, sc.max, sc.values }) |len| try bufs.append(gpu.gpa, try cuda.DeviceBuffer.alloc(gpu.d, len));
    call.out = bufs.items[streams + 2].ptr;
    call.pm = bufs.items[streams + 3].ptr;
    call.pl = bufs.items[streams + 4].ptr;
    call.po = bufs.items[streams + 5].ptr;
    for (0..3) |_| try da.attention(k, s, call);
    var e0 = try cuda.Event.init(gpu.d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(gpu.d, true);
    defer e1.deinit();
    const calls = 20;
    var ms: [2]f64 = undefined;
    for (&ms, 0..) |*m, i| {
        try e0.record(s);
        for (0..calls) |_| if (i == 0) try da.attention(k, s, call) else try da.partVia(k.attn_part, s, call);
        try e1.record(s);
        try e1.synchronize();
        m.* = @as(f64, try cuda.Event.elapsedMs(e0, e1)) / calls;
    }
    return ms;
}

/// The top-32 of `rows` rows of the vocabulary, mean of 20 calls.
fn timeTop32(gpu: check.Gpu, s: cuda.Stream, k: *const da.Kernels, rows: usize) !f64 {
    var call: da.Top32 = .{ .logits = 0, .stride = da.vocab, .rows = rows, .vals = 0, .ids = 0, .part_vals = 0, .part_ids = 0 };
    var bufs: [5]cuda.DeviceBuffer = undefined;
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| b.free();
    for ([_]usize{ rows * da.vocab * 2, rows * da.top * 4, rows * da.top * 4, da.top32Scratch(call), da.top32Scratch(call) }, 0..) |len, i| {
        bufs[i] = try cuda.DeviceBuffer.alloc(gpu.d, len);
        made += 1;
    }
    try bufs[0].fill8(0x3f, s.handle);
    call.logits = bufs[0].ptr;
    call.vals = bufs[1].ptr;
    call.ids = bufs[2].ptr;
    call.part_vals = bufs[3].ptr;
    call.part_ids = bufs[4].ptr;
    for (0..3) |_| try da.top32(k, s, call);
    var e0 = try cuda.Event.init(gpu.d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(gpu.d, true);
    defer e1.deinit();
    const calls = 20;
    try e0.record(s);
    for (0..calls) |_| try da.top32(k, s, call);
    try e1.record(s);
    try e1.synchronize();
    return @as(f64, try cuda.Event.elapsedMs(e0, e1)) / calls;
}

/// Gates 1-4 of the draft attention and top-32 and their controls, then the timings.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x726364, &t);
    std.debug.print("SUMMARY ricursive dflash2 attention: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    for ([_]Form{ .sliding, .global }) |f| for ([_]usize{ 1, 8 }) |streams| {
        const span = spanOf(f);
        const ms = try timeAttn(gpu, s, &set.k, span, streams);
        const keys = streams * kv_heads * (span + block - 1);
        const kv_mb = @as(f64, @floatFromInt(keys * D * 2 * 2)) / 1e6;
        std.debug.print("TIME ricursive dflash2 attention {t} (window {d}) at {d} stream(s), full rings: {d:.3} ms a call (part {d:.3} + merge {d:.3}); K and V read {d:.1} MB, {d:.0} GB/s\n", .{ f, span, streams, ms[0], ms[1], ms[0] - ms[1], kv_mb, kv_mb / ms[0] });
    };
    for ([_]usize{ 15, 128 }) |rows| std.debug.print("TIME ricursive dflash2 top-32 at {d} rows of {d}: {d:.3} ms a call (part + merge)\n", .{ rows, da.vocab, try timeTop32(gpu, s, &set.k, rows) });
    try check.expect(t.problems() == 0, "ricursive-dflash2-attn-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "the controls: every planted fault of the image registered once, each against a served kernel's shapes" {
    const t = std.testing;
    const faults = [_][]const u8{ "tf_rcd_ctl_attn_causal", "tf_rcd_ctl_attn_floor", "tf_rcd_ctl_attn_no_block", "tf_rcd_ctl_attn_merge_trunc", "tf_rcd_ctl_top32_part_ties", "tf_rcd_ctl_top32_merge_ties", "tf_rcd_ctl_top32_merge_drop" };
    for (faults) |f| {
        var n: usize = 0;
        for (attn_entries) |e| n += @intFromBool(std.mem.eql(u8, e.part, f)) + @intFromBool(std.mem.eql(u8, e.merge, f));
        for (top_entries) |e| n += @intFromBool(std.mem.eql(u8, e.part, f)) + @intFromBool(std.mem.eql(u8, e.merge, f));
        try t.expectEqual(@as(usize, 1), n);
    }
    var controls: usize = 0;
    for (attn_entries) |e| controls += @intFromBool(isControl(e));
    try t.expectEqual(@as(usize, 4), controls);
    // the rings the brief names, a wrapped ring each, a from-floor above the window start on odd streams
    try t.expectEqual(@as(usize, 4224), da.ringRows(spanOf(.sliding)));
    try t.expectEqual(@as(usize, 16512), da.ringRows(spanOf(.global)));
    try t.expect(anchorOf(sliding_served[0].k, 0) + block > da.ringRows(da.sliding_span));
    try t.expect(anchorOf(global_served[1].k, 0) + block > da.ringRows(da.global_span));
    const a1 = anchorOf(sliding_served[0].k, 1);
    try t.expect(fromOf(a1, da.sliding_span, 1) > a1 + 1 - da.sliding_span);
}
