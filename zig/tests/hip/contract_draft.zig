//! The DFlash2 drafter's kernels under the contract on HIP: exact copies and conv, float64 attention; a control each.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;
const draft = hip.launches.draft;

// Qwen3.8-27B's DFlash2: 32 query and 8 KV heads of 128, hidden 5120 in groups of 16, a block of 8, a 2048 window
const hd = draft.head;
const heads = 32;
const kv_heads = 8;
const kv_width = kv_heads * hd;
const group = 16;
const block = 8;
const window = 2048;
const ring_rows = window + 16;
const tap_rows = 2048 + 1024; // the target's tap ring: the window and a prompt chunk
const layers = 2; // the drafter ring's layers; the attention reads the second

fn none(_: c.Shape) usize {
    return 0;
}

/// A copy or conv checked by its exact steps, or the block attention.
const Case = enum { gather, gather_rows, ring_put, mask_rows, conv, conv_res, attention };

/// Each case, then a planted control of each kernel: the real kernel launched wrongly, which must go red at gate 3.
pub const Mode = enum { gather, gather_rows, ring_put, mask_rows, conv, conv_res, attention, no_wrap, unrounded, no_residual, first_row, other_layer, lost_partial };

/// The case a mode runs: itself, or the one its control plants a fault in.
fn caseOf(m: Mode) Case {
    return switch (m) {
        .no_wrap => .gather,
        .unrounded => .ring_put,
        .no_residual => .conv_res,
        .first_row => .mask_rows,
        .other_layer, .lost_partial => .attention,
        inline else => |x| @field(Case, @tagName(x)),
    };
}

/// A control's name, null for a real case.
fn controlName(m: Mode) ?[]const u8 {
    return switch (m) {
        .no_wrap => "control: tf_draft_gather skips the ring wrap",
        .unrounded => "control: tf_draft_ring_put writes keys unrounded",
        .no_residual => "control: tf_draft_dyn_conv drops the residual",
        .first_row => "control: tf_draft_mask_rows keeps each stream's first row",
        .other_layer => "control: tf_draft_attn_partial reads the ring's other layer",
        .lost_partial => "control: tf_draft_attn_merge ignores each row's first partial",
        else => null,
    };
}

const exact = "every output is the bytes its fp32 steps round to, NaN wherever they give NaN";

/// A mode's spec: its case's, under the control's name.
pub fn spec(m: Mode) c.Spec {
    var s = caseSpec(caseOf(m));
    if (controlName(m)) |name| s.name = name;
    return s;
}

fn caseSpec(m: Case) c.Spec {
    const raw: c.Spec = .{ .name = "", .accum = .none, .depth = none, .out = .raw, .nan = .{ .stated = exact }, .served = &.{} };
    var s = raw;
    switch (m) {
        .gather => {
            s.name = "hip tf_draft_gather (rows in order)";
            // a block's taps, a stream's prompt chunk: rows over the 27B's five taps
            s.served = &.{ .{ .rows = 8, .n = 5, .k = 5120 }, .{ .rows = 512, .n = 5, .k = 5120 } };
        },
        .gather_rows => {
            s.name = "hip tf_draft_gather (a kept path by value)";
            s.served = &.{ .{ .rows = 7, .n = 5, .k = 5120 }, .{ .rows = 64, .n = 5, .k = 5120 } };
        },
        .ring_put => {
            s.name = "hip tf_draft_ring_put";
            s.served = &.{ .{ .rows = 8, .n = kv_heads, .k = hd }, .{ .rows = 512, .n = kv_heads, .k = hd } };
        },
        .mask_rows => {
            s.name = "hip tf_draft_mask_rows";
            // one stream's block and four streams'
            s.served = &.{ .{ .rows = 8, .n = 1, .k = 5120 }, .{ .rows = 32, .n = 1, .k = 5120 } };
        },
        .conv, .conv_res => {
            s.name = if (m == .conv) "hip tf_draft_dyn_conv (branch 0)" else "hip tf_draft_dyn_conv (branch 1, residual)";
            s.served = &.{ .{ .rows = 8, .n = 1, .k = 5120 }, .{ .rows = 32, .n = 1, .k = 5120 } };
        },
        .attention => return .{
            .name = if (kit.act() == .bf16) "hip tf_draft_attn_partial + merge (bf16)" else "hip tf_draft_attn_partial + merge (fp16)",
            .accum = .f32,
            .depth = depth,
            .ops = 4,
            .out = kit.elem(kit.act()),
            // one stream's block over a full window, four streams', and one over a short context
            .served = &.{
                .{ .rows = 8, .n = heads, .k = 2048 },
                .{ .rows = 32, .n = heads, .k = 2048 },
                .{ .rows = 8, .n = heads, .k = 64 },
            },
        },
    }
    return s;
}

/// A lane's 128-term dot, a wave's 64 scores and their tree, four waves, nine partials, and the values likewise.
fn depth(_: c.Shape) usize {
    return hd + 2 * (draft.chunk / 4 + 5 + 4 + draft.chunks) + 4;
}

pub const Draft = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    m: Case,
    /// The mode as registered: m, or a control planted in m.
    plant: Mode,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    in: [4]u64 = @splat(0),
    out: u64 = 0,
    scratch: u64 = 0,
    // the inputs as float64 (attention: q, kb, vb, then each stream's ring)
    x: [4][]f64 = @splat(&.{}),
    base: []f32 = &.{},
    path: [draft.most_rows]u32 = undefined,
    segs: [4]draft.Segment = undefined,
    rings: [4][]f64 = @splat(&.{}),
    streams: usize = 0,
    /// The attention's block rows: 8, or a shorter block when fewer rows are asked (gate 4's four).
    blk: usize = block,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, m: Mode) Draft {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .m = caseOf(m), .plant = m, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Draft) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Draft) g.Kernel {
        const control: ?g.Gate = if (controlName(self.plant) != null) .shapes else null;
        const vt = if (self.m == .attention) &vt_attend else &vt_exact;
        return .{ .spec = spec(self.plant), .ctx = self, .vt = vt, .control = control };
    }

    /// The copies and convs check their own exact steps; the attention takes float64's bound and gate 4.
    const vt_exact = blk: {
        var v = g.hooks(Draft);
        v.simulate = null;
        break :blk v;
    };
    const vt_attend = blk: {
        var v = g.hooks(Draft);
        v.check = null;
        break :blk v;
    };

    pub fn prepare(self: *Draft, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        return if (self.m == .attention) Attend.prepare(self, shape, fill, seed) else Exact.prepare(self, shape, fill, seed);
    }

    pub fn run(self: *Draft) !void {
        return if (self.m == .attention) Attend.run(self) else Exact.run(self);
    }

    pub fn reference(self: *Draft, want: []f64, mag: []f64) !void {
        if (self.m == .attention) return Attend.reference(self, want, mag);
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(self: *Draft, _: g.Gate) !?[]const u8 {
        return Exact.check(self);
    }

    pub fn simulate(self: *Draft, acc: c.Accum, order: g.Order, out: []f64) !void {
        return Attend.simulate(self, acc, order, out);
    }

    /// Random values of the kind (non-finite: NaN and both infinities planted), as float64 and uploaded.
    fn act(self: *Draft, rng: std.Random, n: usize, lo: f64, hi: f64, fill: g.Fill, at: usize) !struct { []f64, u64 } {
        const v, const b = try kit.values(self.arena.allocator(), rng, self.k, n, lo, hi);
        if (fill == .nonfinite) for ([_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }, 0..) |p, i| {
            const j = (at + i * 2711) % n;
            v[j] = p;
            kit.put(self.k, b, j, p);
        };
        return .{ v, try self.rig.input(b) };
    }

    fn width(self: *const Draft) usize {
        return self.shape.n * self.shape.k;
    }

    pub fn count(self: *Draft) usize {
        const s = self.shape;
        return switch (self.m) {
            .gather, .gather_rows, .conv, .conv_res => s.rows * self.width(),
            .ring_put => 2 * ring_rows * self.width(),
            .mask_rows => s.rows / block * (block - 1) * s.k,
            .attention => s.rows * heads * hd,
        };
    }

    fn outBytes(self: *const Draft) usize {
        return switch (self.m) {
            .ring_put => 2 * ring_rows * self.width() * 2,
            .gather, .gather_rows, .conv, .conv_res => self.shape.rows * self.width() * 2,
            .mask_rows => self.shape.rows / block * (block - 1) * self.shape.k * 2,
            .attention => self.shape.rows * heads * hd * 2,
        };
    }

    fn start(self: *Draft) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        return true;
    }
};

/// The copies and the conv: run, then each output against its exact host steps.
const Exact = struct {
    fn prepare(d: *Draft, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const s = shape;
        if (s.rows == 0 or s.n == 0 or s.k == 0 or s.k % 2 != 0) return false;
        if ((d.m == .mask_rows or d.m == .conv or d.m == .conv_res) and s.rows % block != 0) return false;
        if (d.m == .gather_rows and s.rows > draft.most_rows) return false;
        if (d.m == .ring_put and s.rows > ring_rows) return false;
        if ((d.m == .conv or d.m == .conv_res) and s.k % group != 0) return false;
        _ = try d.start();
        d.shape = s;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const w = d.width();
        switch (d.m) {
            .gather, .gather_rows => {
                const past: usize = if (d.plant == .no_wrap) s.rows else 0; // the control's rows past the ring's end
                d.x[0], d.in[0] = try d.act(rng, (tap_rows + past) * w, -4, 4, fill, 3);
                if (d.m == .gather_rows) for (d.path[0..s.rows], 0..) |*p, i| {
                    p.* = @intCast((i * 7 + 3) % (2 * s.rows + 1));
                };
            },
            .ring_put => {
                const keys = try kit.floats(d.arena.allocator(), rng, s.rows * w, -3, 3);
                if (fill == .nonfinite) keys[s.rows * w / 2] = std.math.nan(f64);
                d.x[0] = keys;
                const sent = if (d.plant == .unrounded) try truncated(d, keys) else keys;
                d.in[0] = try d.rig.input(try kit.f32Bytes(d.arena.allocator(), sent));
                d.x[1], d.in[1] = try d.act(rng, s.rows * w, -2, 2, fill, 1);
            },
            .mask_rows => {
                d.x[0], d.in[0] = try d.act(rng, s.rows * s.k, -4, 4, fill, 5);
            },
            .conv, .conv_res => {
                d.x[0], d.in[0] = try d.act(rng, s.rows * s.k, -2, 2, fill, 2);
                d.x[1], d.in[1] = try d.act(rng, s.rows * 4 * (s.k / group), -1, 1, .random, 0);
                d.x[2], d.in[2] = try d.act(rng, s.rows * s.k, -4, 4, .random, 0);
                const base = try kit.floats(d.arena.allocator(), rng, 4 * s.k, -0.5, 0.5);
                d.base = try d.arena.allocator().alloc(f32, base.len);
                for (d.base, base) |*b, v| b.* = @floatCast(v);
                d.in[3] = try d.rig.input(std.mem.sliceAsBytes(d.base));
            },
            .attention => unreachable,
        }
        d.out = try d.rig.output(d.outBytes());
        return true;
    }

    /// The rows a gather reads: first = tap_rows - 3, so a block's rows wrap the ring.
    const first = 3 * tap_rows - 3;
    /// A ring put's first position: its rows wrap the drafter ring.
    const pos0 = 5 * ring_rows - 5;

    /// Each key cut toward zero to the kind: the kernel's rounding keeps them, so the ring holds them unrounded.
    fn truncated(d: *Draft, keys: []const f64) ![]f64 {
        const t = try d.arena.allocator().alloc(f64, keys.len);
        for (t, keys) |*o, v| {
            o.* = kit.round(d.k, v);
            if (!(@abs(o.*) > @abs(v))) continue;
            var b: [2]u8 = undefined;
            kit.put(d.k, &b, 0, o.*);
            std.mem.writeInt(u16, &b, std.mem.readInt(u16, &b, .little) - 1, .little); // one step toward zero
            o.* = kit.get(d.k, &b, 0);
        }
        return t;
    }

    fn run(d: *Draft) !void {
        const s = d.shape;
        const st = d.rig.s.handle;
        const kind = @backingInt(d.k);
        switch (d.m) {
            .gather => {
                // the control enters a ring s.rows rows longer at first's row: its rows run past the end, unwrapped
                const wrap = d.plant != .no_wrap;
                const rows = if (wrap) tap_rows else tap_rows + s.rows;
                const start: u64 = if (wrap) first else first % tap_rows;
                try d.l.draftGather(d.in[0], rows, start, null, s.rows, d.width() / 2, d.out, st);
            },
            .gather_rows => try d.l.draftGather(d.in[0], tap_rows, first, d.path[0..s.rows], s.rows, d.width() / 2, d.out, st),
            .ring_put => {
                const half = ring_rows * d.width() * 2;
                try d.l.draftRingPut(d.in[0], d.in[1], kind, d.out, d.out + half, s.rows, d.width(), pos0, ring_rows, st);
            },
            .mask_rows => {
                // the control reads one row early: each stream's first row kept, its last dropped
                const x = if (d.plant == .first_row) d.in[0] - 2 * s.k else d.in[0];
                try d.l.draftMaskRows(x, d.out, s.rows / block, s.k, block, st);
            },
            .conv => try d.l.draftDynConv(d.in[0], d.in[1], d.in[3], 0, d.out, kind, s.rows, s.k, group, 0, block, st),
            .conv_res => {
                const res = if (d.plant == .no_residual) 0 else d.in[2];
                try d.l.draftDynConv(d.in[0], d.in[1], d.in[3], res, d.out, kind, s.rows, s.k, group, 1, block, st);
            },
            .attention => unreachable,
        }
        try d.rig.s.synchronize();
    }

    /// `v` rounded to the kind as the device rounds an fp32 value.
    fn round(d: *const Draft, v: f32) f32 {
        return @floatCast(kit.round(d.k, v));
    }

    /// Output `i`'s value by the kernel's own steps; null: the poison it must keep (a ring slot no row writes).
    fn want(d: *Draft, i: usize) ?f64 {
        const s = d.shape;
        const w = d.width();
        switch (d.m) {
            .gather, .gather_rows => {
                const r = i / w;
                const off: usize = if (d.m == .gather) r else d.path[r];
                return d.x[0][((first + off) % tap_rows) * w + i % w];
            },
            .ring_put => {
                const values = i >= ring_rows * w;
                const j = i % (ring_rows * w);
                const slot = j / w;
                const row = (slot + ring_rows - pos0 % ring_rows) % ring_rows;
                if (row >= s.rows) return null;
                const v = d.x[@intFromBool(values)][row * w + j % w];
                return if (values) v else kit.round(d.k, v);
            },
            .mask_rows => {
                const r = i / s.k;
                const source = r / (block - 1) * block + 1 + r % (block - 1);
                return d.x[0][source * s.k + i % s.k];
            },
            .conv, .conv_res => {
                const t = i / s.k;
                const ch = i % s.k;
                const groups = s.k / group;
                const branch: usize = @intFromBool(d.m == .conv_res);
                var total: f32 = 0;
                for (0..2) |l| {
                    if (t % block < l) continue;
                    const x: f32 = @floatCast(d.x[0][(t - l) * s.k + ch]);
                    const dyn: f32 = @floatCast(d.x[1][t * 4 * groups + (branch * 2 + l) * groups + ch / group]);
                    total = round(d, total + round(d, d.base[(branch * 2 + l) * s.k + ch] * x));
                    total = round(d, total + round(d, dyn * x));
                }
                if (d.m == .conv_res) total = round(d, total + @as(f32, @floatCast(d.x[2][i])));
                return total;
            },
            .attention => unreachable,
        }
    }

    fn check(d: *Draft) !?[]const u8 {
        const got = try d.rig.bytes(d.out, d.outBytes());
        for (0..d.count()) |i| {
            const bits = std.mem.readInt(u16, got[2 * i ..][0..2], .little);
            const w = want(d, i) orelse {
                if (bits == 0xffff) continue;
                return try std.fmt.bufPrint(&d.msg, "output {d} is 0x{x:0>4}, a slot no row writes", .{ i, bits });
            };
            const v = kit.get(d.k, got, i);
            if (std.math.isNan(w) and std.math.isNan(v)) continue;
            var b: [2]u8 = undefined;
            kit.put(d.k, &b, 0, w);
            if (std.mem.readInt(u16, &b, .little) == bits) continue;
            return try std.fmt.bufPrint(&d.msg, "output {d} is {e} (0x{x:0>4}), its steps give {e}", .{ i, v, bits, w });
        }
        return null;
    }
};

/// The block attention: q and block keys fp32 as qkRope gives them, values and rings of the kind.
const Attend = struct {
    /// Stream s's context: stream 0 (all under stress) sees shape.k keys a row; others wrap the ring past the window.
    fn segment(d: *Draft, s: usize, fill: g.Fill) struct { begin: usize, position: usize } {
        const k = d.shape.k;
        const position = 3 * ring_rows + 41 * s + 7;
        if (s == 0 or fill == .stress) return .{ .begin = position - (k - d.blk), .position = position };
        return .{ .begin = 0, .position = position };
    }

    fn prepare(d: *Draft, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const s = shape;
        d.blk = if (s.rows < block) s.rows else block;
        if (s.rows == 0 or s.rows % d.blk != 0 or s.rows / d.blk > 4 or s.n != heads) return false;
        if (s.k < d.blk or s.k > window) return false;
        _ = try d.start();
        d.shape = s;
        const a = d.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        d.streams = s.rows / d.blk;
        const lo: f64 = if (fill == .stress) 0 else -1;
        const q = try a.alloc(f64, s.rows * heads * hd);
        for (q) |*v| v.* = if (fill == .stress) 0 else kit.round(d.k, rng.float(f64) * 2 - 1);
        if (fill == .nonfinite) q[0] = std.math.nan(f64);
        const kb = try a.alloc(f64, s.rows * kv_width);
        for (kb) |*v| v.* = kit.round(d.k, rng.float(f64) * 2 - 1);
        d.x[0] = q;
        d.x[1] = kb;
        d.in[0] = try d.rig.input(try kit.f32Bytes(a, q));
        d.in[1] = try d.rig.input(try kit.f32Bytes(a, kb));
        d.x[2], d.in[2] = try d.act(rng, s.rows * kv_width, lo, 1, .random, 0);
        if (fill == .stress) {
            // whole block values, so the stress sums stay exact in fp32
            const b = try a.alloc(u8, d.x[2].len * 2);
            for (d.x[2], 0..) |*v, i| {
                v.* = @floatFromInt(1 + rng.uintLessThan(u32, 4));
                kit.put(d.k, b, i, v.*);
            }
            d.in[2] = try d.rig.input(b);
        }
        const ring_len = layers * 2 * ring_rows * kv_width;
        for (0..d.streams) |i| {
            const v, const b = try kit.values(a, rng, d.k, ring_len, lo, 1);
            d.rings[i] = v;
            const sg = segment(d, i, fill);
            if (fill == .stress) stress(d, rng, i, v, b);
            if (fill == .nonfinite and i == 0) {
                // an infinite key and a -infinite value inside stream 0's window, in the read layer
                const kr = ring_rows * kv_width * 2;
                const at = ((sg.position - 3) % ring_rows) * kv_width;
                v[kr + at + 5] = std.math.inf(f64);
                kit.put(d.k, b, kr + at + 5, v[kr + at + 5]);
                const vr = kr + ring_rows * kv_width;
                const at2 = ((sg.position - 9) % ring_rows) * kv_width + hd + 7;
                v[vr + at2] = -std.math.inf(f64);
                kit.put(d.k, b, vr + at2, v[vr + at2]);
            }
            d.segs[i] = .{ .ring = try d.rig.input(b), .begin = @intCast(sg.begin), .position = @intCast(sg.position) };
        }
        d.in[3] = try d.rig.input(std.mem.sliceAsBytes(d.segs[0..d.streams]));
        d.scratch = try d.rig.scratch(draft.Partials.bytes(s.rows, heads));
        d.out = try d.rig.output(d.outBytes());
        return true;
    }

    /// Gate 4: q = 0, so each key weighs 1 / shape.k; values to 255 whose halves cancel but for eight: exact in fp32.
    fn stress(d: *Draft, rng: std.Random, s: usize, v: []f64, b: []u8) void {
        const sg = segment(d, s, .stress);
        const vr = ring_rows * kv_width * 2 + ring_rows * kv_width;
        const ring_keys = sg.position - sg.begin;
        for (0..kv_heads) |h| for (0..ring_keys / 2) |row| for (0..hd) |j| {
            const x: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 255));
            const p = ((sg.begin + row) % ring_rows) * kv_width + h * hd + j;
            const n = ((sg.begin + ring_keys / 2 + row) % ring_rows) * kv_width + h * hd + j;
            v[vr + p] = x;
            v[vr + n] = -x + @as(f64, if (row < 8) 1 else 0);
            kit.put(d.k, b, vr + p, v[vr + p]);
            kit.put(d.k, b, vr + n, v[vr + n]);
        };
    }

    fn run(d: *Draft) !void {
        const s = d.shape;
        const layer: usize = if (d.plant == .other_layer) 0 else layers - 1; // the control scores the first layer
        const at: draft.Attention = .{ .rows = s.rows, .block_rows = d.blk, .heads = heads, .kv_heads = kv_heads, .ring_rows = ring_rows, .window = window, .layer = layer };
        const p = draft.Partials.at(d.scratch, s.rows, heads);
        try d.l.draftAttention(d.in[0], d.in[1], d.in[2], 0, d.in[3], @backingInt(d.k), at, p, d.out, d.rig.s.handle);
        if (d.plant == .lost_partial) try dropFirst(d, p);
        try d.rig.s.synchronize();
    }

    /// The control: each row and head's first partial emptied to the zero sum the merge skips, then merged again.
    fn dropFirst(d: *Draft, p: draft.Partials) !void {
        const n = d.shape.rows * heads;
        const sums = try d.rig.bytes(p.l, n * draft.chunks * 4);
        for (0..n) |i| std.mem.writeInt(u32, sums[i * draft.chunks * 4 ..][0..4], 0, .little);
        try d.rig.put(p.l, sums);
        var b: hip.launch.Args = .{};
        for ([_]u64{ p.m, p.l, p.o }) |v| try b.add(v);
        for ([_]c_int{ heads, @backingInt(d.k) }) |v| try b.add(v);
        try b.add(d.out);
        const grid: hip.launch.Dim3 = .{ .x = @intCast(d.shape.rows), .y = heads };
        try hip.launch.launch(d.l.draft.attn_merge, .{ .grid = grid, .block = .{ .x = 32 } }, d.rig.s, &b);
    }

    /// Row m's key i (ring keys first, then the block's): its key and value rows' offsets in their arrays.
    const Key = struct { k: []const f64, v: []const f64, at: usize };

    fn key(d: *Draft, m: usize, i: usize, ring_keys: usize, first: usize, kvh: usize) Key {
        const s = m / d.blk;
        if (i < ring_keys) {
            const kr = ring_rows * kv_width * 2;
            const r = d.rings[s];
            const at = ((first + i) % ring_rows) * kv_width + kvh * hd;
            return .{ .k = r[kr..], .v = r[kr + ring_rows * kv_width ..], .at = at };
        }
        const at = (s * d.blk + i - ring_keys) * kv_width + kvh * hd;
        return .{ .k = d.x[1], .v = d.x[2], .at = at };
    }

    fn windowOf(d: *Draft, m: usize) struct { usize, usize } {
        const sg = d.segs[m / d.blk];
        const position: usize = @intCast(sg.position);
        const first = @min(position, @max(@as(usize, @intCast(sg.begin)), (position + m % d.blk + 1) -| window));
        return .{ first, position - first };
    }

    /// softmax(q k / sqrt(128)) v in float64 a row and head, each output's magnitude through its scores and sums.
    fn reference(d: *Draft, want: []f64, mag: []f64) !void {
        const a = d.rig.a;
        const most = window + d.blk;
        const p = try a.alloc(f64, 2 * most);
        defer a.free(p);
        const sens = p[most..];
        const scale: f64 = @as(f32, 1 / @sqrt(@as(f32, hd)));
        for (0..d.shape.rows) |m| for (0..heads) |h| {
            const first, const ring_keys = windowOf(d, m);
            const keys = ring_keys + d.blk;
            const kvh = h / (heads / kv_heads);
            const qv = d.x[0][(m * heads + h) * hd ..][0..hd];
            var top = -std.math.inf(f64);
            for (0..keys) |i| {
                const ky = key(d, m, i, ring_keys, first, kvh);
                var dot: f64 = 0;
                var abs: f64 = 0;
                for (0..hd) |j| {
                    dot += qv[j] * ky.k[ky.at + j];
                    abs += @abs(qv[j] * ky.k[ky.at + j]);
                }
                p[i] = dot * scale;
                sens[i] = 2 + abs * scale;
                top = if (std.math.isNan(p[i]) or std.math.isNan(top)) std.math.nan(f64) else @max(top, p[i]);
            }
            var sum: f64 = 0;
            for (p[0..keys]) |*x| {
                x.* = @exp(x.* - top);
                sum += x.*;
            }
            const out = want[(m * heads + h) * hd ..][0..hd];
            const mg = mag[(m * heads + h) * hd ..][0..hd];
            @memset(out, 0);
            for (0..keys) |i| {
                const ky = key(d, m, i, ring_keys, first, kvh);
                for (0..hd) |j| out[j] += p[i] / sum * ky.v[ky.at + j];
            }
            @memset(mg, 0);
            for (0..keys) |i| {
                const ky = key(d, m, i, ring_keys, first, kvh);
                for (0..hd) |j| mg[j] += p[i] / sum * (@abs(ky.v[ky.at + j]) + @abs(out[j])) * sens[i];
            }
        };
    }

    /// Gate 4's outputs: every weight 1 / keys, the values summed in the accumulator and order.
    fn simulate(d: *Draft, acc: c.Accum, order: g.Order, out: []f64) !void {
        const t = try d.rig.a.alloc(f64, window + d.blk);
        defer d.rig.a.free(t);
        for (0..d.shape.rows) |m| for (0..heads) |h| for (0..hd) |j| {
            const first, const ring_keys = windowOf(d, m);
            const keys = ring_keys + d.blk;
            const w = 1 / @as(f64, @floatFromInt(keys));
            const kvh = h / (heads / kv_heads);
            for (t[0..keys], 0..) |*x, i| {
                const ky = key(d, m, i, ring_keys, first, kvh);
                x.* = w * ky.v[ky.at + j];
            }
            out[(m * heads + h) * hd + j] = rows_.sumIn(acc, order, t[0..keys]);
        };
    }
};
