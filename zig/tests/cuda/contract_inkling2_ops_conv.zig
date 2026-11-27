//! Inkling 2's short convolutions under the contract: tf_ink2_conv's sites and states, and tf_ink2_conv_rewind.
const std = @import("std");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_inkling2_ops_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;
const ops = ref.ops;
const step = ref.step;

/// forward.zig's four sites: k and v (sliding: static taps; global: dynamic), attention's and the MLP's residual.
pub const Form = enum { kv_static, kv_dynamic, attn, attn_boundary, mlp };

/// What a registration reads: the site's output, or the states and the inputs it keeps (hist).
pub const View = enum { out, state };

/// The served kernel, or a control: every stream on slot 0's state, or rank 0's partial read in rank 1's place.
pub const Fault = enum { none, one_slot, same_part };

fn width(f: Form) usize {
    return if (f == .kv_static or f == .kv_dynamic) ref.kv_width else ref.hidden;
}

fn parts(f: Form) usize {
    return if (f == .attn or f == .attn_boundary or f == .mlp) 2 else 1;
}

fn mode(f: Form) i32 {
    return switch (f) {
        .kv_static, .kv_dynamic => 0,
        .attn, .attn_boundary => 1,
        .mlp => 2,
    };
}

/// The ranks' partials added (one add), then the four taps' products and x added in order.
fn depth(_: c.Shape) usize {
    return 1 + 4;
}

const names = [5][2][]const u8{
    .{ "inkling2 ops tf_ink2_conv k/v static taps (sliding layers), bf16 out", "inkling2 ops tf_ink2_conv k/v static taps, states and kept inputs" },
    .{ "inkling2 ops tf_ink2_conv k/v dynamic taps (global layers), bf16 out", "" },
    .{ "inkling2 ops tf_ink2_conv attention site (rank partials, base + bf16(y))", "inkling2 ops tf_ink2_conv attention site, states and kept inputs" },
    .{ "inkling2 ops tf_ink2_conv attention site at a block boundary (no base)", "" },
    .{ "inkling2 ops tf_ink2_conv MLP site (rank partials, bf16(resid + bf16(y)))", "" },
};

fn servedAt(comptime C: usize) []const c.Shape {
    return &.{ .{ .rows = 16, .n = C, .k = 1 }, .{ .rows = 1, .n = C, .k = 1 }, .{ .rows = 64, .n = C, .k = 4 }, .{ .rows = 128, .n = C, .k = 8 }, .{ .rows = 8, .n = C, .k = 8 }, .{ .rows = 2048, .n = C, .k = 1 } };
}

/// Rows of the call x the site's width x streams (each its rows, its state slot): decode, windows, prompt chunks.
pub fn convSpec(f: Form, v: View, fault: Fault) c.Spec {
    const served = if (width(f) == ref.kv_width) servedAt(ref.kv_width) else servedAt(ref.hidden);
    var s: c.Spec = .{
        .name = names[@backingInt(f)][@backingInt(v)],
        .accum = if (v == .state and parts(f) == 1) .none else .f32, // a plain site keeps its inputs: no sums
        .depth = depth,
        .out = if (v == .out and mode(f) != 1) .bf16 else .f32,
        .in_place = v == .state,
        .tiles = .{ .{ .size = 32 }, .{}, .{} },
        .served = served,
    };
    if (fault == .one_slot) {
        s.name = "control: inkling2 ops tf_ink2_conv k/v static taps, every stream on slot 0's state";
        s.served = served[2..4];
        s.tiles = .{ .{}, .{}, .{} };
    }
    if (fault == .same_part) {
        s.name = "control: inkling2 ops tf_ink2_conv attention site, rank 0's partial read in rank 1's place";
        s.served = served[0..1];
        s.tiles = .{ .{}, .{}, .{} };
    }
    return s;
}

/// One conv launch over `k` streams' rows: inputs uploaded, the states restored before every launch.
pub const Conv = struct {
    kern: *const ops.Kernels,
    form: Form,
    view: View = .out,
    fault: Fault = .none,
    shape: c.Shape = undefined,
    x: []f32 = &.{}, // [parts][rows][C]
    w: []u16 = &.{}, // static [C][4], dynamic [4][C]
    dyn: []f32 = &.{}, // [rows][heads][4]
    base: []u16 = &.{}, // [rows][C]
    resid: []f32 = &.{}, // [rows][C]: the MLP site's input
    state: []f32 = &.{}, // [slots][3][C]
    segs: [][4]i32 = &.{},
    args: ops.ConvArgs = undefined,
    streams: usize = 0,

    pub const sums = {};

    pub fn spec(self: *const Conv) c.Spec {
        return convSpec(self.form, self.view, self.fault);
    }

    pub fn control(self: *const Conv) ?g.Gate {
        return if (self.fault == .none) null else .shapes;
    }

    pub fn free(self: *Conv, a: std.mem.Allocator) void {
        inline for (.{ "x", "w", "dyn", "base", "resid", "state", "segs" }) |f| {
            a.free(@field(self, f));
            @field(self, f) = &.{};
        }
    }

    const heads = 4;

    /// Values in [-2, 2); stress: products near 2^16 adding to 2^18 (fp16 overflows), partials near 2^17 cancelling.
    pub fn prepare(self: *Conv, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        const C = width(self.form);
        if (shape.rows == 0 or shape.n != C or shape.k == 0) return false;
        self.free(a);
        self.shape = shape;
        const rows = shape.rows;
        const np = parts(self.form);
        const stress = fill == .stress;
        const round = np == 2;
        self.x = try a.alloc(f32, np * rows * C);
        for (0..rows * C) |i| {
            const v = if (stress) ref.int(rng, -256, 256) else rng.float(f32) * 4 - 2;
            if (np == 1) self.x[i] = v else {
                const big = if (stress) ref.int(rng, 40000, 120000) else rng.float(f32) * 2 - 1;
                self.x[i] = big;
                self.x[rows * C + i] = v - big;
            }
        }
        if (fill == .nonfinite) ref.plant(f32, self.x, @min(rows, 4), C, ref.asF32);
        self.w = try a.alloc(u16, 4 * C);
        for (self.w) |*v| v.* = c.toBf16(if (stress) ref.int(rng, 64, 255) * (if (rng.boolean()) @as(f32, -1) else 1) else rng.float(f32) * 2 - 1);
        self.dyn = try a.alloc(f32, rows * heads * 4);
        for (self.dyn) |*v| v.* = if (stress) ref.int(rng, -8, 8) else rng.float(f32) - 0.5;
        self.base = try ref.bfs(a, rng, rows * C, -4, 4);
        self.resid = try ref.f32s(a, rng, rows * C, -4, 4);
        self.streams = @min(shape.k, rows); // a call has no more streams than rows (gate 4 keeps four rows)
        const ns = self.streams;
        const slots = ns + 1;
        self.state = try a.alloc(f32, slots * 3 * C);
        for (self.state) |*v| v.* = if (stress) ref.int(rng, -256, 256) else rng.float(f32) * 4 - 2;
        if (round) for (self.state) |*v| {
            v.* = c.fromBf16(c.toBf16(v.*));
        };
        // streams of uneven lengths (weights 1, 2, ...), each its own slot: stream i in slot k - i
        self.segs = try a.alloc([4]i32, ns);
        const run_segs = try a.dupe([4]i32, self.segs);
        defer a.free(run_segs);
        var row0: usize = 0;
        const tri = ns * (ns + 1) / 2;
        for (self.segs, run_segs, 0..) |*sg, *rs, i| {
            const n = if (i + 1 == ns) rows - row0 else 1 + (rows - ns) * (i + 1) / tri;
            sg.* = .{ @intCast(row0), @intCast(n), @intCast(ns - i), 0 };
            rs.* = sg.*;
            if (self.fault == .one_slot) rs[2] = 0; // the control's kernel only: the reference keeps each stream's slot
            row0 += n;
        }
        const out_bytes = rows * C * @as(usize, if (mode(self.form) == 1) 4 else 2);
        const mine = self.view == .out;
        self.args = .{
            .x = try ref.input(rig, self.x),
            .state = 0,
            .w = try ref.input(rig, self.w),
            .dyn = if (self.form == .kv_dynamic) try ref.input(rig, self.dyn) else 0,
            .base = if (self.form == .attn) try ref.input(rig, self.base) else 0,
            .resid = switch (mode(self.form)) {
                0 => 0,
                1 => if (mine) try rig.output(out_bytes) else try rig.scratch(out_bytes),
                else => try ref.input(rig, self.resid),
            },
            .out = if (mode(self.form) == 1) 0 else if (mine) try rig.output(out_bytes) else try rig.scratch(out_bytes),
            .segs = try ref.input(rig, run_segs),
            .hist = 0,
            .part_stride = if (self.fault == .same_part) 0 else @intCast(rows * C), // the control adds part 0 twice
            .state_stride = @intCast(3 * C),
            .parts = @intCast(np),
            .round_in = @intFromBool(round),
            .heads = heads,
            .c = @intCast(C),
            .rows = @intCast(rows),
            .mode = mode(self.form),
        };
        self.args.state = if (mine) try rig.scratch(self.state.len * 4) else try rig.output(self.state.len * 4);
        self.args.hist = if (mine) try rig.scratch(rows * C * 4) else try rig.output(rows * C * 4);
        return true;
    }

    pub fn run(self: *Conv, rig: *rig_.Rig) !void {
        try ref.restore(rig, self.args.state, std.mem.sliceAsBytes(self.state));
        try ops.conv(self.kern, rig.s, self.args, self.streams);
    }

    pub fn count(self: *const Conv) usize {
        const C = width(self.form);
        return if (self.view == .out) self.shape.rows * C else self.state.len + self.shape.rows * C;
    }

    /// Row t's input at column col: the ranks' partials added in order, rounded to bf16 at the residual sites.
    fn input(self: *const Conv, s: Sums, t: usize, col: usize) Iv {
        const C = width(self.form);
        const rows = self.shape.rows;
        var x = Iv.of(self.x[t * C + col]);
        if (parts(self.form) == 2) x = step(s, x, Iv.of(self.x[(rows + t) * C + col]), false).bf();
        return x;
    }

    pub fn solve(self: *const Conv, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const C = width(self.form);
        const hist = out[if (self.view == .state) self.state.len else 0..];
        if (self.view == .state) for (out[0..self.state.len], self.state) |*o, v| {
            o.* = Iv.of(v);
        };
        for (self.segs) |sg| for (0..C) |col| {
            const r0: usize = @intCast(sg[0]);
            const slot: usize = @intCast(sg[2]);
            const st = self.state[slot * 3 * C ..];
            var h = [3]Iv{ Iv.of(st[col]), Iv.of(st[C + col]), Iv.of(st[2 * C + col]) };
            for (r0..r0 + @as(usize, @intCast(sg[1]))) |t| {
                const x = self.input(s, t, col);
                if (self.view == .state) hist[t * C + col] = x else out[t * C + col] = self.y(s, t, col, h, x);
                h = .{ h[1], h[2], x };
            }
            if (self.view == .state) for (h, 0..) |v, k| {
                out[slot * 3 * C + k * C + col] = v;
            };
        };
    }

    fn tap(self: *const Conv, k: usize, col: usize) Iv {
        const C = width(self.form);
        return Iv.of(ref.bf(self.w[if (self.form == .kv_dynamic) k * C + col else col * 4 + k]));
    }

    /// The site's output at row t from its three prior inputs h (oldest first) and x.
    fn y(self: *const Conv, s: Sums, t: usize, col: usize, h: [3]Iv, x: Iv) Iv {
        const C = width(self.form);
        var v: Iv = undefined;
        if (self.form == .kv_dynamic) {
            const d = self.dyn[(t * heads + col / (C / heads)) * 4 ..][0..4];
            var w: [4]Iv = undefined;
            for (&w, 0..) |*wk, k| wk.* = self.tap(k, col).add(Iv.of(d[k]));
            v = step(s, w[0].mul(x), w[1].mul(h[2]), false);
            v = step(s, v, w[2].mul(h[1]), false);
            v = step(s, v, w[3].mul(h[0]), false);
            v = step(s, v, x, false).div(Iv.of(3));
        } else {
            v = step(s, self.tap(0, col).mul(h[0]), self.tap(1, col).mul(h[1]), false);
            v = step(s, v, self.tap(2, col).mul(h[2]), false);
            v = step(s, v, self.tap(3, col).mul(x), false);
            v = step(s, v, x, false).div(Iv.of(@as(f32, 2.2360679774997896)));
        }
        return switch (self.form) {
            .kv_static, .kv_dynamic => v.bf(),
            .attn => Iv.of(ref.bf(self.base[t * C + col])).add(v.bf()),
            .attn_boundary => v.bf(),
            .mlp => Iv.of(self.resid[t * C + col]).add(v.bf()).bf(),
        };
    }
};

/// Kept rows x the widest site x the slot's first row in hist: four sites, two 512 wide and two 4,608.
pub fn rewindSpec(long: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 3, .n = ref.hidden, .k = 0 }, .{ .rows = 1, .n = ref.hidden, .k = 5 }, .{ .rows = 2, .n = ref.hidden, .k = 0 }, .{ .rows = 4, .n = ref.hidden, .k = 3 }, .{ .rows = 17, .n = ref.hidden, .k = 5 }, .{ .rows = 128, .n = ref.hidden, .k = 0 } };
    const name = if (long) "control: inkling2 ops tf_ink2_conv_rewind keeping one row more" else "inkling2 ops tf_ink2_conv_rewind (a slot's states from its backup and kept rows)";
    return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .in_place = true, .nan = .{ .stated = "a move: every bit copied, NaN payloads too" }, .served = if (long) served[0..2] else served };
}

/// Each site's state of one slot: the last three of (its backup's three inputs, the kept rows' inputs in hist).
pub const Rewind = struct {
    kern: *const ops.Kernels,
    long: bool = false,
    shape: c.Shape = undefined,
    hist: []f32 = &.{},
    backup: []f32 = &.{},
    states: [4][]f32 = .{ &.{}, &.{}, &.{}, &.{} },
    dev: [3]u64 = @splat(0),
    out: [4]u64 = @splat(0),

    const widths = [4]usize{ ref.kv_width, ref.kv_width, ref.hidden, ref.hidden };
    const slots = 3;
    const slot = 1;

    pub fn spec(self: *const Rewind) c.Spec {
        return rewindSpec(self.long);
    }

    pub fn control(self: *const Rewind) ?g.Gate {
        return if (self.long) .shapes else null;
    }

    pub fn free(self: *Rewind, a: std.mem.Allocator) void {
        a.free(self.hist);
        a.free(self.backup);
        for (&self.states) |*s| a.free(s.*);
        self.hist = &.{};
        self.backup = &.{};
        self.states = .{ &.{}, &.{}, &.{}, &.{} };
    }

    fn histRows(self: *const Rewind) usize {
        return self.shape.k + self.shape.rows + 1;
    }

    pub fn prepare(self: *Rewind, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != ref.hidden) return false;
        self.free(a);
        self.shape = shape;
        const hr = self.histRows();
        var total: usize = 0;
        for (widths) |w| total += w;
        self.hist = try ref.f32s(a, rng, hr * total, -4, 4);
        self.backup = try ref.f32s(a, rng, 3 * total, -4, 4);
        if (fill == .nonfinite) {
            ref.plant(f32, self.backup, 3, total, ref.asF32);
            ref.plant(f32, self.hist, @min(hr, 4), total, ref.asF32);
        }
        var sites: [4]ops.RewindSite = undefined;
        var h_off: usize = 0;
        var b_off: usize = 0;
        for (&sites, &self.states, &self.out, widths) |*st, *s, *o, w| {
            s.* = try ref.f32s(a, rng, slots * 3 * w, -4, 4);
            o.* = try rig.output(s.len * 4);
            st.* = .{ .state = o.*, .hist = @intCast(h_off * 4), .backup = @intCast(b_off * 4), .c = @intCast(w) };
            h_off += hr * w;
            b_off += 3 * w;
        }
        self.dev = .{ try ref.input(rig, std.mem.sliceAsBytes(&sites)), try ref.input(rig, self.backup), try ref.input(rig, self.hist) };
        return true;
    }

    pub fn run(self: *Rewind, rig: *rig_.Rig) !void {
        for (self.states, self.out) |s, o| try ref.restore(rig, o, std.mem.sliceAsBytes(s));
        const n = self.shape.rows + @intFromBool(self.long);
        try ops.rewind(self.kern, rig.s, self.dev[0], 4, ref.hidden, self.dev[1], self.dev[2], slot, self.shape.k, n);
    }

    pub fn count(self: *const Rewind) usize {
        var n: usize = 0;
        for (self.states) |s| n += s.len * 4;
        return n;
    }

    pub fn check(self: *const Rewind, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const n = self.shape.rows;
        const row0 = self.shape.k;
        const hr = self.histRows();
        var at: usize = 0;
        var h_off: usize = 0;
        var b_off: usize = 0;
        for (self.states, widths, 0..) |s, w, site| {
            for (s, 0..) |old, e| {
                const sl = e / (3 * w);
                const k = e / w % 3;
                const col = e % w;
                const i = @as(isize, @intCast(n)) - 3 + @as(isize, @intCast(k));
                const want = if (sl != slot) old else if (i < 0) self.backup[b_off + @as(usize, @intCast(3 + i)) * w + col] else self.hist[h_off + (row0 + @as(usize, @intCast(i))) * w + col];
                const have = std.mem.readInt(u32, got[at + 4 * e ..][0..4], .little);
                if (have != @as(u32, @bitCast(want))) return try std.fmt.bufPrint(msg, "site {d} slot {d} input {d} column {d}: {x} where {x} is due", .{ site, sl, k, col, have, @as(u32, @bitCast(want)) });
            }
            at += s.len * 4;
            h_off += hr * w;
            b_off += 3 * w;
        }
        return null;
    }
};
