//! Routing under the contract: core tf_route_logits at each family's width, and Flint's fl_route_pick and fl_sort.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const fl = @import("metal_flint.zig");

const c = contract.spec;
const g = contract.gates;
const bf = fl.bf;
const Range = fl.Range;
const route = tf.moe_route;

// --- tf_route_logits: logits [rows, E] = x W^T over a packed bf16 router, fp32 sums ---------------------------------

const Router = struct { label: []const u8, shape: route.Shape };

/// Every family's router: Flint's (bf16 logits, as MLX rounds them), GLM's two, Step-5's.
const routers = [_]Router{
    .{ .label = "flint", .shape = .{ .hidden = fl.hidden, .experts = fl.experts, .topk = fl.topk, .round_logits = true } },
    .{ .label = "glm", .shape = .{ .hidden = 4096, .experts = 288, .topk = 8 } },
    .{ .label = "glm igate", .shape = .{ .hidden = 4096, .experts = 128, .topk = 8, .out_bf16 = true } },
    .{ .label = "step5", .shape = .{ .hidden = 4096, .experts = 352, .topk = 8 } },
};

/// A lane's chain (hidden / 1024 slices of 4 products each), the simd_sum tree, then the eight partials in order.
fn logitsDepth(s: c.Shape) usize {
    return 4 * (s.k / 1024) + 5 + 7;
}

fn routeText(a: std.mem.Allocator, s: route.Shape, drop_round: bool) ![]u8 {
    var x = s;
    if (drop_round) x.round_logits = false;
    return route.source(a, x);
}

pub const Logits = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    r: Router,
    served: [3]c.Shape,
    control: ?g.Gate = null,
    base_buf: [160]u8 = undefined,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    rows: usize = 0,
    fill: g.Fill = .random,
    x: []u16 = &.{},
    w: []u16 = &.{},
    packed_w: []u16 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn base(r: Router, buf: []u8) []const u8 {
        const s = r.shape;
        const how: []const u8 = if (s.round_logits) "bf16 logits in fp32" else if (s.out_bf16) "bf16 out" else "fp32 out";
        return std.fmt.bufPrint(buf, "metal moe_route tf_route_logits {s} hidden {d} experts {d}, {s}", .{ r.label, s.hidden, s.experts, how }) catch unreachable;
    }

    /// The launch's rows: 16 at most (a window, or 16 of a prompt chunk's), one, and a part-filled block of four.
    fn shapes(r: Router) [3]c.Shape {
        const H: usize = r.shape.hidden;
        const E: usize = r.shape.experts;
        return .{ .{ .rows = 16, .n = E, .k = H }, .{ .rows = 1, .n = E, .k = H }, .{ .rows = 6, .n = E, .k = H } };
    }

    /// Rows: the launch's rows; n: experts; k: the hidden width.
    pub fn spec(name: []const u8, r: Router, served: []const c.Shape) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = logitsDepth,
            .out = if (r.shape.out_bf16) .bf16 else .f32,
            .tiles = .{ .{ .size = 4, .step = 1, .min = 1 }, .{}, .{} },
            .served = served,
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, r: Router, control: bool) !Logits {
        const text = try routeText(a, r.shape, control);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        defer lib.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try mtl.Pipeline.init(device, lib, route.names[0], false), .r = r, .served = shapes(r), .control = if (control) .shapes else null };
    }
    pub fn deinit(t: *Logits) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Logits) void {
        inline for (.{ "x", "w", "packed_w" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Logits) g.Kernel {
        t.name = fl.named(base(t.r, &t.base_buf), if (t.control != null) "the logits' bf16 rounding dropped" else null, &t.label);
        return .{ .spec = spec(t.name, t.r, &t.served), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Logits);

    pub fn prepare(t: *Logits, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const H: usize = t.r.shape.hidden;
        const E: usize = t.r.shape.experts;
        if (shape.rows == 0 or shape.rows > 16 or shape.n != E or shape.k != H) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.fill = fill;
        t.x = try t.a.alloc(u16, shape.rows * H);
        t.w = try t.a.alloc(u16, E * H);
        t.packed_w = try t.a.alloc(u16, E * H);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // integers whose dots are exact in fp32 in any order: the second half mirrors the first, all but one nudged
            const half = H / 2;
            for (0..shape.rows) |r| for (0..half) |k| {
                const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 63));
                t.x[r * H + k] = c.toBf16(v);
                t.x[r * H + half + k] = c.toBf16(v);
            };
            for (0..E) |e| for (0..half) |k| {
                const v: f32 = @floatFromInt(@as(i32, @intCast(1 + rng.uintLessThan(u32, 63))) * (if (rng.boolean()) @as(i32, 1) else -1));
                t.w[e * H + k] = c.toBf16(v);
                t.w[e * H + half + k] = c.toBf16(-v + @as(f32, if (k == e % 7) 1 else 0));
            };
        } else {
            for (t.x) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 3);
            for (t.w) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.05);
        }
        if (fill == .nonfinite) {
            // NaN in row 0, +Inf in row 1, -Inf in row 2, both infinities in row 3
            const at = [_][2]usize{ .{ 0, 3 }, .{ 1, 5 }, .{ 2, 9 }, .{ 3, 11 }, .{ 3, 700 } };
            const vals = [_]f32{ fl.planted[0], fl.planted[1], fl.planted[2], fl.planted[1], fl.planted[2] };
            for (at, vals) |p, v| if (p[0] < shape.rows) {
                t.x[p[0] * H + p[1]] = c.toBf16(v);
            };
        }
        route.pack(t.w, t.packed_w, E, H);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.packed_w));
        t.ins[2] = try t.rig.input(std.mem.asBytes(&@as(i32, @intCast(shape.rows))));
        t.out = try t.rig.output(shape.rows * E * @as(usize, if (t.r.shape.out_bf16) 2 else 4));
        return true;
    }

    pub fn run(t: *Logits) !void {
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        e.setBuffer(t.ins[2].buf, t.ins[2].off, 2);
        e.setBuffer(t.out.buf, t.out.off, 3);
        // as route.logits: a 4-expert block for four rows a threadgroup of 256
        e.dispatchGroups(mtl.Size.of(t.r.shape.experts / 4, (t.rows + 3) / 4, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Logits) usize {
        return t.rows * t.r.shape.experts;
    }

    fn rounds(t: *const Logits) bool {
        return t.r.shape.round_logits or t.r.shape.out_bf16;
    }

    fn dot(t: *const Logits, i: usize, acc: ?struct { a: c.Accum, o: g.Order }) [2]f64 {
        const H: usize = t.r.shape.hidden;
        const r = i / t.r.shape.experts;
        const e = i % t.r.shape.experts;
        var terms: [4096]f64 = undefined;
        var sum: f64 = 0;
        var mag: f64 = 0;
        for (terms[0..H], 0..) |*x, k| {
            x.* = @as(f64, c.fromBf16(t.x[r * H + k])) * c.fromBf16(t.w[e * H + k]);
            sum += x.*;
            mag += @abs(x.*);
        }
        if (acc) |s| sum = contract.rows.sumIn(s.a, s.o, terms[0..H]);
        return .{ sum, mag };
    }

    pub fn reference(t: *Logits, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            const d = t.dot(i, null);
            w.* = if (t.rounds()) bf(d[0]) else d[0];
            m.* = if (std.math.isFinite(d[1])) d[1] else 0;
        }
    }

    /// The fp32 sums within gamma(depth) of their terms' size (none on gate 4's exact inputs), then the bf16 point.
    pub fn intervals(t: *Logits, low: []f64, high: []f64) !void {
        const gam = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(logitsDepth(.{ .rows = t.rows, .n = t.r.shape.experts, .k = t.r.shape.hidden })), fl.u32_ulp);
        for (low, high, 0..) |*l, *h, i| {
            const d = t.dot(i, null);
            var r = Range.point(d[0]).widen(gam * d[1] + fl.tiny);
            if (t.rounds()) r = r.bf16();
            l.* = r.lo;
            h.* = r.hi;
        }
    }

    pub fn bounds(t: *Logits, want: []const f64, _: []const f64, allow: []f64) !void {
        try fl.allowFrom(t.a, t, want, allow);
    }

    pub fn simulate(t: *Logits, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*o, i| {
            const s = t.dot(i, .{ .a = acc, .o = order })[0];
            o.* = if (t.rounds()) bf(s) else s;
        }
    }
};

// --- fl_route_pick: MLX's sigmoid scores, top 6 (ties to the lower id), weights the picks over their fp32 sum -------

/// MLX's sigmoid in fp32 as the kernel computes it (1 / (1 + exp(|x|)) below 0, one less that above).
fn sigmoid32(x: f32) f32 {
    const y = 1.0 / (1.0 + @exp(@abs(x)));
    return if (x < 0) y else 1.0 - y;
}

pub const Pick = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    control: ?g.Gate = null,
    label: [160]u8 = undefined,
    name: []const u8 = "",
    rows: usize = 0,
    logits: []f32 = &.{},
    in: rig_.Ref = undefined,
    picks: rig_.Ref = undefined,
    wts: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    const base = "metal flint fl_route_pick experts 256 top 6 sigmoid";
    const last_tie: fl.Edit = .{ .kernel = "fl_route_pick", .from = "(sc[j] == best && id < bid)", .to = "(sc[j] == best && id > bid)" };

    /// Rows: the router launch's rows (16 at most); n: experts.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = fl.noSums,
            .out = .raw,
            .nan = .{ .stated = "MLX's fp32 sigmoid scores; the top 6 by score, ties to the lower id, a NaN score never picked; weights each pick's score over the picks' fp32 sum, within 8 ulps" },
            .tiles = .{ .{ .size = 4, .step = 1, .min = 1 }, .{}, .{} },
            .served = &.{ .{ .rows = 16, .n = fl.experts, .k = 1 }, .{ .rows = 1, .n = fl.experts, .k = 1 } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !Pick {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try fl.gluePipe(a, device, "fl_route_pick", if (control) last_tie else null), .control = if (control) .shapes else null };
    }
    pub fn deinit(t: *Pick) void {
        t.a.free(t.logits);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Pick) g.Kernel {
        t.name = fl.named(base, if (t.control != null) "a lane's tie goes to the higher id" else null, &t.label);
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Pick);

    pub fn prepare(t: *Pick, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != fl.experts) return false;
        t.rig.reset();
        t.a.free(t.logits);
        t.rows = shape.rows;
        t.logits = try t.a.alloc(f32, shape.rows * fl.experts);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // bf16 logits in quarters from -8 to 8: many exact ties, distinct scores far apart; a few saturate to 1
        for (t.logits) |*v| v.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 64))) / 4 - 8;
        for (0..shape.rows) |r| if (r % 3 == 1) {
            t.logits[r * fl.experts + 200] = 40;
            t.logits[r * fl.experts + 17] = 60;
        };
        if (fill == .nonfinite) for (0..shape.rows) |r| {
            const row = t.logits[r * fl.experts ..][0..fl.experts];
            row[(r * 37) % fl.experts] = fl.nan32;
            row[(r * 53 + 5) % fl.experts] = if (r % 2 == 0) fl.inf32 else -fl.inf32;
        };
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.logits));
        t.picks = try t.rig.output(shape.rows * fl.topk * 4);
        t.wts = try t.rig.output(shape.rows * fl.topk * 4);
        return true;
    }

    pub fn run(t: *Pick) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.in.buf, t.in.off, 0);
        p.e.setValue(fl.Args{ .rows = @intCast(t.rows), .width = fl.experts, .heads = fl.topk }, 1);
        p.e.setBuffer(t.picks.buf, t.picks.off, 2);
        p.e.setBuffer(t.wts.buf, t.wts.off, 3);
        p.e.dispatchGroups(mtl.Size.of(t.rows, 1, 1), mtl.Size.of(32, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Pick) usize {
        return t.rows * fl.topk * 8;
    }
    pub fn reference(_: *Pick, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    /// Row r's top 6 on the host: the rule's order over fp32 scores, which every distinct logit here keeps apart.
    fn hostPicks(row: []const f32, out: *[fl.topk]u32, scores: *[fl.topk]f32) void {
        var taken: [fl.experts]bool = @splat(false);
        for (out, scores) |*o, *s| {
            var best: f32 = -std.math.inf(f32);
            var bid: u32 = fl.experts;
            for (row, 0..) |x, e| {
                const sc = sigmoid32(x);
                if (taken[e] or std.math.isNan(sc)) continue;
                if (sc > best or (sc == best and e < bid)) {
                    best = sc;
                    bid = @intCast(e);
                }
            }
            taken[bid] = true;
            o.* = bid;
            s.* = best;
        }
    }

    pub fn check(t: *Pick, _: g.Gate) !?[]const u8 {
        const picks = std.mem.bytesAsSlice(i32, t.rig.bytes(t.picks, t.rows * fl.topk * 4));
        const wts = std.mem.bytesAsSlice(f32, t.rig.bytes(t.wts, t.rows * fl.topk * 4));
        for (0..t.rows) |r| {
            var ids: [fl.topk]u32 = undefined;
            var sc: [fl.topk]f32 = undefined;
            hostPicks(t.logits[r * fl.experts ..][0..fl.experts], &ids, &sc);
            var total: f32 = sc[0];
            for (sc[1..]) |s| total += s;
            for (0..fl.topk) |k| {
                const got = picks[r * fl.topk + k];
                if (got != ids[k]) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d} is expert {d} where the rule picks {d}", .{ r, k, got, ids[k] });
                const want = sc[k] / total;
                const w = wts[r * fl.topk + k];
                if (!(@abs(w - want) <= 8 * fl.u32_ulp * want)) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d}'s weight is {e} where the rule gives {e}", .{ r, k, w, want });
            }
        }
        return null;
    }
};

// --- fl_sort: a prompt chunk's picks by expert, each expert's in slot order ------------------------------------------

pub const Sort = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    control: ?g.Gate = null,
    label: [160]u8 = undefined,
    name: []const u8 = "",
    slots: usize = 0,
    picks: []i32 = &.{},
    in: rig_.Ref = undefined,
    offsets: rig_.Ref = undefined,
    order: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    const base = "metal flint fl_sort experts 256";
    const shifted: fl.Edit = .{ .kernel = "fl_sort", .from = "offsets[t] = at;", .to = "offsets[t] = at + 1;" };

    /// Rows: slots (6 a token; the gates' small calls end mid-token); n: experts.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = fl.noSums,
            .out = .raw,
            .nan = .{ .stated = "the host's stable sort: offsets its exclusive prefix counts, order each expert's slots ascending inside its own offsets" },
            .tiles = .{ .{ .size = 1024, .step = fl.topk, .min = fl.topk * (fl.batch_rows + 1) }, .{}, .{} },
            // the prompt path's chunks: 33 rows (past a shared decode forward) to 2,048
            .served = &.{ .{ .rows = fl.prompt_rows * fl.topk, .n = fl.experts, .k = 1 }, .{ .rows = (fl.batch_rows + 1) * fl.topk, .n = fl.experts, .k = 1 } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !Sort {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try fl.gluePipe(a, device, "fl_sort", if (control) shifted else null), .control = if (control) .shapes else null };
    }
    pub fn deinit(t: *Sort) void {
        t.a.free(t.picks);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Sort) g.Kernel {
        t.name = fl.named(base, if (t.control != null) "every offset one past its first pair" else null, &t.label);
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Sort);

    pub fn prepare(t: *Sort, shape: c.Shape, _: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != fl.experts) return false;
        t.rig.reset();
        t.a.free(t.picks);
        t.slots = shape.rows;
        t.picks = try t.a.alloc(i32, shape.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // each token's 6 distinct experts; a few hot experts take most tokens, so empty and crowded experts both occur
        for (0..(shape.rows + fl.topk - 1) / fl.topk) |tok| {
            var used: [fl.experts]bool = @splat(false);
            for (t.picks[tok * fl.topk .. @min(shape.rows, tok * fl.topk + fl.topk)]) |*p| {
                var e: u32 = if (rng.boolean()) rng.uintLessThan(u32, 12) else rng.uintLessThan(u32, fl.experts);
                while (used[e]) e = (e + 1) % fl.experts;
                used[e] = true;
                p.* = @intCast(e);
            }
        }
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.picks));
        t.offsets = try t.rig.output(fl.experts * 4);
        t.order = try t.rig.output(shape.rows * 4);
        return true;
    }

    pub fn run(t: *Sort) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.in.buf, t.in.off, 0);
        p.e.setValue(fl.Args{ .rows = @intCast(t.slots), .width = fl.experts }, 1);
        p.e.setBuffer(t.offsets.buf, t.offsets.off, 2);
        p.e.setBuffer(t.order.buf, t.order.off, 3);
        p.e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Sort) usize {
        return (fl.experts + t.slots) * 4;
    }
    pub fn reference(_: *Sort, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    pub fn check(t: *Sort, _: g.Gate) !?[]const u8 {
        const offs = std.mem.bytesAsSlice(i32, t.rig.bytes(t.offsets, fl.experts * 4));
        const order = std.mem.bytesAsSlice(i32, t.rig.bytes(t.order, t.slots * 4));
        var want: [fl.experts + 1]i32 = @splat(0);
        for (t.picks) |e| want[@intCast(e + 1)] += 1;
        for (1..fl.experts + 1) |e| want[e] += want[e - 1];
        for (0..fl.experts) |e| if (offs[e] != want[e]) return try std.fmt.bufPrint(&t.msg, "expert {d}'s offset is {d} where the host counts {d}", .{ e, offs[e], want[e] });
        var i: usize = 0;
        for (0..fl.experts) |e| for (t.picks, 0..) |p, slot| if (p == e) {
            if (order[i] != slot) return try std.fmt.bufPrint(&t.msg, "order[{d}] = {d} where the host's stable sort gives slot {d} (expert {d})", .{ i, order[i], slot, e });
            i += 1;
        };
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    logits: fl.List(Logits) = .{},
    picks: fl.List(Pick) = .{},
    sorts: fl.List(Sort) = .{},

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        var buf: [160]u8 = undefined;
        for (routers) |r| for ([_]bool{ false, true }) |control| {
            // the control drops the rounding only Flint's router has
            if (control and !r.shape.round_logits) continue;
            if (fl.wanted(only, Logits.base(r, &buf))) try s.logits.add(s.a, s.kernels, try Logits.init(s.a, s.device, s.queue, r, control));
        };
        for ([_]bool{ false, true }) |control| {
            if (fl.wanted(only, Pick.base)) try s.picks.add(s.a, s.kernels, try Pick.init(s.a, s.device, s.queue, control));
            if (fl.wanted(only, Sort.base)) try s.sorts.add(s.a, s.kernels, try Sort.init(s.a, s.device, s.queue, control));
        }
    }
    pub fn deinit(s: *Suite) void {
        s.logits.deinit(s.a);
        s.picks.deinit(s.a);
        s.sorts.deinit(s.a);
    }
};

/// Gate 5's texts for `moe_route logits <label> ...`: that family's router (fl_route_pick and fl_sort are glue's).
pub fn limitsText(a: std.mem.Allocator, _: mtl.Device, name: []const u8) !?fl.Texts {
    const head = "moe_route logits ";
    if (!std.mem.startsWith(u8, name, head)) return null;
    const label = name[head.len..std.mem.lastIndexOfScalar(u8, name, ' ').?];
    for (routers) |r| if (std.mem.eql(u8, r.label, label)) {
        const text = try routeText(a, r.shape, false);
        return .{ .here = text, .pre_m5 = text };
    };
    return error.UnknownLaunch;
}

test "the host's picks: top 6 by fp32 score, saturated scores tied to the lower id, NaN never" {
    var row: [fl.experts]f32 = @splat(-8);
    row[9] = 3;
    row[3] = fl.nan32;
    row[70] = 60;
    row[40] = 40;
    row[41] = 1;
    row[42] = 1;
    row[5] = 0.5;
    var ids: [fl.topk]u32 = undefined;
    var sc: [fl.topk]f32 = undefined;
    Pick.hostPicks(&row, &ids, &sc);
    try std.testing.expectEqualSlices(u32, &.{ 40, 70, 9, 41, 42, 5 }, &ids);
}
