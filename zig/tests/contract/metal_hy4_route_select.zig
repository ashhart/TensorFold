//! Hy4-flash's expert picks: core tf_route_select (windows) and route_rows.metal (prompt chunks) against the rule.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const rt = @import("metal_hy4_route.zig");

const c = contract.spec;
const g = contract.gates;
const fl = h.fl;
const route = tf.moe_route;
const Dims = rt.Dims;

/// The core router's limits (moe_route.zig's source: 512 experts at most; 16 picks fill a window's 256 slots).
pub const max_experts = 512;
pub const max_topk = 16;
const max_rows = h.hy.metal_routed.max_rows;

pub const rule = "fp32 sigmoid scores, keys score + bias; the top k by key, ties to the lower id, a NaN key never " ++
    "picked (a picked key turns -inf, so a row short of non-NaN keys repeats its lowest-id pick, and a row of none " ++
    "picks id E at weight 0); weights each pick's score over the picks' sum times the scale, within 64 fp32 roundoffs " ++
    "of float64";

/// The routers' fp32 score as tf_sigmoid_precise computes it: 1 / (1 + exp|x|) below 0, one less that above.
fn score32(x: f32) f32 {
    const y = 1.0 / (1.0 + @exp(@abs(x)));
    return if (x < 0) y else 1.0 - y;
}

/// The score in float64, the weights' reference.
fn score64(x: f32) f64 {
    const v: f64 = x;
    if (std.math.isNan(v)) return v;
    if (v >= 0) return 1.0 / (1.0 + @exp(-v));
    const e = @exp(v);
    return e / (1.0 + e);
}

/// A row's picks by the rule over fp32 keys: the largest non-NaN key, the lower id on a tie; a pick's key turns -inf.
fn rulePicks(logits: []const f32, bias: []const f32, pick: []u32) void {
    var key: [max_experts]f32 = undefined;
    const E = logits.len;
    for (key[0..E], logits, bias) |*k, x, b| k.* = score32(x) + b;
    for (pick) |*p| {
        var best = -std.math.inf(f32);
        var bid: usize = E;
        for (key[0..E], 0..) |k, e| if (k > best or (k == best and e < bid)) {
            best = k;
            bid = e;
        };
        p.* = @intCast(bid);
        if (bid < E) key[bid] = -std.math.inf(f32);
    }
}

/// Distinct finite keys 2^-16 apart or more, far past the scores' few ulps: the host's order is the device's.
fn separated(logits: []const f32, bias: []const f32, buf: *[max_experts]f32) ?[2]f32 {
    var n: usize = 0;
    for (logits, bias) |x, b| {
        const k = score32(x) + b;
        if (std.math.isFinite(k)) {
            buf[n] = k;
            n += 1;
        }
    }
    if (n < 2) return null;
    std.mem.sort(f32, buf[0..n], {}, std.sort.asc(f32));
    for (buf[0 .. n - 1], buf[1..n]) |lo, hi| if (hi > lo and hi - lo < 0x1p-16) return .{ lo, hi };
    return null;
}

/// Each row's picks and weights against the rule; null when they hold, else what failed (written to `msg`).
pub fn checkPicks(d: Dims, logits: []const f32, bias: []const f32, rows: usize, pick: []const i32, wts: []const f32, msg: []u8) !?[]const u8 {
    const E: usize = d.experts;
    const K: usize = d.topk;
    var buf: [max_experts]f32 = undefined;
    for (0..rows) |r| {
        const row = logits[r * E ..][0..E];
        if (separated(row, bias, &buf)) |p| return try std.fmt.bufPrint(msg, "row {d}'s keys {e} and {e} are too close for the fill to order", .{ r, p[0], p[1] });
        var want: [max_topk]u32 = undefined;
        rulePicks(row, bias, want[0..K]);
        var s: [max_topk]f64 = undefined;
        var total: f64 = 0;
        for (want[0..K], s[0..K]) |p, *v| {
            v.* = if (p < E) score64(row[p]) else 0;
            total += v.*;
        }
        for (0..K) |k| {
            const got = pick[r * K + k];
            if (got != want[k]) return try std.fmt.bufPrint(msg, "row {d} pick {d} is expert {d} where the rule picks {d}", .{ r, k, got, want[k] });
            const ref = s[k] / total * d.scale;
            const w: f64 = wts[r * K + k];
            const ok = if (std.math.isNan(ref)) std.math.isNan(w) else @abs(w - ref) <= 64 * fl.u32_ulp * @abs(ref) + fl.tiny;
            if (!ok) return try std.fmt.bufPrint(msg, "row {d} pick {d}'s weight is {e} where float64 gives {e}", .{ r, k, w, ref });
        }
    }
    return null;
}

/// Tied experts: across lanes (5, 200) and within lane 17 (17, 49, 81), each saturated to score 1 at bias 2.
const tied = [_]usize{ 5, 17, 49, 81, 200 };

/// Logits in quarters, biases -2, 0 or 2, the tied set on top; gate 1: NaN and infinite rows, scattered, -inf biases.
pub fn fillRoute(rng: std.Random, logits: []f32, bias: []f32, rows: usize, fill: g.Fill) void {
    const E = bias.len;
    for (bias) |*b| b.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 3))) * 2 - 2;
    for (tied) |e| bias[e] = 2;
    for (logits) |*v| v.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 64))) / 4 - 8;
    for (0..rows) |r| if (r % 3 != 2) for (tied, 0..) |e, i| {
        logits[r * E + e] = if (i % 2 == 0) 60 else 40;
    };
    if (fill != .nonfinite) return;
    bias[7] = -fl.inf32;
    bias[300 % E] = -fl.inf32;
    for (0..rows) |r| {
        const row = logits[r * E ..][0..E];
        switch (r) {
            0 => @memset(row, fl.nan32),
            1 => for (row, 0..) |*v, e| {
                v.* = fl.planted[(e + 1) % 3];
            },
            else => {
                row[(r * 37) % E] = fl.nan32;
                row[(r * 53 + 5) % E] = if (r % 2 == 0) fl.inf32 else -fl.inf32;
            },
        }
    }
}

fn ints(rig: *rig_.Rig, ref: rig_.Ref, n: usize) []const i32 {
    return @alignCast(std.mem.bytesAsSlice(i32, rig.bytes(ref, n * 4)));
}
fn floats(rig: *rig_.Rig, ref: rig_.Ref, n: usize) []const f32 {
    return @alignCast(std.mem.bytesAsSlice(f32, rig.bytes(ref, n * 4)));
}

// --- tf_route_select: a window's picks and weights, then the held experts' groups and both Macs' pick lists ---------

pub const Select = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    d: Dims,
    base_buf: [128]u8 = undefined,
    name: []const u8 = "",
    rows: usize = 0,
    logits: []f32 = &.{},
    bias: []f32 = &.{},
    ins: [2]rig_.Ref = undefined,
    outs: [9]rig_.Ref = undefined,
    msg: [240]u8 = undefined,

    pub fn base(d: Dims, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "metal hy4 moe_route tf_route_select experts {d} top {d}", .{ d.experts, d.topk }) catch unreachable;
    }

    /// Rows: the window's rows (16 at most); n: experts. Hy4 holds every expert on one Mac.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = fl.noSums,
            .out = .raw,
            .nan = .{ .stated = rule ++ "; the groups: held experts ascending with their member picks in row order, the pick lists ascending, counts and word" },
            .served = &served,
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "the weights' sum of 16 scores is held to the rule's bound; the rest compares and counts" },
        };
    }
    const served = [_]c.Shape{
        .{ .rows = max_rows, .n = 512, .k = 1 },
        .{ .rows = h.decode_rows, .n = 512, .k = 1 },
        .{ .rows = h.window_rows, .n = 512, .k = 1 },
        .{ .rows = 11, .n = 512, .k = 1 },
    };
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, d: Dims) !Select {
        if (d.experts != 512) return error.NotHy4Router;
        const text = try rt.routeText(a, d, null);
        defer a.free(text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rt.compile(device, text, route.names[1]), .d = d };
    }
    pub fn deinit(t: *Select) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Select) void {
        t.a.free(t.logits);
        t.a.free(t.bias);
        t.logits = &.{};
        t.bias = &.{};
    }
    pub fn kernel(t: *Select) g.Kernel {
        t.name = base(t.d, &t.base_buf);
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Select);

    /// Output elements in the loader's Scratch order: picks, weights, ids, members, count, mine, theirs, counts, word.
    fn sizes(t: *const Select) [9]usize {
        const RK = t.rows * t.d.topk;
        const E: usize = t.d.experts;
        return .{ RK, RK, E, E * max_rows, 1, RK, RK, 2, 1 };
    }

    pub fn prepare(t: *Select, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const E: usize = t.d.experts;
        if (shape.rows == 0 or shape.rows > max_rows or shape.n != E) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.logits = try t.a.alloc(f32, shape.rows * E);
        t.bias = try t.a.alloc(f32, E);
        var prng = std.Random.DefaultPrng.init(seed);
        fillRoute(prng.random(), t.logits, t.bias, shape.rows, fill);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.logits));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.bias));
        for (&t.outs, t.sizes()) |*o, n| o.* = try t.rig.output(n * 4);
        return true;
    }

    pub fn run(t: *Select) !void {
        const p = t.rig.begin();
        const o = t.outs;
        // the loader's call: every expert held here
        route.select(p.e, t.pipe, t.ins[0], t.ins[1], t.d.scale, @intCast(t.rows), .{ 0, t.d.experts }, route.Outputs(rig_.Ref){ .pick = o[0], .wts = o[1], .ids = o[2], .members = o[3], .count = o[4], .mine = o[5], .theirs = o[6], .counts = o[7], .word = o[8] });
        try t.rig.finish(p);
    }

    pub fn count(t: *Select) usize {
        var n: usize = 0;
        for (t.sizes()) |s| n += 4 * s;
        return n;
    }
    pub fn reference(_: *Select, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    pub fn check(t: *Select, _: g.Gate) !?[]const u8 {
        const R = t.rows;
        const E: usize = t.d.experts;
        const K: usize = t.d.topk;
        const z = t.sizes();
        const pick = ints(&t.rig, t.outs[0], z[0]);
        if (try checkPicks(t.d, t.logits, t.bias, R, pick, floats(&t.rig, t.outs[1], z[1]), &t.msg)) |why| return why;
        // the picks hold; each row's slot of each expert (its last pick of it), as the kernel's slot table
        var slot: [max_rows][max_experts]u8 = undefined;
        for (0..R) |r| {
            @memset(&slot[r], 0xff);
            for (0..K) |k| if (pick[r * K + k] < E) {
                slot[r][@intCast(pick[r * K + k])] = @intCast(k);
            };
        }
        const ids = ints(&t.rig, t.outs[2], z[2]);
        const members = ints(&t.rig, t.outs[3], z[3]);
        var u: usize = 0;
        for (0..E) |e| {
            var j: usize = 0;
            for (0..R) |r| if (slot[r][e] != 0xff) {
                if (j == 0) {
                    if (ids[u] != e) return try std.fmt.bufPrint(&t.msg, "held expert {d} is {d} where the host's ascending groups give {d}", .{ u, ids[u], e });
                }
                const m: i32 = @intCast(r * K + slot[r][e]);
                if (members[u * max_rows + j] != m) return try std.fmt.bufPrint(&t.msg, "expert {d}'s member {d} is {d} where the host gives {d}", .{ e, j, members[u * max_rows + j], m });
                j += 1;
            };
            if (j == 0) continue;
            for (j..max_rows) |q| if (members[u * max_rows + q] != -1) return try std.fmt.bufPrint(&t.msg, "expert {d}'s member {d} is {d} where the host pads -1", .{ e, q, members[u * max_rows + q] });
            u += 1;
        }
        const held = ints(&t.rig, t.outs[4], 1)[0];
        if (held != u) return try std.fmt.bufPrint(&t.msg, "the held count is {d} where the host counts {d}", .{ held, u });
        const mine = ints(&t.rig, t.outs[5], z[5]);
        const theirs = ints(&t.rig, t.outs[6], z[6]);
        var n = [2]usize{ 0, 0 };
        for (pick, 0..) |p, i| {
            const here: usize = if (p >= 0 and p < E) 0 else 1;
            const list = if (here == 0) mine else theirs;
            if (list[n[here]] != i) return try std.fmt.bufPrint(&t.msg, "{s} list entry {d} is pick {d} where the host gives {d}", .{ if (here == 0) "this Mac's" else "the peer's", n[here], list[n[here]], i });
            n[here] += 1;
        }
        const counts = ints(&t.rig, t.outs[7], 2);
        const word = ints(&t.rig, t.outs[8], 1)[0];
        if (counts[0] != n[0] or counts[1] != n[1] or word != n[0]) return try std.fmt.bufPrint(&t.msg, "counts {d}, {d} and word {d} where the host counts {d}, {d}", .{ counts[0], counts[1], word, n[0], n[1] });
        return null;
    }
};

// --- tf_hy4_route_select_rows and _rank: a prompt chunk's picks and weights, a threadgroup a row ---------------------

pub const Which = enum { rows, rank };

/// The rank kernel's gate 1 finding (seen on the M5 Ultra): a NaN key's every compare is false, so it ranks 0.
const rank_nan = "a NaN key ranks 0 (each compare with it is false) and races the row's best expert for pick 0: one NaN " ++
    "logit at expert 3 took pick 0 where the rule picks 5, and by the source its NaN score enters the row's weights; a " ++
    "row of NaN keys writes pick 0 only (expert 448 where the rule picks 512), leaving picks 1-15 and their weights " ++
    "unwritten. The core select and the rows kernel never pick a NaN key";

pub const Prompt = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    d: Dims,
    which: Which,
    control: ?g.Gate = null,
    base_buf: [128]u8 = undefined,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    rows: usize = 0,
    logits: []f32 = &.{},
    bias: []f32 = &.{},
    ins: [2]rig_.Ref = undefined,
    outs: [2]rig_.Ref = undefined,
    msg: [240]u8 = undefined,

    fn function(which: Which) []const u8 {
        return if (which == .rows) "tf_hy4_route_select_rows" else "tf_hy4_route_select_rank";
    }
    pub fn base(d: Dims, which: Which, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "metal hy4 route_rows {s} experts {d} top {d}", .{ function(which), d.experts, d.topk }) catch unreachable;
    }

    /// Rows: a prompt chunk's (17 to 512); n: experts. Hy4's 512 experts take the rank kernel; the rows one is built.
    pub fn spec(name: []const u8, known: ?c.Known) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = fl.noSums,
            .out = .raw,
            .nan = .{ .stated = rule },
            .served = &served,
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "the weights' sum of 16 scores is held to the rule's bound; the rest compares" },
            .known = known,
        };
    }
    const served = [_]c.Shape{
        .{ .rows = max_rows + 1, .n = 512, .k = 1 },
        .{ .rows = h.prompt_rows, .n = 512, .k = 1 },
        .{ .rows = 100, .n = 512, .k = 1 },
    };
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, d: Dims, which: Which, control: bool) !Prompt {
        if (d.experts != 512) return error.NotHy4Router;
        const edit: ?fl.Edit = if (!control) null else if (which == .rows) rt.nan_picked else rt.ties_high;
        const text = try rt.routeText(a, d, edit);
        defer a.free(text);
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .pipe = try rt.compile(device, text, function(which)),
            .d = d,
            .which = which,
            .control = if (!control) null else if (which == .rows) .nonfinite else .shapes,
        };
    }
    pub fn deinit(t: *Prompt) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Prompt) void {
        t.a.free(t.logits);
        t.a.free(t.bias);
        t.logits = &.{};
        t.bias = &.{};
    }
    pub fn kernel(t: *Prompt) g.Kernel {
        const what: ?[]const u8 = if (t.control == null) null else if (t.which == .rows) "a NaN key picked" else "rank ties to the higher id";
        t.name = fl.named(base(t.d, t.which, &t.base_buf), what, &t.label);
        const known: ?c.Known = if (t.which == .rank and t.control == null) .{ .gate = .nonfinite, .finding = rank_nan } else null;
        return .{ .spec = spec(t.name, known), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Prompt);

    pub fn prepare(t: *Prompt, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const E: usize = t.d.experts;
        if (shape.rows == 0 or shape.rows > h.prompt_rows or shape.n != E) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.logits = try t.a.alloc(f32, shape.rows * E);
        t.bias = try t.a.alloc(f32, E);
        var prng = std.Random.DefaultPrng.init(seed);
        fillRoute(prng.random(), t.logits, t.bias, shape.rows, fill);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.logits));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.bias));
        for (&t.outs) |*o| o.* = try t.rig.output(shape.rows * t.d.topk * 4);
        return true;
    }

    pub fn run(t: *Prompt) !void {
        const p = t.rig.begin();
        const e = p.e;
        // metal_routed.zig's prompt branch: a threadgroup a row, 512 threads for ranks, a simdgroup for the rows kernel
        e.setPipeline(t.pipe);
        e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        e.setValue(route.Args{ .rows = @intCast(t.rows), .lo = 0, .hi = @intCast(t.d.experts), .scale = t.d.scale }, 2);
        e.setBuffer(t.outs[0].buf, t.outs[0].off, 3);
        e.setBuffer(t.outs[1].buf, t.outs[1].off, 4);
        e.dispatchGroups(mtl.Size.of(t.rows, 1, 1), mtl.Size.of(if (t.which == .rank) 512 else 32, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Prompt) usize {
        return 2 * 4 * t.rows * t.d.topk;
    }
    pub fn reference(_: *Prompt, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    pub fn check(t: *Prompt, _: g.Gate) !?[]const u8 {
        const n = t.rows * t.d.topk;
        return checkPicks(t.d, t.logits, t.bias, t.rows, ints(&t.rig, t.outs[0], n), floats(&t.rig, t.outs[1], n), &t.msg);
    }
};
