//! Qwen3.6-35B-A3B's routed-expert kernels under the contract on Metal: select, sort and combine against CPU oracles.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const q = tf.qwen27;
const mw = tf.qwen35_experts;

const E = 256;
const TOPK = 8;
const WS = TOPK + 1;
const D = 2048;

/// The 35B's routing shape, as its config.json gives it.
pub fn config() q.config.Config {
    return .{ .hidden = D, .intermediate = 0, .layers = 40, .vocab = 248320, .heads = 16, .kv_heads = 2, .head_dim = 256, .k_heads = 16, .v_heads = 32, .dk = 128, .dv = 128, .conv_kernel = 4, .eps = 1e-6, .rope_dims = 64, .rope_theta = 1e7, .experts = E, .topk = TOPK, .moe_intermediate = 512, .shared_intermediate = 512 };
}
const LS = (E + 1 + 3) / 4 * 4;

fn noSums(_: c.Shape) usize {
    return 0;
}

/// Lane 0's chain over the picks' exponentials: the weights' denominator.
fn pickChain(_: c.Shape) usize {
    return TOPK;
}

/// One fma chain a column over the picks' rows, then the shared expert's.
fn slotChain(_: c.Shape) usize {
    return WS;
}

pub const Which = enum { select, sort, combine };
const select_rule = "NaN and -Inf logits rank as -FLT_MAX: k distinct experts; a +Inf top or a NaN gate gives NaN weights";
const sort_rule = "ids only: no float input carries NaN or an infinity";
const combine_rule = "the f32 fma chain's own: a NaN or infinite weight or row gives what IEEE fma gives";
const select_why = "eight exponentials in (0, 1], the first 1, each weight rounded to bf16: no accumulator's error survives 100x";
/// Each weight's bf16 rounding is 2^16 fp32 roundings, after the exponential and the division.
const select_ops = 0x1p16 + 64;

pub fn spec(w: Which) c.Spec {
    const served: []const c.Shape = &.{ .{ .rows = 1, .n = TOPK, .k = E }, .{ .rows = 16, .n = TOPK, .k = E }, .{ .rows = 128, .n = TOPK, .k = E } };
    return switch (w) {
        .select => .{ .name = "metal qwen27 moe select q27_moe_select", .accum = .f32, .depth = pickChain, .ops = select_ops, .out = .f32, .nan = .{ .stated = select_rule }, .served = served, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = select_why } },
        .sort => .{ .name = "metal qwen27 moe sort q27_moe_sort", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = sort_rule }, .served = served },
        .combine => .{ .name = "metal qwen27 moe combine q27_moe_combine", .accum = .f32, .depth = slotChain, .out = .bf16, .nan = .{ .stated = combine_rule }, .served = served },
    };
}

pub const Moe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    pipe: mtl.Pipeline,
    exp: mtl.Pipeline,
    rows: usize = 0,
    sorted: u32 = 0,
    f: []f32 = &.{},
    ids: []i32 = &.{},
    y: []u16 = &.{},
    ins: [4]rig_.Ref = undefined,
    outs: [4]rig_.Ref = undefined,
    out_len: [4]usize = @splat(0),
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which) !Moe {
        const text = try q.moe.source(a, config());
        defer a.free(text);
        const probe = "#include <metal_stdlib>\nusing namespace metal;\n[[kernel]] void probe(device float* X [[buffer(0)]], uint i [[thread_position_in_grid]]) { X[i] = precise::exp(X[i]); }";
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = w, .pipe = try rig_.pipeline(device, text, spec(w).name[std.mem.lastIndexOfScalar(u8, spec(w).name, ' ').? + 1 ..]), .exp = try rig_.pipeline(device, probe, "probe") };
    }

    pub fn deinit(t: *Moe) void {
        t.free();
        t.pipe.deinit();
        t.exp.deinit();
        t.rig.deinit();
    }

    fn free(t: *Moe) void {
        t.a.free(t.f);
        t.a.free(t.ids);
        t.a.free(t.y);
        t.f = &.{};
        t.ids = &.{};
        t.y = &.{};
    }

    pub fn kernel(t: *Moe) g.Kernel {
        return .{ .spec = spec(t.which), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(Moe);

    fn eighths(rng: std.Random) f32 {
        return @as(f32, @floatFromInt(rng.uintLessThan(u32, 128))) / 8 - 8; // eighths: ties
    }

    pub fn prepare(t: *Moe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.out_len = @splat(0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const slots = t.rows * TOPK;
        switch (t.which) {
            .select => {
                t.f = try t.a.alloc(f32, t.rows * LS);
                for (t.f) |*v| v.* = eighths(rng);
                if (fill == .nonfinite) {
                    const nan = std.math.nan(f32);
                    const inf = std.math.inf(f32);
                    const at = [_][2]usize{ .{ 0, 3 }, .{ 1, 100 }, .{ 2, 7 }, .{ 3, 0 }, .{ 3, E - 1 }, .{ 0, E } };
                    const put = [_]f32{ nan, inf, -inf, inf, -inf, nan };
                    for (at, put) |p, v| if (p[0] < t.rows) {
                        t.f[p[0] * LS + p[1]] = v;
                    };
                    if (t.rows > 4) for (3..E) |id| {
                        t.f[4 * LS + id] = nan; // fewer finite logits than picks
                    };
                }
                t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.f));
                t.out_len[0] = slots * 4;
                t.out_len[1] = t.rows * WS * 4;
            },
            .sort => {
                t.ids = try t.a.alloc(i32, slots);
                for (t.ids, 0..) |*v, i| v.* = @intCast(if (i % 3 == 0) rng.uintLessThan(u32, 4) else rng.uintLessThan(u32, E));
                t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.ids));
                t.out_len = .{ (E + 1) * 4, slots * 4, slots * 4, 2 * 4 };
            },
            .combine => {
                t.sorted = @intFromBool(seed % 2 == 1);
                t.y = try t.a.alloc(u16, (slots + t.rows) * D);
                for (t.y) |*v| v.* = c.toBf16(eighths(rng));
                t.f = try t.a.alloc(f32, t.rows * WS);
                for (t.f) |*v| v.* = mw.float(mw.bf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 256))) / 256));
                t.ids = try t.a.alloc(i32, slots);
                for (t.ids, 0..) |*v, i| v.* = @intCast(if (t.sorted != 0) (i * 7 + 3) % slots else i);
                if (fill == .nonfinite) {
                    const at = [_]usize{ 0, D + 5, 2 * D + 9 };
                    const put = [_]u16{ 0x7fc1, 0x7f80, 0xff80 };
                    for (at, put) |p, v| if (p < slots * D) {
                        t.y[p] = v;
                    };
                    if (t.rows > 1) t.f[WS + TOPK] = std.math.nan(f32);
                }
                if (fill == .stress) t.stress(rng);
                t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.y[0 .. slots * D]));
                t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.y[slots * D ..]));
                t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.f));
                t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.ids));
                t.out_len[0] = t.rows * D * 2;
            },
        }
        const order: [4]usize = if (t.which == .select) .{ 1, 0, 2, 3 } else .{ 0, 1, 2, 3 }; // gate 4 reads the weights
        for (order) |i| if (t.out_len[i] > 0) {
            t.outs[i] = try t.rig.output(t.out_len[i]);
        };
        return true;
    }

    /// Gate 4's combine: slots +2^15, s, s', -2^15 at weight 1 (s, s' in 1-7), the rest 0: fp16 and bf16 lose both.
    fn stress(t: *Moe, rng: std.Random) void {
        @memset(t.y, 0);
        @memset(t.f, 1);
        for (0..t.rows) |r| for (0..D) |d| {
            const big: f32 = if (d % 2 == 0) 0x1p15 else -0x1p15;
            const small = [2]f32{ @floatFromInt(1 + rng.uintLessThan(u32, 7)), @floatFromInt(1 + rng.uintLessThan(u32, 7)) };
            for ([_]f32{ big, small[0], small[1], -big }, 0..) |v, k| t.y[t.row(r * TOPK + k) * D + d] = c.toBf16(v);
        };
    }

    /// The routed row a slot's output sits in: sorted by expert, or in slot order.
    fn row(t: *const Moe, slot: usize) usize {
        return if (t.sorted != 0) @intCast(t.ids[slot]) else slot;
    }

    pub fn run(t: *Moe) !void {
        const pass = t.rig.begin();
        const e = pass.e;
        e.setPipeline(t.pipe);
        switch (t.which) {
            .select => {
                for ([_]rig_.Ref{ t.ins[0], t.outs[0], t.outs[1] }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.dispatchGroups(mtl.Size.of(t.rows, 1, 1), mtl.Size.of(32, 1, 1));
            },
            .sort => {
                e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
                e.setValue(@as(u32, @intCast(t.rows * TOPK)), 1);
                for (t.outs, 2..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(E * 4, 1, 1));
            },
            .combine => {
                for (t.ins[0..3], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.setBuffer(t.ins[3].buf, t.ins[3].off, 3);
                e.setValue(t.sorted, 4);
                e.setBuffer(t.outs[0].buf, t.outs[0].off, 5);
                e.dispatchThreads(mtl.Size.of(D, t.rows, 1), mtl.Size.of(256, 1, 1));
            },
        }
        try t.rig.finish(pass);
    }

    /// The outputs gate 4 reads: select's weights, combine's columns; sort's bytes.
    pub fn count(t: *Moe) usize {
        return switch (t.which) {
            .select => t.rows * WS,
            .combine => t.rows * D,
            .sort => (E + 1 + 2 * t.rows * TOPK + 2) * 4,
        };
    }

    /// Row r's picks by the stated rule (NaN and -Inf as -FLT_MAX, ties to the lower id) and their ranked logits.
    fn ranked(t: *const Moe, r: usize, ids: *[TOPK]usize, vals: *[TOPK]f32) void {
        var rank: [E]f32 = undefined;
        for (&rank, t.f[r * LS ..][0..E]) |*v, l| v.* = if (std.math.isNan(l) or l == -std.math.inf(f32)) -std.math.floatMax(f32) else l;
        var taken: [E]bool = @splat(false);
        for (ids, vals) |*id, *v| {
            var best: usize = E;
            for (0..E) |e| if (!taken[e] and (best == E or rank[e] > rank[best])) {
                best = e;
            };
            taken[best] = true;
            id.* = best;
            v.* = rank[best];
        }
    }

    /// Column d of row r's summed terms: each pick's weight times its row, then the shared expert's.
    fn terms(t: *const Moe, r: usize, d: usize, out: *[WS]f64) void {
        for (0..TOPK) |k| out[k] = @as(f64, t.f[r * WS + k]) * mw.float(t.y[t.row(r * TOPK + k) * D + d]);
        out[TOPK] = @as(f64, t.f[r * WS + TOPK]) * mw.float(t.y[(t.rows * TOPK + r) * D + d]);
    }

    /// Float64 weights (exp(l - l_top) over their sum, the gate's sigmoid) or columns, as if every sum ran in `acc`.
    fn outputs(t: *const Moe, acc: ?c.Accum, order: g.Order, want: []f64, mag: []f64) void {
        for (0..t.rows) |r| switch (t.which) {
            .select => {
                var ids: [TOPK]usize = undefined;
                var vals: [TOPK]f32 = undefined;
                t.ranked(r, &ids, &vals);
                var ex: [TOPK]f64 = undefined;
                for (&ex, vals) |*e, v| e.* = @exp(@as(f64, v) - vals[0]);
                var total: f64 = 0;
                for (ex) |e| total += e;
                if (acc) |a| total = rows.sumIn(a, order, &ex);
                const gate: f64 = mw.float(mw.bf16(t.f[r * LS + E]));
                for (0..WS) |k| want[r * WS + k] = if (k < TOPK) ex[k] / total else 1 / (1 + @exp(-gate));
            },
            .combine => for (0..D) |d| {
                var x: [WS]f64 = undefined;
                t.terms(r, d, &x);
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (x) |v| {
                    sum += v;
                    abs += @abs(v);
                }
                want[r * D + d] = if (acc) |a| rows.sumIn(a, order, &x) else sum;
                mag[r * D + d] = abs;
            },
            .sort => {},
        };
        if (t.which == .select) for (mag, want) |*m, w| {
            m.* = @abs(w);
        };
    }

    pub fn reference(t: *Moe, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
        t.outputs(null, .sequential, want, mag);
    }

    pub fn simulate(t: *Moe, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (t.which == .sort) return error.NoSums;
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        t.outputs(acc, order, out, mag);
    }

    /// The GPU's precise::exp on each value, so the oracle's weights round as the kernel's do.
    fn gpuExp(t: *Moe, xs: []f32) !void {
        const b = try t.rig.device.buffer(xs.len * 4, mtl.ResourceOptions.shared);
        defer b.deinit();
        @memcpy(b.slice(f32, xs.len), xs);
        const pass = t.rig.begin();
        pass.e.setPipeline(t.exp);
        pass.e.setBuffer(b, 0, 0);
        pass.e.dispatchThreads(mtl.Size.of(xs.len, 1, 1), mtl.Size.of(32, 1, 1));
        try t.rig.finish(pass);
        @memcpy(xs, b.slice(f32, xs.len));
    }

    /// Equal bits, or both NaN (the payload is not part of the rule).
    fn same32(x: u32, y: u32) bool {
        const nx = x & 0x7f800000 == 0x7f800000 and x & 0x7fffff != 0;
        const ny = y & 0x7f800000 == 0x7f800000 and y & 0x7fffff != 0;
        return x == y or (nx and ny);
    }

    fn selectCheck(t: *Moe, at: g.Gate) !?[]const u8 {
        const picks = std.mem.bytesAsSlice(i32, t.rig.bytes(t.outs[0], t.out_len[0]));
        const wts = std.mem.bytesAsSlice(u32, t.rig.bytes(t.outs[1], t.out_len[1]));
        const args = try t.a.alloc(f32, t.rows * WS);
        defer t.a.free(args);
        const want = try t.a.alloc(i32, t.rows * TOPK);
        defer t.a.free(want);
        for (0..t.rows) |r| {
            var ids: [TOPK]usize = undefined;
            var vals: [TOPK]f32 = undefined;
            t.ranked(r, &ids, &vals);
            for (0..TOPK) |k| {
                want[r * TOPK + k] = @intCast(ids[k]);
                args[r * WS + k] = vals[k] - vals[0];
            }
            args[r * WS + TOPK] = @abs(mw.float(mw.bf16(t.f[r * LS + E])));
        }
        try t.gpuExp(args);
        for (0..t.rows) |r| {
            for (0..TOPK) |k| if (picks[r * TOPK + k] != want[r * TOPK + k] or picks[r * TOPK + k] >= E)
                return try std.fmt.bufPrint(&t.msg, "row {d} pick {d}: {d}, the oracle {d}", .{ r, k, picks[r * TOPK + k], want[r * TOPK + k] });
            const ex = args[r * WS ..][0..WS];
            var total: f32 = 0;
            for (ex[0..TOPK]) |v| total += v;
            var w: [WS]f32 = undefined;
            for (0..TOPK) |k| w[k] = mw.float(mw.bf16(ex[k] / total));
            const gate = mw.float(mw.bf16(t.f[r * LS + E]));
            const y = 1.0 / (1.0 + ex[TOPK]);
            w[TOPK] = mw.float(mw.bf16(if (gate < 0) y else 1.0 - y));
            for (0..WS) |k| if (!same32(wts[r * WS + k], @bitCast(w[k])))
                return try std.fmt.bufPrint(&t.msg, "row {d} weight {d}: 0x{x}, the oracle 0x{x}", .{ r, k, wts[r * WS + k], @as(u32, @bitCast(w[k])) });
        }
        if (at == .nonfinite and t.rows > 4) for (picks[4 * TOPK ..][0..TOPK]) |p| if (p >= E) return "row 4 picked past the experts";
        return null;
    }

    fn sortCheck(t: *Moe) !?[]const u8 {
        const slots = t.rows * TOPK;
        var starts: [E + 1]u32 = @splat(0);
        for (t.ids) |id| starts[@as(usize, @intCast(id)) + 1] += 1;
        for (1..E + 1) |e| starts[e] += starts[e - 1];
        if (!std.mem.eql(u8, std.mem.asBytes(&starts), t.rig.bytes(t.outs[0], t.out_len[0]))) return "STARTS differ from the counting sort";
        const pos = std.mem.bytesAsSlice(u32, t.rig.bytes(t.outs[1], t.out_len[1]));
        const owner = std.mem.bytesAsSlice(u32, t.rig.bytes(t.outs[2], t.out_len[2]));
        var next = starts;
        for (t.ids, 0..) |id, j| {
            const at = next[@intCast(id)];
            next[@intCast(id)] += 1;
            if (pos[j] != at or owner[at] != j / TOPK) return try std.fmt.bufPrint(&t.msg, "slot {d} of {d}: at {d} row {d}, the oracle {d}", .{ j, slots, pos[j], owner[at], at });
        }
        const one = std.mem.bytesAsSlice(u32, t.rig.bytes(t.outs[3], t.out_len[3]));
        if (one[0] != 0 or one[1] != t.rows) return "ONE is not {0, rows}";
        return null;
    }

    fn combineCheck(t: *Moe) !?[]const u8 {
        const got = std.mem.bytesAsSlice(u16, t.rig.bytes(t.outs[0], t.out_len[0]));
        const slots = t.rows * TOPK;
        for (0..t.rows) |r| for (0..D) |d| {
            var acc: f32 = 0;
            for (0..TOPK) |k| acc = @mulAdd(f32, t.f[r * WS + k], mw.float(t.y[t.row(r * TOPK + k) * D + d]), acc);
            acc = @mulAdd(f32, t.f[r * WS + TOPK], mw.float(t.y[(slots + r) * D + d]), acc);
            const want = mw.bf16(acc);
            const w = got[r * D + d];
            const both_nan = want & 0x7fff > 0x7f80 and w & 0x7fff > 0x7f80;
            if (w != want and !both_nan) return try std.fmt.bufPrint(&t.msg, "row {d} col {d}: 0x{x}, the fp32 fma chain 0x{x}", .{ r, d, w, want });
        };
        return null;
    }

    /// Every output against its oracle; at the non-finite gate, the stated rule on the planted rows.
    pub fn check(t: *Moe, at: g.Gate) !?[]const u8 {
        return switch (t.which) {
            .select => t.selectCheck(at),
            .sort => t.sortCheck(),
            .combine => t.combineCheck(),
        };
    }
};
