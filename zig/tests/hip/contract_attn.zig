//! Attention kernels under the contract on HIP: their fp32 outputs against core/attention_ref.zig, head by head.
const std = @import("std");
const core = @import("core");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const ref = core.attention_ref;

/// Why gates 1 and 3 run attention_ref's bound: P enters P x V in the cache type; the spec's bound takes fp32 terms.
pub const why = "P is rounded to the cache type before the P x V dot2: gates 1 and 3 hold each output to " ++
    "attention_ref's float64 bound";

/// One output head's float64 inputs, slices of the test's: its query, its visible keys' K and V rows ([keys][d]).
pub const Head = struct { q: []const f64 = &.{}, k: []const f64 = &.{}, v: []const f64 = &.{}, n: usize = 0 };

/// The float64 side of an attention test: `t.load(i, *Head)` points output head i at its inputs (fp32 outputs).
pub const Ref = struct {
    a: std.mem.Allocator,
    d: usize,
    head: Head = .{},
    keys: []usize,
    p: []f64,
    want: []f64,
    tol: []f64,
    msg: [240]u8 = undefined,

    pub fn init(a: std.mem.Allocator, d: usize, most: usize) !Ref {
        const keys = try a.alloc(usize, most);
        errdefer a.free(keys);
        for (keys, 0..) |*k, i| k.* = i;
        const p = try a.alloc(f64, most);
        errdefer a.free(p);
        const want = try a.alloc(f64, d);
        errdefer a.free(want);
        return .{ .a = a, .d = d, .keys = keys, .p = p, .want = want, .tol = try a.alloc(f64, d) };
    }

    pub fn deinit(r: *Ref) void {
        for ([_][]f64{ r.p, r.want, r.tol }) |x| r.a.free(x);
        r.a.free(r.keys);
    }

    /// Gates 1 and 3: every output's class equals float64's, and a finite one is within attention_ref's bound.
    pub fn check(r: *Ref, t: anytype, heads: usize, scale: f64, out: []const u8, at: g.Gate) !?[]const u8 {
        var planted: usize = 0;
        var differ: usize = 0;
        var first: struct { i: usize = 0, e: usize = 0, got: c.Class = .finite, want: c.Class = .finite } = .{};
        for (0..heads) |i| {
            try t.load(i, &r.head);
            ref.head(r.head.q, r.head.k, r.head.v, r.keys[0..r.head.n], r.d, scale, r.want, r.tol);
            for (r.want, r.tol, 0..) |w, tl, e| {
                const got = c.read(.f32, out, i * r.d + e);
                if (c.classOf(w) != .finite) planted += 1;
                if (c.classOf(got) != c.classOf(w)) {
                    if (differ == 0) first = .{ .i = i, .e = e, .got = c.classOf(got), .want = c.classOf(w) };
                    differ += 1;
                    continue;
                }
                if (at != .nonfinite and std.math.isFinite(w) and !(@abs(got - w) <= tl)) {
                    const say = "head {d} element {d} is {e:.6} where float64 gives {e:.6}: {d:.2} of " ++
                        "attention_ref's bound";
                    return try std.fmt.bufPrint(&r.msg, say, .{ i, e, got, w, @abs(got - w) / tl });
                }
            }
        }
        if (differ > 0) {
            const say = "{d} of {d} outputs differ in class; first at head {d} element {d}: {t} where float64 " ++
                "gives {t}";
            return try std.fmt.bufPrint(&r.msg, say, .{ differ, heads * r.d, first.i, first.e, first.got, first.want });
        }
        if (at == .nonfinite and (planted == 0 or planted == heads * r.d)) {
            const say = "the planted values make {d} of {d} reference outputs non-finite: no proof";
            return try std.fmt.bufPrint(&r.msg, say, .{ planted, heads * r.d });
        }
        return null;
    }

    fn weights(r: *Ref, scale: f64) f64 {
        const h = r.head;
        var m: f64 = -std.math.inf(f64);
        for (r.p[0..h.n], 0..) |*p, j| {
            var s: f64 = 0;
            for (0..r.d) |e| s += h.q[e] * h.k[j * r.d + e];
            p.* = s * scale;
            m = @max(m, p.*);
        }
        var l: f64 = 0;
        for (r.p[0..h.n]) |*p| {
            p.* = @exp(p.* - m);
            l += p.*;
        }
        return l;
    }

    /// Float64 outputs and each one's term magnitude, sum p_j |v_j| / sum p_j (gate 4 reads them).
    pub fn reference(r: *Ref, t: anytype, heads: usize, scale: f64, want: []f64, mag: []f64) !void {
        for (0..heads) |i| {
            try t.load(i, &r.head);
            const l = r.weights(scale);
            for (0..r.d) |e| {
                var s: f64 = 0;
                var abs: f64 = 0;
                for (r.p[0..r.head.n], 0..) |p, j| {
                    s += p * r.head.v[j * r.d + e];
                    abs += p * @abs(r.head.v[j * r.d + e]);
                }
                want[i * r.d + e] = s / l;
                mag[i * r.d + e] = abs / l;
            }
        }
    }

    /// Gate 4's sums in an accumulator: its keys score alike, so every p is 1 and the sums are V's terms and l's ones.
    pub fn simulate(r: *Ref, t: anytype, heads: usize, scale: f64, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try r.a.alloc(f64, r.p.len);
        defer r.a.free(terms);
        const ones = try r.a.alloc(f64, r.p.len);
        defer r.a.free(ones);
        @memset(ones, 1);
        for (0..heads) |i| {
            try t.load(i, &r.head);
            _ = r.weights(scale);
            for (r.p[0..r.head.n]) |p| if (p != 1) return error.KeysScoreApart;
            const l = rows.sumIn(acc, order, ones[0..r.head.n]);
            for (0..r.d) |e| {
                for (terms[0..r.head.n], 0..) |*x, j| x.* = r.head.v[j * r.d + e];
                out[i * r.d + e] = rows.sumIn(acc, order, terms[0..r.head.n]) / l;
            }
        }
    }
};

/// Gate 4's values: odd integers, every other one times 64, the second half cancelling the first (before rounding).
pub fn mirrored(rng: std.Random, out: []f64) void {
    const h = out.len / 2;
    for (0..h) |i| {
        const v: f64 = @floatFromInt((1 + 2 * rng.uintLessThan(u32, 128)) * @as(u32, if (i % 2 == 1) 64 else 1));
        out[i] = v;
        out[h + i] = (if (i < 8) @as(f64, 1) else 0) - v;
    }
    if (out.len % 2 == 1) out[out.len - 1] = 0;
}
