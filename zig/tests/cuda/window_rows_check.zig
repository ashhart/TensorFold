//! window_rows.cu's calls on the host: each output against core/attention_range.zig's float64 point and range.
const std = @import("std");
const core = @import("core");

const ar = core.attention_range;
const bfr = core.bf16_round;

/// One decode or prompt call as the host holds it: q [rows, h, d], caches [cache rows, hk, d] (bf16 bits).
pub const Call = struct {
    h: usize,
    hk: usize,
    d: usize,
    win: usize,
    ring: usize,
    scale: f32,
    prompt: bool,
    p32: bool = false,
    rows: usize,
    q: []const u16,
    kc: []const u16,
    vc: []const u16,
    /// Decode: each row's position and stream slot (cache rows slot * ring + key % ring).
    pos: []const i32 = &.{},
    slot: []const i32 = &.{},
    /// Prompt: the first row's position and each row's last key (null: its own position); ring 0 is a linear cache.
    p0: usize = 0,
    hi: ?[]const i32 = null,

    /// Row r's position, its last key and its stream's first cache row.
    pub fn at(c: Call, r: usize) struct { p: usize, last: usize, base: usize } {
        if (c.prompt) return .{ .p = c.p0 + r, .last = if (c.hi) |h| @intCast(h[r]) else c.p0 + r, .base = 0 };
        const p: usize = @intCast(c.pos[r]);
        return .{ .p = p, .last = p, .base = @as(usize, @intCast(c.slot[r])) * c.ring };
    }

    pub fn first(c: Call, p: usize) usize {
        return if (c.win > 0 and p + 1 > c.win) p + 1 - c.win else 0;
    }

    pub fn cacheRow(c: Call, base: usize, key: usize) usize {
        return base + if (c.ring > 0) key % c.ring else key;
    }

    pub fn form(c: Call, keys: usize) ar.Form {
        return if (c.prompt) ar.prompt(c.d, keys, c.p32) else ar.decode(c.d, keys);
    }
};

/// Scratch for a call's heads: one (row, key head)'s keys and values as float64, and a head's weights and ends.
pub const Work = struct {
    a: std.mem.Allocator,
    k: []f64,
    v: []f64,
    q: []f64,
    w: []f64,
    t: []f64,
    point: []f64,
    half: []f64,

    pub fn init(a: std.mem.Allocator, d: usize, most: usize) !Work {
        var self: Work = undefined;
        self.a = a;
        inline for (.{ "k", "v" }) |n| @field(self, n) = try a.alloc(f64, most * d);
        inline for (.{ "w", "t" }) |n| @field(self, n) = try a.alloc(f64, most);
        inline for (.{ "q", "point", "half" }) |n| @field(self, n) = try a.alloc(f64, d);
        return self;
    }

    pub fn deinit(self: *Work) void {
        inline for (.{ "k", "v", "w", "t", "q", "point", "half" }) |n| self.a.free(@field(self, n));
    }
};

/// The most keys any row of the call sees.
pub fn mostKeys(c: Call) usize {
    var n: usize = 1;
    for (0..c.rows) |r| {
        const w = c.at(r);
        n = @max(n, w.last + 1 - c.first(w.p));
    }
    return n;
}

/// Each output head's float64 point and half range, in out's order ([rows, h, d]); `each` sees every head's.
fn heads(c: Call, wk: *Work, ctx: anytype, comptime each: fn (@TypeOf(ctx), usize, []const f64, []const f64) anyerror!void) !void {
    const d = c.d;
    const g = c.h / c.hk;
    for (0..c.rows) |r| {
        const w = c.at(r);
        const lo = c.first(w.p);
        const n = w.last + 1 - lo;
        for (0..c.hk) |kh| {
            for (lo..w.last + 1, 0..) |key, j| for (0..d) |e| {
                const at = (c.cacheRow(w.base, key) * c.hk + kh) * d + e;
                wk.k[j * d + e] = bfr.value(c.kc[at]);
                wk.v[j * d + e] = bfr.value(c.vc[at]);
            };
            for (0..g) |hh| {
                const head = kh * g + hh;
                for (wk.q, 0..) |*x, e| x.* = bfr.value(c.q[(r * c.h + head) * d + e]);
                ar.head(wk.q, wk.k, wk.v, n, d, c.scale, c.form(n), wk.w, wk.t, wk.point, wk.half);
                try each(ctx, (r * c.h + head) * d, wk.point, wk.half);
            }
        }
    }
}

const Verify = struct { out: []const u16, stats: *ar.Stats, msg: []u8, d: usize, h: usize, bad: ?[]const u8 = null };

fn verifyOne(v: *Verify, at: usize, point: []const f64, half: []const f64) !void {
    if (v.bad != null) return;
    for (point, half, 0..) |p, h, e| {
        const got = v.out[at + e];
        if (ar.within(p, h, got, v.stats)) continue;
        v.bad = try std.fmt.bufPrint(v.msg, "row {d} head {d} out[{d}] = {e} (0x{x:0>4}) outside float64's {e} +- {e}", .{ at / v.d / v.h, at / v.d % v.h, e, bfr.value(got), got, p, h });
        return;
    }
}

/// Every output within its range (NaN where float64 gives NaN, infinities by their bits); null, or what failed.
pub fn verify(a: std.mem.Allocator, c: Call, out: []const u16, stats: *ar.Stats, msg: []u8) !?[]const u8 {
    var wk = try Work.init(a, c.d, mostKeys(c));
    defer wk.deinit();
    var v: Verify = .{ .out = out, .stats = stats, .msg = msg, .d = c.d, .h = c.h };
    try heads(c, &wk, &v, verifyOne);
    return v.bad;
}

const Ref = struct { want: []f64, half: []f64 };

fn refOne(r: *Ref, at: usize, point: []const f64, half: []const f64) !void {
    @memcpy(r.want[at..][0..point.len], point);
    @memcpy(r.half[at..][0..half.len], half);
}

/// Every output's float64 point and half range, in out's order.
pub fn reference(a: std.mem.Allocator, c: Call, want: []f64, half: []f64) !void {
    var wk = try Work.init(a, c.d, mostKeys(c));
    defer wk.deinit();
    var r: Ref = .{ .want = want, .half = half };
    try heads(c, &wk, &r, refOne);
}

/// A row of d bf16 values of RMS `rms` (a unit-RMS normal row times `rms`, as a norm with a constant weight leaves it).
pub fn row(rng: std.Random, x: []u16, rms: f32) void {
    var ss: f64 = 0;
    var buf: [512]f32 = undefined;
    for (buf[0..x.len]) |*v| {
        v.* = rng.floatNorm(f32);
        ss += @as(f64, v.*) * v.*;
    }
    const inv: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(x.len)) / ss));
    for (x, buf[0..x.len]) |*h, v| h.* = bfr.bits(bfr.round32(v * inv) * rms);
}

/// Gemma 4's q and k weights (all equal within a layer: q 0.95 to 1.05, k 0.06 to 0.13; mlx-community 12B 4-bit).
pub const q_rms = [2]f32{ 0.95, 1.05 };
pub const k_rms = [2]f32{ 0.06, 0.13 };

test "a row's RMS is as asked" {
    var prng = std.Random.DefaultPrng.init(1);
    var x: [256]u16 = undefined;
    row(prng.random(), &x, 0.125);
    var ss: f64 = 0;
    for (x) |h| ss += bfr.value(h) * bfr.value(h);
    try std.testing.expectApproxEqRel(@as(f64, 0.125), @sqrt(ss / 256), 0.01);
}
