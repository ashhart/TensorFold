//! GigaChat 4's GatedNorm (gigachat4/gated_norm.cu) under the contract: float64 at the reference's bf16 points.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const gc4 = @import("gigachat4_norm_cuda");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const iv = @import("contract_interval.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const gn = gc4.gated_norm;
const P = gn.Point;
const Iv = iv.Iv;
const Sums = iv.Sums;

const eps: f32 = 1e-6;
/// The config's layernorm_gating_weight, the same in both releases.
const scale: f32 = 2.0;

/// A registration: the served kernel (mask all), or a control that keeps every rounding point but one.
pub const Entry = struct { name: []const u8, mask: u32 = P.all };

pub const entries = [_]Entry{
    .{ .name = "cuda gigachat4 tf_gc4_gated_norm (GatedNorm: RMSNorm, the 16-wide sigmoid gate, bf16 at torch's points)" },
    .{ .name = "control: gigachat4 GatedNorm without the normed row's rounding", .mask = P.all & ~@as(u32, P.normed) },
    .{ .name = "control: gigachat4 GatedNorm without up's rounding", .mask = P.all & ~@as(u32, P.up) },
    .{ .name = "control: gigachat4 GatedNorm without silu's rounding", .mask = P.all & ~@as(u32, P.silu) },
    .{ .name = "control: gigachat4 GatedNorm without down's rounding", .mask = P.all & ~@as(u32, P.down) },
    .{ .name = "control: gigachat4 GatedNorm without the sigmoid's rounding", .mask = P.all & ~@as(u32, P.sigmoid) },
};

/// rows x width: the 37B's hidden (decode, a verify window, prompt chunks), q_a (1536) and kv_a (512).
pub const served = &[_]c.Shape{
    .{ .rows = 1, .n = 2560, .k = 1 },  .{ .rows = 7, .n = 2560, .k = 1 }, .{ .rows = 16, .n = 2560, .k = 1 },
    .{ .rows = 64, .n = 1536, .k = 1 }, .{ .rows = 64, .n = 512, .k = 1 }, .{ .rows = 512, .n = 2560, .k = 1 },
};

/// up's dot: a lane's n / 32 fmas and the warp's 5 shuffles; the squares (n / 256 + 5 + 8) and down's 16 are shorter.
fn depth(s: c.Shape) usize {
    return (s.n + 31) / 32 + 5;
}

fn squaresDepth(n: usize) usize {
    return (n + 255) / 256 + 5 + 8;
}

pub fn spec(e: Entry) c.Spec {
    return .{
        .name = e.name,
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .tiles = .{ .{}, .{ .size = gn.threads, .step = 32, .min = 32 }, .{} },
        .served = if (e.mask == P.all) served else served[0..3],
    };
}

/// The value `v` rounded at point `p` when `mask` keeps it.
fn at(v: Iv, mask: u32, p: u32) Iv {
    return if (mask & p != 0) v.bf() else v;
}

/// A dot of bf16 values in fp32 fmas: each product exact before its add, the sum's spread from its chain.
fn dot(s: Sums, a: []const Iv, b: []const f64, depth_: usize, t: []Iv, scratch: []f64) Iv {
    for (t[0..a.len], a, 0..) |*q, x, i| {
        const w = b[i];
        q.* = .{ .m = x.m * w, .lo = @min(x.lo * w, x.hi * w), .hi = @max(x.lo * w, x.hi * w) };
    }
    return s.sum(t[0..a.len], depth_, scratch);
}

pub const Norm = struct {
    rig: rig_.Rig,
    entry: Entry,
    f: cuda.Function,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []f64 = &.{},
    up: []f64 = &.{},
    down: []f64 = &.{},
    bits: [3][]u16 = .{ &.{}, &.{}, &.{} },
    cache: []Iv = &.{},
    fresh: bool = false,
    exact: bool = false,
    /// A checkpoint site's weight, up and down bits in place of drawn ones (runCheckpoint).
    given: ?[3][]const u16 = null,
    dev: [4]u64 = @splat(0),
    out: u64 = 0,

    const vt = g.hooks(Norm);

    pub fn kernel(self: *Norm) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.mask == P.all) null else .shapes };
    }

    fn free(self: *Norm) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        for ([_][]f64{ self.w, self.up, self.down }) |v| a.free(v);
        for (self.bits) |v| a.free(v);
        a.free(self.cache);
        self.* = .{ .rig = self.rig, .entry = self.entry, .f = self.f, .given = self.given };
    }

    pub fn deinit(self: *Norm) void {
        self.free();
        self.rig.deinit();
    }

    /// `len` bf16 values uniform in [lo, hi).
    fn values(a: std.mem.Allocator, rng: std.Random, len: usize, lo: f32, hi: f32) ![]u16 {
        const v = try a.alloc(u16, len);
        for (v) |*b| b.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
        return v;
    }

    /// x in [-2, 2), one in 64 twenty times larger; stress: integers in [-32, 32] and the gate's weights zero.
    pub fn prepare(self: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const n = shape.n;
        if (shape.rows == 0 or n == 0 or n > gn.max_width) return false;
        self.rig.reset();
        self.free();
        self.shape = shape;
        self.exact = fill == .stress;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try a.alloc(u16, shape.rows * n);
        for (self.x, 0..) |*v, i| v.* = c.toBf16(if (fill == .stress)
            @as(f32, @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 65))) - 32))
        else
            (rng.float(f32) * 4 - 2) * (if (i % 64 == 7) @as(f32, 20) else 1));
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, usize, f32 }{ .{ 0, n / 3, iv_nan }, .{ 1, n / 2, iv_inf }, .{ 2, n - 1, -iv_inf }, .{ 3, 0, iv_inf }, .{ 3, n - 1, -iv_inf } };
            for (planted) |p| if (p[0] < shape.rows) {
                self.x[p[0] * n + p[1]] = c.toBf16(p[2]);
            };
        }
        self.bits = .{ try values(a, rng, n, 0.5, 1.5), try values(a, rng, gn.rank * n, -0.1, 0.1), try values(a, rng, n * gn.rank, -1, 1) };
        if (self.given) |gv| for (&self.bits, gv) |*dst, src| {
            if (dst.len != src.len) return false;
            @memcpy(dst.*, src);
        };
        if (fill == .stress) for (self.bits[1..]) |v| @memset(v, 0);
        self.w = try a.alloc(f64, n);
        self.up = try a.alloc(f64, gn.rank * n);
        self.down = try a.alloc(f64, n * gn.rank);
        for ([_][]f64{ self.w, self.up, self.down }, self.bits) |dst, src| for (dst, src) |*d, b| {
            d.* = c.fromBf16(b);
        };
        self.dev = .{ try self.rig.input(std.mem.sliceAsBytes(self.x)), try self.rig.input(std.mem.sliceAsBytes(self.bits[0])), try self.rig.input(std.mem.sliceAsBytes(self.bits[1])), try self.rig.input(std.mem.sliceAsBytes(self.bits[2])) };
        self.out = try self.rig.output(shape.rows * n * 2);
        return true;
    }

    pub fn run(self: *Norm) !void {
        const n = self.shape.n;
        const mask: ?u32 = if (self.entry.mask == P.all) null else self.entry.mask;
        try gc4.launch(self.f, mask, self.rig.s, self.shape.rows, n, self.dev[0], n, self.dev[1], self.dev[2], self.dev[3], self.out, n, eps, scale);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Norm) usize {
        return self.shape.rows * self.shape.n;
    }

    /// Every row in float64 at the kernel's steps, each value with the range its fp32 steps allow.
    fn solve(self: *Norm, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const a = self.rig.gpu.gpa;
        const buf = try a.alloc(Iv, 2 * n);
        defer a.free(buf);
        const scratch = try a.alloc(f64, n);
        defer a.free(scratch);
        const y = buf[0..n];
        const t = buf[n..];
        for (0..self.shape.rows) |r| {
            const xr = self.x[r * n ..][0..n];
            for (t, xr) |*q, b| {
                const v = c.fromBf16(b);
                q.* = Iv.of(v * v);
            }
            const ss = s.sum(t, squaresDepth(n), scratch);
            const inv = ss.div(Iv.of(@floatFromInt(n))).add(Iv.of(eps)).rsqrt();
            for (y, xr, self.w) |*o, b, wv| o.* = Iv.of(c.fromBf16(b)).mul(inv).mul(Iv.of(wv)).bf();
            var h: [gn.rank]Iv = undefined;
            for (&h, 0..) |*hv, j| {
                const u = dot(s, y, self.up[j * n ..], depth(self.shape), t, scratch).bf();
                hv.* = u.div(Iv.of(1).add(u.neg().exp())).bf();
            }
            for (out[r * n ..][0..n], y, 0..) |*o, yv, i| {
                const v = dot(s, &h, self.down[i * gn.rank ..], gn.rank, t, scratch).bf();
                o.* = yv.mul(Iv.of(scale)).bf().mul(v.sigmoid().bf()).bf();
            }
        }
    }

    fn solved(self: *Norm) ![]const Iv {
        if (self.fresh) return self.cache;
        const a = self.rig.gpu.gpa;
        a.free(self.cache);
        self.cache = &.{};
        self.cache = try a.alloc(Iv, self.count());
        try self.solve(.{ .exact = self.exact }, self.cache);
        self.fresh = true;
        return self.cache;
    }

    pub fn reference(self: *Norm, want: []f64, mag: []f64) !void {
        self.fresh = false;
        for (want, mag, try self.solved()) |*w, *m, v| {
            w.* = v.m;
            m.* = 0;
        }
    }

    /// The farthest end of each output's range from float64's value.
    pub fn bounds(self: *Norm, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, try self.solved()) |*l, w, v| l.* = if (std.math.isFinite(w)) @max(0, @max(w - v.lo, v.hi - w)) + 0x1p-140 else std.math.floatMax(f64);
    }

    pub fn intervals(self: *Norm, low: []f64, high: []f64) !void {
        for (low, high, try self.solved()) |*l, *h, v| {
            l.* = v.lo;
            h.* = v.hi;
        }
    }

    /// The rows with every sum in `acc`, in `order` (gate 4's replays).
    pub fn simulate(self: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const tmp = try a.alloc(Iv, out.len);
        defer a.free(tmp);
        try self.solve(.{ .sim = .{ .acc = acc, .order = order }, .exact = true }, tmp);
        for (out, tmp) |*o, v| o.* = v.m;
    }
};

const iv_nan = std.math.nan(f32);
const iv_inf = std.math.inf(f32);

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    norm: gc4.Norm,
    items: [entries.len]Norm = undefined,

    /// The set, its kernels appended to `list` (each control after the kernel it breaks).
    pub fn init(gpu: check_.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var m = try gc4.Norm.load(gpu.d);
        errdefer m.unload();
        set.* = .{ .gpa = gpu.gpa, .norm = m };
        for (&set.items, entries) |*x, e| {
            const sym: [:0]const u8 = if (e.mask == P.all) "tf_gc4_gated_norm" else "tf_ctl_gc4_gated_norm";
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = try set.norm.mod.function(sym) };
        }
        for (&set.items) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.items) |*x| x.deinit();
        set.norm.unload();
        set.gpa.destroy(set);
    }
};

/// The GatedNorm image's gates 1-4 and its controls alone.
pub fn run(gpu: check_.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x6763346e, &t);
    std.debug.print("SUMMARY gigachat4 norm: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed });
    try check_.expect(t.problems() == 0, "gigachat4-norm contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

/// The released checkpoint's GatedNorms: each site's own weight, up and down, 16 drawn rows, inside float64's range.
pub fn runCheckpoint(gpu: check_.Gpu, dir: []const u8) !void {
    const a = gpu.gpa;
    var ck = try core.Checkpoint.openModel(a, gpu.io, dir);
    defer ck.close();
    var prefixes: std.ArrayList([]const u8) = .empty;
    defer prefixes.deinit(a);
    const tail = ".gate_up_projection.weight";
    for (ck.files.items) |*f| for (f.names.keys()) |k| if (std.mem.endsWith(u8, k, tail)) try prefixes.append(a, k[0 .. k.len - tail.len]);
    std.mem.sort([]const u8, prefixes.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var m = try gc4.Norm.load(gpu.d);
    defer m.unload();
    var x: Norm = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = entries[0], .f = m.gated_fn };
    defer x.deinit();
    var outputs: usize = 0;
    var widths = [_]usize{ 0, 0, 0 };
    var name: [320]u8 = undefined;
    for (prefixes.items, 0..) |pre, i| {
        var bits: [3][]const u16 = undefined;
        for ([_][]const u8{ ".weight", ".gate_up_projection.weight", ".gate_down_projection.weight" }, &bits) |suf, *b| {
            const t = (try ck.stored(try std.fmt.bufPrint(&name, "{s}{s}", .{ pre, suf }))) orelse return error.MissingTensor;
            if (t.dtype != .bf16) return error.UnexpectedTensor;
            b.* = std.mem.bytesAsSlice(u16, @as([]align(2) const u8, @alignCast(t.bytes)));
        }
        const n = bits[0].len;
        if (bits[1].len != gn.rank * n or bits[2].len != n * gn.rank) return error.UnexpectedTensor;
        x.given = bits;
        if (!try x.prepare(.{ .rows = 16, .n = n, .k = 1 }, .random, 0x6763 + i)) return error.UnexpectedTensor;
        try x.run();
        const got = try a.dupe(u8, try x.rig.outputBytes());
        defer a.free(got);
        const want = try x.solved();
        for (want, 0..) |v, j| {
            const o = c.fromBf16(std.mem.readInt(u16, got[2 * j ..][0..2], .little));
            if (!(v.lo <= o and o <= v.hi)) {
                std.debug.print("FAIL {s} (n {d}): row {d} out[{d}] = {e} outside [{e}, {e}]\n", .{ pre, n, j / n, j % n, o, v.lo, v.hi });
                return error.TestFailed;
            }
        }
        outputs += want.len;
        widths[if (n == 512) 0 else if (n == 1536) 1 else 2] += 1;
    }
    std.debug.print("PASS gigachat4 norm checkpoint: {d} GatedNorm sites (n 512: {d}, 1536: {d}, 2560: {d}), {d} outputs inside float64's range\n", .{ prefixes.items.len, widths[0], widths[1], widths[2], outputs });
}

test "every control drops one rounding point, each point once, and is named a control" {
    var seen: u32 = 0;
    for (entries) |e| {
        try std.testing.expectEqual(e.mask != P.all, std.mem.startsWith(u8, e.name, "control: "));
        if (e.mask == P.all) continue;
        const dropped = P.all & ~e.mask;
        try std.testing.expectEqual(@as(u32, 1), @popCount(dropped));
        try std.testing.expect(seen & dropped == 0);
        seen |= dropped;
    }
    try std.testing.expectEqual(@as(u32, P.all), seen);
}

test "the served shapes fit the kernel's launch check" {
    for (served) |x| try gn.check(x.rows, x.n, x.n, x.n);
}
