//! The grouped EXL3 experts' contract fixture on the host: specs, random trellises and rows, float64 sums, replays.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

const fmt = core.exl3_format;
const c = core.contract.spec;
const rows_ = core.contract.rows;

pub const Codebook = fmt.Codebook;

/// The load-ahead, then gate 3's planted faults (contract_controls.cu, mul1): loads tile i + 2, or stale last words.
pub const Fault = enum { none, skip, stale };

pub const experts = 4;
pub const slots = 2; // distinct experts a token row
const warps = 4; // grouped_body's W
const step: f32 = 256; // gate 4's x: multiples of 256 to 8,192, so fp16 sums of the products overflow

/// Each expert's half-bits a value: lane_words 2 and 3 (3.5 bits, 5 bits) and the packs' 2, 3 and 4 bits.
pub fn halves(cb: Codebook) [experts]u16 {
    return if (cb == .mul1) .{ 6, 8, 7, 10 } else .{ 8, 4, 6, 10 };
}

/// The engine's K splits (exl3_experts.plan): gate and up where K > N, as at every served model's shapes, else down.
pub fn split(s: c.Shape) usize {
    return cuda.exl3_experts.plan(s.k, s.n, s.k > s.n) catch 1;
}

/// A split's 16-input mma steps, the 16 products inside one, then the 4 warps' partials added.
fn depth(s: c.Shape) usize {
    return s.k / split(s) / 16 + 16 + warps;
}

const rule = "as main's loop gives them: NaN and infinite inputs";
/// rows, n, k: 3, 2, 1 k tiles a warp; GLM-5.3-Flash's and MiMo-V2.6-Flash's experts on one rank and two; Qwen3.5's
const served = [_]c.Shape{
    .{ .rows = 4, .n = 128, .k = 768 },   .{ .rows = 4, .n = 128, .k = 512 },   .{ .rows = 4, .n = 128, .k = 256 },
    .{ .rows = 1, .n = 2048, .k = 4096 }, .{ .rows = 4, .n = 4096, .k = 2048 }, .{ .rows = 17, .n = 1024, .k = 4096 },
    .{ .rows = 4, .n = 4096, .k = 1024 }, .{ .rows = 4, .n = 512, .k = 2048 },  .{ .rows = 4, .n = 2048, .k = 512 },
};
const tiles = [3]c.Tile{ .{ .size = 16 }, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 256, .step = 64, .min = 64 } };

pub fn spec(cb: Codebook, fault: Fault) c.Spec {
    const name = switch (fault) {
        .none => switch (cb) {
            .inst3 => "cuda exl3 grouped experts tf_exl3_grouped_0 (3inst), a tile ahead on sm_120 up",
            .mcg => "cuda exl3 grouped experts tf_exl3_grouped_1 (mcg), a tile ahead on sm_120 up",
            .mul1 => "cuda exl3 grouped experts tf_exl3_grouped_2 (mul1), a tile ahead on sm_120 up",
        },
        .skip => "control: exl3 grouped load-ahead reads tile i + 2",
        .stale => "control: exl3 grouped load-ahead decodes stale words on the last tile",
    };
    const old = switch (cb) {
        .inst3 => "tf_ctl_exl3_grouped_old_0 (main's loop)",
        .mcg => "tf_ctl_exl3_grouped_old_1 (main's loop)",
        .mul1 => "tf_ctl_exl3_grouped_old_2 (main's loop)",
    };
    // EXL3 weights are no integers: gate 4's fp32 sums meet the gamma bound, where fp16 overflows and bf16 misses 240x
    return .{ .name = name, .accum = .f32, .depth = depth, .out = .f32, .replaces = old, .nan = .{ .stated = rule }, .tiles = tiles, .served = &served, .sums = .{ .exact = false } };
}

/// One call's host side: each expert's trellis, the pairs' fp16 rows, each pair's expert; outputs [split][pair][n].
pub const Fixture = struct {
    a: std.mem.Allocator,
    cb: Codebook,
    shape: c.Shape,
    sk: usize,
    trellis: [experts][]i16,
    x: []u16, // fp16 bits [pairs][k], the rotated rows
    pick: []i32, // [pairs]: pair p = row * slots + slot
    w: []f16 = &.{}, // each expert's W_q [k][n], decoded on first use

    /// false: a shape the kernel is not built for (N a whole number of 128s, K of 64s).
    pub fn valid(s: c.Shape) bool {
        return s.rows > 0 and s.n > 0 and s.n % 128 == 0 and s.k > 0 and s.k % 64 == 0;
    }

    pub fn init(a: std.mem.Allocator, cb: Codebook, s: c.Shape, fill: rows_.Fill, seed: u64) !Fixture {
        var f: Fixture = .{ .a = a, .cb = cb, .shape = s, .sk = split(s), .trellis = @splat(@as([]i16, &.{})), .x = &.{}, .pick = &.{} };
        errdefer f.deinit();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const hv = halves(cb);
        for (&f.trellis, hv) |*t, h| {
            t.* = try a.alloc(i16, s.k / 16 * (s.n / 16) * @as(usize, h) * 8);
            for (t.*) |*v| v.* = rng.int(i16);
        }
        const p = s.rows * slots;
        f.pick = try a.alloc(i32, p);
        for (0..s.rows) |r| {
            const first = rng.uintLessThan(usize, experts);
            for (0..slots) |i| f.pick[r * slots + i] = @intCast((first + i) % experts);
        }
        f.x = try a.alloc(u16, p * s.k);
        for (f.x) |*v| v.* = bits(rng.float(f32) * 4 - 2);
        switch (fill) {
            .random => {},
            .nonfinite => {
                const k = s.k;
                const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
                const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
                for (at, put) |q, v| if (q[0] < p) {
                    f.x[q[0] * k + q[1]] = bits(v);
                };
            },
            .stress => for (f.x) |*v| {
                v.* = bits(step * @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, 32))));
            },
        }
        return f;
    }

    pub fn deinit(f: *Fixture) void {
        for (f.trellis) |t| f.a.free(t);
        f.a.free(f.x);
        f.a.free(f.pick);
        f.a.free(f.w);
        f.* = undefined;
    }

    pub fn pairs(f: *const Fixture) usize {
        return f.shape.rows * slots;
    }

    pub fn outputs(f: *const Fixture) usize {
        return f.sk * f.pairs() * f.shape.n;
    }

    /// Each expert's W_q [k][n] as the trellis decodes it.
    fn weights(f: *Fixture) ![]const f16 {
        if (f.w.len > 0) return f.w;
        const s = f.shape;
        f.w = try f.a.alloc(f16, experts * s.k * s.n);
        for (f.trellis, halves(f.cb), 0..) |t, hv, e| fmt.unpack(.{ .halves = hv }, f.cb, t, s.k / 16, s.n / 16, f.w[e * s.k * s.n ..][0 .. s.k * s.n]);
        return f.w;
    }

    /// The terms output (split, pair, column) sums: x times W_q over the split's K range, in K order.
    fn terms(f: *const Fixture, w: []const f16, sp: usize, p: usize, j: usize, t: []f64) void {
        const s = f.shape;
        const e: usize = @intCast(f.pick[p]);
        const per = s.k / f.sk;
        for (t, sp * per..) |*o, i| o.* = value(f.x[p * s.k + i]) * @as(f64, w[(e * s.k + i) * s.n + j]);
    }

    pub fn reference(f: *Fixture, want: []f64, mag: []f64) !void {
        const w = try f.weights();
        const t = try f.a.alloc(f64, f.shape.k / f.sk);
        defer f.a.free(t);
        for (0..f.sk) |sp| for (0..f.pairs()) |p| for (0..f.shape.n) |j| {
            f.terms(w, sp, p, j, t);
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (t) |v| {
                sum += v;
                abs += @abs(v);
            }
            const o = (sp * f.pairs() + p) * f.shape.n + j;
            want[o] = sum;
            mag[o] = abs;
        };
    }

    pub fn simulate(f: *Fixture, acc: c.Accum, order: rows_.Order, out: []f64) !void {
        const w = try f.weights();
        const t = try f.a.alloc(f64, f.shape.k / f.sk);
        defer f.a.free(t);
        for (0..f.sk) |sp| for (0..f.pairs()) |p| for (0..f.shape.n) |j| {
            f.terms(w, sp, p, j, t);
            out[(sp * f.pairs() + p) * f.shape.n + j] = rows_.sumIn(acc, order, t);
        };
    }
};

fn bits(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}

fn value(h: u16) f64 {
    return @as(f16, @bitCast(h));
}

test "exl3 grouped contract: one, two and three k tiles a warp, and a split for every served shape" {
    const want = [_]usize{ 3, 2, 1 };
    for (served[0..3], want) |s, n| try std.testing.expectEqual(n, s.k / split(s) / 16 / warps);
    for (served) |s| {
        try std.testing.expect(Fixture.valid(s));
        try std.testing.expectEqual(@as(usize, 0), s.k % (split(s) * 16 * warps));
    }
}

/// A perfect fp32 kernel on the host (its outputs: the pairwise fp32 replay), so gate 4 judges its stress rows alone.
const HostSums = struct {
    a: std.mem.Allocator,
    cb: Codebook,
    f: ?Fixture = null,
    out: []u8 = &.{},

    fn of(ctx: *anyopaque) *HostSums {
        return @ptrCast(@alignCast(ctx));
    }
    fn prepare(ctx: *anyopaque, s: c.Shape, fill: rows_.Fill, seed: u64) anyerror!bool {
        const h = of(ctx);
        if (h.f) |*f| f.deinit();
        h.f = null;
        h.f = try Fixture.init(h.a, h.cb, s, fill, seed);
        return true;
    }
    fn run(ctx: *anyopaque) anyerror!void {
        const h = of(ctx);
        const f = &h.f.?;
        const got = try h.a.alloc(f64, f.outputs());
        defer h.a.free(got);
        try f.simulate(.f32, .pairwise, got);
        h.a.free(h.out);
        h.out = try h.a.alloc(u8, 4 * got.len);
        for (got, 0..) |v, i| std.mem.writeInt(u32, h.out[4 * i ..][0..4], @bitCast(@as(f32, @floatCast(v))), .little);
    }
    fn output(ctx: *anyopaque) anyerror![]const u8 {
        return of(ctx).out;
    }
    fn inputs(_: *anyopaque) anyerror![]const u8 {
        return &.{};
    }
    fn intact(_: *anyopaque) anyerror!bool {
        return true;
    }
    fn overlap(_: *anyopaque) bool {
        return false;
    }
    fn count(ctx: *anyopaque) usize {
        return of(ctx).f.?.outputs();
    }
    fn reference(ctx: *anyopaque, want: []f64, mag: []f64) anyerror!void {
        return of(ctx).f.?.reference(want, mag);
    }
    fn simulate(ctx: *anyopaque, acc: c.Accum, order: rows_.Order, out: []f64) anyerror!void {
        return of(ctx).f.?.simulate(acc, order, out);
    }
    const vt: core.contract.gates.VTable = .{ .prepare = prepare, .run = run, .output = output, .inputs = inputs, .intact = intact, .overlap = overlap, .count = count, .reference = reference, .simulate = simulate };
};

test "exl3 grouped contract: gate 4's stress rows hold fp32 replays and make fp16 and bf16 sums miss" {
    const gates = core.contract.gates;
    for ([_]Codebook{ .inst3, .mcg, .mul1 }) |cb| {
        var h: HostSums = .{ .a = std.testing.allocator, .cb = cb };
        defer {
            if (h.f) |*f| f.deinit();
            std.testing.allocator.free(h.out);
        }
        const k: gates.Kernel = .{ .spec = spec(cb, .none), .ctx = &h, .vt = &HostSums.vt };
        // the runner's seed for this kernel (gates.runAll), so these are the rows the device run sees
        try std.testing.expectEqual(gates.Verdict.pass, try gates.sums(std.testing.allocator, k, std.hash.Wyhash.hash(0x6b63, k.spec.name)));
    }
}
