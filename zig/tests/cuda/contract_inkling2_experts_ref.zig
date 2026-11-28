//! Inkling 2's grouped experts on the host: specs, Inkling's routing and trellises, float64 sums, replays.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ink = @import("inkling2");

const fmt = core.exl3_format;
const c = core.contract.spec;
const rows_ = core.contract.rows;

pub const experts = 384;
pub const slots = 8; // top 8
pub const hidden = 4608;
pub const width = 1152; // a rank's intermediate at TP=2: 2304 cut in 128-column blocks
const pool = 16; // distinct trellises a shape: expert e's gate takes pool[e % 16], its up pool[(7e + 3) % 16]
const warps = 4; // grouped_body's W
const step: f32 = 256; // gate 4's x: multiples of 256 to 8,192, so fp16 sums of the products overflow

/// The served kernels: decode rows (128 columns, 16 members), prompt chunks (64 columns, 32).
pub const Form = enum { rows, wide };
/// Gate 3's planted faults: the call's last item dropped, or the up rows read as gate rows.
pub const Fault = enum { none, short, swapped };

/// Baseline calls from this many rows list items rather than using one item an expert.
pub const listed_from = 17;

/// Whether a baseline call takes an item list: the prompt kernel always, the rows kernel from 17 rows.
pub fn listed(form: Form, rows: usize) bool {
    return form == .wide or rows >= listed_from;
}

/// Half-bits a value of pool entry i: the release's 2- and 3-bit trellises in turn.
pub fn halves(i: usize) u16 {
    return if (i % 2 == 0) 4 else 6;
}

/// exl3_experts.plan's K splits as experts.run takes them: gate and up where K > N, down otherwise.
pub fn split(s: c.Shape) usize {
    return cuda.exl3_experts.plan(s.k, s.n, s.k > s.n) catch 1;
}

/// Gate and up (two matrices, K > N), or down (one).
pub fn mats(s: c.Shape) usize {
    return if (s.k > s.n) 2 else 1;
}

/// Items experts.zig makes room for: runs of 16 members, plus one partial run a distinct expert.
pub fn maxItems(p: usize) usize {
    return (p + 15) / 16 + @min(p, experts);
}
/// Pairs in one item: 32 on the wide kernel, 16 on core's standard grouped kernel.
pub fn per(form: Form) usize {
    return if (form == .wide) 32 else 16;
}

/// A split's 16-input mma steps, the 16 products inside one, then the 4 warps' partials added in warp order.
fn depth(s: c.Shape) usize {
    return s.k / split(s) / 16 + 16 + warps;
}

fn served(rows: []const usize, comptime n: usize) [2 * n]c.Shape {
    var out: [2 * n]c.Shape = undefined;
    for (rows[0..n], 0..) |r, i| {
        out[2 * i] = .{ .rows = r, .n = width, .k = hidden };
        out[2 * i + 1] = .{ .rows = r, .n = hidden, .k = width };
    }
    return out;
}

const rows_served = served(&.{ 8, 1, 15, 16, 17, 128, 511 }, 7);
const wide_served = served(&.{ 512, 513, 2048 }, 3);
const rule = "as tf_exl3_grouped_2 gives them: NaN and infinite inputs";

pub fn spec(form: Form, fault: Fault) c.Spec {
    const name = switch (fault) {
        .none => if (form == .rows) "inkling2 exl3 tf_exl3_grouped_2 (decode rows to 511, 128 columns by 16 members)" else "inkling2 exl3 tf_exl3_grouped_wide_2 (prompt chunks from 512, 64 columns by 32 members)",
        .short => "inkling2 exl3 control: tf_exl3_grouped_2 launched one item short",
        .swapped => "inkling2 exl3 control: tf_exl3_grouped_wide_2 reading the up rows as gate rows",
    };
    const edge: c.Tile = if (form == .rows) .{ .size = 16 } else .{ .size = ink.experts.wide_from, .min = ink.experts.wide_from };
    const tiles = [3]c.Tile{ edge, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 256, .step = 64, .min = 64 } };
    const list: []const c.Shape = if (form == .rows) &rows_served else &wide_served;
    return .{ .name = name, .accum = .f32, .depth = depth, .out = .f32, .replaces = "tf_exl3_grouped_2 (main's grouped experts, mul1)", .nan = .{ .stated = rule }, .tiles = tiles, .served = list, .sums = .{ .exact = false } };
}

/// Mini's baseline rows and wide kernels, including the two planted launch faults.
pub fn baselineSpec(form: Form, fault: Fault) c.Spec {
    const name = switch (fault) {
        .none => if (form == .rows) "inkling2 experts tf_ink2x_rows (decode rows to 511, 128 columns by 16 members)" else "inkling2 experts tf_ink2x_wide (prompt chunks from 512, 64 columns by 32 members)",
        .short => "inkling2 experts control: tf_ink2x_rows launched one item short",
        .swapped => "inkling2 experts control: tf_ink2x_wide reading the up rows as gate rows",
    };
    const edge: c.Tile = if (form == .rows) .{ .size = 16 } else .{ .size = ink.experts.wide_from, .min = ink.experts.wide_from };
    const tiles = [3]c.Tile{ edge, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 256, .step = 64, .min = 64 } };
    const list: []const c.Shape = if (form == .rows) &rows_served else &wide_served;
    return .{ .name = name, .accum = .f32, .depth = depth, .out = .f32, .replaces = "tf_exl3_grouped_2 (main's grouped experts, mul1)", .nan = .{ .stated = rule }, .tiles = tiles, .served = list, .sums = .{ .exact = false } };
}

/// Each row's 8 distinct experts: expert 0 in every row (long runs and a partial one), the rest at random.
pub fn picks(rng: std.Random, out: []i32) void {
    for (0..out.len / slots) |r| {
        const row = out[r * slots ..][0..slots];
        row[0] = 0;
        var i: usize = 1;
        while (i < slots) {
            const e: i32 = @intCast(rng.uintLessThan(u32, experts));
            if (std.mem.indexOfScalar(i32, row[0..i], e) != null) continue;
            row[i] = e;
            i += 1;
        }
    }
}

/// One projection's host side: the pool's trellises, the pairs' fp16 rows per matrix, each pair's expert.
pub const Fixture = struct {
    a: std.mem.Allocator,
    shape: c.Shape,
    sk: usize,
    trellis: [pool][]i16,
    x: []u16, // fp16 bits [mats][pairs][k], the rotated rows
    pick: []i32, // [pairs]: pair p = row * slots + slot
    w: [pool][]f16 = @splat(@as([]f16, &.{})), // each entry's W_q [k][n], decoded on first use

    /// false: a shape the kernels are not built for (plan() finds no split, N not a whole number of 128s).
    pub fn valid(s: c.Shape) bool {
        if (s.rows == 0 or s.n == 0 or s.k == 0 or s.n % 128 != 0) return false;
        _ = cuda.exl3_experts.plan(s.k, s.n, s.k > s.n) catch return false;
        return true;
    }

    pub fn init(a: std.mem.Allocator, s: c.Shape, fill: rows_.Fill, seed: u64) !Fixture {
        var f: Fixture = .{ .a = a, .shape = s, .sk = split(s), .trellis = @splat(@as([]i16, &.{})), .x = &.{}, .pick = &.{} };
        errdefer f.deinit();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (&f.trellis, 0..) |*t, i| {
            t.* = try a.alloc(i16, s.k / 16 * (s.n / 16) * @as(usize, halves(i)) * 8);
            for (t.*) |*v| v.* = rng.int(i16);
        }
        const p = f.pairs();
        f.pick = try a.alloc(i32, p);
        picks(rng, f.pick);
        f.x = try a.alloc(u16, mats(s) * p * s.k);
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
        for (f.w) |w| f.a.free(w);
        f.a.free(f.x);
        f.a.free(f.pick);
        f.* = undefined;
    }

    pub fn pairs(f: *const Fixture) usize {
        return f.shape.rows * slots;
    }

    pub fn outputs(f: *const Fixture) usize {
        return mats(f.shape) * f.sk * f.pairs() * f.shape.n;
    }

    /// The pool entry matrix `mat` of expert e reads.
    pub fn entry(e: usize, mat: usize) usize {
        return if (mat == 0) e % pool else (7 * e + 3) % pool;
    }

    fn weights(f: *Fixture, i: usize) ![]const f16 {
        if (f.w[i].len > 0) return f.w[i];
        const s = f.shape;
        f.w[i] = try f.a.alloc(f16, s.k * s.n);
        fmt.unpack(.{ .halves = halves(i) }, .mul1, f.trellis[i], s.k / 16, s.n / 16, f.w[i]);
        return f.w[i];
    }

    /// The terms output (mat, split, pair, column) sums: x times W_q over the split's K range, in K order.
    fn terms(f: *Fixture, mat: usize, sp: usize, p: usize, j: usize, t: []f64) !void {
        const s = f.shape;
        const e: usize = @intCast(f.pick[p]);
        const w = try f.weights(entry(e, mat));
        const per_split = s.k / f.sk;
        const x = f.x[(mat * f.pairs() + p) * s.k ..];
        for (t, sp * per_split..) |*o, i| o.* = value(x[i]) * @as(f64, w[i * s.n + j]);
    }

    fn each(f: *Fixture, ctx: anytype, comptime body: fn (@TypeOf(ctx), usize, []const f64) void) !void {
        const t = try f.a.alloc(f64, f.shape.k / f.sk);
        defer f.a.free(t);
        for (0..mats(f.shape)) |mat| for (0..f.sk) |sp| for (0..f.pairs()) |p| for (0..f.shape.n) |j| {
            try f.terms(mat, sp, p, j, t);
            body(ctx, (((mat * f.sk) + sp) * f.pairs() + p) * f.shape.n + j, t);
        };
    }

    pub fn reference(f: *Fixture, want: []f64, mag: []f64) !void {
        const Ctx = struct { want: []f64, mag: []f64 };
        try f.each(Ctx{ .want = want, .mag = mag }, struct {
            fn body(x: Ctx, o: usize, t: []const f64) void {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (t) |v| {
                    sum += v;
                    abs += @abs(v);
                }
                x.want[o] = sum;
                x.mag[o] = abs;
            }
        }.body);
    }

    pub fn simulate(f: *Fixture, acc: c.Accum, order: rows_.Order, out: []f64) !void {
        const Ctx = struct { acc: c.Accum, order: rows_.Order, out: []f64 };
        try f.each(Ctx{ .acc = acc, .order = order, .out = out }, struct {
            fn body(x: Ctx, o: usize, t: []const f64) void {
                x.out[o] = rows_.sumIn(x.acc, x.order, t);
            }
        }.body);
    }
};

/// The lists tf_exl3_group writes for `pick`: distinct experts in id order, members row * 32 + slot, -1 after them.
pub const Lists = struct {
    uids: []i32,
    count: i32,
    members: []i32, // [min(pairs, experts)][rows]

    pub fn init(a: std.mem.Allocator, pick: []const i32, rows: usize) !Lists {
        const maxu = @min(pick.len, experts);
        const uids = try a.alloc(i32, maxu);
        errdefer a.free(uids);
        const members = try a.alloc(i32, maxu * rows);
        @memset(members, -1);
        var used: usize = 0;
        for (0..experts) |e| {
            var m: usize = 0;
            for (pick, 0..) |pe, p| if (pe == e) {
                members[used * rows + m] = @intCast(p / slots * 32 + p % slots);
                m += 1;
            };
            if (m == 0) continue;
            uids[used] = @intCast(e);
            used += 1;
        }
        return .{ .uids = uids, .count = @intCast(used), .members = members };
    }

    pub fn deinit(l: *Lists, a: std.mem.Allocator) void {
        a.free(l.uids);
        a.free(l.members);
    }

    /// Items for runs of `run` members: (u, first, count) in expert order, -1 words after them.
    pub fn items(l: Lists, rows: usize, run: usize, out: []i32) i32 {
        @memset(out, -1);
        var n: usize = 0;
        for (0..@intCast(l.count)) |u| {
            const m = l.members[u * rows ..][0..rows];
            const cnt = std.mem.indexOfScalar(i32, m, -1) orelse rows;
            var first: usize = 0;
            while (first < cnt) : (first += run) {
                out[3 * n ..][0..3].* = .{ @intCast(u), @intCast(first), @intCast(@min(run, cnt - first)) };
                n += 1;
            }
        }
        return @intCast(n);
    }
};

fn bits(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}

fn value(h: u16) f64 {
    return @as(f16, @bitCast(h));
}

test "inkling2 experts contract: every served shape has a split, items fit experts.zig's room" {
    for (rows_served ++ wide_served) |s| {
        try std.testing.expect(Fixture.valid(s));
        try std.testing.expectEqual(@as(usize, 0), s.k % (split(s) * 16 * warps));
    }
    try std.testing.expectEqual(@as(usize, 4), split(rows_served[0]));
    var prng = std.Random.DefaultPrng.init(1);
    const pick = try std.testing.allocator.alloc(i32, 2048 * slots);
    defer std.testing.allocator.free(pick);
    picks(prng.random(), pick);
    var l = try Lists.init(std.testing.allocator, pick, 2048);
    defer l.deinit(std.testing.allocator);
    const out = try std.testing.allocator.alloc(i32, 3 * maxItems(pick.len));
    defer std.testing.allocator.free(out);
    try std.testing.expect(l.items(2048, 16, out) <= maxItems(pick.len));
    try std.testing.expect(l.items(2048, 32, out) <= maxItems(pick.len));
    try std.testing.expectEqual(@as(i32, 0), l.uids[0]);
}
