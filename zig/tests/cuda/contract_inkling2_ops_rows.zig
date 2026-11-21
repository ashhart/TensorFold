//! Inkling 2's row kernels under the contract: tf_ink2_embed, tf_ink2_rowdot (tf_ink2_rowdot_tiles its twin), softcap.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_inkling2_ops_ref.zig");
const M = @import("contract_inkling2_model.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;
const ops = ref.ops;

/// Rows of the fixture's table: ids are a bijection of the rows, so planted rows stay in their rows.
const table_rows = 4096;

fn embedDepth(s: c.Shape) usize {
    return ref.blockDepth(s.k, 256);
}

fn embedServed(comptime C: usize) []const c.Shape {
    return &.{ .{ .rows = 16, .n = C, .k = C }, .{ .rows = 1, .n = C, .k = C }, .{ .rows = 128, .n = C, .k = C }, .{ .rows = 2048, .n = C, .k = C } };
}

/// Rows x hidden x hidden: a prompt chunk, a verify window, one decode row (the drafter's anchor embeds one too).
pub fn embedSpec(shifted: bool, mini: bool) c.Spec {
    const served = if (mini) embedServed(M.mini.hidden) else embedServed(M.small.hidden);
    if (shifted) return .{ .name = M.named("control: inkling2 ops tf_ink2_embed gathering the next row", mini), .accum = .f32, .depth = embedDepth, .out = .bf16, .served = served[0..2] };
    return .{ .name = M.named("inkling2 ops tf_ink2_embed (embed_norm, rounded once)", mini), .accum = .f32, .depth = embedDepth, .out = .bf16, .served = served };
}

/// out = bf16(w * (e * rsqrtf(mean(e^2) + eps))) of each id's embedding row.
pub const Embed = struct {
    kern: *const ops.Kernels,
    shifted: bool = false,
    mini: bool = false,
    shape: c.Shape = undefined,
    table: []u16 = &.{}, // [table_rows + 1][C]: the control reads one row on
    w: []u16 = &.{},
    ids: []i32 = &.{},
    dev: [3]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Embed) c.Spec {
        return embedSpec(self.shifted, self.mini);
    }

    pub fn control(self: *const Embed) ?g.Gate {
        return if (self.shifted) .shapes else null;
    }

    pub fn free(self: *Embed, a: std.mem.Allocator) void {
        a.free(self.table);
        a.free(self.w);
        a.free(self.ids);
        self.table = &.{};
        self.w = &.{};
        self.ids = &.{};
    }

    /// Embeddings in [-1, 1); stress: integers in [-48, 48] (squares to 2^21.7: fp16 overflows, bf16 loses).
    pub fn prepare(self: *Embed, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.rows > table_rows or shape.n != shape.k or shape.n % 8 != 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.n;
        self.table = try a.alloc(u16, (table_rows + 1) * n);
        for (self.table) |*v| v.* = c.toBf16(if (fill == .stress) ref.int(rng, -48, 48) else rng.float(f32) * 2 - 1);
        self.w = try ref.bfs(a, rng, n, 0.5, 1.5);
        self.ids = try a.alloc(i32, shape.rows);
        for (self.ids, 0..) |*id, r| id.* = @intCast((r * 1031 + 17) % table_rows);
        if (fill == .nonfinite) for (0..@min(shape.rows, 4)) |r| {
            const row = self.table[@as(usize, @intCast(self.ids[r])) * n ..][0..n];
            const at = [_]usize{ n / 3, n / 2, n - 1, 0 };
            const put = [_]f32{ ref.nan, ref.inf, -ref.inf, ref.inf };
            row[at[r]] = c.toBf16(put[r]);
            if (r == 3) row[n - 1] = c.toBf16(-ref.inf);
        };
        self.dev = .{ try ref.input(rig, self.ids), try ref.input(rig, self.table), try ref.input(rig, self.w) };
        self.out = try rig.output(shape.rows * n * 2);
        return true;
    }

    pub fn run(self: *Embed, rig: *rig_.Rig) !void {
        const table = self.dev[1] + if (self.shifted) self.shape.n * 2 else 0;
        try ops.embed(self.kern, rig.s, self.dev[0], self.shape.rows, table, self.dev[2], self.out, self.shape.n, ref.eps);
    }

    pub fn count(self: *const Embed) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Embed, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const terms = try a.alloc(Iv, n);
        defer a.free(terms);
        const scratch = try a.alloc(f64, n);
        defer a.free(scratch);
        for (0..self.shape.rows) |r| {
            const e = self.table[@as(usize, @intCast(self.ids[r])) * n ..][0..n];
            for (terms, e) |*t, v| t.* = Iv.of(ref.bf(v) * ref.bf(v));
            const inv = ref.rms(s.sum(terms, embedDepth(self.shape), scratch), n, ref.eps);
            for (out[r * n ..][0..n], e, self.w) |*o, v, wv| o.* = Iv.of(ref.bf(wv)).mul(Iv.of(ref.bf(v)).mul(inv)).bf();
        }
    }
};

/// The row dots forward.zig runs: the router (fp32 sigmoid scores), the sinks (bf16 sigmoid), the dynamic conv taps.
pub const Dot = enum { router, sink, taps };

/// A lane's columns 32 apart, then the warp's 5 shuffles.
fn dotDepth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5;
}

/// A model's row-dot calls, plain then tiled: router (experts), sinks (query heads), dynamic taps (4 a key head).
fn dotServed(comptime m: M.Model, d: Dot) [2][]const c.Shape {
    const C = m.hidden;
    const E = m.experts;
    const H = m.heads;
    const T = comptime m.taps();
    const plain: []const c.Shape = switch (d) { // comptime arrays: a spec outlives this call
        .router => &.{ .{ .rows = 16, .n = E, .k = C }, .{ .rows = 1, .n = E, .k = C }, .{ .rows = 31, .n = E, .k = C }, .{ .rows = 7, .n = E, .k = C } },
        .sink => &.{ .{ .rows = 16, .n = H, .k = C }, .{ .rows = 1, .n = H, .k = C }, .{ .rows = 128, .n = H, .k = C }, .{ .rows = 2048, .n = H, .k = C } },
        .taps => &.{ .{ .rows = 16, .n = T, .k = C }, .{ .rows = 1, .n = T, .k = C }, .{ .rows = 31, .n = T, .k = C } },
    };
    const wide: []const c.Shape = if (d == .router)
        &.{ .{ .rows = 64, .n = E, .k = C }, .{ .rows = 32, .n = E, .k = C }, .{ .rows = 100, .n = E, .k = C }, .{ .rows = 2048, .n = E, .k = C } }
    else
        &.{ .{ .rows = 64, .n = T, .k = C }, .{ .rows = 32, .n = T, .k = C }, .{ .rows = 100, .n = T, .k = C }, .{ .rows = 2048, .n = T, .k = C } };
    return .{ plain, wide };
}

/// Rows x outputs x hidden; tiles: calls of 32 rows or more with fp32 out (router, taps), bit-equal to tf_ink2_rowdot.
pub fn dotSpec(d: Dot, tiles: bool, short: bool, mini: bool) c.Spec {
    const both = if (mini) dotServed(M.mini, d) else dotServed(M.small, d);
    const plain = both[0];
    const wide = both[1];
    const names = [3][2][]const u8{
        .{ "inkling2 ops tf_ink2_rowdot router (fp32 sigmoid scores)", "inkling2 ops tf_ink2_rowdot_tiles router (fp32 sigmoid scores)" },
        .{ "inkling2 ops tf_ink2_rowdot attention sinks (bf16 sigmoid)", "" },
        .{ "inkling2 ops tf_ink2_rowdot dynamic conv taps (fp32)", "inkling2 ops tf_ink2_rowdot_tiles dynamic conv taps (fp32)" },
    };
    const flat = names[0] ++ names[1] ++ names[2];
    var s: c.Spec = .{ .name = M.namedAt(&flat, @as(usize, @backingInt(d)) * 2 + @intFromBool(tiles), mini), .accum = .f32, .depth = dotDepth, .out = if (d == .sink) .bf16 else .f32, .served = if (tiles) wide else plain };
    s.tiles = if (tiles) .{ .{ .size = 32, .min = 32 }, .{ .size = 16 }, .{} } else .{ .{}, .{ .size = 8 }, .{} };
    if (tiles) s.replaces = "tf_ink2_rowdot";
    if (short) {
        s.name = M.named("control: inkling2 ops tf_ink2_rowdot router one 32-column step short", mini);
        s.tiles = .{ .{}, .{}, .{} };
        s.served = plain[0..2];
    }
    return s;
}

/// out[row, j] = act(x[row] . W[j]) in fp32 over bf16 inputs.
pub const Rowdot = struct {
    kern: *const ops.Kernels,
    dot: Dot,
    tiles: bool = false,
    short: bool = false,
    mini: bool = false,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u16 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,
    base: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Rowdot) c.Spec {
        return dotSpec(self.dot, self.tiles, self.short, self.mini);
    }

    pub fn control(self: *const Rowdot) ?g.Gate {
        return if (self.short) .shapes else null;
    }

    pub fn free(self: *Rowdot, a: std.mem.Allocator) void {
        a.free(self.x);
        a.free(self.w);
        self.x = &.{};
        self.w = &.{};
    }

    fn outBytes(self: *const Rowdot) usize {
        return self.count() * @as(usize, if (self.dot == .sink) 2 else 4);
    }

    /// Values in [-1, 1); stress: integers in [8, 16], each row's second half its first's mirror one step on.
    pub fn prepare(self: *Rowdot, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k % 128 != 0) return false;
        self.free(a);
        self.shape = shape;
        const K = shape.k;
        const wl: f32 = if (self.dot == .router) 0.1 else 1;
        self.x = try ref.bfs(a, rng, shape.rows * K, -1, 1);
        self.w = try ref.bfs(a, rng, shape.n * K, -wl, wl);
        if (fill == .stress) {
            const h = K / 2;
            for ([_][]u16{ self.x, self.w }, 0..) |m, mi| for (0..m.len / K) |r| {
                const row = m[r * K ..][0..K];
                for (0..h) |i| {
                    const v = ref.int(rng, 8, 16);
                    row[i] = c.toBf16(v);
                    row[h + (i + 1) % h] = c.toBf16(if (mi == 0) v else -v);
                }
            };
        }
        if (fill == .nonfinite) ref.plant(u16, self.x, shape.rows, K, ref.asBf);
        self.dev = .{ try ref.input(rig, self.x), try ref.input(rig, self.w) };
        self.out = try rig.output(self.outBytes());
        if (self.tiles) self.base = try rig.scratch(self.outBytes());
        return true;
    }

    /// tf_ink2_rowdot's launch as ops.rowdot makes it below 32 rows.
    fn plain(self: *const Rowdot, s: cuda.Stream, out: u64, k: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ self.dev[0], self.dev[1], out }) |v| a.add(v);
        for ([_]usize{ self.shape.n, k, @intFromBool(self.dot == .sink), @intFromBool(self.dot != .taps) }) |v| a.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(self.kern.rowdot, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast((self.shape.n + 7) / 8) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn run(self: *Rowdot, rig: *rig_.Rig) !void {
        if (self.short) return self.plain(rig.s, self.out, self.shape.k - 32);
        if (!self.tiles) return self.plain(rig.s, self.out, self.shape.k);
        const d = self.dot;
        try ops.rowdot(self.kern, rig.s, self.dev[0], self.dev[1], self.out, self.shape.rows, self.shape.n, self.shape.k, d == .sink, d != .taps);
    }

    pub fn replaced(self: *Rowdot, rig: *rig_.Rig) ![]const u8 {
        try self.plain(rig.s, self.base, self.shape.k);
        return rig.bytes(self.base, self.outBytes());
    }

    pub fn count(self: *const Rowdot) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Rowdot, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const K = self.shape.k;
        const terms = try a.alloc(Iv, K);
        defer a.free(terms);
        const scratch = try a.alloc(f64, K);
        defer a.free(scratch);
        for (0..self.shape.rows) |r| for (0..self.shape.n) |j| {
            for (terms, self.x[r * K ..][0..K], self.w[j * K ..][0..K]) |*t, xv, wv| t.* = Iv.of(ref.bf(xv) * ref.bf(wv));
            var v = s.sum(terms, dotDepth(self.shape), scratch);
            if (self.dot != .taps) v = ref.sigmoid(v);
            out[r * self.shape.n + j] = if (self.dot == .sink) v.bf() else v;
        };
    }
};

fn capServed(comptime V: usize) []const c.Shape {
    return &.{ .{ .rows = 16, .n = V, .k = 1 }, .{ .rows = 1, .n = V, .k = 1 }, .{ .rows = 128, .n = V, .k = 1 } };
}

/// Rows x a rank's vocabulary share x 1: the head's fp32 logits, in place (Mini: the whole 201,024 at one rank).
pub fn capSpec(wrong: bool, mini: bool) c.Spec {
    const served = if (mini) capServed(M.mini.share()) else capServed(M.small.share());
    const name = if (wrong) M.named("control: inkling2 ops tf_ink2_softcap at cap 20", mini) else M.named("inkling2 ops tf_ink2_softcap (final logits, cap 30, in place)", mini);
    return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .f32, .in_place = true, .served = if (wrong) served[0..1] else served };
}

/// x = tanhf(x / 30) * 30 over the logits in place.
pub const Softcap = struct {
    kern: *const ops.Kernels,
    wrong: bool = false,
    mini: bool = false,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    out: u64 = 0,

    pub fn spec(self: *const Softcap) c.Spec {
        return capSpec(self.wrong, self.mini);
    }

    pub fn control(self: *const Softcap) ?g.Gate {
        return if (self.wrong) .shapes else null;
    }

    pub fn free(self: *Softcap, a: std.mem.Allocator) void {
        a.free(self.x);
        self.x = &.{};
    }

    /// Logits in [-90, 90): past the cap's knee both ways.
    pub fn prepare(self: *Softcap, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        self.x = try ref.f32s(a, rng, shape.rows * shape.n, -90, 90);
        if (fill == .nonfinite) ref.plant(f32, self.x, shape.rows, shape.n, ref.asF32);
        self.out = try rig.output(self.x.len * 4);
        return true;
    }

    pub fn run(self: *Softcap, rig: *rig_.Rig) !void {
        try ref.restore(rig, self.out, std.mem.sliceAsBytes(self.x));
        try ops.softcap(self.kern, rig.s, self.out, self.x.len, if (self.wrong) 20 else 30);
    }

    pub fn count(self: *const Softcap) usize {
        return self.x.len;
    }

    pub fn solve(self: *const Softcap, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        for (out, self.x) |*o, v| o.* = ref.tanh(Iv.of(v).div(Iv.of(30))).mul(Iv.of(30));
    }
};
