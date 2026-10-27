//! MiMo-V2's torch glue under the contract (glue.cu): embed, the ranks' add, the RoPE tables and the head's argmax.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const mu = @import("contract_mimo_util.zig");

const c = core.contract.spec;
const g = core.contract.gates;

const theta_full: f64 = 10000000; // rope_theta
const theta_swa: f64 = 10000; // swa_rope_theta, and the drafter's rope_theta
const vocab_rows = 300; // the test's embedding table: ids index it as the vocabulary's

fn noSums(_: c.Shape) usize {
    return 0;
}

pub const Kind = enum { embed, add2, add2_in_place, rope_full, rope_swa, argmax };

const argmax_rule = "the first NaN, else the first maximum, over the ranks' slices in rank order (torch.argmax)";

pub fn spec(kind: Kind) c.Spec {
    const hidden = &[_]c.Shape{ .{ .rows = 8, .n = 4096, .k = 1 }, .{ .rows = 1, .n = 4096, .k = 1 }, .{ .rows = 128, .n = 4096, .k = 1 } };
    const sums = &[_]c.Shape{ .{ .rows = 8, .n = 4096, .k = 1 }, .{ .rows = 1, .n = 4096, .k = 1 }, .{ .rows = 128, .n = 4096, .k = 1 }, .{ .rows = 32, .n = 4096, .k = 1 } };
    const tables = &[_]c.Shape{ .{ .rows = 262144, .n = 64, .k = 1 }, .{ .rows = 4104, .n = 64, .k = 1 } };
    const t256: [3]c.Tile = .{ .{}, .{ .size = 256 }, .{} };
    const ctx: [3]c.Tile = .{ .{ .size = 65536 }, .{}, .{} };
    return switch (kind) {
        .embed => .{ .name = "cuda mimo mimo_embed_f32 (bf16 rows to fp32)", .accum = .none, .depth = noSums, .out = .f32, .tiles = t256, .served = hidden },
        .add2 => .{ .name = "cuda mimo mimo_add2_f32 (two ranks' fp32 partials)", .accum = .none, .depth = noSums, .out = .f32, .tiles = t256, .served = sums },
        .add2_in_place => .{ .name = "cuda mimo mimo_add2_f32 (in place: the drafter's residual)", .accum = .none, .depth = noSums, .out = .f32, .in_place = true, .tiles = t256, .served = sums },
        .rope_full => .{ .name = "cuda mimo mimo_rope_tables (theta 1e7, fp64 then fp32)", .accum = .none, .depth = noSums, .out = .f32, .tiles = ctx, .served = tables },
        .rope_swa => .{ .name = "cuda mimo mimo_rope_tables (theta 1e4: sliding layers, drafter)", .accum = .none, .depth = noSums, .out = .f32, .tiles = ctx, .served = tables },
        .argmax => .{ .name = "cuda mimo mimo_argmax (the ranks' head slices, NaN first)", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = argmax_rule }, .tiles = t256, .served = &.{ .{ .rows = 8, .n = 76288, .k = 2 }, .{ .rows = 1, .n = 76288, .k = 2 }, .{ .rows = 8, .n = 152576, .k = 1 } } },
    };
}

pub const Glue = struct {
    rig: rig_.Rig,
    kind: Kind,
    f: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []f64 = &.{}, // the table, a, or the gathered logits [rows, world * vr]
    y: []f64 = &.{}, // b
    ids: []i64 = &.{},
    th: f64 = 0,
    dev: [4]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Glue) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    const vt_ref = blk: {
        var v = g.hooks(Glue);
        v.check = null;
        break :blk v;
    };
    const vt_rule = g.hooks(Glue);

    pub fn kernel(self: *Glue) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = if (self.kind == .argmax) &vt_rule else &vt_ref };
    }

    pub fn prepare(self: *Glue, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        const ga = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const nf = fill == .nonfinite;
        const nan = std.math.nan(f64);
        const inf = std.math.inf(f64);
        const n = shape.rows;
        const D = shape.n;
        switch (self.kind) {
            .embed => {
                self.x = try a.alloc(f64, vocab_rows * D);
                for (self.x) |*v| v.* = mu.uni(rng, u16, -2, 2);
                self.ids = try a.alloc(i64, n);
                for (self.ids, 0..) |*t, r| t.* = @intCast((r * 37 + 11) % vocab_rows);
                if (nf) for ([_]f64{ nan, inf, -inf }, 0..) |v, r| if (r < n) {
                    self.x[@as(usize, @intCast(self.ids[r])) * D + r + 1] = v;
                };
                self.dev[0] = try mu.put(&self.rig, ga, u16, self.x);
                self.dev[1] = try mu.ints(&self.rig, i64, self.ids);
                self.dev[2] = try self.rig.output(n * D * 4);
            },
            .add2, .add2_in_place => {
                self.x = try a.alloc(f64, n * D);
                self.y = try a.alloc(f64, n * D);
                for (self.x, self.y) |*p, *q| {
                    p.* = mu.uni(rng, f32, -4, 4);
                    q.* = mu.uni(rng, f32, -4, 4);
                }
                if (nf and n > 3) {
                    self.x[0] = nan;
                    self.x[D + 1] = inf;
                    self.y[2 * D + 2] = -inf;
                    self.x[3 * D + 3] = inf; // inf + -inf: NaN
                    self.y[3 * D + 3] = -inf;
                }
                self.dev[0] = try mu.put(&self.rig, ga, f32, self.x);
                self.dev[1] = try mu.put(&self.rig, ga, f32, self.y);
                self.dev[2] = try self.rig.output(n * D * 4);
            },
            .rope_full, .rope_swa => {
                if (D % 2 != 0) return false;
                self.th = if (nf) nan else if (self.kind == .rope_full) theta_full else theta_swa; // column 0: theta^0 = 1
                self.dev[0] = try self.rig.output(n * D * 4);
            },
            .argmax => {
                const world = shape.k;
                const cols = world * D;
                self.x = try a.alloc(f64, n * cols);
                for (self.x) |*v| v.* = mu.int(rng, -1024, 1024) / 16; // sixteenths: ties
                if (nf) {
                    self.x[cols / 2] = nan;
                    self.x[cols - 1] = nan; // the first NaN wins
                    if (n > 1) self.x[cols + cols / 3] = inf;
                    if (n > 2) self.x[2 * cols + 7] = -inf;
                }
                const lay = try a.alloc(f64, n * cols); // recv [world, rows, vr]: rank k's slice of row r
                for (0..n) |r| for (0..world) |k| for (0..D) |j| {
                    lay[(k * n + r) * D + j] = self.x[r * cols + k * D + j];
                };
                self.dev[0] = try mu.put(&self.rig, ga, f32, lay);
                self.dev[1] = try self.rig.output(n * 4);
            },
        }
        return true;
    }

    /// forward.zig's launches: embed and argmax a block a row, add2 over at most 1,024 blocks, the tables on 256.
    pub fn run(self: *Glue) !void {
        const s = self.rig.s;
        const sh = self.shape;
        const d = self.dev;
        const n = sh.rows * sh.n;
        var a: cuda.Args = .{};
        var grid: u32 = @intCast(sh.rows);
        switch (self.kind) {
            .embed => {
                inline for (.{ d[0], d[1], d[2] }) |p| a.add(p);
                a.add(@as(c_int, @intCast(sh.n)));
            },
            .add2, .add2_in_place => {
                const in_place = self.kind == .add2_in_place;
                if (in_place) try mu.copy(&self.rig, d[2], d[0], n * 4);
                a.add(if (in_place) d[2] else d[0]);
                a.add(d[1]);
                a.add(d[2]);
                a.add(@as(i64, @intCast(n)));
                grid = @intCast(@min((n + 255) / 256, 1024));
            },
            .rope_full, .rope_swa => {
                a.add(d[0]);
                a.add(d[0] + n / 2 * 4);
                a.add(@as(c_int, @intCast(sh.rows)));
                a.add(self.th);
                a.add(@as(c_int, @intCast(sh.n)));
                grid = 256;
            },
            .argmax => {
                inline for (.{ d[0], d[1] }) |p| a.add(p);
                inline for (.{ sh.k, sh.rows, sh.n }) |v| a.add(@as(c_int, @intCast(v)));
            },
        }
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = grid, .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
        try s.synchronize();
    }

    pub fn count(self: *Glue) usize {
        const sh = self.shape;
        return if (self.kind == .argmax) sh.rows * 4 else sh.rows * sh.n;
    }

    pub fn reference(self: *Glue, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        const D = self.shape.n;
        switch (self.kind) {
            .embed => for (0..self.shape.rows) |r| {
                const at: usize = @intCast(self.ids[r]);
                @memcpy(want[r * D ..][0..D], self.x[at * D ..][0..D]);
            },
            .add2, .add2_in_place => for (want, self.x, self.y) |*w, p, q| {
                w.* = p + q;
            },
            .rope_full, .rope_swa => {
                const nh = want.len / 2;
                for (0..nh) |i| {
                    const ang = self.angle(i);
                    want[i] = @cos(ang);
                    want[nh + i] = @sin(ang);
                }
            },
            .argmax => return error.StatedRule, // gates 1 and 3 run check
        }
    }

    fn angle(self: *const Glue, i: usize) f64 {
        const half = self.shape.n / 2;
        const j: f64 = @floatFromInt(i % half);
        const p: f64 = @floatFromInt(i / half);
        return p * (1 / std.math.pow(f64, self.th, 2 * j / @as(f64, @floatFromInt(self.shape.n))));
    }

    /// The tables: libdevice's fp64 pow / cos / sin within a few double ulps of the angle, then the fp32 rounding.
    pub fn bounds(self: *Glue, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = mag;
        if (self.kind != .rope_full and self.kind != .rope_swa) return;
        const nh = want.len / 2;
        for (allow, want, 0..) |*l, w, i| l.* = mu.allow(.f32, w, 0x1p-50 * (@abs(self.angle(i % nh)) + 1));
    }

    pub fn check(self: *Glue, _: g.Gate) !?[]const u8 {
        const got = try self.rig.bytes(self.dev[1], self.shape.rows * 4);
        const cols = self.shape.k * self.shape.n;
        for (0..self.shape.rows) |r| {
            const row = self.x[r * cols ..][0..cols];
            var best: usize = 0;
            for (row, 0..) |v, i| {
                const b = row[best];
                if (std.math.isNan(b)) break;
                if (std.math.isNan(v) or v > b) best = i;
            }
            const id = std.mem.readInt(i32, got[r * 4 ..][0..4], .little);
            if (id != @as(i32, @intCast(best))) return try std.fmt.bufPrint(&self.msg, "row {d}: column {d} where the rule gives {d}", .{ r, id, best });
        }
        return null;
    }
};
