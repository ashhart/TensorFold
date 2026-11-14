//! The joint schema head's span kernels under the contract on HIP: span sums over piece or table rows, span means.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Fns = hip.joint_schema.Fns;

pub const Which = enum { sums, table_sums, means };

/// A span's rows in this piece, one chain from the sum it carries in.
fn chain(s: c.Shape) usize {
    return s.rows + 1;
}

fn one(_: c.Shape) usize {
    return 1;
}

/// rows: a piece's rows (the table's ids), n: spans, k: the hidden width; a 1,024-row piece of Clef-Flash, and less.
const served = [_]c.Shape{ .{ .rows = 1024, .n = 24, .k = 4096 }, .{ .rows = 64, .n = 9, .k = 1024 }, .{ .rows = 1, .n = 3, .k = 4096 } };
const dim_tiles = [3]c.Tile{ .{}, .{}, .{ .size = 256 } };
/// Where the piece starts in the prompt: spans begin before it, inside it and after it.
const first = 300;
const vocab = 97;

pub fn spec(which: Which) c.Spec {
    return switch (which) {
        .sums, .table_sums => .{
            .name = if (which == .sums) "hip joint schema tf_js_span_sums (piece rows)" else "hip joint schema tf_js_span_sums (table rows by id)",
            .accum = .f32,
            .depth = chain,
            .out = .f32,
            .in_place = true,
            .tiles = dim_tiles,
            .served = &served,
            .nan = .propagate,
        },
        .means => .{ .name = "hip joint schema tf_js_span_means", .accum = .none, .depth = one, .ops = 1, .out = .bf16, .tiles = dim_tiles, .served = &served },
    };
}

pub const Spans = struct {
    rig: rig_.Rig,
    which: Which,
    fns: Fns,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    rows: []u16 = &.{}, // piece rows, or the table
    ids: []i32 = &.{},
    spans: []i32 = &.{},
    start: []f32 = &.{}, // the sums carried in, or the means' sums
    d_rows: u64 = 0,
    d_ids: u64 = 0,
    d_spans: u64 = 0,
    out: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, fns: Fns, which: Which) Spans {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .which = which, .fns = fns, .arena = .init(a) };
    }

    pub fn deinit(self: *Spans) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Spans) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Spans);

    /// Span ends around the piece [first, first + rows): before, across its start, inside, across its end, after.
    fn spanOf(i: usize, rows: usize, rng: std.Random) [2]i32 {
        const r: i32 = @intCast(rows);
        const f: i32 = first;
        const lo: i32, const hi: i32 = switch (i % 5) {
            0 => .{ f - 40, f - 3 },
            1 => .{ f - 7, f + @min(r, 5) },
            2 => .{ f + @divFloor(r, 3), f + @divFloor(r, 3) + 1 + @as(i32, @intCast(rng.uintLessThan(u32, @intCast(@max(1, @divFloor(r, 2)))))) },
            3 => .{ f + r - @min(r, 4), f + r + 20 },
            else => .{ f + r, f + r + 9 },
        };
        return .{ lo, @max(hi, lo + 1) };
    }

    pub fn prepare(self: *Spans, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.spans = try a.alloc(i32, 2 * shape.n);
        for (0..shape.n) |i| {
            const sp = spanOf(i, shape.rows, rng);
            self.spans[2 * i] = sp[0];
            self.spans[2 * i + 1] = sp[1];
        }
        self.start = try a.alloc(f32, shape.n * shape.k);
        self.ids = try a.alloc(i32, first + shape.rows + 64);
        for (self.ids, 0..) |*v, p| v.* = @intCast(p % vocab);
        const table = self.which == .table_sums;
        self.rows = try a.alloc(u16, (if (table) vocab else shape.rows) * shape.k);
        for (self.rows) |*v| v.* = c.toBf16(switch (fill) {
            .stress => 0,
            else => rng.float(f32) * 8 - 4,
        });
        for (self.start, 0..) |*v, i| v.* = if (self.which == .means) @floatFromInt(@as(i32, @intCast(i % 41)) - 20) else if (fill == .stress) 0 else rng.float(f32) * 2 - 1;
        // gate 4's rows: 2048, a small integer, -2048, a small one: fp32 keeps the small ones, fp16 and bf16 lose them
        if (fill == .stress and self.which != .means) for (0..shape.rows) |i| {
            const r = self.rows[self.row(first + i) * shape.k ..][0..shape.k];
            for (r, 0..) |*v, d| {
                const big: f32 = if (d % 2 == 0) 2048 else -2048;
                v.* = c.toBf16(switch (i % 4) {
                    0 => big,
                    2 => -big,
                    else => @floatFromInt(1 + d % 7),
                });
            }
        };
        if (fill == .nonfinite) {
            const plant = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            if (self.which == .means) {
                for (plant, 0..) |p, i| self.start[i * 7 % self.start.len] = p;
            } else for (plant, 0..) |p, i| {
                // a piece row the second span covers: its first rows
                self.rows[self.row(first + i % shape.rows) * shape.k + i * 7 % shape.k] = c.toBf16(p);
            }
        }
        self.d_spans = try self.rig.input(std.mem.sliceAsBytes(self.spans));
        if (self.which == .means) {
            self.d_rows = try self.rig.input(std.mem.sliceAsBytes(self.start));
            self.out = try self.rig.output(shape.n * shape.k * 2);
            return true;
        }
        self.d_rows = try self.rig.input(std.mem.sliceAsBytes(self.rows));
        self.d_ids = if (table) try self.rig.input(std.mem.sliceAsBytes(self.ids)) else 0;
        self.out = try self.rig.output(self.start.len * 4);
        return true;
    }

    pub fn run(self: *Spans) !void {
        const s = self.shape;
        if (self.which == .means) {
            try self.fns.spanMeans(self.rig.s, self.d_rows, self.d_spans, s.n, s.k, self.out);
        } else {
            try self.rig.put(self.out, std.mem.sliceAsBytes(self.start));
            try self.fns.spanSums(self.rig.s, self.d_rows, self.d_ids, first, s.rows, s.k, self.d_spans, s.n, self.out);
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Spans) usize {
        return self.shape.n * self.shape.k;
    }

    /// The table row (or the piece row) prompt row p reads.
    fn row(self: *Spans, p: usize) usize {
        return if (self.which == .table_sums) @intCast(self.ids[p]) else p - first;
    }

    fn overlapOf(self: *Spans, i: usize) [2]usize {
        const lo: usize = @intCast(@max(self.spans[2 * i], first));
        const hi: usize = @intCast(@min(self.spans[2 * i + 1], first + @as(i32, @intCast(self.shape.rows))));
        return .{ lo, @max(lo, hi) };
    }

    pub fn reference(self: *Spans, want: []f64, mag: []f64) !void {
        const k = self.shape.k;
        for (0..self.shape.n) |i| for (0..k) |d| {
            const at = i * k + d;
            if (self.which == .means) {
                want[at] = self.start[at] / @as(f64, @floatFromInt(self.spans[2 * i + 1] - self.spans[2 * i]));
                mag[at] = 0;
                continue;
            }
            var s: f64 = self.start[at];
            var m: f64 = @abs(s);
            const o = self.overlapOf(i);
            for (o[0]..o[1]) |p| {
                const v = c.fromBf16(self.rows[self.row(p) * k + d]);
                s += v;
                m += @abs(v);
            }
            want[at] = s;
            mag[at] = m;
        };
    }

    /// The kernel's one chain a span, from the carried sum, in `acc` (every order is that chain).
    pub fn simulate(self: *Spans, acc: c.Accum, _: g.Order, out: []f64) !void {
        const k = self.shape.k;
        for (0..self.shape.n) |i| for (0..k) |d| {
            var s = rnd(acc, self.start[i * k + d]);
            const o = self.overlapOf(i);
            for (o[0]..o[1]) |p| s = rnd(acc, s + rnd(acc, c.fromBf16(self.rows[self.row(p) * k + d])));
            out[i * k + d] = s;
        };
    }
};

fn rnd(acc: c.Accum, v: f64) f64 {
    return switch (acc) {
        .f32 => @as(f32, @floatCast(v)),
        .f16 => @as(f16, @floatCast(v)),
        .bf16 => c.fromBf16(c.toBf16(@floatCast(v))),
        .f64, .none => v,
    };
}
