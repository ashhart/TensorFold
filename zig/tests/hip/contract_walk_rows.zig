//! plan_rows.hip's two-launch score step under the contract: slots, poisoned caches, tf_plan_score256_*'s bits.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const sums = contract.rows;

const page = 64;
const d = 256;
/// Spans the d = 256 score kernels take from (launches/plan.zig's score_span).
const long_span = 16384;
/// Gate 4's keys repeat every `period` positions, so its float64 sums are worked once a period.
const period = 61;

/// Query heads over KV heads: Qwen3.6-35B-A3B's 16 over 2 and Qwen3.8-27B's 24 over 4.
pub const Case = struct { heads: usize, kv: usize };
pub const cases = [_]Case{ .{ .heads = 16, .kv = 2 }, .{ .heads = 24, .kv = 4 } };

/// served[0] takes gates 1-2 (slots of 3, 1 and 4 rows, a span with a scalar tail), gate 4 the 262,144-key span.
const served = blk: {
    var t: [cases.len][7]c.Shape = undefined;
    for (cases, 0..) |x, i| t[i] = .{
        .{ .rows = 8, .n = x.heads, .k = 16387 },
        .{ .rows = 16, .n = x.heads, .k = 65536 },
        .{ .rows = 4, .n = x.heads, .k = 131072 },
        .{ .rows = 2, .n = x.heads, .k = 262144 },
        .{ .rows = 64, .n = x.heads, .k = 16384 },
        .{ .rows = 5, .n = x.heads, .k = 65536 },
        .{ .rows = 7, .n = x.heads, .k = 131072 },
    };
    break :blk t;
};

/// The real launches, or a control's: chunks led across slots (first 0, count every row), or the chunks' launch alone.
pub const Fault = enum { none, crossing, alone };

/// The registrations: each case in fp16 and bf16 caches, then each control in each.
pub const count_all = 2 * cases.len + 4;

const kernel_names = [2][]const u8{ "tf_plan_rows_score256_f16", "tf_plan_rows_score256_bf16" };
const lone_names = [2][]const u8{ "tf_plan_lone_score256_f16", "tf_plan_lone_score256_bf16" };
const replaced_names = [2][]const u8{ "tf_plan_score256_f16", "tf_plan_score256_bf16" };

/// Each case's names [case][bf16], and the controls' [bf16].
const names = blk: {
    var t: [cases.len][2][]const u8 = undefined;
    for (cases, 0..) |x, i| for (0..2) |bf| {
        const what = "hip {s} then {s}, {d} heads over {d}: one-row slots alone, the others' rows on one read of their keys";
        t[i][bf] = std.fmt.comptimePrint(what, .{ lone_names[bf], kernel_names[bf], x.heads, x.kv });
    };
    break :blk t;
};
const control_names = blk: {
    var t: [2][2][]const u8 = undefined;
    for (0..2) |bf| {
        t[0][bf] = "control: hip " ++ kernel_names[bf] ++ ", chunks led across slots (first 0, count every row)";
        t[1][bf] = "control: hip " ++ kernel_names[bf] ++ " alone, the one-row slots' launch left out";
    }
    break :blk t;
};

/// Case `i`'s spec in kind `k`: the bits of the one-row kernel it replaces, edges at a chunk's rows and a block's keys.
pub fn spec(i: usize, k: kit.Kind, fault: Fault) c.Spec {
    const bf = @intFromBool(k == .bf16);
    return .{
        .name = if (fault == .none) names[i][bf] else control_names[@intFromBool(fault == .alone)][bf],
        .accum = .f32,
        .depth = depth,
        .ops = 1, // the scale
        .out = .f32,
        .replaces = replaced_names[bf],
        .tiles = .{ .{ .size = hip.launches.score_rows }, .{}, .{ .size = long_span + 128, .min = long_span } },
        .served = &served[i],
    };
}

/// A lane's eight products and sums, the five folds of the wave.
fn depth(_: c.Shape) usize {
    return 2 * 8 + 5;
}

/// Rows a slot holds: one each for an odd count from 5 (streams with no drafts), else cycling 3, 1, 4, 2.
fn counts(a: std.mem.Allocator, rows: usize) ![]usize {
    var out: std.ArrayList(usize) = .empty;
    const cycle: []const usize = if (rows % 2 == 1 and rows >= 5) &.{1} else &.{ 3, 1, 4, 2 };
    var left = rows;
    while (left > 0) : (left -= out.items[out.items.len - 1]) try out.append(a, @min(left, cycle[out.items.len % cycle.len]));
    return out.toOwnedSlice(a);
}

/// Slot s's last row's visible keys: the span's end, mid-span, one past a block's first key, or the span's start.
fn lastVisible(s: usize, rows: usize, k: usize) usize {
    const v = switch (s % 4) {
        0 => k - s / 4 * 7,
        1 => k / 2 + 5 + s / 4 * 64,
        2 => k / 4 / 128 * 128 + 1 + s / 4,
        else => rows + s / 4,
    };
    return std.math.clamp(v, rows, k);
}

fn mix(x: u32) u32 {
    var h = x;
    h ^= h >> 16;
    h *%= 0x7feb352d;
    h ^= h >> 15;
    h *%= 0x846ca68b;
    h ^= h >> 16;
    return h;
}

/// A finite value of the kind, any sign and mantissa, magnitude in [2^-4, 2^2), as its 16 bits.
fn finite(k: kit.Kind, h: u32) u16 {
    const sign: u16 = @intCast(h >> 31);
    return if (k == .bf16)
        sign << 15 | @as(u16, @intCast(123 + (h >> 8) % 6)) << 7 | @as(u16, @intCast(h & 0x7f))
    else
        sign << 15 | @as(u16, @intCast(11 + (h >> 10) % 6)) << 10 | @as(u16, @intCast(h & 0x3ff));
}

/// A NaN of payload h (quiet or signaling) or an infinity of either sign, as the kind's 16 bits.
fn nonFinite(k: kit.Kind, h: u32) u16 {
    const top: u16 = if (k == .bf16) 0x7f80 else 0x7c00;
    const mant: u32 = if (k == .bf16) 0x7f else 0x3ff;
    return switch (h % 3) {
        0 => @as(u16, @intCast((h >> 31) << 15)) | top | @as(u16, @intCast(1 + (h >> 8) % mant)),
        1 => top,
        else => 0x8000 | top,
    };
}

/// Gate 4's key element: integers in [1, 255] of either sign, the second half negated but for one, repeating.
fn stressKey(k: kit.Kind, h: usize, pos: usize, j: usize) u16 {
    const r = mix(@intCast((h * period + pos % period) * 128 + j % 128));
    const v: f64 = @floatFromInt(1 + r % 255);
    const x = if (r >> 31 == 1) -v else v;
    var b: [2]u8 = undefined;
    kit.put(k, &b, 0, if (j < 128) x else if (j == 128) 1 - x else -x); // integers of 256 or less are exact in both
    return std.mem.readInt(u16, &b, .little);
}

/// Gate 4's query element: integers in [1, 63], the second half equal to the first, so a dot is its first query.
fn stressQuery(i: usize) f64 {
    return @floatFromInt(1 + mix(@intCast(i / d * 128 + i % d % 128)) % 63);
}

fn scale() f32 {
    return 1 / @sqrt(@as(f32, d));
}

/// shape.rows rows over slots of `counts` rows, each slot's own pages in one pool; the scores of every row and head.
pub const RowsScore = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    case: usize,
    fault: Fault = .none,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    stress: bool = false,
    q: []f64 = &.{}, // [rows][heads][d]
    pool: []u16 = &.{}, // the keys in the kind, [kv][pages][page][d]
    pages: usize = 0,
    /// Each slot's page table, and each row's slot and visible keys.
    tables: [][]u32 = &.{},
    slot: []usize = &.{},
    vis: []usize = &.{},
    plan: hip.launches.PlanRef = undefined,
    /// The control's plan: every slot's first 0 and its count every row.
    crossing: hip.launches.PlanRef = undefined,
    q32: u64 = 0,
    out: u64 = 0,
    tree: u64 = 0, // the replaced kernel's scores

    /// Registration i of `count_all`: each case in fp16 then bf16 caches, then each control likewise.
    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, i: usize) RowsScore {
        const control = i >= 2 * cases.len;
        return .{
            .rig = .{ .a = a, .r = rt, .s = s },
            .l = l,
            .k = if (i % 2 == 1) .bf16 else .f16,
            .case = if (control) 0 else i / 2,
            .fault = if (!control) .none else if (i < 2 * cases.len + 2) .crossing else .alone,
            .arena = .init(a),
        };
    }

    pub fn deinit(self: *RowsScore) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    /// A control keeps its case's spec under its own name and must go red at gate 3.
    pub fn kernel(self: *RowsScore) g.Kernel {
        return .{ .spec = spec(self.case, self.k, self.fault), .ctx = self, .vt = &vt, .control = if (self.fault != .none) .shapes else null };
    }

    const vt = g.hooks(RowsScore);

    /// Element 0 of position `pos` of KV head `h` in slot `s`'s pages.
    fn at(self: *const RowsScore, s: usize, h: usize, pos: usize) usize {
        return ((h * self.pages + self.tables[s][pos / page]) * page + pos % page) * d;
    }

    pub fn prepare(self: *RowsScore, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const kv = cases[self.case].kv;
        if (shape.rows == 0 or shape.k < long_span or shape.n % kv != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.free_all);
        self.shape = shape;
        self.stress = fill == .stress;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const keys = shape.k;
        const per = try counts(a, shape.rows);
        const slots = per.len;
        // each slot's rows at consecutive positions up to its last row's, and its pages, permuted over one pool
        self.slot = try a.alloc(usize, shape.rows);
        self.vis = try a.alloc(usize, shape.rows);
        const first = try a.alloc(i32, slots);
        const many = try a.alloc(i32, slots);
        const need = try a.alloc(usize, slots);
        var row: usize = 0;
        self.pages = 0;
        for (per, 0..) |n, s| {
            const last = lastVisible(s, n, keys);
            first[s] = @intCast(row);
            many[s] = @intCast(n);
            for (0..n) |i| {
                self.slot[row + i] = s;
                self.vis[row + i] = last - (n - 1) + i;
            }
            row += n;
            need[s] = (last + page - 1) / page;
            self.pages += need[s];
        }
        self.tables = try a.alloc([]u32, slots);
        var next: usize = 0;
        for (self.tables, need) |*t, n| {
            t.* = try a.alloc(u32, n);
            for (t.*, next..) |*p, i| p.* = @intCast(if (self.pages % 5 == 0) self.pages - 1 - i else (i * 5 + 3) % self.pages);
            next += n;
        }
        self.pool = try a.alloc(u16, kv * self.pages * page * d);
        const salt: u32 = @truncate(seed);
        for (self.tables, 0..) |t, s| for (0..kv) |h| for (0..t.len * page) |pos| {
            const row_at = self.at(s, h, pos);
            const hk = mix(@as(u32, @truncate((s * kv + h) * keys + pos)) *% 0x27d4eb2f +% salt);
            for (self.pool[row_at..][0..d], 0..) |*x, j| {
                x.* = if (self.stress) stressKey(self.k, h, pos, j) else finite(self.k, mix(hk ^ @as(u32, @intCast(j)) *% 0x9e3779b1));
            }
            // one key in 41 holds a NaN or an infinity at one element, seen by some rows and past others
            if (!self.stress and hk % 41 == 0) self.pool[row_at + hk / 41 % d] = nonFinite(self.k, mix(hk));
        };
        if (!self.stress) for (per, need, 0..) |n, pages, s| {
            const last = lastVisible(s, n, keys);
            // the key only a slot's last row sees, and the one past all its rows, both non-finite
            for ([_]usize{ last - 1, last }) |pos| if (pos < pages * page) for (0..kv) |h| {
                self.pool[self.at(s, h, pos) + (pos + h) % d] = nonFinite(self.k, mix(@intCast(pos + h)));
            };
        };
        // queries: fp32 values in [-1, 1) away from zero, gate 4's integers, or one NaN at gate 1
        self.q = try a.alloc(f64, shape.rows * shape.n * d);
        for (self.q, 0..) |*v, i| {
            const x = rng.float(f64) * 2 - 1;
            v.* = if (self.stress) stressQuery(i) else @as(f32, @floatCast(if (@abs(x) < 0x1p-8) 0.5 else x));
        }
        if (fill == .nonfinite) self.q[0] = std.math.nan(f64); // row 0, head 0
        const qb = try kit.f32Bytes(a, self.q);
        // quiet NaNs of distinct payloads in row 1's head 1
        if (!self.stress and shape.rows > 1) for (0..4) |e| {
            const i = (shape.n + 1) * d + e * 37;
            std.mem.writeInt(u32, qb[4 * i ..][0..4], 0x7fc00000 | @as(u32, @intCast(e * 0x1235 + 1)), .little);
            self.q[i] = std.math.nan(f64);
        };
        self.q32 = try self.rig.input(qb);
        const pool = try self.rig.input(std.mem.sliceAsBytes(self.pool));
        // each slot's descriptor [cached, kept, k pool, v pool], then its page table
        const words = 4;
        const descs = try a.alloc(u64, slots);
        for (self.tables, descs) |t, *dev| {
            const desc = try a.alloc(u64, words + (t.len + 1) / 2);
            @memset(desc, 0);
            desc[2] = pool;
            @memcpy(std.mem.sliceAsBytes(desc[words..])[0 .. t.len * 4], std.mem.sliceAsBytes(t));
            dev.* = try self.rig.input(std.mem.sliceAsBytes(desc));
        }
        const pos = try a.alloc(i32, shape.rows);
        const slot = try a.alloc(i32, shape.rows);
        for (pos, slot, self.vis, self.slot) |*p, *s, v, x| {
            p.* = @intCast(v - 1);
            s.* = @intCast(x);
        }
        const args: hip.launches.PlanArgs = .{
            .pos = try self.rig.input(std.mem.sliceAsBytes(pos)),
            .slot = try self.rig.input(std.mem.sliceAsBytes(slot)),
            .first = try self.rig.input(std.mem.sliceAsBytes(first)),
            .count = try self.rig.input(std.mem.sliceAsBytes(many)),
            .desc = try self.rig.input(std.mem.sliceAsBytes(descs)),
            .pages = words,
            .pool = @intCast(self.pages),
        };
        self.plan = .{ .rows = shape.rows, .slots = slots, .args = args };
        self.crossing = self.plan;
        if (self.fault == .crossing) {
            const zero = try a.alloc(i32, slots);
            @memset(zero, 0);
            const all = try a.alloc(i32, slots);
            @memset(all, @intCast(shape.rows));
            self.crossing.args.first = try self.rig.input(std.mem.sliceAsBytes(zero));
            self.crossing.args.count = try self.rig.input(std.mem.sliceAsBytes(all));
        }
        self.out = try self.rig.output(self.count() * 4);
        self.tree = try self.rig.scratch(self.count() * 4);
        return true;
    }

    pub fn run(self: *RowsScore) !void {
        const s = self.shape;
        if (self.fault == .alone) return self.chunksAlone();
        const p = if (self.fault == .crossing) self.crossing else self.plan;
        try self.l.planScore(self.q32, self.out, p, 0, s.n, cases[self.case].kv, d, s.k, scale(), @backingInt(self.k), false, .shared, self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    /// The control's launch: planScore's tf_plan_rows_score256_* on its grid and LDS, without tf_plan_lone_score256_*.
    fn chunksAlone(self: *RowsScore) !void {
        const s = self.shape;
        const kv = cases[self.case].kv;
        var a: hip.launch.Args = .{};
        try a.add(self.q32);
        try a.add(self.out);
        for ([_]usize{ s.k, s.n, kv, d }) |x| try a.add(@as(c_int, @intCast(x)));
        try a.add(scale());
        try a.add(self.plan.args);
        try a.add(@as(c_int, 0));
        const grid: hip.launch.Dim3 = .{ .x = @intCast((s.k + 127) / 128), .y = @intCast(kv), .z = @intCast(s.rows) };
        const lds: u32 = @intCast(hip.launches.score_rows * s.n / kv * d * 4);
        const f = self.l.plan.score256_rows[@intCast(@backingInt(self.k) - 1)];
        try hip.launch.launch(f, .{ .grid = grid, .block = .{ .x = 128 }, .shared = lds }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    /// tf_plan_score256_*'s bytes on the same plan: a block a row.
    pub fn replaced(self: *RowsScore) ![]const u8 {
        const s = self.shape;
        try self.l.planScore(self.q32, self.tree, self.plan, 0, s.n, cases[self.case].kv, d, s.k, scale(), @backingInt(self.k), false, .one, self.rig.s.handle);
        return self.rig.bytes(self.tree, self.count() * 4);
    }

    pub fn count(self: *RowsScore) usize {
        return self.shape.rows * self.shape.n * self.shape.k;
    }

    /// q k * scale in float64 for the keys a row sees, -inf past them; each score's magnitude is its terms' sum.
    pub fn reference(self: *RowsScore, want: []f64, mag: []f64) !void {
        try self.work(want, mag, null);
    }

    /// Gate 4's scores with each dot's products summed in the accumulator and order, then scaled.
    pub fn simulate(self: *RowsScore, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.work(out, null, .{ acc, order });
    }

    /// Every row's scores, each dot in float64 or (`sim`) as an accumulator sums it; gate 4's once a period.
    fn work(self: *RowsScore, out: []f64, mag: ?[]f64, sim: ?struct { c.Accum, g.Order }) !void {
        const s = self.shape;
        const sc: f64 = scale();
        const kv = cases[self.case].kv;
        const heads = s.n / kv;
        for (0..s.rows) |r| for (0..s.n) |h| {
            const base = (r * s.n + h) * s.k;
            @memset(out[base + self.vis[r] .. base + s.k], -std.math.inf(f64));
            if (mag) |m| @memset(m[base..][0..s.k], 0);
        };
        var krow: [d]f64 = undefined;
        var terms: [d]f64 = undefined;
        var r0: usize = 0;
        while (r0 < s.rows) {
            var r1 = r0 + 1;
            while (r1 < s.rows and self.slot[r1] == self.slot[r0]) r1 += 1;
            var most: usize = 0;
            for (self.vis[r0..r1]) |v| most = @max(most, v);
            const reach = if (self.stress) @min(most, period) else most;
            for (0..kv) |hk| for (0..reach) |pos| {
                const row_at = self.at(self.slot[r0], hk, pos);
                for (&krow, self.pool[row_at..][0..d]) |*x, bits| x.* = kit.get(self.k, std.mem.asBytes(&bits), 0);
                for (r0..r1) |r| if (pos < self.vis[r]) for (hk * heads..(hk + 1) * heads) |h| {
                    var dot: f64 = 0;
                    var abs: f64 = 0;
                    for (self.q[(r * s.n + h) * d ..][0..d], krow, &terms) |x, y, *t| {
                        t.* = x * y;
                        dot += t.*;
                        abs += @abs(t.*);
                    }
                    if (sim) |how| dot = sums.sumIn(how[0], how[1], &terms);
                    const i = (r * s.n + h) * s.k + pos;
                    out[i] = dot * sc;
                    if (mag) |m| m[i] = abs * sc;
                };
            };
            if (self.stress) for (r0..r1) |r| for (0..s.n) |h| {
                const base = (r * s.n + h) * s.k;
                for (@min(period, self.vis[r])..self.vis[r]) |pos| {
                    out[base + pos] = out[base + pos - period];
                    if (mag) |m| m[base + pos] = m[base + pos - period];
                }
            };
            r0 = r1;
        }
    }
};
