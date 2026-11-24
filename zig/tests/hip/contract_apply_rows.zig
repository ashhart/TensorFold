//! plan_rows.hip's two-launch values step under the contract: slots, poisoned values, tf_plan_apply_*'s bits.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");
const walk_rows = @import("contract_walk_rows.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const sums = contract.rows;

const page = 64;
const d = 256;
/// Keys a values tile sums (plan_apply_body's).
const tile = 128;
/// Spans the shared-row values kernels take from (launches/plan.zig's score_span).
const long_span = 16384;

/// Query heads over KV heads: Qwen3.6-35B-A3B's 16 over 2 and Qwen3.8-27B's 24 over 4, the score's cases.
pub const cases = walk_rows.cases;

/// served[0] takes gates 1-2 (slots of 3, 1 and 4 rows, a span with a part tile), gate 4 the 262,144-key span.
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

const kernel_names = [2][]const u8{ "tf_plan_rows_apply_f16", "tf_plan_rows_apply_bf16" };
const lone_names = [2][]const u8{ "tf_plan_lone_apply_f16", "tf_plan_lone_apply_bf16" };
const replaced_names = [2][]const u8{ "tf_plan_apply_f16", "tf_plan_apply_bf16" };

/// Each case's names [case][bf16], and the controls' [bf16].
const names = blk: {
    var t: [cases.len][2][]const u8 = undefined;
    for (cases, 0..) |x, i| for (0..2) |bf| {
        const what = "hip {s} then {s}, {d} heads over {d}: one-row slots alone, the others' rows on one read of their values";
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

/// Case `i`'s spec in kind `k`: the bits of the one-row kernel it replaces, edges at a chunk's rows and a tile's keys.
pub fn spec(i: usize, k: kit.Kind, fault: Fault) c.Spec {
    const bf = @intFromBool(k == .bf16);
    return .{
        .name = if (fault == .none) names[i][bf] else control_names[@intFromBool(fault == .alone)][bf],
        .accum = .f32,
        .depth = depth,
        .out = .f32,
        .replaces = replaced_names[bf],
        .tiles = .{ .{ .size = hip.launches.apply_rows }, .{}, .{ .size = long_span + tile, .min = long_span } },
        .served = &served[i],
    };
}

/// A tile's adds, after each term's subtract, exp (two roundings), divide and product.
fn depth(_: c.Shape) usize {
    return tile + 5;
}

/// Rows a slot holds: one each for an odd count from 5 (streams with no drafts), else cycling 3, 1, 4, 2 and 6.
fn counts(a: std.mem.Allocator, rows: usize) ![]usize {
    var out: std.ArrayList(usize) = .empty;
    const cycle: []const usize = if (rows % 2 == 1 and rows >= 5) &.{1} else &.{ 3, 1, 4, 2, 6 };
    var left = rows;
    while (left > 0) : (left -= out.items[out.items.len - 1]) try out.append(a, @min(left, cycle[out.items.len % cycle.len]));
    return out.toOwnedSlice(a);
}

/// Slot s's last row's visible keys: the span's end, mid-span, one past a tile's first key, or the span's start.
fn lastVisible(s: usize, rows: usize, k: usize) usize {
    const v = switch (s % 4) {
        0 => k - s / 4 * 7,
        1 => k / 2 + 5 + s / 4 * 64,
        2 => k / 4 / tile * tile + 1 + s / 4,
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

/// Gate 4's value: an integer in [1, 255] of either sign, times 16 for half of them, the same in every tile.
fn stressValue(k: kit.Kind, h: usize, key: usize, j: usize) u16 {
    const r = mix(@intCast((h * tile + key % tile) * d + j));
    const v: f64 = @floatFromInt((1 + r % 255) * @as(u32, if (r >> 9 & 1 == 1) 16 else 1));
    var b: [2]u8 = undefined;
    kit.put(k, &b, 0, if (r >> 31 == 1) -v else v); // eight significant bits: exact in both kinds
    return std.mem.readInt(u16, &b, .little);
}

/// shape.rows rows over slots of `counts` rows, each slot's own pages in one pool; every row's partials of every tile.
pub const RowsApply = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    case: usize,
    fault: Fault = .none,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    stress: bool = false,
    scores: []f32 = &.{}, // [rows][heads][span], NaN past a row's keys
    stats: []f32 = &.{}, // [rows][heads][2]: the max and the sum the weights divide by
    pool: []u16 = &.{}, // the values in the kind, [kv][pages][page][d]
    pages: usize = 0,
    /// Each slot's page table, and each row's slot and visible keys.
    tables: [][]u32 = &.{},
    slot: []usize = &.{},
    vis: []usize = &.{},
    plan: hip.launches.PlanRef = undefined,
    /// The control's plan: every slot's first 0 and its count every row.
    crossing: hip.launches.PlanRef = undefined,
    sc32: u64 = 0,
    st32: u64 = 0,
    out: u64 = 0,
    tree: u64 = 0, // the replaced kernel's partials

    /// Registration i of `count_all`: each case in fp16 then bf16 caches, then each control likewise.
    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, i: usize) RowsApply {
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

    pub fn deinit(self: *RowsApply) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    /// A control keeps its case's spec under its own name and must go red at gate 3.
    pub fn kernel(self: *RowsApply) g.Kernel {
        return .{ .spec = spec(self.case, self.k, self.fault), .ctx = self, .vt = &vt, .control = if (self.fault != .none) .shapes else null };
    }

    const vt = g.hooks(RowsApply);

    /// Element 0 of position `pos` of KV head `h` in slot `s`'s pages.
    fn at(self: *const RowsApply, s: usize, h: usize, pos: usize) usize {
        return ((h * self.pages + self.tables[s][pos / page]) * page + pos % page) * d;
    }

    fn tiles(self: *const RowsApply) usize {
        return (self.shape.k + tile - 1) / tile;
    }

    pub fn prepare(self: *RowsApply, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
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
                x.* = if (self.stress) stressValue(self.k, h, pos, j) else finite(self.k, mix(hk ^ @as(u32, @intCast(j)) *% 0x9e3779b1));
            }
            // one key in 41 holds a NaN or an infinity at one element, seen by some rows and past others
            if (!self.stress and hk % 41 == 0) self.pool[row_at + hk / 41 % d] = nonFinite(self.k, mix(hk));
        };
        if (!self.stress) for (per, need, 0..) |n, pages, s| {
            const last = lastVisible(s, n, keys);
            // the value only a slot's last row weighs, and the one past all its rows, both non-finite
            for ([_]usize{ last - 1, last }) |pos| if (pos < pages * page) for (0..kv) |h| {
                self.pool[self.at(s, h, pos) + (pos + h) % d] = nonFinite(self.k, mix(@intCast(pos + h)));
            };
        };
        try self.weights(rng, fill);
        self.sc32 = try self.rig.input(std.mem.sliceAsBytes(self.scores));
        self.st32 = try self.rig.input(std.mem.sliceAsBytes(self.stats));
        const pool = try self.rig.input(std.mem.sliceAsBytes(self.pool));
        // each slot's descriptor [cached, kept, k pool, v pool], then its page table
        const words = 4;
        const descs = try a.alloc(u64, slots);
        for (self.tables, descs) |t, *dev| {
            const desc = try a.alloc(u64, words + (t.len + 1) / 2);
            @memset(desc, 0);
            desc[3] = pool;
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

    /// The fp32 scores and stats: weights in (e^-4 / 64, 1], gate 4's 1, 1/2 or 1/4, gate 1's NaN and infinities.
    fn weights(self: *RowsApply, rng: std.Random, fill: g.Fill) !void {
        const s = self.shape;
        const a = self.arena.allocator();
        self.scores = try a.alloc(f32, s.rows * s.n * s.k);
        self.stats = try a.alloc(f32, s.rows * s.n * 2);
        for (0..s.rows) |r| for (0..s.n) |h| {
            const i = r * s.n + h;
            const top: f32 = if (self.stress) 0.5 else @floatCast(rng.float(f64) * 4 - 2);
            self.stats[2 * i] = top;
            self.stats[2 * i + 1] = if (self.stress) @floatFromInt(@as(u32, 1) << @intCast(h % 3)) else @floatCast(1 + rng.float(f64) * 63);
            for (self.scores[i * s.k ..][0..s.k], 0..) |*x, key| {
                x.* = if (key >= self.vis[r]) std.math.nan(f32) else if (self.stress) top else @floatCast(@as(f64, top) - rng.float(f64) * 4);
            }
        };
        if (self.stress) return;
        // a NaN, an infinite and a negatively infinite score at keys some rows of a chunk weigh, and a NaN max
        for ([_][3]usize{ .{ 0, 0, 5 }, .{ 1, 1, 200 }, .{ 2, 2, 300 } }, [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |x, v| {
            if (x[0] < s.rows and x[2] < self.vis[x[0]]) self.scores[(x[0] * s.n + x[1]) * s.k + x[2]] = v;
        }
        if (fill == .nonfinite and s.rows > 3) self.stats[2 * (3 * s.n + 3)] = std.math.nan(f32);
    }

    pub fn run(self: *RowsApply) !void {
        const s = self.shape;
        if (self.fault == .alone) return self.chunksAlone();
        const p = if (self.fault == .crossing) self.crossing else self.plan;
        try self.l.planApply(self.sc32, self.st32, self.out, p, 0, s.n, cases[self.case].kv, d, s.k, @backingInt(self.k), false, .shared, self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    /// The control's launch: planApply's tf_plan_rows_apply_* on its grid, without tf_plan_lone_apply_*.
    fn chunksAlone(self: *RowsApply) !void {
        const s = self.shape;
        const kv = cases[self.case].kv;
        var a: hip.launch.Args = .{};
        try a.add(self.sc32);
        try a.add(self.st32);
        try a.add(self.out);
        for ([_]usize{ s.k, s.n, kv, d }) |x| try a.add(@as(c_int, @intCast(x)));
        try a.add(self.plan.args);
        try a.add(@as(c_int, 0));
        const grid: hip.launch.Dim3 = .{ .x = @intCast(self.tiles()), .y = @intCast(kv), .z = @intCast(s.rows) };
        const f = self.l.plan.apply_rows[@intCast(@backingInt(self.k) - 1)];
        try hip.launch.launch(f, .{ .grid = grid, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    /// tf_plan_apply_*'s bytes on the same plan: a block a row.
    pub fn replaced(self: *RowsApply) ![]const u8 {
        const s = self.shape;
        try self.l.planApply(self.sc32, self.st32, self.tree, self.plan, 0, s.n, cases[self.case].kv, d, s.k, @backingInt(self.k), false, .one, self.rig.s.handle);
        return self.rig.bytes(self.tree, self.count() * 4);
    }

    pub fn count(self: *RowsApply) usize {
        return self.shape.rows * self.shape.n * self.tiles() * d;
    }

    /// Each tile's sum of exp(score - max) / sum times the values over the keys a row sees, 0 past them, in float64.
    pub fn reference(self: *RowsApply, want: []f64, mag: []f64) !void {
        try self.work(want, mag, null);
    }

    /// Gate 4's partials with each tile's products summed in the accumulator and order.
    pub fn simulate(self: *RowsApply, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.work(out, null, .{ acc, order });
    }

    /// Element j of key `pos` of KV head `h` in slot `s`'s pages, as float64.
    fn value(self: *const RowsApply, s: usize, h: usize, pos: usize, j: usize) f64 {
        const bits = self.pool[self.at(s, h, pos) + j];
        return if (self.k == .bf16) @as(f64, c.fromBf16(bits)) else @as(f64, @as(f16, @bitCast(bits)));
    }

    /// Every row's partials, in float64 or (`sim`) as an accumulator sums a tile; gate 4's whole tiles are worked once.
    fn work(self: *RowsApply, out: []f64, mag: ?[]f64, sim: ?struct { c.Accum, g.Order }) !void {
        const s = self.shape;
        const a = self.rig.a;
        const kv = cases[self.case].kv;
        const group = s.n / kv;
        const n = self.tiles();
        @memset(out, 0);
        if (mag) |m| @memset(m, 0);
        const vals = try a.alloc(f64, tile * d); // a tile's values, [key][j]
        defer a.free(vals);
        const w = try a.alloc(f64, tile);
        defer a.free(w);
        const terms = try a.alloc(f64, tile);
        defer a.free(terms);
        var r0: usize = 0;
        while (r0 < s.rows) {
            var r1 = r0 + 1;
            while (r1 < s.rows and self.slot[r1] == self.slot[r0]) r1 += 1;
            var most: usize = 0;
            for (self.vis[r0..r1]) |v| most = @max(most, v);
            const sl = self.slot[r0];
            for (0..kv) |hk| {
                // gate 4: the first whole tile's partials of each row and head, copied to its other whole tiles
                var whole: ?usize = null;
                for (0..n) |t| {
                    const key0 = t * tile;
                    if (key0 >= most) break;
                    const top = @min(key0 + tile, most);
                    if (!self.stress or t == 0 or top - key0 < tile or whole == null) {
                        for (key0..top) |pos| for (0..d) |j| {
                            vals[(pos - key0) * d + j] = self.value(sl, hk, if (self.stress) pos % tile else pos, j);
                        };
                    }
                    for (r0..r1) |r| for (hk * group..(hk + 1) * group) |h| {
                        const stop = @min(key0 + tile, self.vis[r]);
                        if (stop <= key0) continue;
                        const base = ((r * s.n + h) * n + t) * d;
                        if (self.stress and stop - key0 == tile) {
                            if (whole) |from| {
                                const src = ((r * s.n + h) * n + from) * d;
                                @memcpy(out[base..][0..d], out[src..][0..d]);
                                if (mag) |m| @memcpy(m[base..][0..d], m[src..][0..d]);
                                continue;
                            }
                        }
                        const i = r * s.n + h;
                        const top_s: f64 = self.stats[2 * i];
                        const sum: f64 = self.stats[2 * i + 1];
                        for (w[0 .. stop - key0], key0..) |*x, pos| x.* = @exp(@as(f64, self.scores[i * s.k + pos]) - top_s) / sum;
                        for (0..d) |j| {
                            var acc: f64 = 0;
                            var abs: f64 = 0;
                            for (terms[0 .. stop - key0], w[0 .. stop - key0], 0..) |*x, wk, e| {
                                x.* = wk * vals[e * d + j];
                                acc += x.*;
                                abs += @abs(x.*);
                            }
                            if (sim) |how| acc = sums.sumIn(how[0], how[1], terms[0 .. stop - key0]);
                            out[base + j] = acc;
                            if (mag) |m| m[base + j] = abs;
                        }
                    };
                    if (self.stress and top - key0 == tile and whole == null) whole = t;
                }
            }
            r0 = r1;
        }
    }
};
