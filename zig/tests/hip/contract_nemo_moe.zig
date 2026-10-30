//! Nemotron-H's MoE kernels under the contract on HIP: both picks, relu^2, local ids, the sum.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

pub const Which = enum { pass, rank, relu2, localize, combine, combine_f32 };

fn none(_: c.Shape) usize {
    return 0;
}

/// Each slot's product and add in order, then the shared row's add.
fn slotChain(s: c.Shape) usize {
    return 2 * s.k + 1;
}

const scaling: f32 = 2.5;
const odd = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
const sum_names = [3][]const u8{
    "hip nemotron tf_nemo_combine (fp16)",
    "hip nemotron tf_nemo_combine (bf16)",
    "hip nemotron tf_nemo_combine (fp32)",
};
const pass_name = "hip nemotron tf_nemo_select (a pick at a time)";
const pick_rule = "the top-k of sigmoid(logit) + bias, a NaN never ahead of a number, ties to the lower expert; " ++
    "weights within (k + 8) fp32 ulps of float64";
const ids_rule = "pick - first for the rank's own experts, else own";

/// Nano's 128 experts and top 6, a decode row to a 2048-row prompt chunk; its 1,856-wide experts and 2,688 hidden.
pub fn spec(which: Which, k: kit.Kind) c.Spec {
    const picks = &[_]c.Shape{
        .{ .rows = 1, .n = 128, .k = 6 },
        .{ .rows = 16, .n = 128, .k = 6 },
        .{ .rows = 2048, .n = 128, .k = 6 },
    };
    const sums = &[_]c.Shape{
        .{ .rows = 1, .n = 2688, .k = 6 },
        .{ .rows = 16, .n = 2688, .k = 6 },
        .{ .rows = 2048, .n = 2688, .k = 6 },
    };
    return switch (which) {
        .pass, .rank => .{
            .name = if (which == .pass) pass_name else "hip nemotron tf_nemo_select_rank",
            .accum = .none,
            .depth = none,
            .out = .raw,
            .replaces = if (which == .rank) "tf_nemo_select" else null,
            .nan = .{ .stated = pick_rule },
            .tiles = .{ .{}, .{ .size = 32, .min = 8 }, .{} },
            .served = picks,
        },
        .relu2 => .{
            .name = if (k == .bf16) "hip nemotron tf_nemo_relu2 (bf16)" else "hip nemotron tf_nemo_relu2 (fp16)",
            .accum = .none,
            .depth = none,
            .ops = 1,
            // up rounded to the kind, then squared
            .mids = 2,
            .out = kit.elem(k),
            .tiles = .{ .{}, .{ .size = 256 }, .{} },
            .served = &.{
                .{ .rows = 6, .n = 1856, .k = 0 },
                .{ .rows = 1, .n = 3712, .k = 0 },
                .{ .rows = 96, .n = 1856, .k = 0 },
            },
        },
        .localize => .{
            .name = "hip nemotron tf_nemo_localize",
            .accum = .none,
            .depth = none,
            .out = .raw,
            .nan = .{ .stated = ids_rule },
            .tiles = .{ .{}, .{ .size = 256 }, .{} },
            .served = &.{
                .{ .rows = 1, .n = 6, .k = 64 },
                .{ .rows = 1, .n = 96, .k = 32 },
                .{ .rows = 1, .n = 12288, .k = 64 },
            },
        },
        .combine, .combine_f32 => .{
            .name = sum_names[if (which == .combine_f32) 2 else @intFromBool(k == .bf16)],
            .accum = .f32,
            .depth = slotChain,
            .out = if (which == .combine_f32) .f32 else kit.elem(k),
            .tiles = .{ .{}, .{ .size = 256 }, .{} },
            .served = sums,
        },
    };
}

/// The pick, relu^2, local ids and the sum; the picks and the ids check their stated rule themselves.
pub const Moe = struct {
    rig: rig_.Rig,
    which: Which,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []f64 = &.{}, // logits [rows][experts], up values, picks, or y [rows][slots][d]
    w: []f64 = &.{}, // the bias [experts], or the weights [rows][slots]
    shared: []f64 = &.{},
    first: c_int = 0,
    in: [3]u64 = @splat(0),
    out: [4]u64 = @splat(0),
    base: [4]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, w: Which) Moe {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .which = w, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Moe) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Moe) g.Kernel {
        const own = self.pick() or self.which == .localize;
        return .{ .spec = spec(self.which, self.k), .ctx = self, .vt = if (own) &vt else &vt_float };
    }

    const vt = g.hooks(Moe);
    /// relu^2 and the sum meet float64 through the harness
    const vt_float = blk: {
        var v = vt;
        v.check = null;
        break :blk v;
    };

    fn pick(self: *Moe) bool {
        return self.which == .pass or self.which == .rank;
    }

    /// The plan's items and members: one row's, so only for one row.
    fn planned(self: *Moe) bool {
        return self.pick() and self.shape.rows == 1;
    }

    pub fn prepare(self: *Moe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        if (self.pick() and (shape.n > 1024 or shape.k > 32 or shape.k > shape.n)) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.in = @splat(0);
        self.out = @splat(0);
        self.base = @splat(0);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * shape.n;
        switch (self.which) {
            .pass, .rank => {
                // sixteenths and sixty-fourths, so equal scores happen
                self.x = try a.alloc(f64, n);
                for (self.x) |*v| v.* = @as(f64, @floatFromInt(rng.uintLessThan(u32, 64))) / 16 - 2;
                self.w = try a.alloc(f64, shape.n);
                for (self.w) |*v| v.* = @as(f64, @floatFromInt(rng.uintLessThan(u32, 8))) / 64;
                if (fill == .nonfinite) for (odd, 0..) |v, i| {
                    if (i < shape.rows) self.x[i * shape.n + 2 * i + 1] = v;
                };
                self.in[0] = try self.rig.input(try kit.f32Bytes(a, self.x));
                self.in[1] = try self.rig.input(try kit.f32Bytes(a, self.w));
                const slots = shape.rows * shape.k * 4;
                for ([_]usize{ slots, slots, 3 * 32 * 4, 32 * 4 }, 0..) |len, i| {
                    if (i >= 2 and !self.planned()) break;
                    self.out[i] = try self.rig.output(len);
                    if (self.which != .rank) continue;
                    self.base[i] = try self.rig.scratch(len);
                    const poison = try a.alloc(u8, len);
                    @memset(poison, 0xff); // as the outputs start, so slots neither writes compare equal
                    try self.rig.put(self.base[i], poison);
                }
            },
            .relu2 => {
                self.x = try a.alloc(f64, n);
                for (self.x) |*v| {
                    const m = 0.25 + rng.float(f64) * 7.75;
                    v.* = @as(f32, @floatCast(if (rng.boolean()) m else -m));
                }
                if (fill == .nonfinite) for (odd, 0..) |v, i| {
                    self.x[(i * 37) % n] = v;
                };
                self.in[0] = try self.rig.input(try kit.f32Bytes(a, self.x));
                self.out[0] = try self.rig.output(n * kit.size(self.k));
            },
            .localize => {
                const own = shape.k;
                self.first = @intCast(own * rng.uintLessThan(usize, 2));
                const ids = try a.alloc(u8, n * 4);
                self.x = try a.alloc(f64, n);
                for (self.x, 0..) |*v, i| {
                    v.* = @floatFromInt(rng.uintLessThan(usize, 2 * own));
                    std.mem.writeInt(i32, ids[4 * i ..][0..4], @intFromFloat(v.*), .little);
                }
                self.in[0] = try self.rig.input(ids);
                self.out[0] = try self.rig.output(n * 4);
            },
            .combine, .combine_f32 => try self.prepareCombine(rng, fill),
        }
        return true;
    }

    /// y [rows][slots][d] and weights [rows][slots] around a unit; gate 4: integers whose slots 3 to 5 undo 0 to 2.
    fn prepareCombine(self: *Moe, rng: std.Random, fill: g.Fill) !void {
        const a = self.arena.allocator();
        const s = self.shape;
        const slots = s.k;
        self.x = try a.alloc(f64, s.rows * slots * s.n);
        self.w = try a.alloc(f64, s.rows * slots);
        self.shared = try a.alloc(f64, s.rows * s.n);
        for (self.x) |*v| v.* = @as(f32, @floatCast(rng.float(f64) * 4 - 2));
        for (self.w) |*v| v.* = @as(f32, @floatCast(rng.float(f64)));
        for (self.shared) |*v| v.* = @as(f32, @floatCast(rng.float(f64) * 2 - 1));
        if (fill == .stress) for (0..s.rows) |r| {
            for (0..slots) |j| self.w[r * slots + j] = @floatFromInt(200 + j % (slots / 2));
            for (0..s.n) |col| {
                self.shared[r * s.n + col] = @floatFromInt(rng.uintLessThan(u32, 8));
                for (0..slots) |j| {
                    const y = &self.x[(r * slots + j) * s.n + col];
                    const nudge: f64 = if (j == slots / 2) 1 else 0;
                    const big: f64 = @floatFromInt(500 + rng.uintLessThan(u32, 600));
                    y.* = if (j < slots / 2) big else nudge - self.x[(r * slots + j - slots / 2) * s.n + col];
                }
            }
        };
        if (fill == .nonfinite) {
            const at = [_]usize{ 0, s.n + 3, 2 * s.n + 9 };
            for (odd, at) |v, i| self.x[i % self.x.len] = v;
            self.shared[s.n - 1] = std.math.inf(f64);
        }
        self.in[0] = try self.rig.input(try kit.f32Bytes(a, self.x));
        self.in[1] = try self.rig.input(try kit.f32Bytes(a, self.w));
        self.in[2] = try self.rig.input(try kit.f32Bytes(a, self.shared));
        const kind: kit.Kind = if (self.which == .combine_f32) .f32 else self.k;
        self.out[0] = try self.rig.output(s.rows * s.n * kit.size(kind));
    }

    fn launchPick(self: *Moe, f: hip.Function, threads: u32, out: [4]u64) !void {
        const s = self.shape;
        var a: hip.launch.Args = .{};
        for ([_]u64{ self.in[0], self.in[1], out[0], out[1], out[2], out[3] }) |v| try a.add(v);
        for ([_]c_int{ 32, @intCast(s.n), @intCast(s.k) }) |v| try a.add(v);
        try a.add(scaling);
        try a.add(@as(c_int, 1));
        const grid: hip.launch.Dim3 = .{ .x = @intCast(s.rows) };
        try hip.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, self.rig.s, &a);
    }

    pub fn run(self: *Moe) !void {
        const s = self.shape;
        const nemo = self.l.nemo;
        var a: hip.launch.Args = .{};
        const blocks: u32 = @intCast((s.rows * s.n + 255) / 256);
        const flat: hip.launch.Config = .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 } };
        switch (self.which) {
            .pass => try self.launchPick(nemo.select_pass, 32, self.out),
            .rank => try self.launchPick(nemo.select, 512, self.out),
            .relu2 => {
                for ([_]u64{ self.in[0], self.out[0] }) |v| try a.add(v);
                try a.add(@backingInt(self.k));
                try a.add(@as(c_longlong, @intCast(s.rows * s.n)));
                try hip.launch.launch(nemo.relu2, flat, self.rig.s, &a);
            },
            .localize => {
                for ([_]u64{ self.in[0], self.out[0] }) |v| try a.add(v);
                for ([_]c_int{ @intCast(s.rows * s.n), self.first, @intCast(s.k) }) |v| try a.add(v);
                try hip.launch.launch(nemo.localize, flat, self.rig.s, &a);
            },
            .combine, .combine_f32 => {
                for ([_]u64{ self.in[0], self.in[1], self.in[2], self.out[0] }) |v| try a.add(v);
                const kind: c_int = if (self.which == .combine_f32) 0 else @backingInt(self.k);
                for ([_]c_int{ kind, @intCast(s.k), @intCast(s.n) }) |v| try a.add(v);
                const grid: hip.launch.Dim3 = .{ .x = @intCast((s.n + 255) / 256), .y = @intCast(s.rows) };
                try hip.launch.launch(nemo.combine, .{ .grid = grid, .block = .{ .x = 256 } }, self.rig.s, &a);
            },
        }
        try self.rig.s.synchronize();
    }

    /// The pass's bytes on the same inputs, the rank pick's baseline.
    pub fn replaced(self: *Moe) ![]const u8 {
        try self.launchPick(self.l.nemo.select_pass, 32, self.base);
        const a = self.arena.allocator();
        var all: std.ArrayList(u8) = .empty;
        const s = self.shape;
        const lens = [_]usize{ s.rows * s.k * 4, s.rows * s.k * 4, 3 * 32 * 4, 32 * 4 };
        for (self.base, lens, 0..) |at, len, i| {
            if (i >= 2 and !self.planned()) break;
            try all.appendSlice(a, try self.rig.bytes(at, len));
        }
        return all.items;
    }

    pub fn count(self: *Moe) usize {
        const s = self.shape;
        return switch (self.which) {
            .pass, .rank => 2 * s.rows * s.k * 4 + if (self.planned()) @as(usize, 4 * 32 * 4) else 0,
            .localize => s.rows * s.n * 4,
            else => s.rows * s.n,
        };
    }

    pub fn reference(self: *Moe, want: []f64, mag: []f64) !void {
        const s = self.shape;
        @memset(mag, 0);
        switch (self.which) {
            .relu2 => for (self.x, want) |v, *o| {
                const r = if (std.math.isNan(v)) v else @max(v, 0);
                o.* = r * r;
            },
            .combine, .combine_f32 => for (0..s.rows) |r| for (0..s.n) |col| {
                var acc: f64 = 0;
                var abs: f64 = 0;
                for (0..s.k) |j| {
                    const t = self.x[(r * s.k + j) * s.n + col] * self.w[r * s.k + j];
                    acc += t;
                    abs += @abs(t);
                }
                acc += self.shared[r * s.n + col];
                want[r * s.n + col] = acc;
                mag[r * s.n + col] = abs + @abs(self.shared[r * s.n + col]);
            },
            else => @memset(want, 0),
        }
    }

    pub fn simulate(self: *Moe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = self.shape;
        var t: [33]f64 = undefined;
        for (0..s.rows) |r| for (0..s.n) |col| {
            for (0..s.k) |j| t[j] = self.x[(r * s.k + j) * s.n + col] * self.w[r * s.k + j];
            t[s.k] = self.shared[r * s.n + col];
            out[r * s.n + col] = rows.sumIn(acc, order, t[0 .. s.k + 1]);
        };
    }

    /// The pick's rule against float64 scores (near-ties within 2^-20 may swap), and the local ids exactly.
    pub fn check(self: *Moe, _: g.Gate) !?[]const u8 {
        const s = self.shape;
        if (self.which == .localize) {
            const got = try self.rig.bytes(self.out[0], s.rows * s.n * 4);
            const own: i32 = @intCast(s.k);
            for (self.x, 0..) |v, i| {
                const local = @as(i32, @intFromFloat(v)) - self.first;
                const want = if (local >= 0 and local < own) local else own;
                const id = std.mem.readInt(i32, got[4 * i ..][0..4], .little);
                if (id != want) return self.say("id {d} is {d} where the rule gives {d}", .{ i, id, want });
            }
            return null;
        }
        return self.checkPicks();
    }

    fn say(self: *Moe, comptime fmt: []const u8, args: anytype) !?[]const u8 {
        return try std.fmt.bufPrint(&self.msg, fmt, args);
    }

    fn checkPicks(self: *Moe) !?[]const u8 {
        const a = self.arena.allocator();
        const s = self.shape;
        const n = s.n;
        const ids = try a.dupe(u8, try self.rig.bytes(self.out[0], s.rows * s.k * 4));
        const wts = try a.dupe(u8, try self.rig.bytes(self.out[1], s.rows * s.k * 4));
        const sel = try a.alloc(f64, n);
        const score = try a.alloc(f64, n);
        const taken = try a.alloc(bool, n);
        for (0..s.rows) |r| {
            for (sel, score, 0..) |*v, *p, e| {
                p.* = 1 / (1 + @exp(-self.x[r * n + e]));
                v.* = if (std.math.isNan(p.* + self.w[e])) -std.math.inf(f64) else p.* + self.w[e];
            }
            @memset(taken, false);
            var total: f64 = 0;
            for (0..s.k) |k| {
                const raw = std.mem.readInt(i32, ids[(r * s.k + k) * 4 ..][0..4], .little);
                const bad = raw < 0 or raw >= n or taken[@intCast(raw)];
                if (bad) return self.say("row {d} pick {d}: expert {d} out of range or repeated", .{ r, k, raw });
                const id: usize = @intCast(raw);
                var best: f64 = -std.math.inf(f64);
                for (sel, taken) |v, t| if (!t) {
                    best = @max(best, v);
                };
                const tol = 0x1p-20 * (@abs(best) + 1);
                if (!(sel[id] >= best - tol) and !(best == -std.math.inf(f64) and sel[id] == best))
                    return self.say("row {d} pick {d}: expert {d} at {d:.6}, one left at {d:.6}", .{
                        r,
                        k,
                        id,
                        sel[id],
                        best,
                    });
                for (0..id) |e| if (!taken[e] and sel[e] == sel[id])
                    return self.say("row {d} pick {d}: expert {d} ahead of tied expert {d}", .{ r, k, id, e });
                taken[id] = true;
                total += score[id];
            }
            for (0..s.k) |k| {
                const id: usize = @intCast(std.mem.readInt(i32, ids[(r * s.k + k) * 4 ..][0..4], .little));
                const w: f64 = @as(f32, @bitCast(std.mem.readInt(u32, wts[(r * s.k + k) * 4 ..][0..4], .little)));
                const want = score[id] / (total + 1e-20) * scaling;
                const tol = @as(f64, @floatFromInt(s.k + 8)) * 0x1p-23 * @abs(want) + 1e-30;
                const off = "row {d} slot {d}: weight {d:.7}, float64 {d:.7}";
                if (!(@abs(w - want) <= tol)) return self.say(off, .{ r, k, w, want });
            }
        }
        if (!self.planned()) return null;
        const items = try self.rig.bytes(self.out[2], 3 * 32 * 4);
        for (0..32) |k| {
            const real = k < s.k;
            const id = if (real) std.mem.readInt(i32, ids[4 * k ..][0..4], .little) else 0;
            const want = [3]i32{ id, if (real) @intCast(k) else 0, @intFromBool(real) };
            for (want, 0..) |v, j| if (std.mem.readInt(i32, items[(3 * k + j) * 4 ..][0..4], .little) != v)
                return self.say("plan item {d} field {d} is not {d}", .{ k, j, v });
        }
        const members = try self.rig.bytes(self.out[3], 32 * 4);
        for (0..s.k) |k| if (std.mem.readInt(i32, members[4 * k ..][0..4], .little) != @as(i32, @intCast(k)))
            return self.say("plan member {d} is not {d}", .{ k, k });
        return null;
    }
};
