//! The MoE pick under the contract on HIP: a prompt's picks a wave a row, and the router, pick and plan in one launch.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");
const route = @import("contract_route.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;

pub const Which = enum { rows, router };

const top_k = 10; // Flash Next's routed picks a row; its shared expert takes the slot after them
const slots = top_k + 1;
const tile = 8; // moe.zig tileOf: a verify window's plan tile
const waves = 4; // decode.hip kRouterWaves: rows a block of tf_select_rows

fn none(_: c.Shape) usize {
    return 0;
}

/// A lane's fma chain over a d / 32 slice of the dot, then the wave's 5 shuffle adds.
fn chain(s: c.Shape) usize {
    return s.k / 32 + 5;
}

/// moe.zig capacity(bucket(rows) * slots, e): a decode call's items, 16 rows' worth.
fn capacity(e: usize) usize {
    return @min(16 * slots, e) + 16 * slots / 8;
}

const pick_rule = "every slot of every row written: the top k logits, larger first, equal ones to the lower id, a NaN " ++
    "below every number; weights the softmax of the picks (NaN where a NaN or inf - inf enters) within bf16's half " ++
    "ulp and 2 (k + 8) fp32 ulps of float64; the shared slot sigmoid(bf16(gate))";
const select_known = "select_body (upstream): a NaN shares rank 0 with the top number (slot 0 races) and a row with " ++
    "fewer than k numbers leaves slots unwritten; bytes kept equal to upstream's";
const router_rule = "logits in float64's classes; picks and weights by tf_select_rows' rule; the plan tf_moe_route's " ++
    "of the picks (one row: an item a slot)";
const router_names = [2][]const u8{ "hip tf_router_select_f16", "hip tf_router_select_bf16" };

pub fn spec(which: Which, k: kit.Kind) c.Spec {
    return switch (which) {
        .rows => .{
            .name = "hip tf_select_rows",
            .accum = .none,
            .depth = none,
            .out = .raw,
            .replaces = "tf_select_decode (select_body)",
            .nan = .{ .stated = pick_rule },
            .known = .{ .gate = .nonfinite, .finding = select_known },
            // a block's 4 rows; select_row_lanes' 16 logits a lane to 512 experts
            .tiles = .{ .{ .size = waves }, .{ .size = 512, .min = 16 }, .{} },
            // Flash Next's 512 experts, top 10: a one-row prompt and a prompt chunk
            .served = &.{
                .{ .rows = 1, .n = 512, .k = top_k },
                .{ .rows = 2048, .n = 512, .k = top_k },
            },
        },
        .router => .{
            .name = router_names[@intFromBool(k == .bf16)],
            .accum = .f32,
            .depth = chain,
            // the logits, its first output: gate 4 reads them; gate 3 compares every output's bytes
            .out = .f32,
            .replaces = "tf_router_decode, tf_select_decode or tf_select_rows, then tf_moe_route",
            .nan = .{ .stated = router_rule },
            .known = .{ .gate = .nonfinite, .finding = select_known },
            .tiles = .{ .{ .size = waves }, .{}, .{} },
            // Flash Next's 512 experts and shared gate over 2560 wide rows: a decode row, a verify window, 16 rows
            .served = &.{
                .{ .rows = 1, .n = 513, .k = 2560 },
                .{ .rows = 8, .n = 513, .k = 2560 },
                .{ .rows = 16, .n = 513, .k = 2560 },
            },
        },
    };
}

fn say(msg: []u8, comptime fmt: []const u8, args: anytype) !?[]const u8 {
    return try std.fmt.bufPrint(msg, fmt, args);
}

fn word(b: []const u8, i: usize) i32 {
    return std.mem.readInt(i32, b[4 * i ..][0..4], .little);
}

/// A weight within bf16's half ulp of a value within 2 (k + 8) fp32 ulps of float64; classes equal.
fn close(got: f64, want: f64, k: usize) bool {
    if (std.math.isNan(got) or std.math.isNan(want)) return std.math.isNan(got) and std.math.isNan(want);
    if (got == want) return true;
    const tol = (0x1p-8 + 2 * @as(f64, @floatFromInt(k + 8)) * 0x1p-23) * @abs(want) + 0x1p-120;
    return @abs(got - want) <= tol;
}

/// The pick rule over one call's logits [rows][experts + 1] (the gate's last): null when every row keeps it.
fn pickRule(msg: []u8, a: std.mem.Allocator, logits: []const f64, rows: usize, experts: usize, k: usize, ids: []const u8, wts: []const u8) !?[]const u8 {
    const order = try a.alloc(usize, experts);
    const want = try a.alloc(f64, k);
    const By = struct {
        fn ahead(v: []const f64, x: usize, y: usize) bool {
            const nx = std.math.isNan(v[x]);
            const ny = std.math.isNan(v[y]);
            if (nx or ny) return if (nx and ny) x < y else ny;
            return v[x] > v[y] or (v[x] == v[y] and x < y);
        }
    };
    for (0..rows) |r| {
        const row = logits[r * (experts + 1) ..][0 .. experts + 1];
        for (order, 0..) |*v, i| v.* = i;
        std.sort.pdq(usize, order, row, By.ahead);
        const at = r * (k + 1);
        for (0..k) |j| {
            const id = word(ids, at + j);
            if (id != @as(i32, @intCast(order[j])))
                return say(msg, "row {d} slot {d} holds expert {d}, the rule gives {d}", .{ r, j, id, order[j] });
        }
        if (word(ids, at + k) != @as(i32, @intCast(experts)))
            return say(msg, "row {d}: the shared slot is not expert {d}", .{ r, experts });
        var total: f64 = 0;
        for (want, order[0..k]) |*w, i| {
            w.* = @exp(row[i] - row[order[0]]);
            total += w.*;
        }
        for (want, 0..) |w, j| {
            const got = c.read(.f32, wts, at + j);
            if (!close(got, w / total, k))
                return say(msg, "row {d} slot {d}: weight {e:.6}, float64 {e:.6}", .{ r, j, got, w / total });
        }
        const gate = kit.round(.bf16, row[experts]);
        const shared = c.read(.f32, wts, at + k);
        if (!close(shared, 1 / (1 + @exp(-gate)), 1))
            return say(msg, "row {d}: shared weight {e:.6} for gate {e:.6}", .{ r, shared, gate });
    }
    return null;
}

/// NaN, both infinities and tied +Inf among a call's logits [rows][experts + 1], and a row of fewer than k numbers.
fn plantLogits(x: []f64, rows: usize, experts: usize, k: usize) void {
    const nan = std.math.nan(f64);
    const inf = std.math.inf(f64);
    const at = [_][2]usize{ .{ 0, 7 }, .{ 1, 5 }, .{ 1, 6 }, .{ 2, 3 }, .{ 3, 1 }, .{ 3, 2 }, .{ 3, 9 }, .{ 4, experts }, .{ 4, 0 }, .{ 4, 11 } };
    const put = [_]f64{ nan, inf, inf, -inf, inf, -inf, nan, nan, nan, nan };
    for (at, put) |p, v| if (p[0] < rows) {
        x[p[0] * (experts + 1) + p[1]] = v;
    };
    if (rows > 4) for (x[4 * (experts + 1) + k / 2 ..][0 .. experts - k / 2]) |*v| {
        v.* = nan;
    };
}

/// Infinities in router rows, x's sign or 0 there by row: NaN logits, both infinities, a row of 5 numbers.
fn plantRouter(x: []f64, w: []f64, rows: usize, e: usize, d: usize) void {
    const inf = std.math.inf(f64);
    w[d] = inf;
    w[2 * d] = -inf;
    w[3 * d] = inf;
    w[3 * d + 4] = -inf;
    w[(e - 1) * d + 4] = inf;
    for (8..e - 1) |j| w[j * d + 8] = inf;
    const signs = [5][2]f64{ .{ 0.5, 0.5 }, .{ -0.5, 0.5 }, .{ 0.5, -0.5 }, .{ 0, 0.5 }, .{ 0.5, 0 } };
    for (0..rows) |r| {
        x[r * d] = signs[r % 5][0];
        x[r * d + 4] = signs[r % 5][1];
        x[r * d + 8] = if (r % 5 == 3) 0 else 0.5;
    }
}

/// tf_select_rows on fp32 logits, or tf_router_select on x of the kind and bf16 router rows; both check the rule.
pub const Select = struct {
    rig: rig_.Rig,
    which: Which,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []f64 = &.{}, // the logits [rows][experts + 1], or x [rows][d]
    w: []f64 = &.{}, // the router's rows [e][d]
    in: [3]u64 = @splat(0),
    out: [5]u64 = @splat(0),
    base: [5]u64 = @splat(0),
    lens: [5]usize = @splat(0),
    outs: usize = 0,
    done: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, w: Which) Select {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .which = w, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Select) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Select) g.Kernel {
        return .{ .spec = spec(self.which, self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Select);

    pub fn prepare(self: *Select, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const s = shape;
        if (s.rows == 0) return false;
        switch (self.which) {
            .rows => if (s.n < 12 or s.n > 1024 or s.k == 0 or s.k > 31 or s.k > s.n) return false,
            .router => if (s.rows > 16 or s.n < top_k + 2 or s.n > 1025 or s.k < 16 or s.k % 4 != 0) return false,
        }
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = s;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        switch (self.which) {
            .rows => try self.prepareRows(rng, fill),
            .router => try self.prepareRouter(rng, fill),
        }
        const a = self.arena.allocator();
        for (self.out[0..self.outs], self.base[0..self.outs], self.lens[0..self.outs]) |*o, *b, len| {
            o.* = try self.rig.output(len);
            b.* = try self.rig.scratch(len);
            const poison = try a.alloc(u8, len);
            @memset(poison, 0xff); // as the outputs start
            try self.rig.put(b.*, poison);
        }
        return true;
    }

    /// Logits in sixteenths, so equal logits are common; the gate's logit last in each row.
    fn prepareRows(self: *Select, rng: std.Random, fill: g.Fill) !void {
        const a = self.arena.allocator();
        const s = self.shape;
        self.x = try a.alloc(f64, s.rows * (s.n + 1));
        for (self.x) |*v| v.* = @as(f64, @floatFromInt(rng.uintLessThan(u32, 64))) / 16 - 2;
        if (fill == .nonfinite) plantLogits(self.x, s.rows, s.n, s.k);
        self.in[0] = try self.rig.input(try kit.f32Bytes(a, self.x));
        const picks = s.rows * (s.k + 1) * 4;
        self.lens = .{ picks, picks, 0, 0, 0 };
        self.outs = 2;
    }

    /// x in [-1, 1) and router rows in [-1/8, 1/8), three rows equal to expert 0's; gate 4: mirrored integers.
    fn prepareRouter(self: *Select, rng: std.Random, fill: g.Fill) !void {
        const a = self.arena.allocator();
        const s = self.shape;
        const e = s.n;
        const d = s.k;
        self.x = try a.alloc(f64, s.rows * d);
        self.w = try a.alloc(f64, e * d);
        if (fill == .stress) {
            // the second half undoes the first, but for one: every fp32 partial sum exact, each logit small
            const h = d / 2;
            for (0..s.rows) |r| for (0..h) |i| {
                const v: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 57));
                self.x[r * d + i] = v;
                self.x[r * d + h + i] = v;
            };
            for (0..e) |j| {
                for (0..h) |i| {
                    const v: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 57));
                    self.w[j * d + i] = v;
                    self.w[j * d + h + i] = -v;
                }
                self.w[j * d + h] += 1;
            }
        } else {
            for (self.x) |*v| {
                v.* = kit.round(self.k, rng.float(f64) * 2 - 1);
                if (self.k == .f16 and @abs(v.*) < 0x1p-14) v.* = 0; // fp16 normals only, as kit.values gives
            }
            for (self.w) |*v| v.* = kit.round(.bf16, rng.float(f64) / 4 - 0.125);
            for ([_]usize{ 3, 9, e / 2 }) |j| if (j < e - 1) @memcpy(self.w[j * d ..][0..d], self.w[0..d]);
            if (fill == .nonfinite) plantRouter(self.x, self.w, s.rows, e, d);
        }
        const xb = try a.alloc(u8, self.x.len * 2);
        for (self.x, 0..) |v, i| kit.put(self.k, xb, i, v);
        const w16 = try a.alloc(u8, self.w.len * 2);
        for (self.w, 0..) |v, i| kit.put(.bf16, w16, i, v);
        self.in = .{ try self.rig.input(xb), try self.rig.input(w16), try self.rig.input(try kit.f32Bytes(a, self.w)) };
        self.done = try self.rig.scratch(4);
        const picks = s.rows * slots * 4;
        self.lens = .{ s.rows * e * 4, picks, picks, capacity(e) * 12, picks };
        self.outs = if (s.rows <= 8) 5 else 3;
    }

    /// tf_select_rows (4 rows a block), or with `body` select_body's block a row with no plan (tf_select_decode).
    fn launchPick(self: *Select, body: bool, out: [5]u64) !void {
        const s = self.shape;
        var a: hip.launch.Args = .{};
        for ([_]u64{ self.in[0], out[0], out[1] }) |v| try a.add(v);
        const n: c_int = @intCast(s.n);
        const k: c_int = @intCast(s.k);
        const rows: u32 = @intCast(s.rows);
        if (body) {
            for ([_]u64{ 0, 0 }) |v| try a.add(v);
            for ([_]c_int{ 0, n, k }) |v| try a.add(v);
            const cfg: hip.launch.Config = .{ .grid = .{ .x = rows }, .block = .{ .x = 512 } };
            return hip.launch.launch(self.l.dec.select, cfg, self.rig.s, &a);
        }
        for ([_]c_int{ @intCast(s.rows), n, k }) |v| try a.add(v);
        const cfg: hip.launch.Config = .{ .grid = .{ .x = (rows + waves - 1) / waves }, .block = .{ .x = 32 * waves } };
        try hip.launch.launch(self.l.dec.select_rows, cfg, self.rig.s, &a);
    }

    pub fn run(self: *Select) !void {
        const s = self.shape;
        switch (self.which) {
            .rows => try self.launchPick(false, self.out),
            .router => {
                const planned = self.outs == 5;
                const items: ?[*]i32 = if (planned) @ptrFromInt(self.out[3]) else null;
                const members: ?[*]i32 = if (planned) @ptrFromInt(self.out[4]) else null;
                const ran = try self.l.routerSelect(
                    @ptrFromInt(self.in[0]),
                    @backingInt(self.k),
                    @ptrFromInt(self.in[1]),
                    @ptrFromInt(self.out[0]),
                    @ptrFromInt(self.out[1]),
                    @ptrFromInt(self.out[2]),
                    items,
                    members,
                    @intCast(capacity(s.n)),
                    @intCast(s.rows),
                    @intCast(s.k),
                    @intCast(s.n),
                    top_k,
                    @as(c_int, if (s.rows == 1) 0 else tile),
                    self.done,
                    self.rig.s.handle,
                );
                if (!ran) return error.RouterSelectRefused;
            },
        }
        try self.rig.s.synchronize();
    }

    /// tf_router_decode on the rows widened to fp32, the pick (one row: select_body with its plan), then tf_moe_route.
    fn routerBaseline(self: *Select) !void {
        const s = self.shape;
        const b = self.base;
        const h = self.rig.s.handle;
        const one = s.rows == 1;
        const cap: c_int = @intCast(capacity(s.n));
        const items: ?[*]i32 = if (one) @ptrFromInt(b[3]) else null;
        const members: ?[*]i32 = if (one) @ptrFromInt(b[4]) else null;
        const r: c_int = @intCast(s.rows);
        const e: c_int = @intCast(s.n);
        try self.l.tf_moe_router(@ptrFromInt(self.in[0]), @backingInt(self.k), @ptrFromInt(self.in[2]), @ptrFromInt(b[0]), r, @intCast(s.k), e, h);
        try self.l.tf_moe_select(@ptrFromInt(b[0]), @ptrFromInt(b[1]), @ptrFromInt(b[2]), items, members, cap, r, e - 1, top_k, h);
        if (self.outs == 5 and !one) try self.l.tf_moe_route(@ptrFromInt(b[1]), r * slots, e, tile, @ptrFromInt(b[4]), @ptrFromInt(b[3]), cap, h);
    }

    /// The launches it replaces on the same inputs, every output's bytes in order.
    pub fn replaced(self: *Select) ![]const u8 {
        switch (self.which) {
            .rows => try self.launchPick(true, self.base),
            .router => try self.routerBaseline(),
        }
        const a = self.arena.allocator();
        var all: std.ArrayList(u8) = .empty;
        for (self.base[0..self.outs], self.lens[0..self.outs]) |at, len| try all.appendSlice(a, try self.rig.bytes(at, len));
        return all.items;
    }

    pub fn count(self: *Select) usize {
        const s = self.shape;
        return if (self.which == .router) s.rows * s.n else self.lens[0] + self.lens[1];
    }

    pub fn reference(self: *Select, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
        if (self.which == .rows) return;
        const s = self.shape;
        for (0..s.rows) |r| for (0..s.n) |j| {
            const at = r * s.n + j;
            for (self.x[r * s.k ..][0..s.k], self.w[j * s.k ..][0..s.k]) |x, w| {
                want[at] += x * w;
                mag[at] += @abs(x * w);
            }
        };
    }

    /// The router's dots in an accumulator and order (gate 4).
    pub fn simulate(self: *Select, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = self.shape;
        const t = try self.arena.allocator().alloc(f64, s.k);
        for (0..s.rows) |r| for (0..s.n) |j| {
            for (t, self.x[r * s.k ..][0..s.k], self.w[j * s.k ..][0..s.k]) |*v, x, w| v.* = x * w;
            out[r * s.n + j] = rows_.sumIn(acc, order, t);
        };
    }

    /// The pick rule on the logits it read, or wrote (the router: classes against float64), then the plan.
    pub fn check(self: *Select, _: g.Gate) !?[]const u8 {
        const a = self.arena.allocator();
        const s = self.shape;
        const first: usize = if (self.which == .rows) 0 else 1; // the router's logits come first
        const ids = try a.dupe(u8, try self.rig.bytes(self.out[first], self.lens[1]));
        const wts = try a.dupe(u8, try self.rig.bytes(self.out[first + 1], self.lens[1]));
        if (self.which == .rows) return pickRule(&self.msg, a, self.x, s.rows, s.n, s.k, ids, wts);
        const n = s.rows * s.n;
        const want = try a.alloc(f64, 2 * n);
        try self.reference(want[0..n], want[n..]);
        const raw = try self.rig.bytes(self.out[0], n * 4);
        const logits = try a.alloc(f64, n);
        for (logits, want[0..n], 0..) |*v, f, i| {
            v.* = c.read(.f32, raw, i);
            if (c.classOf(v.*) != c.classOf(f))
                return say(&self.msg, "logit {d} is {t} where float64 gives {t}", .{ i, c.classOf(v.*), c.classOf(f) });
        }
        if (try pickRule(&self.msg, a, logits, s.rows, s.n - 1, top_k, ids, wts)) |why| return why;
        return if (self.outs == 5) self.checkPlan(ids) else null;
    }

    /// One row: an item a slot (expert, slot, 1) and dead items after; more: tf_moe_route's plan of the picks.
    fn checkPlan(self: *Select, ids: []const u8) !?[]const u8 {
        const a = self.arena.allocator();
        const s = self.shape;
        const cap = capacity(s.n);
        const picks = try a.alloc(i32, s.rows * slots);
        for (picks, 0..) |*p, i| p.* = word(ids, i);
        var members: []i32 = undefined;
        var items: []i32 = undefined;
        if (s.rows == 1) {
            items = try a.alloc(i32, cap * 3);
            @memset(items, 0);
            members = try a.alloc(i32, slots);
            for (0..slots) |k| {
                items[3 * k ..][0..3].* = .{ picks[k], @intCast(k), 1 };
                members[k] = @intCast(k);
            }
        } else {
            members, items = try route.plan(a, picks, s.n, tile, cap);
        }
        if (!std.mem.eql(u8, try self.rig.bytes(self.out[3], cap * 12), std.mem.sliceAsBytes(items)))
            return say(&self.msg, "the plan's items differ from the rule's ({d} rows)", .{s.rows});
        if (!std.mem.eql(u8, try self.rig.bytes(self.out[4], picks.len * 4), std.mem.sliceAsBytes(members)))
            return say(&self.msg, "the plan's members differ from the rule's ({d} rows)", .{s.rows});
        return null;
    }
};
