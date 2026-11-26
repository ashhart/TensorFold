//! Core's 16/32-member grouped projections and Router on Inkling fixtures, with exact baselines and controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ink = @import("inkling2");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_inkling2_experts_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;

fn i32s(v: usize) i32 {
    return @intCast(v);
}

/// One projection as exl3_experts.runWith launches it on 16/32-member items; the baseline runs tf_exl3_grouped_2.
pub const Grouped = struct {
    rig: rig_.Rig,
    k: *const ink.experts.Kernels,
    form: ref.Form,
    fault: ref.Fault = .none,
    fix: ?ref.Fixture = null,
    tp: [2]u64 = .{ 0, 0 },
    k2: [2]u64 = .{ 0, 0 },
    x: [2]u64 = .{ 0, 0 },
    uids: u64 = 0,
    ucount: u64 = 0,
    members: u64 = 0,
    plan: u64 = 0, // the item plan's members: pair p = row * slots + slot
    items: u64 = 0,
    nitems: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Grouped) void {
        self.drop();
        self.rig.deinit();
    }

    fn drop(self: *Grouped) void {
        if (self.fix) |*f| f.deinit();
        self.fix = null;
        self.rig.reset();
    }

    pub fn kernel(self: *Grouped) g.Kernel {
        return .{ .spec = ref.spec(self.form, self.fault), .ctx = self, .vt = &vt, .control = if (self.fault == .none) null else .shapes };
    }

    const vt = g.hooks(Grouped);

    fn put(self: *Grouped, x: anytype) !u64 {
        return self.rig.input(std.mem.sliceAsBytes(x));
    }

    pub fn prepare(self: *Grouped, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!ref.Fixture.valid(shape)) return false;
        self.drop();
        const a = self.rig.gpu.gpa;
        self.fix = try ref.Fixture.init(a, shape, fill, seed);
        const f = &self.fix.?;
        var pool: [f.trellis.len]u64 = undefined;
        for (&pool, f.trellis) |*p, t| p.* = try self.put(t);
        const m = ref.mats(shape);
        for (0..2) |mat| {
            var tp: [ref.experts]u64 = undefined;
            var k2: [ref.experts]i32 = undefined;
            for (&tp, &k2, 0..) |*t, *h, e| {
                const i = ref.Fixture.entry(e, @min(mat, m - 1));
                t.* = pool[i];
                h.* = ref.halves(i);
            }
            self.tp[mat] = try self.put(&tp);
            self.k2[mat] = try self.put(&k2);
        }
        const p = f.pairs();
        const xs = p * shape.k;
        self.x[0] = try self.put(f.x[0..xs]);
        self.x[1] = if (m == 2) try self.put(f.x[xs..]) else self.x[0];
        var l = try ref.Lists.init(a, f.pick, shape.rows);
        defer l.deinit(a);
        self.uids = try self.put(l.uids);
        self.ucount = try self.put(&[1]i32{l.count});
        self.members = try self.put(l.members);
        const items = try a.alloc(i32, 3 * ref.maxItems(p));
        defer a.free(items);
        const n = l.items(shape.rows, ref.per(self.form), items);
        const plan = try a.alloc(i32, l.members.len);
        defer a.free(plan);
        for (plan, l.members) |*q, code| q.* = if (code < 0) -1 else @divTrunc(code, 32) * ref.slots + @mod(code, 32);
        for (0..@intCast(n)) |i| { // (expert, first member in the plan, count), as grouped.Router.route writes them
            const u: usize = @intCast(items[3 * i]);
            items[3 * i] = l.uids[u];
            items[3 * i + 1] += @intCast(u * shape.rows);
        }
        self.plan = try self.put(plan);
        self.items = try self.put(items);
        const short: i32 = @intFromBool(self.fault == .short);
        self.nitems = try self.put(&[1]i32{n - short});
        self.out = try self.rig.output(f.outputs() * 4);
        self.base = try self.rig.scratch(f.outputs() * 4);
        return true;
    }

    /// exl3_experts.runWith's launch: standard mode, 16 or 32 members by 128 or 64 columns.
    fn launch(self: *Grouped) !void {
        const sh = self.fix.?.shape;
        const p = self.fix.?.pairs();
        const sk = self.fix.?.sk;
        const m = ref.mats(sh);
        const per = ref.per(self.form);
        const nx = cuda.grouped.maxItems(p, ref.experts + 1, per);
        const x0 = if (self.fault == .swapped) self.x[1] else self.x[0];
        var a: cuda.Args = .{};
        for ([_]u64{ x0, self.x[1], self.tp[0], self.tp[1], self.k2[0], self.k2[1], self.items, self.nitems, self.plan, self.out }) |v| a.add(v);
        for ([_]usize{ sh.k, sh.n, p, sk, per, ref.slots, nx, m, 0, 1 }) |v| a.add(i32s(v));
        a.add(@as(i32, ref.experts));
        const function = if (self.form == .wide) self.k.main.wide else self.k.main.grouped[2];
        const columns: usize = if (self.form == .wide) 64 else 128;
        try cuda.launch.launch(function, .{ .grid = .{ .x = @intCast(nx), .y = @intCast(sh.n / columns), .z = @intCast(m * sk) }, .block = .{ .x = 128 } }, self.rig.s, &a);
    }

    pub fn run(self: *Grouped) !void {
        try self.launch();
        try self.rig.s.synchronize();
    }

    /// exl3_experts.Experts.run's grouped launch on the same lists: GroupedArgs, one member tile a grid z step.
    pub fn replaced(self: *Grouped) ![]const u8 {
        const sh = self.fix.?.shape;
        const p = self.fix.?.pairs();
        const sk = self.fix.?.sk;
        const maxu = @min(p, ref.experts);
        const m = ref.mats(sh);
        var a: cuda.Args = .{};
        for ([_]u64{ self.x[0], self.x[1], self.tp[0], self.tp[1], self.k2[0], self.k2[1], self.uids, self.ucount, self.members, self.base }) |v| a.add(v);
        for ([_]usize{ sh.k, sh.n, p, sk, sh.rows, ref.slots, maxu, m, 0, 0 }) |v| a.add(i32s(v));
        a.add(@as(i32, -1));
        const mt = (sh.rows + 15) / 16;
        try cuda.launch.launch(self.k.main.grouped[2], .{ .grid = .{ .x = @intCast(maxu), .y = @intCast(sh.n / 128), .z = @intCast(m * sk * mt) }, .block = .{ .x = 128 } }, self.rig.s, &a);
        return self.rig.bytes(self.base, self.outputs() * 4);
    }

    /// Gate 1: tf_exl3_grouped_2's bytes, and some output non-finite: the planted values.
    pub fn check(self: *Grouped, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from tf_exl3_grouped_2's", .{ at, got.len });
        for (0..got.len / 4) |i| if (c.classOf(c.read(.f32, got, i)) != .finite) return null;
        return try std.fmt.bufPrint(&self.msg, "no output is non-finite: the planted values prove nothing", .{});
    }

    fn outputs(self: *Grouped) usize {
        return self.fix.?.outputs();
    }

    pub fn count(self: *Grouped) usize {
        return self.outputs();
    }

    pub fn reference(self: *Grouped, want: []f64, mag: []f64) !void {
        return self.fix.?.reference(want, mag);
    }

    pub fn simulate(self: *Grouped, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.fix.?.simulate(acc, order, out);
    }
};

/// Shape: token rows x slots (8) x experts (384), including both Router.plan_small and Router.wide.
fn groupSpec(control: bool) c.Spec {
    const served = &[_]c.Shape{
        .{ .rows = 8, .n = ref.slots, .k = ref.experts },
        .{ .rows = 1, .n = ref.slots, .k = ref.experts },
        .{ .rows = 15, .n = ref.slots, .k = ref.experts },
        .{ .rows = 16, .n = ref.slots, .k = ref.experts },
        .{ .rows = 17, .n = ref.slots, .k = ref.experts },
        .{ .rows = 128, .n = ref.slots, .k = ref.experts },
        .{ .rows = 511, .n = ref.slots, .k = ref.experts },
        .{ .rows = 512, .n = ref.slots, .k = ref.experts },
        .{ .rows = 2048, .n = ref.slots, .k = ref.experts },
    };
    const name = if (control) "inkling2 Router control: grouped.Router.route with 16-member items on wide rows" else "inkling2 Router: grouped.Router.route members, items and counts";
    return .{
        .name = name,
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .replaces = "tf_exl3_group's lists, transformed to core's item plan",
        .nan = .{ .stated = "picks outside [0, experts) are left out, as tf_exl3_group leaves them" },
        .tiles = .{ .{ .size = 16 }, .{}, .{} },
        .served = served,
    };
}

fn noSums(_: c.Shape) usize {
    return 0;
}

/// grouped.Router.route as experts.run launches it; tf_exl3_group supplies the exact baseline lists.
pub const Group = struct {
    rig: rig_.Rig,
    k: *const ink.experts.Kernels,
    wrong_run: bool = false,
    shape: c.Shape = undefined,
    pick: []i32 = &.{},
    dev: u64 = 0,
    outs: [3]u64 = @splat(0), // pair members, item triples, item/used counts
    base: [3]u64 = @splat(0), // expert ids, used count, row*32+slot members
    rank: u64 = 0,
    hist: u64 = 0,
    want: std.ArrayList(u8) = .empty,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Group) void {
        const a = self.rig.gpu.gpa;
        a.free(self.pick);
        self.want.deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *Group) g.Kernel {
        return .{ .spec = groupSpec(self.wrong_run), .ctx = self, .vt = &vt, .control = if (self.wrong_run) .shapes else null };
    }

    const vt = g.hooks(Group);

    fn maxu(self: *const Group) usize {
        return @min(self.shape.rows * ref.slots, ref.experts);
    }

    fn sizes(self: *const Group) [3]usize {
        const p = self.shape.rows * ref.slots;
        return .{ p * 4, 3 * ref.maxItems(p) * 4, 8 };
    }

    fn baseSizes(self: *const Group) [3]usize {
        return .{ self.maxu() * 4, 4, self.maxu() * self.shape.rows * 4 };
    }

    pub fn prepare(self: *Group, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != ref.slots or shape.k != ref.experts or shape.rows * shape.n * 4 > 64 * 1024) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        a.free(self.pick);
        self.pick = &.{};
        self.pick = try a.alloc(i32, shape.rows * ref.slots);
        var prng = std.Random.DefaultPrng.init(seed);
        ref.picks(prng.random(), self.pick);
        if (fill == .nonfinite) {
            const put = [_]i32{ -1, ref.experts, 1 << 20 };
            for (put, 0..) |v, i| self.pick[(i * 5 + 1) % self.pick.len] = v;
        }
        self.dev = try self.rig.input(std.mem.sliceAsBytes(self.pick));
        for (&self.outs, self.sizes()) |*o, n| o.* = try self.rig.output(n);
        for (&self.base, self.baseSizes()) |*o, n| o.* = try self.rig.scratch(n);
        const p = shape.rows * ref.slots;
        self.rank = if (p > cuda.grouped.small) try self.rig.scratch(p * 4) else 0;
        const blocks = if (p > cuda.grouped.small) (p + cuda.grouped.small - 1) / cuda.grouped.small else 0;
        self.hist = if (blocks != 0) try self.rig.scratch(blocks * (ref.experts + 1) * 4) else 0;
        return true;
    }

    pub fn run(self: *Group) !void {
        const rows = self.shape.rows;
        const tile: usize = if (rows >= ink.experts.wide_from and !self.wrong_run) 32 else 16;
        const p: cuda.grouped.Plan = .{
            .members = self.outs[0],
            .items = self.outs[1],
            .counts = self.outs[2],
            .rank = self.rank,
            .hist = self.hist,
        };
        try self.k.router.route(self.rig.s, self.dev, rows * ref.slots, ref.experts, tile, p);
        try self.rig.s.synchronize();
    }

    /// Core member codes and global item offsets, with unused output tails left poisoned like the candidate's.
    pub fn replaced(self: *Group) ![]const u8 {
        const a = self.rig.gpu.gpa;
        const rows = self.shape.rows;
        const p = rows * ref.slots;
        var args: cuda.Args = .{};
        for ([_]u64{ self.dev, self.base[0], self.base[1], self.base[2] }) |v| args.add(v);
        for ([_]usize{ rows, ref.slots, ref.experts, rows }) |v| args.add(i32s(v));
        try cuda.launch.launch(self.k.main.group, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 }, .shared = @intCast(p * 4) }, self.rig.s, &args);
        const cnt = std.mem.readInt(i32, (try self.rig.bytes(self.base[1], 4))[0..4], .little);
        if (cnt < 0 or @as(usize, @intCast(cnt)) > self.maxu()) return error.BadBaselineCount;
        const used: usize = @intCast(cnt);
        const sz = self.sizes();
        self.want.clearRetainingCapacity();
        try self.want.appendNTimes(a, 0xff, sz[0] + sz[1] + sz[2]);
        const w = self.want.items;
        const uids = try a.alloc(i32, self.maxu());
        defer a.free(uids);
        @memcpy(std.mem.sliceAsBytes(uids), try self.rig.bytes(self.base[0], self.maxu() * 4));
        const members = try a.alloc(i32, self.maxu() * rows);
        defer a.free(members);
        @memcpy(std.mem.sliceAsBytes(members), try self.rig.bytes(self.base[2], self.maxu() * rows * 4));
        var at: usize = 0;
        for (0..used) |u| {
            const old = members[u * rows ..][0..rows];
            const n = std.mem.indexOfScalar(i32, old, -1) orelse rows;
            for (old[0..n]) |code| {
                const pair = @divTrunc(code, 32) * ref.slots + @mod(code, 32);
                std.mem.writeInt(i32, w[4 * at ..][0..4], pair, .little);
                at += 1;
            }
        }
        var valid: usize = 0;
        for (self.pick) |e| valid += @intFromBool(e >= 0 and e < ref.experts);
        if (at != valid) return error.BadBaselineMembers;
        const lists: ref.Lists = .{ .uids = uids, .count = cnt, .members = members };
        const items = try a.alloc(i32, sz[1] / 4);
        defer a.free(items);
        const n = lists.items(rows, if (rows >= ink.experts.wide_from) 32 else 16, items);
        for (0..@intCast(n)) |i| {
            const u: usize = @intCast(items[3 * i]);
            const first = items[3 * i + 1];
            const nmem = items[3 * i + 2];
            const offset = sz[0] + 12 * i;
            std.mem.writeInt(i32, w[offset..][0..4], uids[u], .little);
            std.mem.writeInt(i32, w[offset + 4 ..][0..4], @as(i32, @intCast(u * rows)) + first, .little);
            std.mem.writeInt(i32, w[offset + 8 ..][0..4], nmem, .little);
        }
        std.mem.writeInt(i32, w[sz[0] + sz[1] ..][0..4], n, .little);
        std.mem.writeInt(i32, w[sz[0] + sz[1] + 4 ..][0..4], cnt, .little);
        return w;
    }

    /// Gate 1: tf_exl3_group's lists converted to core's plan, on valid serving picks.
    pub fn check(self: *Group, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from tf_exl3_group's plan", .{ at, got.len });
        return null;
    }

    pub fn count(self: *Group) usize {
        var n: usize = 0;
        for (self.sizes()) |s| n += s;
        return n;
    }

    pub fn reference(_: *Group, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
};
