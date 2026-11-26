//! Mini's baseline rows, wide and grouping kernels, plus core's shared wide comparison, under the contract.
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

/// The baseline module beside main's exl3 experts, retained for the old kernels' contract cases.
pub const BaselineKernels = struct {
    main: cuda.exl3_experts.Experts,
    mod: cuda.Module,
    group: cuda.Function,
    rows: cuda.Function,
    wide: cuda.Function,

    pub fn load(d: *const cuda.Driver) !BaselineKernels {
        var main = try cuda.exl3_experts.Experts.load(d);
        errdefer main.unload();
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_experts);
        errdefer mod.unload();
        return .{
            .main = main,
            .mod = mod,
            .group = try mod.function("tf_ink2x_group"),
            .rows = try mod.function("tf_ink2x_rows"),
            .wide = try mod.function("tf_ink2x_wide"),
        };
    }

    pub fn unload(k: *BaselineKernels) void {
        k.mod.unload();
        k.main.unload();
    }
};

/// One projection as experts.run launches it, on the lists tf_exl3_group writes; the baseline runs tf_exl3_grouped_2.
pub const Grouped = struct {
    rig: rig_.Rig,
    k: *const BaselineKernels,
    form: ref.Form,
    fault: ref.Fault = .none,
    fix: ?ref.Fixture = null,
    tp: [2]u64 = .{ 0, 0 },
    k2: [2]u64 = .{ 0, 0 },
    x: [2]u64 = .{ 0, 0 },
    uids: u64 = 0,
    ucount: u64 = 0,
    members: u64 = 0,
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
        self.items = try self.put(items);
        const listed = ref.listed(self.form, shape.rows);
        const short: i32 = @intFromBool(self.fault == .short);
        self.nitems = try self.put(&[1]i32{(if (listed) n else l.count) - short});
        self.out = try self.rig.output(f.outputs() * 4);
        self.base = try self.rig.scratch(f.outputs() * 4);
        return true;
    }

    /// experts.zig's launch: WideArgs in its add order, items and their count (rows kernel below 17 rows: the groups).
    fn launch(self: *Grouped) !void {
        const sh = self.fix.?.shape;
        const p = self.fix.?.pairs();
        const sk = self.fix.?.sk;
        const listed = ref.listed(self.form, sh.rows);
        const per = ref.per(self.form);
        const x0 = if (self.fault == .swapped) self.x[1] else self.x[0];
        var a: cuda.Args = .{};
        for ([_]u64{ x0, self.x[1], self.tp[0], self.tp[1], self.k2[0], self.k2[1], self.uids, if (listed) self.items else 0, self.nitems, self.members, self.out }) |v| a.add(v);
        for ([_]usize{ sh.k, sh.n, p, sk, sh.rows, ref.slots }) |v| a.add(i32s(v));
        const maxu = @min(p, ref.experts);
        const gy = if (listed) (p + per - 1) / per + maxu else maxu;
        const cols: usize = if (self.form == .wide) 64 else 128;
        const f = if (self.form == .wide) self.k.wide else self.k.rows;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(sh.n / cols), .y = @intCast(gy), .z = @intCast(ref.mats(sh) * sk) }, .block = .{ .x = 128 } }, self.rig.s, &a);
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

/// Shape: token rows x slots (8) x experts (384); short calls pad members with -1, longer ones list item runs.
fn groupSpec(control: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 8, .n = ref.slots, .k = ref.experts }, .{ .rows = 1, .n = ref.slots, .k = ref.experts }, .{ .rows = 15, .n = ref.slots, .k = ref.experts }, .{ .rows = 16, .n = ref.slots, .k = ref.experts }, .{ .rows = 17, .n = ref.slots, .k = ref.experts }, .{ .rows = 128, .n = ref.slots, .k = ref.experts }, .{ .rows = 511, .n = ref.slots, .k = ref.experts }, .{ .rows = 512, .n = ref.slots, .k = ref.experts }, .{ .rows = 2048, .n = ref.slots, .k = ref.experts } };
    const name = if (control) "inkling2 experts control: tf_ink2x_group listing 16-member runs for the prompt kernel" else "inkling2 experts tf_ink2x_group (tf_exl3_group's lists, then item runs)";
    return .{ .name = name, .accum = .none, .depth = noSums, .out = .raw, .replaces = "tf_exl3_group (main's grouping), the runs its counts give", .nan = .{ .stated = "picks outside [0, experts) are left out, as tf_exl3_group leaves them" }, .tiles = .{ .{ .size = 16 }, .{}, .{} }, .served = served };
}

fn noSums(_: c.Shape) usize {
    return 0;
}

/// tf_ink2x_group as experts.run launches it; the baseline runs tf_exl3_group and adds the runs on the host.
pub const Group = struct {
    rig: rig_.Rig,
    k: *const BaselineKernels,
    wrong_run: bool = false,
    shape: c.Shape = undefined,
    pick: []i32 = &.{},
    dev: u64 = 0,
    outs: [5]u64 = @splat(0), // uids, count, members, items, item count
    base: [3]u64 = @splat(0),
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

    fn sizes(self: *const Group) [5]usize {
        const p = self.shape.rows * ref.slots;
        return .{ self.maxu() * 4, 4, self.maxu() * self.shape.rows * 4, 3 * ref.maxItems(p) * 4, 4 };
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
        for (&self.base, self.sizes()[0..3]) |*o, n| o.* = try self.rig.scratch(n);
        return true;
    }

    pub fn run(self: *Group) !void {
        const rows = self.shape.rows;
        const listed = rows >= ref.listed_from;
        const run_: usize = if (rows >= ink.experts.wide_from and !self.wrong_run) 32 else 16;
        var b: cuda.Args = .{};
        for ([_]u64{ self.dev, self.outs[0], self.outs[1], self.outs[2], self.outs[3], self.outs[4] }) |v| b.add(v);
        for ([_]usize{ rows, ref.slots, ref.experts, rows, if (listed) run_ else 0, @intFromBool(!listed) }) |v| b.add(i32s(v));
        const shared: u32 = @intCast(ref.experts * (32 * 2 + 2 * 4));
        try cuda.launch.launch(self.k.group, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 }, .shared = shared }, self.rig.s, &b);
        try self.rig.s.synchronize();
    }

    /// tf_exl3_group's lists as written, words no launch writes as the rig poisons them, and the runs from its counts.
    pub fn replaced(self: *Group) ![]const u8 {
        const a = self.rig.gpu.gpa;
        const rows = self.shape.rows;
        var m: cuda.Args = .{};
        for ([_]u64{ self.dev, self.base[0], self.base[1], self.base[2] }) |v| m.add(v);
        for ([_]usize{ rows, ref.slots, ref.experts, rows }) |v| m.add(i32s(v));
        try cuda.launch.launch(self.k.main.group, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 }, .shared = @intCast(rows * ref.slots * 4) }, self.rig.s, &m);
        const sz = self.sizes();
        self.want.clearRetainingCapacity();
        try self.want.appendNTimes(a, 0xff, sz[0] + sz[1] + sz[2] + sz[3] + sz[4]);
        const w = self.want.items;
        const cnt = std.mem.readInt(i32, (try self.rig.bytes(self.base[1], 4))[0..4], .little);
        if (cnt < 0 or @as(usize, @intCast(cnt)) > self.maxu()) return error.BadBaselineCount;
        const used: usize = @intCast(cnt);
        @memcpy(w[0 .. used * 4], (try self.rig.bytes(self.base[0], sz[0]))[0 .. used * 4]);
        std.mem.writeInt(i32, w[sz[0]..][0..4], cnt, .little);
        const mem = w[sz[0] + 4 ..][0..sz[2]];
        @memcpy(mem[0 .. used * rows * 4], (try self.rig.bytes(self.base[2], sz[2]))[0 .. used * rows * 4]);
        if (rows >= ref.listed_from) {
            const members = try a.alloc(i32, sz[2] / 4);
            defer a.free(members);
            @memcpy(std.mem.sliceAsBytes(members), mem);
            const l: ref.Lists = .{ .uids = &.{}, .count = cnt, .members = members };
            const items = try a.alloc(i32, sz[3] / 4);
            defer a.free(items);
            const n = l.items(rows, if (rows >= ink.experts.wide_from) 32 else 16, items);
            const start = sz[0] + 4 + sz[2];
            for (items, 0..) |v, i| if (v >= 0) std.mem.writeInt(i32, w[start + 4 * i ..][0..4], v, .little);
            std.mem.writeInt(i32, w[start + sz[3] ..][0..4], n, .little);
        }
        return w;
    }

    /// Gate 1: the baseline's bytes with picks outside the experts planted.
    pub fn check(self: *Group, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from tf_exl3_group's lists and their runs", .{ at, got.len });
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
pub const CoreGrouped = struct {
    rig: rig_.Rig,
    k: *const ink.experts.Kernels,
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

    pub fn deinit(self: *CoreGrouped) void {
        self.drop();
        self.rig.deinit();
    }

    fn drop(self: *CoreGrouped) void {
        if (self.fix) |*f| f.deinit();
        self.fix = null;
        self.rig.reset();
    }

    pub fn kernel(self: *CoreGrouped) g.Kernel {
        return .{ .spec = ref.coreSpec(self.fault), .ctx = self, .vt = &vt, .control = if (self.fault == .none) null else .shapes };
    }

    const vt = g.hooks(CoreGrouped);

    fn put(self: *CoreGrouped, x: anytype) !u64 {
        return self.rig.input(std.mem.sliceAsBytes(x));
    }

    pub fn prepare(self: *CoreGrouped, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
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
        const n = l.items(shape.rows, 32, items);
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
        self.nitems = try self.put(&[1]i32{n});
        self.out = try self.rig.output(f.outputs() * 4);
        self.base = try self.rig.scratch(f.outputs() * 4);
        return true;
    }

    /// exl3_experts.runWith's launch on an item plan: GroupedArgs in standard mode, 32 members by 64 columns.
    fn launch(self: *CoreGrouped) !void {
        const sh = self.fix.?.shape;
        const p = self.fix.?.pairs();
        const sk = self.fix.?.sk;
        const m = ref.mats(sh);
        const nx = cuda.grouped.maxItems(p, ref.experts + 1, 32);
        const x0 = if (self.fault == .swapped) self.x[1] else self.x[0];
        var a: cuda.Args = .{};
        for ([_]u64{ x0, self.x[1], self.tp[0], self.tp[1], self.k2[0], self.k2[1], self.items, self.nitems, self.plan, self.out }) |v| a.add(v);
        for ([_]usize{ sh.k, sh.n, p, sk, 32, ref.slots, nx, m, 0, 1 }) |v| a.add(i32s(v));
        a.add(@as(i32, ref.experts));
        try cuda.launch.launch(self.k.main.wide, .{ .grid = .{ .x = @intCast(nx), .y = @intCast(sh.n / 64), .z = @intCast(m * sk) }, .block = .{ .x = 128 } }, self.rig.s, &a);
    }

    pub fn run(self: *CoreGrouped) !void {
        try self.launch();
        try self.rig.s.synchronize();
    }

    /// exl3_experts.Experts.run's grouped launch on the same lists: GroupedArgs, one member tile a grid z step.
    pub fn replaced(self: *CoreGrouped) ![]const u8 {
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
    pub fn check(self: *CoreGrouped, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from tf_exl3_grouped_2's", .{ at, got.len });
        for (0..got.len / 4) |i| if (c.classOf(c.read(.f32, got, i)) != .finite) return null;
        return try std.fmt.bufPrint(&self.msg, "no output is non-finite: the planted values prove nothing", .{});
    }

    fn outputs(self: *CoreGrouped) usize {
        return self.fix.?.outputs();
    }

    pub fn count(self: *CoreGrouped) usize {
        return self.outputs();
    }

    pub fn reference(self: *CoreGrouped, want: []f64, mag: []f64) !void {
        return self.fix.?.reference(want, mag);
    }

    pub fn simulate(self: *CoreGrouped, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.fix.?.simulate(acc, order, out);
    }
};
