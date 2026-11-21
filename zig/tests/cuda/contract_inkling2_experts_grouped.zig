//! Core's tf_exl3_grouped_wide_2 on Inkling 2's prompt chunks: bit-equal to tf_exl3_grouped_2 on the same members.
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

/// One projection as exl3_experts.runWith launches it on 32-member items; the baseline runs tf_exl3_grouped_2.
pub const Grouped = struct {
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
        return .{ .spec = ref.spec(self.fault), .ctx = self, .vt = &vt, .control = if (self.fault == .none) null else .shapes };
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
    fn launch(self: *Grouped) !void {
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
