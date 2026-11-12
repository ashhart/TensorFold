//! The TP prefill launches' host cases through the real gates: the shipped emulation passes, faults go red.
const std = @import("std");
const core = @import("core");
const flash = @import("flashnext");
const ref = @import("contract_flashnext_exl3_tppf_ref.zig");
const rows_ref = @import("contract_flashnext_exl3_tppf_rows_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const canon = flash.exl3_canon;
const t = std.testing;

/// A device-less rig: the buffer the emulated launch writes and the inputs' bytes as the last launch left them.
const FakeRig = struct {
    out: []u8 = &.{},
    seen: std.ArrayList(u8) = .empty,

    pub fn outputBytes(r: *FakeRig) ![]const u8 {
        return r.out;
    }

    pub fn inputBytes(r: *FakeRig) ![]const u8 {
        return r.seen.items;
    }

    pub fn intact(_: *FakeRig) !bool {
        return true;
    }

    pub fn overlap(_: *FakeRig) bool {
        return false;
    }
};

fn Fake(comptime Case: type) type {
    return struct {
        const Self = @This();
        rig: FakeRig = .{},
        case: Case,
        small: []const c.Shape,

        fn deinit(self: *Self) void {
            t.allocator.free(self.rig.out);
            self.rig.seen.deinit(t.allocator);
            self.case.deinit();
        }

        fn snapshot(self: *Self) !void {
            self.rig.seen.clearRetainingCapacity();
            for (self.case.inputs()) |bytes| try self.rig.seen.appendSlice(t.allocator, bytes);
        }

        const vt = g.hooks(Self);

        fn kernel(self: *Self) g.Kernel {
            var k: g.Kernel = .{ .spec = self.case.spec(), .ctx = self, .vt = &vt, .control = ref.controlGate(Case, self.case.fault) };
            k.spec.served = self.small;
            return k;
        }

        pub fn prepare(self: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            if (!try self.case.prepare(shape, fill, seed)) return false;
            var total: usize = 0;
            for (self.case.outputLens()) |len| total += len;
            t.allocator.free(self.rig.out);
            self.rig.out = try t.allocator.alloc(u8, total);
            @memset(self.rig.out, 0xff);
            try self.snapshot();
            return true;
        }

        pub fn run(self: *Self) !void {
            self.case.emulate(self.case.fault, self.rig.out);
            try self.snapshot();
        }

        pub fn count(self: *Self) usize {
            return self.case.count();
        }

        pub fn reference(self: *Self, want: []f64, mag: []f64) !void {
            return self.case.reference(want, mag);
        }

        pub fn simulate(self: *Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            return self.case.simulate(acc, order, out);
        }

        pub fn check(self: *Self, gate: g.Gate) !?[]const u8 {
            return self.case.verify(self.rig.out, gate);
        }
    };
}

/// The shipped case passes gates 1 to 4 (or N/A); a control goes red at its own gate.
fn gates(comptime Case: type, case: Case, small: []const c.Shape) !void {
    var f: Fake(Case) = .{ .case = case, .small = small };
    defer f.deinit();
    const k = f.kernel();
    g.quiet = true;
    defer g.quiet = false;
    if (ref.controlGate(Case, case.fault)) |gate| {
        const v = try g.one(t.allocator, gate, k, 0x51);
        if (v != .fail) std.debug.print("control {s} stayed {t} at {t}\n", .{ k.spec.name, v, gate });
        return t.expectEqual(g.Verdict.fail, v);
    }
    for ([_]g.Gate{ .nonfinite, .repeat, .shapes, .sums }) |gate| {
        const v = try g.one(t.allocator, gate, k, 0x51);
        if (v == .fail) {
            g.quiet = false;
            _ = try g.one(t.allocator, gate, k, 0x51);
            return error.TestUnexpectedResult;
        }
    }
}

const w = 256;
const combine_small = [_]c.Shape{ .{ .rows = 5, .n = w, .k = 2 }, .{ .rows = 1, .n = w, .k = 1 }, .{ .rows = 17, .n = w, .k = 3 }, .{ .rows = 40, .n = w, .k = 5 }, .{ .rows = 6, .n = w, .k = 8 } };
const blocks_small = [_]c.Shape{ .{ .rows = 300, .n = w, .k = 2 }, .{ .rows = 700, .n = w, .k = 5 }, .{ .rows = 257, .n = w, .k = 8 } };
const finish_small = [_]c.Shape{ .{ .rows = 5, .n = w, .k = 2 }, .{ .rows = 44, .n = w, .k = 2 }, .{ .rows = 7, .n = w, .k = 3 }, .{ .rows = 9, .n = w, .k = 5 }, .{ .rows = 6, .n = w, .k = 8 } };
const rotate_small = [_]c.Shape{ .{ .rows = 5, .n = w, .k = 128 }, .{ .rows = 1, .n = w, .k = 128 }, .{ .rows = 30, .n = w, .k = 128 } };
const place_small = [_]c.Shape{ .{ .rows = 5, .n = w, .k = 0 }, .{ .rows = 1, .n = w, .k = 0 }, .{ .rows = 130, .n = w, .k = 0 } };

test "tppf contract: combine passes gates 1 to 3 (4 is N/A) in both region layouts and its five controls go red" {
    for ([_]bool{ false, true }) |blocks| {
        const small: []const c.Shape = if (blocks) &blocks_small else &combine_small;
        try gates(ref.Combine, .{ .gpa = t.allocator, .blocks = blocks }, small);
    }
    for ([_]ref.Fault{ .flat, .nan_dropped, .writes_input, .bf16_adds, .f32_partials }) |f| try gates(ref.Combine, .{ .gpa = t.allocator, .blocks = false, .fault = f }, &combine_small);
}

test "tppf contract: finish passes gates 1 to 4 and its four controls go red" {
    try gates(ref.Finish, .{ .gpa = t.allocator }, &finish_small);
    for ([_]ref.Fault{ .flat, .bf16_adds, .writes_input, .f32_partials }) |f| try gates(ref.Finish, .{ .gpa = t.allocator, .fault = f }, &finish_small);
}

test "tppf contract: place passes gates 1 to 3 (4 is N/A) and its three controls go red" {
    try gates(rows_ref.Place, .{ .gpa = t.allocator }, &place_small);
    for ([_]ref.Fault{ .nan_dropped, .writes_input, .wrong }) |f| try gates(rows_ref.Place, .{ .gpa = t.allocator, .fault = f }, &place_small);
}

test "tppf contract: rotate passes gates 1 to 4 and its four controls go red" {
    try gates(rows_ref.Rotate, .{ .gpa = t.allocator }, &rotate_small);
    for ([_]ref.Fault{ .nan_dropped, .writes_input, .wrong, .bf16_adds }) |f| try gates(rows_ref.Rotate, .{ .gpa = t.allocator, .fault = f }, &rotate_small);
}

/// Every shape the registration lists and its edges are ones the case takes; gates 1, 2 and 4 find their shapes in it.
fn takes(comptime Case: type, case: Case, ranked: bool) !void {
    const s = case.spec();
    const list = try c.shapes(t.allocator, s);
    defer t.allocator.free(list);
    for (list) |x| try t.expect(Case.accepts(x));
    var widest: usize = 0;
    for (s.served) |x| widest = @max(widest, x.k);
    if (!ranked) return;
    try t.expectEqual(@as(usize, 2), s.served[0].k);
    try t.expectEqual(@as(usize, 8), widest);
}

test "tppf contract: every served and edge shape of every registration is one its case takes" {
    try takes(ref.Combine, .{ .gpa = t.allocator, .blocks = false }, true);
    try takes(ref.Combine, .{ .gpa = t.allocator, .blocks = true }, true);
    try takes(ref.Finish, .{ .gpa = t.allocator }, true);
    try takes(rows_ref.Rotate, .{ .gpa = t.allocator }, false);
    try takes(rows_ref.Place, .{ .gpa = t.allocator }, false);
}

fn randomRow(rng: std.Random, y: *[ref.slots]f32, wt: *[ref.slots]f32, p: *[ref.slots]u16) void {
    for (0..ref.slots) |k| {
        y[k] = rng.floatNorm(f32) * 0.05;
        wt[k] = rng.floatNorm(f32) * 0.2;
        p[k] = if (k == ref.slots - 1) 512 else if (rng.boolean()) rng.uintLessThan(u16, 512) else 256 + rng.uintLessThan(u16, 24);
    }
}

test "tppf contract: the block oracle is canon.subtree, and a masked slot is the slot with weight 0" {
    var prng = std.Random.DefaultPrng.init(0x7b10c);
    const rng = prng.random();
    for (0..400) |_| {
        var y: [ref.slots]f32 = undefined;
        var wt: [ref.slots]f32 = undefined;
        var pu: [ref.slots]u16 = undefined;
        randomRow(rng, &y, &wt, &pu);
        var p: [ref.slots]i32 = undefined;
        for (&p, pu) |*d, v| d.* = v;
        for (1..9) |world| for (0..world) |rank| {
            var bl: [canon.max_blocks]canon.Block = undefined;
            const nb = ref.rankBlocks(&bl, world, rank);
            for (bl[0..nb]) |b| try t.expectEqual(@as(u64, @bitCast(canon.subtree(&y, &wt, &pu, b.start, b.log))), @as(u64, @bitCast(ref.blockSum(&y, &wt, &p, b.start, b.log, .none))));
        };
        const k = rng.uintLessThan(usize, ref.slots);
        var masked = p;
        masked[k] = -1;
        var zeroed = wt;
        zeroed[k] = 0;
        try t.expectEqual(@as(u64, @bitCast(canon.subtree(&y, &zeroed, &pu, 0, canon.leaf_log))), @as(u64, @bitCast(ref.blockSum(&y, &wt, &masked, 0, canon.leaf_log, .none))));
    }
}

/// Every row's slots 0 to 3 hold the products 1.25e8, 1, -1.25e8, 1 on leaves `at`, in every column; the other slots are masked.
fn cancelling(cb: *ref.Combine, at: [4]usize) void {
    const y = [4]f32{ 1e9, 8, -1e9, 8 };
    const n = cb.shape.n;
    for (0..cb.shape.rows) |r| for (0..ref.slots) |k| {
        cb.p[r * ref.slots + k] = if (k < 4) @intCast(at[k] * canon.leaf_ids) else -1;
        cb.w[r * ref.slots + k] = 0.125;
        for (0..n) |col| cb.y[(r * ref.slots + k) * n + col] = if (k < 4) y[k] else 0;
    };
}

/// Combine's gather buffer of `world` ranks, each region finished: the rows the finish gives. `cancel` plants the cancelling
/// products of `cancelling`; `fault` is a planted combine fault.
fn gathered(world: usize, blocks: bool, shape: c.Shape, seed: u64, out: []f32, cancel: ?[4]usize, fault: ref.Fault) !void {
    var cb: ref.Combine = .{ .gpa = t.allocator, .blocks = blocks };
    defer cb.deinit();
    var sh = shape;
    sh.k = world;
    try t.expect(try cb.prepare(sh, .random, seed));
    if (cancel) |at| cancelling(&cb, at);
    const gath = try t.allocator.alloc(f64, cb.count());
    defer t.allocator.free(gath);
    cb.emulate(fault, std.mem.sliceAsBytes(gath));
    var at: usize = 0;
    var r0: usize = 0;
    while (r0 < sh.rows) : (r0 += cb.sub) {
        const rows = @min(cb.sub, sh.rows - r0);
        const region = rows * world * cb.m * sh.n;
        var fin: ref.Finish = .{ .gpa = t.allocator };
        defer fin.deinit();
        fin.shape = .{ .rows = rows, .n = sh.n, .k = world };
        fin.m = cb.m;
        fin.np = canon.pieces(&fin.ps, world);
        fin.vals = gath[at..][0..region];
        at += region;
        fin.emulate(.none, std.mem.sliceAsBytes(out[r0 * sh.n ..][0 .. rows * sh.n]));
    }
}

test "tppf contract: the ranks' partials gathered and finished equal the single rank's tree at every degree" {
    const shape: c.Shape = .{ .rows = 300, .n = w, .k = 1 };
    const want = try t.allocator.alloc(f32, shape.rows * w);
    defer t.allocator.free(want);
    const got = try t.allocator.alloc(f32, shape.rows * w);
    defer t.allocator.free(got);
    for ([_]bool{ false, true }) |blocks| {
        try gathered(1, blocks, shape, 0x9a7, want, null, .none);
        for (2..9) |world| {
            try gathered(world, blocks, shape, 0x9a7, got, null, .none);
            for (want, got, 0..) |a, b, i| if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) {
                std.debug.print("world {d} blocks {}: output {d}: {x:0>8} for {x:0>8}\n", .{ world, blocks, i, @as(u32, @bitCast(b)), @as(u32, @bitCast(a)) });
                return error.TestUnexpectedResult;
            };
        }
    }
}

test "tppf contract: cancelling products keep their +1s at every degree: the f64 partials finish to exactly 2" {
    // 1.25e8 + 1 is not an f32 (spacing 8): the f32 tree rounded the +1s away and gave 0 where the exact sum is 2.
    const spread = [_][4]usize{ .{ 0, 1, 2, 3 }, .{ 0, 21, 42, 63 }, .{ 0, 9, 40, 62 } };
    for ([_]bool{ false, true }) |blocks| {
        // the sub-block layout (257 rows: a full region and a one-row one) takes the two fixtures that split the cancellation over ranks
        const first: usize = if (blocks) 1 else 0;
        const shape: c.Shape = .{ .rows = if (blocks) 257 else 3, .n = w, .k = 1 };
        const got = try t.allocator.alloc(f32, shape.rows * w);
        defer t.allocator.free(got);
        for (spread[first..]) |at| for (1..9) |world| {
            try gathered(world, blocks, shape, 0x9a7, got, at, .none);
            for (got, 0..) |v, i| if (v != 2) {
                std.debug.print("world {d} blocks {} leaves {any}: output {d} is {d}, not 2\n", .{ world, blocks, at, i, v });
                return error.TestUnexpectedResult;
            };
        };
    }
}

test "tppf contract: a partial carried through f32 loses what the f64 partial keeps" {
    // Two ranks: rank 0 holds the 1.25e8 and a +1, rank 1 the -1.25e8 and a +1. Each partial is +-1.25e8 +- 1 exactly in f64;
    // rounded to f32 both lose the 1 and the finish gives 0 where it gives 2.
    const shape: c.Shape = .{ .rows = 3, .n = w, .k = 1 };
    const got = try t.allocator.alloc(f32, shape.rows * w);
    defer t.allocator.free(got);
    try gathered(2, false, shape, 0x9a7, got, .{ 0, 21, 42, 63 }, .none);
    for (got) |v| try t.expectEqual(@as(f32, 2), v);
    try gathered(2, false, shape, 0x9a7, got, .{ 0, 21, 42, 63 }, .f32_partials);
    for (got) |v| try t.expectEqual(@as(f32, 0), v);
}

test "tppf contract: rotate's float64 transform is the signed sum of the products" {
    var r: rows_ref.Rotate = .{ .gpa = t.allocator };
    defer r.deinit();
    try t.expect(try r.prepare(.{ .rows = 3, .n = w, .k = 128 }, .random, 0x5eed));
    const n = r.count();
    const buf = try t.allocator.alloc(f64, 3 * n);
    defer t.allocator.free(buf);
    try r.reference(buf[0..n], buf[n .. 2 * n]);
    try r.simulate(.none, .sequential, buf[2 * n ..]);
    for (buf[0..n], buf[2 * n ..]) |a, b| try t.expect(@abs(a - b) <= 1e-12 * @max(1, @abs(a)));
}
