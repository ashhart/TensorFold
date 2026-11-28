//! ERNIE 5.5's router (ernie5_5/moe.cu) under the contract: each row's picks and weights by route.zig's rule.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ernie = @import("ernie5_5_moe_cuda");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rt = ernie.route;
const ne = rt.experts;

/// A planted fault (control entries): one sigmoid of the two logits summed, or weights from score + bias.
pub const Fault = enum { none, one_sigmoid, biased_weights };

pub const Entry = struct { name: []const u8, symbol: [:0]const u8, fault: Fault = .none };

pub const entries = [_]Entry{
    .{ .name = "cuda ernie5_5 moe tf_ernie55_route (two sigmoid scores, top 10 of 512 by score + bias, scaled weights)", .symbol = "tf_ernie55_route" },
    .{ .name = "control: ernie5_5 route scoring one sigmoid of the two logits summed", .symbol = "tf_ctl_ernie55_route_one_sigmoid", .fault = .one_sigmoid },
    .{ .name = "control: ernie5_5 route weighing by score + bias", .symbol = "tf_ctl_ernie55_route_biased_weights", .fault = .biased_weights },
};

/// rows, experts (n), picks (k): a verify window, a decode row, a prompt chunk.
pub const served = &[_]c.Shape{ .{ .rows = 16, .n = ne, .k = rt.topk }, .{ .rows = 1, .n = ne, .k = rt.topk }, .{ .rows = 300, .n = ne, .k = rt.topk } };

fn noSums(_: c.Shape) usize {
    return 0;
}

const rule = "picks by fp32 sigmoid(l0) + sigmoid(l1) + bias, ties to the lower id, a NaN never picked, ids ascending; weights the score's";

pub fn spec(e: Entry) c.Spec {
    return .{ .name = e.name, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = rule }, .served = if (e.fault == .none) served else served[0..1] };
}

pub const Route = struct {
    rig: rig_.Rig,
    entry: Entry,
    f: cuda.Function,
    shape: c.Shape = undefined,
    logits: []f32 = &.{},
    bias: [ne]f32 = undefined,
    scale: [ne]f32 = undefined,
    dev: [3]u64 = @splat(0), // logits, bias, scale
    ids: u64 = 0,
    wts: u64 = 0,
    msg: [200]u8 = undefined,

    const vt = g.hooks(Route);

    pub fn kernel(self: *Route) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.fault == .none) null else .shapes };
    }

    fn free(self: *Route) void {
        self.rig.gpu.gpa.free(self.logits);
        self.logits = &.{};
    }

    pub fn deinit(self: *Route) void {
        self.free();
        self.rig.deinit();
    }

    fn sh(self: *const Route) rt.Shape {
        return .{ .rows = self.shape.rows, .k = self.shape.k };
    }

    /// Logits from N(0, 2), every 7th expert copying its neighbour in both routers (ties); bias 0.25 on every 5th.
    pub fn prepare(self: *Route, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != ne or shape.k == 0 or shape.k > rt.most_topk) return false;
        self.rig.reset();
        self.free();
        self.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.logits = try self.rig.gpu.gpa.alloc(f32, shape.rows * 2 * ne);
        for (0..shape.rows) |r| {
            const l = self.logits[r * 2 * ne ..][0 .. 2 * ne];
            for (l) |*x| x.* = rng.floatNorm(f32) * 2;
            for (0..ne) |e| if (e % 7 == 3) {
                l[e] = l[e - 1];
                l[ne + e] = l[ne + e - 1];
            };
        }
        for (&self.bias, &self.scale, 0..) |*b, *s, e| {
            b.* = if (e % 5 == 0) 0.25 else 0;
            s.* = 0.5 + 2 * rng.float(f32);
        }
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, 3 }, .{ 0, ne + 9 }, .{ 1, 5 }, .{ 2, 7 }, .{ 3, ne + 2 } };
            const put = [_]f32{ std.math.nan(f32), std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32) };
            for (at, put) |p, v| if (p[0] < shape.rows) {
                self.logits[p[0] * 2 * ne + p[1]] = v;
            };
        }
        self.dev = .{ try self.rig.input(std.mem.sliceAsBytes(self.logits)), try self.rig.input(std.mem.sliceAsBytes(&self.bias)), try self.rig.input(std.mem.sliceAsBytes(&self.scale)) };
        self.ids = try self.rig.output(shape.rows * shape.k * 4);
        self.wts = try self.rig.output(shape.rows * shape.k * 4);
        return true;
    }

    pub fn run(self: *Route) !void {
        try ernie.launch(self.f, self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.ids, self.wts, self.sh());
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Route) usize {
        return self.shape.rows * self.shape.k * 8;
    }

    pub fn reference(_: *Route, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    fn fail(self: *Route, comptime fmt: []const u8, args: anytype) !?[]const u8 {
        return try std.fmt.bufPrint(&self.msg, fmt, args);
    }

    /// Ids ascending, ranked as route.rank (near-ties may swap, exact ties to the lower id); weights within 2^-20.
    pub fn check(self: *Route, _: g.Gate) !?[]const u8 {
        const k = self.shape.k;
        const got = try self.rig.outputBytes();
        const n = self.shape.rows * k;
        var taken: [ne]bool = undefined;
        var ids: [rt.most_topk]u32 = undefined;
        var w: [rt.most_topk]f64 = undefined;
        for (0..self.shape.rows) |r| {
            const l = self.logits[r * 2 * ne ..][0 .. 2 * ne];
            @memset(&taken, false);
            for (ids[0..k], 0..) |*id, j| {
                id.* = std.mem.readInt(u32, got[(r * k + j) * 4 ..][0..4], .little);
                if (id.* >= ne or (j > 0 and id.* <= ids[j - 1])) return self.fail("row {d} pick {d}: expert {d}, out of range or not ascending", .{ r, j, id.* });
                taken[id.*] = true;
            }
            var low = std.math.inf(f64); // the lowest picked rank
            var low_id: usize = 0;
            var high = -std.math.inf(f64); // the highest unpicked
            var high_id: usize = 0;
            for (0..ne) |e| {
                const v = rt.rank(l, &self.bias, e);
                if (taken[e] and v < low) low = v;
                if (taken[e] and v <= low) low_id = e;
                if (!taken[e] and v > high) {
                    high = v;
                    high_id = e;
                }
            }
            if (low < high - 0x1p-20 * (@abs(high) + 1)) return self.fail("row {d}: expert {d} ranks {d:.8}, unpicked expert {d} {d:.8}", .{ r, low_id, low, high_id, high });
            for (ids[0..k]) |id| for (0..@as(usize, id)) |e| if (!taken[e] and rt.rank(l, &self.bias, e) == rt.rank(l, &self.bias, id))
                return self.fail("row {d}: expert {d} picked ahead of tied expert {d}", .{ r, id, e });
            rt.weights(l, &self.scale, ids[0..k], w[0..k]);
            for (w[0..k], 0..) |want, j| {
                const v: f64 = @as(f32, @bitCast(std.mem.readInt(u32, got[(n + r * k + j) * 4 ..][0..4], .little)));
                if (!(@abs(v - want) <= @abs(want) * 0x1p-20 + 1e-30)) return self.fail("row {d} slot {d}: weight {d:.8} where float64 gives {d:.8}", .{ r, j, v, want });
            }
        }
        return null;
    }
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    moe: ernie.Moe,
    items: [entries.len]Route = undefined,

    /// The set, its kernels appended to `list` (each control after the kernel it breaks).
    pub fn init(gpu: check_.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var m = try ernie.Moe.load(gpu.d);
        errdefer m.unload();
        set.* = .{ .gpa = gpu.gpa, .moe = m };
        for (&set.items, entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = try set.moe.mod.function(e.symbol) };
        for (&set.items) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.items) |*x| x.deinit();
        set.moe.unload();
        set.gpa.destroy(set);
    }
};

/// The MoE image's gates 1-4 and its controls alone.
pub fn run(gpu: check_.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x6d6f65, &t);
    std.debug.print("SUMMARY ernie5_5 moe: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed });
    try check_.expect(t.problems() == 0, "ernie5_5-moe contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "every control is named one, and only the controls plant a fault" {
    for (entries) |e| try std.testing.expectEqual(e.fault != .none, std.mem.startsWith(u8, e.name, "control: "));
}

test "the served shapes fit the kernel at the released width and top-k" {
    for (served) |x| {
        try rt.check(.{ .rows = x.rows, .k = x.k });
        try std.testing.expectEqual(@as(usize, ne), x.n);
    }
}
