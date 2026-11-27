//! ERNIE 5.5's MultiMax (ernie5_5/head.cu) under the contract on the head's fp32 logits, in place; two controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ernie = @import("ernie5_5_cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const mm = ernie.multimax;

/// A planted fault: each term's multiply and add fused into an fma, or the last term's square dropped.
pub const Fault = enum { none, fma, drop_square };

pub const Entry = struct { name: []const u8, symbol: [:0]const u8, fault: Fault = .none };

pub const entries = [_]Entry{
    .{ .name = "cuda ernie5_5 head tf_ernie55_multimax (fp32 logits, in place)", .symbol = "tf_ernie55_multimax" },
    .{ .name = "control: ernie5_5 MultiMax with each term's multiply and add fused", .symbol = "tf_ctl_ernie55_multimax_fma", .fault = .fma },
    .{ .name = "control: ernie5_5 MultiMax dropping the last term's square", .symbol = "tf_ctl_ernie55_multimax_drop_square", .fault = .drop_square },
};

/// Every branch reached by logits in [-48, 48); bf16 values, as the checkpoint stores them.
const coeffs: mm.Coeffs = .{ .ranges = .{ -12.5, 15.25, -20, 25 }, .ts = .{ 0.375, -0.5, 0.01953125, -0.0078125 } };
/// Gate 4's: whole ranges and power-of-two weights, so every fp32 step is exact on whole logits.
const exact_coeffs: mm.Coeffs = .{ .ranges = .{ -12, 15, -20, 25 }, .ts = .{ 0.375, -0.5, 0.0625, -0.03125 } };

const vocab = 201_216;
/// A decode row and a verify window over the whole vocabulary, and the window over a TP=2 rank's share.
const all: []const c.Shape = &.{ .{ .rows = 1, .n = vocab, .k = 1 }, .{ .rows = 16, .n = vocab, .k = 1 }, .{ .rows = 16, .n = vocab / 2, .k = 1 } };

/// A term's subtract, square and multiply, then the four adds: eight roundings at most.
fn depth(_: c.Shape) usize {
    return 8;
}

const why = "five terms a logit: no sum long enough to tell fp32 from half";

pub fn spec(e: Entry) c.Spec {
    const tiles = [3]c.Tile{ .{}, .{ .size = ernie.threads }, .{} };
    return .{ .name = e.name, .accum = .f32, .depth = depth, .out = .f32, .in_place = true, .tiles = tiles, .served = if (e.fault == .none) all else all[0..1], .sums = .{ .f16 = 0, .bf16 = 0, .why = why } };
}

pub fn control(e: Entry) ?g.Gate {
    return if (e.fault == .none) null else .shapes;
}

/// shape.rows rows of shape.n logits, rewritten in place by every run.
pub const Multimax = struct {
    rig: rig_.Rig,
    entry: Entry,
    f: cuda.Function,
    k: mm.Coeffs = coeffs,
    x: []f32 = &.{},
    out: u64 = 0,

    const vt = g.hooks(Multimax);

    pub fn kernel(self: *Multimax) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = control(self.entry) };
    }

    pub fn deinit(self: *Multimax) void {
        self.rig.gpu.gpa.free(self.x);
        self.x = &.{};
        self.rig.deinit();
    }

    /// Logits in [-48, 48); gate 1 plants NaN, +Inf and -Inf a row each and both infinities in one; gate 4's are whole.
    pub fn prepare(self: *Multimax, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        self.x = try a.alloc(f32, shape.rows * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.k = if (fill == .stress) exact_coeffs else coeffs;
        for (self.x) |*v| v.* = if (fill == .stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 96))) - 48) else (rng.float(f32) * 2 - 1) * 48;
        if (fill == .nonfinite) {
            const n = shape.n;
            const planted = [_]struct { usize, usize, f32 }{ .{ 0, n / 3, std.math.nan(f32) }, .{ 1, n / 2, std.math.inf(f32) }, .{ 2, n - 1, -std.math.inf(f32) }, .{ 3, 0, std.math.inf(f32) }, .{ 3, n - 1, -std.math.inf(f32) } };
            for (planted) |p| if (p[0] < shape.rows) {
                self.x[p[0] * n + p[1]] = p[2];
            };
        }
        self.out = try self.rig.output(self.x.len * 4);
        return true;
    }

    pub fn run(self: *Multimax) !void {
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.out, .len = self.x.len * 4 };
        try b.upload(0, std.mem.sliceAsBytes(self.x));
        try ernie.launch(self.f, self.rig.s, self.out, self.x.len, self.k);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Multimax) usize {
        return self.x.len;
    }

    pub fn reference(self: *Multimax, want: []f64, mag: []f64) !void {
        for (want, mag, self.x) |*w, *m, v| {
            const r = mm.reference(v, self.k);
            w.* = r.value;
            m.* = r.mag;
        }
    }

    /// The vendor's fp32 steps replayed on the host, unfused: each output must be that value.
    pub fn intervals(self: *Multimax, low: []f64, high: []f64) !void {
        for (low, high, self.x) |*l, *h, v| {
            l.* = mm.apply(v, self.k);
            h.* = l.*;
        }
    }

    pub fn simulate(self: *Multimax, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, self.x) |*o, v| {
            const t = mm.terms(v, self.k);
            o.* = contract.rows.sumIn(acc, order, &t);
        }
    }
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    head: ernie.Head,
    items: [entries.len]Multimax = undefined,

    /// The set, its kernels appended to `list` (each control after the kernel it breaks).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var head = try ernie.Head.load(gpu.d);
        errdefer head.unload();
        set.* = .{ .gpa = gpu.gpa, .head = head };
        for (&set.items, entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = try set.head.mod.function(e.symbol) };
        for (&set.items) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.items) |*x| x.deinit();
        set.head.unload();
        set.gpa.destroy(set);
    }
};

/// MultiMax's gates 1-4 and its controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x65726e69, &t);
    std.debug.print("SUMMARY ernie5_5 head: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "ernie5_5-head contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "every control is named one, and only the controls plant a fault" {
    for (entries) |e| try std.testing.expectEqual(e.fault != .none, std.mem.startsWith(u8, e.name, "control: "));
}

test "the coefficients are bf16 values, and gate 4's make every fp32 step exact on whole logits" {
    inline for (.{ coeffs, exact_coeffs }) |k| for (k.ranges ++ k.ts) |v| {
        try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(v)) & 0xffff);
    };
    for (0..96) |i| {
        const x: f32 = @floatFromInt(@as(i32, @intCast(i)) - 48);
        try std.testing.expectEqual(mm.reference(x, exact_coeffs).value, @as(f64, mm.apply(x, exact_coeffs)));
    }
}
