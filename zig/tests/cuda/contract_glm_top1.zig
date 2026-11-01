//! GLM's greedy pick under the contract on CUDA: tf_glm_top1_sums, its pick as tf_glm_top1's, its sum against float64.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

/// A rank's vocabulary slice: TP=2, 3 (both sizes) and 4 cut 154,880 rows in 64-row units; one launch a row.
const served = [_]c.Shape{ .{ .rows = 1, .n = 1, .k = 77440 }, .{ .rows = 1, .n = 1, .k = 51648 }, .{ .rows = 1, .n = 1, .k = 51584 }, .{ .rows = 1, .n = 1, .k = 38720 } };
const tiles = [3]c.Tile{ .{}, .{}, .{ .size = 1024 } };
const lo = 77440; // the slice's first row, as rank 1 of two passes it

const rule = "tf_glm_top1's words, and the host's scan: the first bf16 logit above every earlier one from -inf (a NaN " ++
    "never wins), its fp32 bits and lo plus its row; none above -inf gives -inf and lo";

fn noSums(_: c.Shape) usize {
    return 0;
}

/// Each thread's strided adds, then ten halvings over the block's 1,024 threads.
fn depth(s: c.Shape) usize {
    return (s.k + 1023) / 1024 + 10;
}

pub fn pickSpec() c.Spec {
    return .{ .name = "cuda glm tf_glm_top1_sums pick (tf_glm_top1's words)", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = rule }, .tiles = tiles, .served = &served };
}

pub fn sumSpec() c.Spec {
    return .{ .name = "cuda glm tf_glm_top1_sums softmax sum", .accum = .f32, .depth = depth, .out = .f32, .tiles = tiles, .served = &served };
}

const Top = struct { v: f32, i: ?usize };

/// Each launch's bf16 logits: quarter steps in [-16, 16) with -0 among the zeros, or gate 4's 0s (7 in 8) and -infs.
const Logits = struct {
    host: []u16 = &.{},
    shape: c.Shape = undefined,

    fn deinit(self: *Logits, a: std.mem.Allocator) void {
        a.free(self.host);
        self.host = &.{};
    }

    /// Non-finite: rows 0-3 hold NaN, +inf, -inf and +inf with -inf at four places each, row 4 is all -inf.
    fn fill(self: *Logits, a: std.mem.Allocator, shape: c.Shape, how: g.Fill, seed: u64) !void {
        self.deinit(a);
        self.shape = shape;
        self.host = try a.alloc(u16, shape.rows * shape.k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.host) |*v| v.* = switch (how) {
            .stress => c.toBf16(if (rng.uintLessThan(u32, 8) != 0) 0 else -std.math.inf(f32)),
            else => blk: {
                const level: i32 = @as(i32, @intCast(rng.uintLessThan(u32, 128))) - 64;
                break :blk c.toBf16(if (level == 0 and rng.boolean()) -0.0 else @as(f32, @floatFromInt(level)) * 0.25);
            },
        };
        if (how != .nonfinite) return;
        const k = shape.k;
        const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32) };
        for (put, 0..) |v, r| if (r < shape.rows) {
            for (0..4) |e| self.host[r * k + e * (k / 4)] = c.toBf16(v);
        };
        if (shape.rows > 3) self.host[3 * k + k - 1] = c.toBf16(-std.math.inf(f32));
        if (shape.rows > 4) @memset(self.host[4 * k ..][0..k], c.toBf16(-std.math.inf(f32)));
    }

    fn at(self: *const Logits, r: usize, i: usize) f32 {
        return c.fromBf16(self.host[r * self.shape.k + i]);
    }

    /// The host's scan of row `r`: from -inf, each value above the best so far becomes it, so a NaN never wins.
    fn top(self: *const Logits, r: usize) Top {
        var t: Top = .{ .v = -std.math.inf(f32), .i = null };
        for (0..self.shape.k) |i| {
            const x = self.at(r, i);
            if (x > t.v) t = .{ .v = x, .i = i };
        }
        return t;
    }

    /// Row `r`'s terms exp(x - max) in float64, the max as the pick finds it.
    fn terms(self: *const Logits, r: usize, t: []f64) void {
        const m: f64 = self.top(r).v;
        for (t, 0..) |*o, i| o.* = @exp(@as(f64, self.at(r, i)) - m);
    }
};

/// One launch a row as cuda_engine_pick makes it (a 1,024-thread block), at row `r`'s logits and four words.
fn launch(rig: *rig_.Rig, f: cuda.Function, in: u64, out: u64, shape: c.Shape) !void {
    for (0..shape.rows) |r| {
        var a: cuda.Args = .{};
        inline for (.{ in + @as(u64, r * shape.k * 2), @as(i64, @intCast(shape.k)), @as(u32, lo), out + @as(u64, r * 16) }) |v| a.add(v);
        try cuda.launch.launch(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, rig.s, &a);
    }
    try rig.s.synchronize();
}

/// Bytes `from` to `from + len` of each launch's 16, gathered into `list`.
fn words(list: *std.ArrayList(u8), a: std.mem.Allocator, all: []const u8, from: usize, len: usize) ![]const u8 {
    list.clearRetainingCapacity();
    var i: usize = 0;
    while (i < all.len) : (i += 16) try list.appendSlice(a, all[i + from ..][0..len]);
    return list.items;
}

/// The words the gates read in place of the rig's whole output.
fn withOutput(comptime T: type) g.VTable {
    var v = g.hooks(T);
    v.output = struct {
        fn f(ctx: *anyopaque) anyerror![]const u8 {
            return @as(*T, @ptrCast(@alignCast(ctx))).view();
        }
    }.f;
    return v;
}

/// tf_glm_top1_sums' first two words a launch, the pick; tf_glm_top1's on the same logits must equal them.
pub const Pick = struct {
    rig: rig_.Rig,
    sums: cuda.Function,
    plain: cuda.Function,
    logits: Logits = .{},
    in: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    got: std.ArrayList(u8) = .empty,
    want: std.ArrayList(u8) = .empty,
    msg: [160]u8 = undefined,

    const vt = withOutput(Pick);

    pub fn deinit(self: *Pick) void {
        const a = self.rig.gpu.gpa;
        self.logits.deinit(a);
        self.got.deinit(a);
        self.want.deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *Pick) g.Kernel {
        return .{ .spec = pickSpec(), .ctx = self, .vt = &vt };
    }

    pub fn prepare(self: *Pick, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k == 0 or shape.rows == 0) return false;
        self.rig.reset();
        try self.logits.fill(self.rig.gpu.gpa, shape, fill, seed);
        self.in = try self.rig.input(std.mem.sliceAsBytes(self.logits.host));
        self.out = try self.rig.output(shape.rows * 16);
        self.base = try self.rig.scratch(shape.rows * 16);
        return true;
    }

    pub fn run(self: *Pick) !void {
        try launch(&self.rig, self.sums, self.in, self.out, self.logits.shape);
    }

    pub fn view(self: *Pick) ![]const u8 {
        return words(&self.got, self.rig.gpu.gpa, try self.rig.outputBytes(), 0, 8);
    }

    pub fn count(self: *Pick) usize {
        return self.logits.shape.rows * 8;
    }

    pub fn reference(_: *Pick, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// tf_glm_top1's words on the same logits, then every row's pick against the host's scan.
    pub fn check(self: *Pick, _: g.Gate) !?[]const u8 {
        const got = try self.view();
        try launch(&self.rig, self.plain, self.in, self.base, self.logits.shape);
        const want = try words(&self.want, self.rig.gpu.gpa, try self.rig.bytes(self.base, self.logits.shape.rows * 16), 0, 8);
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from tf_glm_top1's", .{ at, got.len });
        for (0..self.logits.shape.rows) |r| {
            const t = self.logits.top(r);
            const v = std.mem.readInt(u32, got[r * 8 ..][0..4], .little);
            const id = std.mem.readInt(u32, got[r * 8 + 4 ..][0..4], .little);
            const due: u32 = lo + @as(u32, @intCast(t.i orelse 0));
            if (v != @as(u32, @bitCast(t.v)) or id != due) return try std.fmt.bufPrint(&self.msg, "row {d}: {x} at {d}, the host's scan gives {x} at {d}", .{ r, v, id, @as(u32, @bitCast(t.v)), due });
        }
        return null;
    }
};

/// tf_glm_top1_sums' third word a launch: the slice's sum of exp(logit - max), against float64 at the same max.
pub const Sum = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    logits: Logits = .{},
    in: u64 = 0,
    out: u64 = 0,
    got: std.ArrayList(u8) = .empty,

    const vt = withOutput(Sum);

    pub fn deinit(self: *Sum) void {
        const a = self.rig.gpu.gpa;
        self.logits.deinit(a);
        self.got.deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *Sum) g.Kernel {
        return .{ .spec = sumSpec(), .ctx = self, .vt = &vt };
    }

    pub fn prepare(self: *Sum, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k == 0 or shape.rows == 0) return false;
        self.rig.reset();
        try self.logits.fill(self.rig.gpu.gpa, shape, fill, seed);
        self.in = try self.rig.input(std.mem.sliceAsBytes(self.logits.host));
        self.out = try self.rig.output(shape.rows * 16);
        return true;
    }

    pub fn run(self: *Sum) !void {
        try launch(&self.rig, self.f, self.in, self.out, self.logits.shape);
    }

    pub fn view(self: *Sum) ![]const u8 {
        return words(&self.got, self.rig.gpu.gpa, try self.rig.outputBytes(), 8, 4);
    }

    pub fn count(self: *Sum) usize {
        return self.logits.shape.rows;
    }

    pub fn reference(self: *Sum, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        const t = try a.alloc(f64, self.logits.shape.k);
        defer a.free(t);
        for (want, mag, 0..) |*w, *m, r| {
            self.logits.terms(r, t);
            var s: f64 = 0;
            for (t) |x| s += x;
            w.* = s;
            m.* = if (std.math.isFinite(s)) s else 0;
        }
    }

    /// The generic bound plus __expf's error: under 2 + 1.2 |x| ulps of exp(x) by CUDA's intrinsic table, none at 0.
    pub fn bounds(self: *Sum, _: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, 0..) |*l, r| {
            const m: f64 = self.logits.top(r).v;
            for (0..self.logits.shape.k) |i| {
                const x = @as(f64, self.logits.at(r, i)) - m;
                if (std.math.isFinite(x) and x != 0) l.* += (2 + 1.2 * @abs(x)) * 0x1p-23 * @exp(x);
            }
        }
    }

    /// Each row's terms summed in `acc` in `order` (gate 4).
    pub fn simulate(self: *Sum, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const t = try a.alloc(f64, self.logits.shape.k);
        defer a.free(t);
        for (out, 0..) |*o, r| {
            self.logits.terms(r, t);
            o.* = rows.sumIn(acc, order, t);
        }
    }
};

/// The pick and the sum on the heap, so the kernels' contexts keep their address; registered into `kernels`.
pub const Top1 = struct {
    pick: Pick,
    sum: Sum,

    pub fn create(a: std.mem.Allocator, gpu: @import("check.zig").Gpu, s: cuda.Stream, fns: glm.glue.Kernels, kernels: *std.ArrayList(g.Kernel)) !*Top1 {
        const self = try a.create(Top1);
        errdefer a.destroy(self);
        self.* = .{ .pick = .{ .rig = .{ .gpu = gpu, .s = s }, .sums = fns.top1_sums, .plain = fns.top1 }, .sum = .{ .rig = .{ .gpu = gpu, .s = s }, .f = fns.top1_sums } };
        try kernels.append(a, self.pick.kernel());
        try kernels.append(a, self.sum.kernel());
        return self;
    }

    pub fn destroy(self: *Top1, a: std.mem.Allocator) void {
        self.sum.deinit();
        self.pick.deinit();
        a.destroy(self);
    }
};
