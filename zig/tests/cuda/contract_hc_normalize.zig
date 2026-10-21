//! fn_hc_normalize_cols under the contract: fn_hc_normalize's bits; control: a doubled epsilon posing as it.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const contract = core.contract;

pub const Kind = enum { cols, control };

const rule = "as fn_hc_normalize gives them: NaN and infinite rows, a NaN stream sum, a zero sum (infinite inverse)";
const eps: f32 = 1e-6;

/// The stream sum (nc fp32 adds), its root and inverse, two products and the bf16 rounding.
fn depth(s: c.Shape) usize {
    return s.n / 256 + 6;
}

pub fn spec(kind: Kind) c.Spec {
    return .{
        .name = if (kind == .cols) "cuda Flash Next fn_hc_normalize_cols (replaces fn_hc_normalize)" else "control: fn_hc_normalize_cols with a doubled epsilon posing as fn_hc_normalize",
        .accum = .f32,
        .depth = depth,
        .ops = 4,
        .out = .bf16,
        .replaces = "fn_hc_normalize",
        .nan = .{ .stated = rule },
        .tiles = .{ .{ .size = 1 }, .{ .size = 256, .step = 256, .min = 256 }, .{} },
        .served = &.{ .{ .rows = 1, .n = 2560, .k = 4 }, .{ .rows = 17, .n = 2560, .k = 4 }, .{ .rows = 3, .n = 512, .k = 4 } },
        // the only sum is nc fp32 partials; through the root a bf16 sum's error falls under the output's rounding
        .sums = .{ .f16 = 100, .bf16 = 0, .why = "a bf16 sum of the stream partials loses less than the bf16 output rounding after the root" },
    };
}

pub const Normalize = struct {
    rig: rig_.Rig,
    kind: Kind,
    cols_fn: cuda.Function,
    old_fn: cuda.Function,
    arena: ?std.heap.ArenaAllocator = null,
    shape: c.Shape = undefined,
    h: []u16 = &.{},
    pss: []f32 = &.{},
    scale: []f32 = &.{},
    dh: u64 = 0,
    dpss: u64 = 0,
    dscale: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Normalize) void {
        if (self.arena) |*arena| arena.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(Normalize);

    pub fn kernel(self: *Normalize) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt, .control = if (self.kind == .control) .shapes else null };
    }

    fn streams(self: *const Normalize) usize {
        return self.shape.k;
    }

    pub fn prepare(self: *Normalize, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 64 or shape.n == 0 or shape.n % 256 != 0 or shape.k == 0 or shape.k > 8) return false;
        self.rig.reset();
        if (self.arena) |*arena| arena.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        self.shape = shape;
        const nc = shape.n / 256;
        const wide = shape.n * shape.k;
        self.h = try a.alloc(u16, shape.rows * wide);
        self.pss = try a.alloc(f32, shape.rows * nc * shape.k);
        self.scale = try a.alloc(f32, wide);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.h) |*b| b.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (self.scale) |*v| v.* = 0.5 + rng.float(f32);
        for (self.pss) |*v| v.* = if (fill == .stress) @floatFromInt(65536 + rng.uintLessThan(u32, 65536)) else 0.5 + rng.float(f32) * 256;
        if (fill == .nonfinite) {
            self.h[0] = c.toBf16(std.math.nan(f32));
            if (self.h.len > 2) self.h[1] = c.toBf16(std.math.inf(f32));
            self.pss[self.pss.len - 1] = std.math.nan(f32);
            if (shape.rows > 1) @memset(self.pss[0 .. nc * shape.k], 0);
        }
        self.dh = try self.rig.input(std.mem.sliceAsBytes(self.h));
        self.dpss = try self.rig.input(std.mem.sliceAsBytes(self.pss));
        self.dscale = try self.rig.input(std.mem.sliceAsBytes(self.scale));
        self.out = try self.rig.output(shape.rows * wide * 2);
        self.base = try self.rig.scratch(shape.rows * wide * 2);
        return true;
    }

    fn launch(self: *Normalize, cols: bool, e: f32, out: u64) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ self.dh, self.dpss, self.dscale, out }) |v| args.add(v);
        args.add(e);
        args.add(@as(c_int, @intCast(self.shape.n)));
        args.add(@as(c_int, @intCast(self.streams())));
        const grid: cuda.launch.Dim3 = if (cols) .{ .x = @intCast(self.shape.rows), .y = @intCast(self.streams()), .z = @intCast(self.shape.n / 256) } else .{ .x = @intCast(self.shape.rows), .y = @intCast(self.streams()) };
        try cuda.launch.launch(if (cols) self.cols_fn else self.old_fn, .{ .grid = grid, .block = .{ .x = 256 } }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Normalize) !void {
        try self.launch(true, if (self.kind == .control) 2 * eps else eps, self.out);
    }

    pub fn replaced(self: *Normalize) ![]const u8 {
        try self.launch(false, eps, self.base);
        return self.rig.bytes(self.base, self.count() * 2);
    }

    pub fn check(self: *Normalize, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from fn_hc_normalize's", .{ at, got.len });
        for (0..got.len / 2) |i| if (c.classOf(c.read(.bf16, got, i)) != .finite) return null;
        return try std.fmt.bufPrint(&self.msg, "no output is non-finite: the planted values prove nothing", .{});
    }

    pub fn count(self: *const Normalize) usize {
        return self.shape.rows * self.shape.n * self.streams();
    }

    fn rows(self: *const Normalize, acc: ?c.Accum, order: g.Order, out: []f64, mag: ?[]f64) void {
        const n = self.shape.n;
        const ss = self.streams();
        const nc = n / 256;
        var t: [64]f64 = undefined;
        for (0..self.shape.rows) |r| for (0..ss) |st| {
            for (0..nc) |cc| t[cc] = self.pss[(r * nc + cc) * ss + st];
            const total = if (acc) |ac| contract.rows.sumIn(ac, order, t[0..nc]) else blk: {
                var x: f64 = 0;
                for (t[0..nc]) |v| x += v;
                break :blk x;
            };
            const rinv = 1.0 / @sqrt(total / @as(f64, @floatFromInt(n)) + eps);
            for (0..n) |d| {
                const col = st * n + d;
                const at = r * ss * n + col;
                const v = @as(f64, c.fromBf16(self.h[at])) * rinv * self.scale[col];
                out[at] = v;
                if (mag) |m| m[at] = @abs(v);
            }
        };
    }

    pub fn reference(self: *Normalize, want: []f64, mag: []f64) !void {
        self.rows(null, .sequential, want, mag);
    }

    pub fn simulate(self: *Normalize, acc: c.Accum, order: g.Order, out: []f64) !void {
        self.rows(acc, order, out, null);
    }
};
