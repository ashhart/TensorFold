//! Router accumulator gate: post outputs for cancellation terms in a fixed normalized residual.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;
const d = 4096;
const width = 2 * d;
const served = [_]c.Shape{
    .{ .rows = 1, .n = d, .k = width },   .{ .rows = 2, .n = d, .k = width },
    .{ .rows = 7, .n = d, .k = width },   .{ .rows = 16, .n = d, .k = width },
    .{ .rows = 33, .n = d, .k = width },  .{ .rows = 128, .n = d, .k = width },
    .{ .rows = 512, .n = d, .k = width },
};
fn depth(_: c.Shape) usize {
    return 40;
}
fn spec() c.Spec {
    return .{ .name = "cuda qwen35_mhc route FP32 sums", .accum = .f32, .depth = depth, .ops = 8, .out = .f32, .served = &served, .nan = .{ .stated = "nonfinite residual makes every post NaN" }, .sums = .{ .exact = false, .f16 = 10, .bf16 = 10, .why = "sigmoid bounds the dot error" } };
}
fn weight(i: usize) f32 {
    const j = i % d;
    const v: f32 = @as(f32, @floatFromInt(1 + j % 7)) / 16384;
    if (i < d) return v;
    const hash = (j * 37 + (j >> 6) * 23 + ((j * j * 2) >> 8)) % 7;
    return -v + @as(f32, @floatFromInt(@as(i32, @intCast(hash)) - 3)) / 262144;
}
fn dotTerms(t: *[width]f64) void {
    const inv = 1.0 / @sqrt(@as(f64, 1.0 + 1e-6));
    for (t, 0..) |*v, i| v.* = @as(f64, weight(i)) * inv;
}
fn post(sum: f64) f64 {
    return 2 / (1 + @exp(-sum));
}
fn expected() struct { value: f64, mag: f64 } {
    var t: [width]f64 = undefined;
    dotTerms(&t);
    var sum: f64 = 0;
    var mag: f64 = 0;
    for (t) |x| {
        sum += x;
        mag += @abs(x);
    }
    return .{ .value = post(sum), .mag = mag };
}

const Case = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    n: usize = 0,
    nonfinite: bool = false,
    dev: [7]u64 = @splat(0),
    msg: [160]u8 = undefined,
    const vt = g.hooks(Case);
    fn kernel(self: *Case) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }
    fn deinit(self: *Case) void {
        self.rig.deinit();
    }
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, _: u64) !bool {
        if (shape.n != d or shape.k != width or shape.rows == 0 or shape.rows > 512) return false;
        self.rig.reset();
        self.n = shape.rows;
        self.nonfinite = fill == .nonfinite;
        const a = self.rig.gpu.gpa;
        const x = try a.alloc(u16, shape.rows * width);
        defer a.free(x);
        @memset(x, 0x3f80);
        if (self.nonfinite) for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }, 0..) |v, i| {
            if (i < shape.rows) x[i * width + 7] = v;
        };
        const w = try a.alloc(f32, 8 * width);
        defer a.free(w);
        @memset(w, 0);
        for (0..width) |i| {
            w[2 * width + i] = weight(i);
            w[3 * width + i] = weight(i);
        }
        const bias: [8]f32 = @splat(0);
        const params = [3]i32{ 0, 0, @intCast(shape.rows) };
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(x));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(w));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(&bias));
        self.dev[3] = try self.rig.scratch(shape.rows * d * 2);
        self.dev[4] = try self.rig.output(shape.rows * 2 * 4);
        self.dev[5] = try self.rig.scratch(shape.rows * 4 * 4);
        self.dev[6] = try self.rig.input(std.mem.sliceAsBytes(&params));
        return true;
    }
    pub fn run(self: *Case) !void {
        var args: cuda.Args = .{};
        for (self.dev[0..6]) |ptr| args.add(ptr);
        args.add(@as(c_int, d));
        args.add(@as(c_int, 20));
        args.add(@as(f32, 1.0));
        args.add(self.dev[6]);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.n) }, .block = .{ .x = 256 } }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }
    pub fn count(self: *Case) usize {
        return self.n * 2;
    }
    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        const e = expected();
        for (want, mag, 0..) |*v, *m, i| {
            v.* = if (self.nonfinite and i / 2 < 3) std.math.nan(f64) else e.value;
            m.* = e.mag;
        }
    }
    pub fn bounds(_: *Case, _: []const f64, _: []const f64, allow: []f64) !void {
        @memset(allow, 2e-6);
    }
    pub fn simulate(_: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        var t: [width]f64 = undefined;
        dotTerms(&t);
        @memset(out, post(rows.sumIn(acc, order, &t)));
    }
    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const output = try self.rig.outputBytes();
        const want = expected().value;
        for (0..self.count()) |i| {
            const got: f32 = @bitCast(std.mem.readInt(u32, output[i * 4 ..][0..4], .little));
            const ok = if (self.nonfinite and i / 2 < 3) std.math.isNan(got) else @abs(@as(f64, got) - want) <= 2e-6;
            if (!ok) return try std.fmt.bufPrint(&self.msg, "post {d}: {d:.8} expected {d:.8}", .{ i, got, want });
        }
        return null;
    }
};

pub const Set = struct {
    module: ?cuda.Module = null,
    case: ?Case = null,
    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !void {
        self.module = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_mhc);
        self.case = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try self.module.?.function("qwen35_mhc_route") };
        try kernels.append(gpu.gpa, self.case.?.kernel());
    }
    pub fn deinit(self: *Set) void {
        if (self.case) |*item| item.deinit();
        if (self.module) |*module| module.unload();
    }
};

test "FP32 stress separates half sums in each order" {
    const e = expected();
    const shape = served[0];
    const allowance = c.bound(spec(), shape, e.value, e.mag);
    var t: [width]f64 = undefined;
    dotTerms(&t);
    for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
        const precise = post(rows.sumIn(.f32, order, &t));
        try std.testing.expect(@abs(precise - e.value) <= allowance);
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            const low = post(rows.sumIn(acc, order, &t));
            try std.testing.expect(@abs(low - e.value) > 10 * allowance);
        }
    }
}
