//! Contract suite for `hy4_route` (`tf_hy4_route_m4`, `tf_hy4_route_m4_plan1`, and planted controls).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rows_mod = @import("hy4_route_rows_cuda");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const glue = cuda.moe_glue;

fn zeroDepth(_: c.Shape) usize {
    return 0;
}

const rule = "sigmoid(logit) + bias descending, lowest expert id on ties, NaN below -inf, expert 0 past -inf; normalized x scale";
const served = [_]c.Shape{
    .{ .rows = 1, .n = 512, .k = 16 },
    .{ .rows = 16, .n = 512, .k = 16 },
    .{ .rows = 17, .n = 512, .k = 16 },
    .{ .rows = 64, .n = 512, .k = 16 },
};
const served_plan1 = [_]c.Shape{
    .{ .rows = 1, .n = 512, .k = 16 },
};

pub const Fault = enum { none, high_ties, distinct };

pub const Case = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    moe: *const glue.MoeGlue,
    fault: Fault,
    name: []const u8,
    plan1: bool = false,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    scale: f32 = 2.5,
    logits: std.ArrayList(f32) = .empty,
    bias: [512]f32 = @splat(0),
    dev_log: u64 = 0,
    dev_bias: u64 = 0,
    dev_out: u64 = 0,
    base: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Case) void {
        self.logits.deinit(self.rig.gpu.gpa);
        self.base.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        const ctl: ?g.Gate = if (self.fault == .none) null else .shapes;
        return .{
            .spec = .{
                .name = self.name,
                .accum = .none,
                .depth = zeroDepth,
                .out = .raw,
                .replaces = "tf_moe_topk_sigmoid_score_warp",
                .nan = .{ .stated = rule },
                .tiles = .{ if (self.plan1) .{ .size = 1, .step = 1, .min = 1 } else .{}, .{ .size = 512, .step = 512, .min = 512 }, .{ .size = 16, .step = 16, .min = 16 } },
                .served = if (self.plan1) &served_plan1 else &served,
            },
            .ctx = self,
            .vt = &vt,
            .control = ctl,
        };
    }

    const vt = g.hooks(Case);

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != 512 or shape.k != 16) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        self.fill = fill;
        try self.logits.resize(a, shape.rows * 512);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x5234);
        const rng = prng.random();
        for (&self.bias, 0..) |*b, i| b.* = if (i % 8 == 0) 0.125 else -0.0625;
        for (0..shape.rows) |r| {
            const row = self.logits.items[r * 512 ..][0..512];
            if (r == shape.rows - 1 and shape.rows >= 4) {
                @memset(row, -std.math.inf(f32));
                for (0..512) |i| self.bias[i] = if (i < 4) 0.0 else -std.math.inf(f32);
                for (0..4) |i| row[i * 33] = 1.0;
            } else {
                for (row, 0..) |*v, i| {
                    v.* = rng.float(f32) * 6.0 - 3.0;
                    if (i >= 8 and i % 19 == 0) {
                        v.* = row[i - 8];
                        self.bias[i] = self.bias[i - 8];
                    }
                }
            }
        }
        if (fill == .nonfinite) {
            self.logits.items[0] = std.math.nan(f32);
            if (shape.rows > 1) self.logits.items[512 + 17] = std.math.inf(f32);
            if (shape.rows > 2) self.logits.items[2 * 512 + 99] = -std.math.inf(f32);
        }
        self.dev_log = try self.rig.input(std.mem.sliceAsBytes(self.logits.items));
        self.dev_bias = try self.rig.input(std.mem.sliceAsBytes(&self.bias));
        self.dev_out = try self.rig.output(shape.rows * 16 * 8);
        return true;
    }

    pub fn run(self: *Case) !void {
        const half = self.shape.rows * 16 * 4;
        if (self.plan1) {
            var plan_buf = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, 1024 * 4);
            defer plan_buf.free();
            for (0..self.shape.rows) |r| {
                var la: cuda.Args = .{};
                la.add(self.dev_log + r * 512 * 4);
                la.add(self.dev_bias);
                la.add(self.scale);
                la.add(self.dev_out + r * 16 * 4);
                la.add(self.dev_out + half + r * 16 * 4);
                la.add(plan_buf.ptr);
                la.add(plan_buf.ptr + 256);
                la.add(plan_buf.ptr + 512);
                try cuda.launch.launch(self.f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 128 } }, self.rig.s, &la);
            }
        } else {
            try rows_mod.route.launch(self.f, self.rig.s, self.dev_log, self.dev_bias, self.shape.rows, self.scale, self.dev_out, self.dev_out + half);
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *const Case) usize {
        return self.shape.rows * 16 * 8;
    }

    pub fn reference(_: *Case, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn replaced(self: *Case) ![]const u8 {
        const half = self.shape.rows * 16 * 4;
        var tmp = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, half * 2);
        defer tmp.free();
        const route: glue.Route = .{ .experts = 512, .topk = 16, .shared = false, .normalize = true, .scale = self.scale, .score = true };
        try self.moe.topk(self.rig.s, self.dev_log, self.dev_bias, route, tmp.ptr, tmp.ptr + half, self.shape.rows);
        try self.rig.s.synchronize();
        try self.base.resize(self.rig.gpu.gpa, half * 2);
        try tmp.download(0, self.base.items);
        return self.base.items;
    }

    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (!std.mem.eql(u8, got, want)) return "tf_hy4_route_m4 output bits differ from tf_moe_topk_sigmoid_score_warp";
        if (!try self.rig.intact()) return "tf_hy4_route_m4 overwrote a guard region";
        return null;
    }
};
