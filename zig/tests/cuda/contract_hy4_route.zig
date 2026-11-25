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
    .{ .rows = 4, .n = 512, .k = 16 },
};

// Packed per-row layout for plan1: IDS[16] + W[16] + MEMBERS[16] + ITEMS[48] + COUNTS[2] = 98 u32s (392 bytes).
const plan1_row_bytes: usize = (16 + 16 + 16 + 48 + 2) * 4;
const plan1_off_w: usize = 16 * 4;
const plan1_off_members: usize = 32 * 4;
const plan1_off_items: usize = 48 * 4;
const plan1_off_counts: usize = 96 * 4;

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
                .replaces = if (self.plan1) "tf_moe_topk_sigmoid_score_warp + grouped plan" else "tf_moe_topk_sigmoid_score_warp",
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
        self.dev_out = try self.rig.output(self.count());
        return true;
    }

    pub fn run(self: *Case) !void {
        if (self.plan1) {
            // Poison output payload with 0xff (-1 for i32) so unwritten trailing ITEMS slots are deterministic.
            const out_view: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.dev_out, .len = self.count() };
            try out_view.fill8(0xff, self.rig.s.handle);
            for (0..self.shape.rows) |r| {
                const dst = self.dev_out + r * plan1_row_bytes;
                var la: cuda.Args = .{};
                la.add(self.dev_log + r * 512 * 4);
                la.add(self.dev_bias);
                la.add(self.scale);
                la.add(dst);
                la.add(dst + plan1_off_w);
                la.add(dst + plan1_off_members);
                la.add(dst + plan1_off_items);
                la.add(dst + plan1_off_counts);
                try cuda.launch.launch(self.f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 128 } }, self.rig.s, &la);
            }
        } else {
            const half = self.shape.rows * 16 * 4;
            try rows_mod.route.launch(self.f, self.rig.s, self.dev_log, self.dev_bias, self.shape.rows, self.scale, self.dev_out, self.dev_out + half);
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *const Case) usize {
        return self.shape.rows * if (self.plan1) plan1_row_bytes else (16 * 8);
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
        if (!self.plan1) {
            try self.base.resize(self.rig.gpu.gpa, half * 2);
            try tmp.download(0, self.base.items);
            return self.base.items;
        }
        const a = self.rig.gpu.gpa;
        const raw = try a.alloc(u8, half * 2);
        defer a.free(raw);
        try tmp.download(0, raw);
        try self.base.resize(a, self.shape.rows * plan1_row_bytes);
        @memset(self.base.items, 0xff);
        for (0..self.shape.rows) |r| {
            const out_row = self.base.items[r * plan1_row_bytes ..][0..plan1_row_bytes];
            @memcpy(out_row[0..64], raw[r * 64 ..][0..64]);
            @memcpy(out_row[plan1_off_w..][0..64], raw[half + r * 64 ..][0..64]);
            var kv: [16]u32 = undefined;
            for (0..16) |t| {
                const id: u32 = @bitCast(std.mem.readInt(i32, raw[r * 64 + t * 4 ..][0..4], .little));
                kv[t] = (id << 4) | @as(u32, @intCast(t));
            }
            std.mem.sort(u32, &kv, {}, comptime std.sort.asc(u32));
            for (0..16) |t| {
                std.mem.writeInt(i32, out_row[plan1_off_members + t * 4 ..][0..4], @intCast(kv[t] & 15), .little);
            }
            var n_items: usize = 0;
            var t: usize = 0;
            while (t < 16) {
                const eid = kv[t] >> 4;
                var end = t + 1;
                while (end < 16 and (kv[end] >> 4) == eid) : (end += 1) {}
                const off = plan1_off_items + n_items * 12;
                std.mem.writeInt(i32, out_row[off..][0..4], @intCast(eid), .little);
                std.mem.writeInt(i32, out_row[off + 4 ..][0..4], @intCast(t), .little);
                std.mem.writeInt(i32, out_row[off + 8 ..][0..4], @intCast(end - t), .little);
                n_items += 1;
                t = end;
            }
            std.mem.writeInt(i32, out_row[plan1_off_counts..][0..4], @intCast(n_items), .little);
            std.mem.writeInt(i32, out_row[plan1_off_counts + 4 ..][0..4], @intCast(n_items), .little);
        }
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
