//! Two-stream mHC: the BF16 operations and the router's FP32 outputs against the host replay.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const ref = cuda.kernels.mhc_ref;
const d = 4096;
const width = 2 * d;
// Allow FP32 rsqrt and transcendental differences; BF16 comparisons use fixed rounded pre values.
const tol: f32 = 2e-6;
const Kind = enum { enter, route, merge, leave };
const shapes = [_]c.Shape{
    .{ .rows = 1, .n = d, .k = width },   .{ .rows = 2, .n = d, .k = width },
    .{ .rows = 7, .n = d, .k = width },   .{ .rows = 16, .n = d, .k = width },
    .{ .rows = 33, .n = d, .k = width },  .{ .rows = 128, .n = d, .k = width },
    .{ .rows = 512, .n = d, .k = width },
};
fn depth(_: c.Shape) usize {
    return 0;
}
fn name(kind: Kind, control: bool) []const u8 {
    return switch (kind) {
        .enter => if (control) "control: qwen35_mhc enter skips add" else "cuda qwen35_mhc enter",
        .route => if (control) "control: qwen35_mhc route changes post" else "cuda qwen35_mhc route",
        .merge => if (control) "control: qwen35_mhc merge drops update" else "cuda qwen35_mhc merge",
        .leave => if (control) "control: qwen35_mhc leave drops stream" else "cuda qwen35_mhc leave",
    };
}
fn spec(kind: Kind, control: bool, add_enabled: bool, route_dynamic: bool) c.Spec {
    return .{ .name = if (kind == .enter and !add_enabled) "cuda qwen35_mhc enter no add" else if (kind == .route and route_dynamic) "cuda qwen35_mhc route varying pre" else name(kind, control), .accum = .none, .depth = depth, .out = .raw, .in_place = kind == .enter or kind == .merge, .served = &shapes, .nan = .{ .stated = "BF16 host replay, NaNs by class; route post and mix within 2e-6" } };
}
fn randomBf(rng: std.Random) u16 {
    return c.toBf16(@as(f32, @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 513))) - 256)) / 128.0);
}
fn f32at(b: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, b[i * 4 ..][0..4], .little));
}
fn bfAt(b: []const u8, i: usize) u16 {
    return std.mem.readInt(u16, b[i * 2 ..][0..2], .little);
}
fn sameBf(bits: u16, want: f32) bool {
    return bits == c.toBf16(want) or (std.math.isNan(c.fromBf16(bits)) and std.math.isNan(want));
}
fn same(a: f32, b: f32) bool {
    return a == b or (std.math.isNan(a) and std.math.isNan(b));
}

const Case = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    kind: Kind,
    control: bool,
    add_enabled: bool = true,
    route_dynamic: bool = false,
    rows: usize = 0,
    h: []u16 = &.{},
    add: []u16 = &.{},
    residual: []u16 = &.{},
    branch: []u16 = &.{},
    weights: []f32 = &.{},
    post: []f32 = &.{},
    mix: []f32 = &.{},
    dev: [8]u64 = @splat(0),
    msg: [200]u8 = undefined,
    const bias = [8]f32{ 0, 1.0986123, -0.4, 0.3, -0.7, 0.2, 0.4, 0.9 };
    const vt = g.hooks(Case);
    fn kernel(self: *Case) g.Kernel {
        return .{ .spec = spec(self.kind, self.control, self.add_enabled, self.route_dynamic), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }
    fn free(self: *Case) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ "h", "add", "residual", "branch", "weights", "post", "mix" }) |field| {
            a.free(@field(self, field));
            @field(self, field) = &.{};
        }
    }
    fn deinit(self: *Case) void {
        self.free();
        self.rig.deinit();
    }
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != d or shape.k != width or shape.rows == 0 or shape.rows > 512) return false;
        self.rig.reset();
        self.free();
        const a = self.rig.gpu.gpa;
        const n = shape.rows + 1;
        self.rows = shape.rows;
        self.h = try a.alloc(u16, n * d);
        self.add = try a.alloc(u16, n * d);
        self.residual = try a.alloc(u16, n * width);
        self.branch = try a.alloc(u16, n * d);
        self.post = try a.alloc(f32, n * 2);
        self.mix = try a.alloc(f32, n * 4);
        self.weights = try a.alloc(f32, 8 * width);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.h) |*v| v.* = randomBf(rng);
        for (self.add) |*v| v.* = randomBf(rng);
        for (self.residual) |*v| v.* = randomBf(rng);
        for (self.branch) |*v| v.* = randomBf(rng);
        for (self.post, 0..) |*v, i| v.* = if (i % 2 == 0) 0.7 else 1.3;
        for (self.mix, 0..) |*v, i| v.* = ([_]f32{ 0.7, 0.3, 0.2, 0.8 })[i % 4];
        for (self.weights, 0..) |*v, i| {
            v.* = if (i < 2 * width) 0 else @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) / 32768;
        }
        self.residual[0] = 0x8000;
        self.residual[d] = 0x8000;
        self.branch[0] = 0x8000;
        if (self.route_dynamic) {
            for (self.residual) |*v| v.* = if (rng.boolean()) 0x3f80 else 0xbf80;
            self.weights[0] = 1;
            self.weights[width + 1] = 1;
        }
        if (fill == .nonfinite) {
            const vals = [_]u16{ 0x7fc0, 0x7f80, 0xff80 };
            for (vals, 0..) |v, i| if (i < self.rows) {
                self.h[i * d + 7] = v;
                self.residual[i * width + 7] = v;
            };
        }
        const params = [3]i32{ 0, 0, @intCast(self.rows) };
        self.dev[7] = try self.rig.input(std.mem.sliceAsBytes(&params));
        switch (self.kind) {
            .enter => {
                self.dev[0] = try self.rig.output(n * d * 2);
                self.dev[1] = if (self.add_enabled) try self.rig.input(std.mem.sliceAsBytes(self.add)) else 0;
                self.dev[2] = try self.rig.output(n * width * 2);
            },
            .route => {
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.residual));
                self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.weights));
                self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(&bias));
                self.dev[3] = try self.rig.output(n * d * 2);
                self.dev[4] = try self.rig.output(n * 2 * 4);
                self.dev[5] = try self.rig.output(n * 4 * 4);
            },
            .merge => {
                self.dev[0] = try self.rig.output(n * width * 2);
                self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.branch));
                self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.post));
                self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(self.mix));
            },
            .leave => {
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.residual));
                self.dev[1] = try self.rig.output(n * d * 2);
            },
        }
        return true;
    }
    fn upload(self: *Case, ptr: u64, bytes: []const u8) !void {
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = ptr, .len = bytes.len }).upload(0, bytes);
    }
    pub fn run(self: *Case) !void {
        if (self.kind == .enter) try self.upload(self.dev[0], std.mem.sliceAsBytes(self.h));
        if (self.kind == .merge) try self.upload(self.dev[0], std.mem.sliceAsBytes(self.residual));
        var a: cuda.Args = .{};
        const nargs: usize = switch (self.kind) {
            .enter => 3,
            .route => 6,
            .merge => 4,
            .leave => 2,
        };
        for (self.dev[0..nargs]) |ptr| a.add(ptr);
        a.add(@as(c_int, d));
        a.add(self.dev[7]);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.rows + 1) }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }
    pub fn count(self: *Case) usize {
        return self.rows * d;
    }
    pub fn reference(_: *Case, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
    fn fail(self: *Case, row: usize, col: usize, what: []const u8) !?[]const u8 {
        return try std.fmt.bufPrint(&self.msg, "row {d} column {d}: {s}", .{ row, col, what });
    }
    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        const n = self.rows + 1;
        var r: [width]f32 = undefined;
        var h: [d]f32 = undefined;
        var extra: [d]f32 = undefined;
        var b: [d]f32 = undefined;
        for (0..self.rows) |row| {
            for (&r, self.residual[row * width ..][0..width]) |*v, bits| v.* = c.fromBf16(bits);
            for (&h, self.h[row * d ..][0..d]) |*v, bits| v.* = c.fromBf16(bits);
            for (&extra, self.add[row * d ..][0..d]) |*v, bits| v.* = c.fromBf16(bits);
            for (&b, self.branch[row * d ..][0..d]) |*v, bits| v.* = c.fromBf16(bits);
            switch (self.kind) {
                .enter => {
                    ref.enter(&h, if (self.add_enabled) &extra else null, &r);
                    for (h, 0..) |v, i| if (!sameBf(bfAt(out, row * d + i), v)) return self.fail(row, i, "h differs");
                    const base = n * d * 2;
                    for (r, 0..) |v, i| if (!sameBf(bfAt(out[base..], row * width + i), v)) return self.fail(row, i, "stream differs");
                },
                .leave => {
                    ref.leave(&r, &h);
                    for (h, 0..) |v, i| if (!sameBf(bfAt(out, row * d + i), v)) return self.fail(row, i, "mean differs");
                },
                .merge => {
                    ref.merge(&r, &b, self.post[row * 2 ..][0..2].*, self.mix[row * 4 ..][0..4].*);
                    for (r, 0..) |v, i| if (!sameBf(bfAt(out, row * width + i), v)) return self.fail(row, i, "merge differs");
                },
                .route => {
                    var p: [2]f32 = undefined;
                    var m: [4]f32 = undefined;
                    ref.route(&r, self.weights, bias, &b, &p, &m);
                    // Each fixture chooses BF16 coefficients away from boundaries, independent of reduction order.
                    if (std.math.isFinite(r[7])) {
                        const pre = if (self.route_dynamic)
                            [2]f32{ if (r[0] > 0) 0.73046875 else 0.26953125, if (r[1] > 0) 0.890625 else 0.5234375 }
                        else
                            [2]f32{ 0.5, 0.75 };
                        ref.collapse(&r, pre, &b);
                    }
                    for (b, 0..) |v, i| if (!sameBf(bfAt(out, row * d + i), v)) return self.fail(row, i, "collapse differs");
                    const pbytes = out[n * d * 2 ..];
                    const mbytes = pbytes[n * 2 * 4 ..];
                    for (p, 0..) |v, i| if (!same(f32at(pbytes, row * 2 + i), v) and !(@abs(f32at(pbytes, row * 2 + i) - v) <= tol)) return self.fail(row, i, "post differs");
                    for (m, 0..) |v, i| if (!same(f32at(mbytes, row * 4 + i), v) and !(@abs(f32at(mbytes, row * 4 + i) - v) <= tol)) return self.fail(row, i, "mix differs");
                },
            }
        }
        const last = self.rows;
        switch (self.kind) {
            .enter => {
                for (0..d) |i| if (bfAt(out, last * d + i) != self.h[last * d + i]) return self.fail(last, i, "past-row h changed");
                const base = n * d * 2 + last * width * 2;
                if (!std.mem.allEqual(u8, out[base..][0 .. width * 2], 0xff)) return self.fail(last, 0, "past-row streams changed");
            },
            .merge => for (0..width) |i| {
                if (bfAt(out, last * width + i) != self.residual[last * width + i]) return self.fail(last, i, "past-row streams changed");
            },
            .leave => if (!std.mem.allEqual(u8, out[last * d * 2 ..][0 .. d * 2], 0xff)) return self.fail(last, 0, "past-row h changed"),
            .route => {
                if (!std.mem.allEqual(u8, out[last * d * 2 ..][0 .. d * 2], 0xff)) return self.fail(last, 0, "past-row branch changed");
                const p = n * d * 2 + last * 2 * 4;
                const m = n * d * 2 + n * 2 * 4 + last * 4 * 4;
                if (!std.mem.allEqual(u8, out[p..][0..8], 0xff)) return self.fail(last, 0, "past-row post changed");
                if (!std.mem.allEqual(u8, out[m..][0..16], 0xff)) return self.fail(last, 0, "past-row mix changed");
            },
        }
        return null;
    }
};

pub const Set = struct {
    module: ?cuda.Module = null,
    cases: [10]Case = undefined,
    count: usize = 0,
    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, ctl: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        self.module = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_mhc);
        const names = [_][:0]const u8{ "qwen35_mhc_enter", "qwen35_mhc_route", "qwen35_mhc_merge", "qwen35_mhc_leave" };
        const controls = [_][:0]const u8{ "tf_ctl_qwen35_mhc_enter", "tf_ctl_qwen35_mhc_route", "tf_ctl_qwen35_mhc_merge", "tf_ctl_qwen35_mhc_leave" };
        inline for (std.meta.tags(Kind), 0..) |kind, i| {
            for ([_]bool{ false, true }) |bad| {
                const mod = if (bad) ctl else self.module.?;
                const item = &self.cases[self.count];
                item.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try mod.function(if (bad) controls[i] else names[i]), .kind = kind, .control = bad };
                self.count += 1;
                try kernels.append(gpu.gpa, item.kernel());
            }
        }
        const item = &self.cases[self.count];
        item.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try self.module.?.function(names[0]), .kind = .enter, .control = false, .add_enabled = false };
        self.count += 1;
        try kernels.append(gpu.gpa, item.kernel());
        const varied = &self.cases[self.count];
        varied.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try self.module.?.function(names[1]), .kind = .route, .control = false, .route_dynamic = true };
        self.count += 1;
        try kernels.append(gpu.gpa, varied.kernel());
    }
    pub fn deinit(self: *Set) void {
        for (self.cases[0..self.count]) |*item| item.deinit();
        if (self.module) |*module| module.unload();
    }
};
