//! Qwen3.5's raw HF norm weights under gates 1-4: FP32 (1 + w), residual writeback and Q/K head layouts.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const rows_ = core.contract.rows;
const bf = core.bf16_round;
const eps: f32 = 1e-6;

pub const Which = enum { norm, residual, query, key };

fn isHead(which: Which) bool {
    return which == .query or which == .key;
}

fn depth(shape: c.Shape) usize {
    const threads: usize = if (shape.k == 256) 64 else 256;
    return shape.k / threads + 5 + threads / 32;
}

fn served(comptime n: usize, comptime k: usize) []const c.Shape {
    return &.{
        .{ .rows = 1, .n = n, .k = k },
        .{ .rows = 17, .n = n, .k = k },
        .{ .rows = 33, .n = n, .k = k },
        .{ .rows = 65, .n = n, .k = k },
        .{ .rows = 4096, .n = n, .k = k },
    };
}

pub fn spec(which: Which, control: bool) c.Spec {
    const name = switch (which) {
        .norm => "cuda q35 norm shift (tf_q35_norm_shift, no residual)",
        .residual => "cuda q35 norm shift (tf_q35_norm_shift, residual and rows)",
        .query => "cuda q35 norm shift (tf_q35_head_norm_shift, interleaved Q)",
        .key => "cuda q35 norm shift (tf_q35_head_norm_shift, packed K)",
    };
    return .{
        .name = if (control) switch (which) {
            .norm => "control q35 norm shift: unshifted norm on raw w",
            .residual => "control q35 norm shift: unshifted residual norm on raw w",
            .query => "control q35 norm shift: unshifted interleaved Q on raw w",
            .key => "control q35 norm shift: unshifted packed K on raw w",
        } else name,
        .accum = .f32,
        .depth = depth,
        .ops = 8,
        .out = .bf16,
        .in_place = which == .residual,
        .tiles = .{ .{ .size = 16 }, .{}, .{} },
        .served = switch (which) {
            .norm, .residual => served(4096, 4096),
            .query => served(16 * 256, 256),
            .key => served(4 * 256, 256),
        },
        .sums = .{
            .bf16 = 0,
            .why = "BF16 output rounding hides small inverse-norm errors; the FP16 square sum overflows",
        },
    };
}

pub const Case = struct {
    rig: rig_.Rig,
    arena: std.heap.ArenaAllocator,
    which: Which,
    control: bool,
    f: cuda.Function,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    heads: usize = 1,
    stride: usize = 0,
    hstride: usize = 0,
    x: []u16 = &.{},
    residual: []u16 = &.{},
    w: []u16 = &.{},
    xdev: u64 = 0,
    rdev: u64 = 0,
    wdev: u64 = 0,
    hdev: u64 = 0,
    out: u64 = 0,
    params: u64 = 0,

    const vt = blk: {
        var v = g.hooks(Case);
        v.release = release;
        break :blk v;
    };

    fn release(ctx: *anyopaque) void {
        const self: *Case = @ptrCast(@alignCast(ctx));
        cuda.completion.drain(self.rig.gpu.d, self.rig.s, "Qwen norm contract release before completion");
        self.rig.reset();
        _ = self.arena.reset(.free_all);
    }

    pub fn deinit(self: *Case) void {
        release(self);
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        return .{ .spec = spec(self.which, self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    fn inputAt(self: *const Case, unit: usize, col: usize) usize {
        return unit / self.heads * self.stride + unit % self.heads * self.hstride + col;
    }

    fn value(self: *const Case, unit: usize, col: usize) f64 {
        const at = self.inputAt(unit, col);
        const x: f64 = bf.value(self.x[at]);
        return if (self.which == .residual) bf.bf16(x + bf.value(self.residual[at])) else x;
    }

    fn random(rng: std.Random, out: []u16, scale: f32) void {
        for (out) |*x| x.* = bf.bits((rng.float(f32) * 2 - 1) * scale);
    }

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const first = spec(self.which, self.control).served[0];
        if (shape.rows == 0 or shape.rows > 4096 or shape.n != first.n or shape.k != first.k) return false;
        release(self);
        const a = self.arena.allocator();
        self.shape = shape;
        self.heads = if (isHead(self.which)) shape.n / shape.k else 1;
        self.hstride = if (self.which == .query) 2 * shape.k else shape.k;
        self.stride = self.heads * self.hstride;
        self.x = try a.alloc(u16, shape.rows * self.stride);
        self.residual = if (self.which == .residual) try a.alloc(u16, self.x.len) else &.{};
        self.w = try a.alloc(u16, shape.k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        random(rng, self.x, 4);
        random(rng, self.residual, 4);
        random(rng, self.w, 0.5);
        const units = shape.rows * self.heads;
        if (fill == .stress) {
            @memset(self.w, bf.bits(0.5));
            for (0..units) |unit| for (0..shape.k) |col| {
                const at = self.inputAt(unit, col);
                const sign: f32 = if (col % 2 == 0) 1 else -1;
                self.x[at] = bf.bits(sign * @as(f32, @floatFromInt(16 + rng.uintLessThan(u32, 16))));
                if (self.which == .residual)
                    self.residual[at] = bf.bits(sign * @as(f32, @floatFromInt(rng.uintLessThan(u32, 16))));
            };
        } else if (fill == .nonfinite) {
            const values = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32) };
            for (values, 0..) |v, unit| if (unit < units) {
                self.x[self.inputAt(unit, shape.k / 3)] = bf.bits(v);
            };
            if (units > 3) self.x[self.inputAt(3, shape.k - 1)] = bf.bits(-std.math.inf(f32));
            if (self.which == .residual) self.residual[self.inputAt(1, shape.k / 2)] = bf.bits(std.math.inf(f32));
            self.w[5] = bf.bits(std.math.nan(f32));
            self.w[6] = bf.bits(std.math.inf(f32));
            self.w[7] = bf.bits(-std.math.inf(f32));
        }
        self.xdev = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.rdev = if (self.which == .residual) try self.rig.input(std.mem.sliceAsBytes(self.residual)) else 0;
        self.wdev = try self.rig.input(std.mem.sliceAsBytes(self.w));
        self.hdev = if (self.which == .residual) try self.rig.output(self.x.len * 2) else self.xdev;
        self.out = try self.rig.output(shape.rows * shape.n * 2);
        self.params = if (self.which == .residual or self.which == .query) blk: {
            const p = [_]c_int{ 0, 0, @intCast(shape.rows), 1, 0, 0, 0, 0 };
            break :blk try self.rig.input(std.mem.asBytes(&p));
        } else 0;
        return true;
    }

    pub fn run(self: *Case) !void {
        const shape = self.shape;
        var a: cuda.Args = .{};
        if (self.which == .residual)
            try self.rig.gpu.d.check(self.rig.gpu.d.api.cuMemcpyDtoDAsync_v2(self.hdev, self.xdev, self.x.len * 2, self.rig.s.handle), "reset norm residual");
        if (isHead(self.which)) {
            inline for (.{ self.xdev, self.wdev, self.out, @as(c_int, @intCast(self.heads)), @as(c_int, @intCast(self.stride)), @as(c_int, @intCast(self.hstride)), eps, self.params }) |v| a.add(v);
        } else {
            inline for (.{ self.hdev, self.rdev, self.wdev, self.out, eps, @as(c_int, @intCast(shape.k)), self.params, @as(u64, 0) }) |v| a.add(v);
        }
        const grid = (shape.rows + @intFromBool(self.params != 0)) * self.heads;
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = if (isHead(self.which)) 64 else 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Case) usize {
        return self.shape.rows * self.shape.n * @as(usize, if (self.which == .residual) 2 else 1);
    }

    fn outputs(self: *Case, acc: c.Accum, order: g.Order, want: []f64, mag: ?[]f64) !void {
        const shape = self.shape;
        const terms = try self.arena.allocator().alloc(f64, shape.k);
        defer self.arena.allocator().free(terms);
        const skip = if (self.which == .residual) self.x.len else 0;
        if (skip > 0) for (want[0..skip], 0..) |*v, i| {
            v.* = self.value(i / shape.k, i % shape.k);
            if (mag) |m| m[i] = 0;
        };
        for (0..shape.rows * self.heads) |unit| {
            var sum: f64 = 0;
            for (terms, 0..) |*term, col| {
                const v = self.value(unit, col);
                term.* = v * v;
                sum += term.*;
            }
            if (acc != .none) sum = rows_.sumIn(acc, order, terms);
            const inv = 1.0 / @sqrt(sum / @as(f64, @floatFromInt(shape.k)) + eps);
            for (0..shape.k) |col| {
                const at = skip + unit * shape.k + col;
                want[at] = (1.0 + @as(f64, bf.value(self.w[col]))) * (self.value(unit, col) * inv);
                if (mag) |m| m[at] = @abs(want[at]) / 2;
            }
        }
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        try self.outputs(.none, .sequential, want, mag);
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.outputs(acc, order, out, null);
    }

    pub fn bounds(self: *Case, _: []const f64, _: []const f64, allow: []f64) !void {
        if (self.which == .residual) @memset(allow[0..self.x.len], 0);
    }
};

pub const Group = struct {
    a: std.mem.Allocator,
    cases: [8]Case = undefined,
    used: usize = 0,

    pub fn create(a: std.mem.Allocator, gpu: check_.Gpu, s: cuda.Stream, mod: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        self.* = .{ .a = a };
        errdefer self.destroy();
        for (std.enums.values(Which)) |which| for ([_]bool{ false, true }) |control| {
            const symbol: [:0]const u8 = if (isHead(which))
                (if (control) "tf_q35_head_norm" else "tf_q35_head_norm_shift")
            else
                (if (control) "tf_q35_norm" else "tf_q35_norm_shift");
            const f = try mod.function(symbol);
            const item = &self.cases[self.used];
            item.* = .{ .rig = .{ .gpu = gpu, .s = s }, .arena = .init(a), .which = which, .control = control, .f = f };
            self.used += 1;
            try kernels.append(a, item.kernel());
        };
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (self.cases[0..self.used]) |*item| item.deinit();
        self.a.destroy(self);
    }
};
