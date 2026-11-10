//! The joint schema head's row kernels under the contract on CUDA: layer norm, row softmax, GELU and controls.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const Fns = @import("qwen35").joint_schema.Fns;

pub const Which = enum { layer_norm, softmax, gelu, norm_bf16, softmax_safe, gelu_identity };

/// A thread's chain over every 256th value, its wave's xor tree, then the block's eight waves in order.
fn blockDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 5 + 8;
}

fn none(_: c.Shape) usize {
    return 0;
}

const block_tiles = [3]c.Tile{ .{}, .{}, .{ .size = 256 } };
// the hidden norm on a piece's rows and the width norms on options, fields and the memory
const norm_shapes = [_]c.Shape{ .{ .rows = 64, .n = 1024, .k = 1024 }, .{ .rows = 16, .n = 4096, .k = 4096 }, .{ .rows = 1, .n = 1024, .k = 1024 } };
// a query block's 16 heads over a short record, the whole window and one piece
const softmax_shapes = [_]c.Shape{ .{ .rows = 64, .n = 245, .k = 245 }, .{ .rows = 16, .n = 16384, .k = 16384 }, .{ .rows = 64, .n = 1024, .k = 1024 } };
const gelu_shapes = [_]c.Shape{ .{ .rows = 8, .n = 4096, .k = 4096 }, .{ .rows = 64, .n = 1024, .k = 1024 } };
const soft_why = "exp terms: fp32 is not exact on them, so gate 4 shows the fp32 bound alone";

pub fn spec(which: Which) c.Spec {
    return switch (which) {
        .layer_norm, .norm_bf16 => .{
            .name = if (which == .layer_norm) "cuda joint schema tf_js_layer_norm" else "control sums: joint schema layer norm summed in bf16",
            .accum = .f32,
            .depth = blockDepth,
            .ops = 8,
            .out = .bf16,
            .tiles = block_tiles,
            .served = &norm_shapes,
        },
        .softmax, .softmax_safe => .{
            .name = if (which == .softmax) "cuda joint schema tf_js_softmax_rows" else "control nonfinite: joint schema softmax skipping non-finite scores",
            .accum = .f32,
            .depth = blockDepth,
            .ops = 8,
            .out = .f32,
            .in_place = true,
            .tiles = block_tiles,
            .served = &softmax_shapes,
            .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = soft_why },
        },
        .gelu, .gelu_identity => .{ .name = if (which == .gelu) "cuda joint schema tf_js_gelu" else "control shapes: joint schema GELU is identity", .accum = .none, .depth = none, .ops = 4, .out = .bf16, .in_place = true, .served = &gelu_shapes },
    };
}

pub fn control(which: Which) ?g.Gate {
    return switch (which) {
        .norm_bf16 => .sums,
        .softmax_safe => .nonfinite,
        .gelu_identity => .shapes,
        else => null,
    };
}

fn opOf(w: Which) rows.Op {
    return switch (w) {
        .layer_norm, .norm_bf16 => .layer_norm,
        .softmax, .softmax_safe => .softmax,
        .gelu, .gelu_identity => .gelu,
    };
}

pub const Row = struct {
    rig: rig_.Rig,
    which: Which,
    f: cuda.Function,
    r: ?rows.Rows = null,
    x: u64 = 0,
    w: u64 = 0,
    b: u64 = 0,
    out: u64 = 0,
    /// An in-place kernel's input, put back before every run.
    start: []u8 = &.{},
    /// The softmax's FP32 scores, most of them between two BF16 values (a BF16-rounding kernel fails on them).
    scores: []f32 = &.{},

    pub fn init(gpu: check.Gpu, s: cuda.Stream, fns: Fns, ctl: cuda.Module, which: Which) !Row {
        const f = switch (which) {
            .layer_norm => fns.k.layer_norm,
            .softmax => fns.k.softmax_rows,
            .gelu => fns.k.gelu,
            .norm_bf16 => try ctl.function("tf_ctl_js_norm_bf16"),
            .softmax_safe => try ctl.function("tf_ctl_js_softmax_safe"),
            .gelu_identity => try ctl.function("tf_ctl_js_gelu_identity"),
        };
        return .{ .rig = .{ .gpu = gpu, .s = s }, .which = which, .f = f };
    }

    pub fn deinit(self: *Row) void {
        if (self.r) |r| r.deinit();
        self.rig.gpu.gpa.free(self.start);
        self.rig.gpu.gpa.free(self.scores);
        self.rig.deinit();
    }

    pub fn kernel(self: *Row) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt, .control = control(self.which) };
    }

    const vt = g.hooks(Row);

    pub fn prepare(self: *Row, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.r) |r| r.deinit();
        self.r = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const r = try rows.Rows.init(a, opOf(self.which), shape);
        self.r = r;
        r.fill(fill, seed);
        a.free(self.start);
        self.start = &.{};
        a.free(self.scores);
        self.scores = &.{};
        switch (opOf(self.which)) {
            .layer_norm => {
                self.x = try self.rig.input(std.mem.sliceAsBytes(r.x));
                self.w = try self.rig.input(std.mem.sliceAsBytes(r.w));
                self.b = try self.rig.input(std.mem.sliceAsBytes(r.b));
                self.out = try self.rig.output(r.outputs() * 2);
            },
            .softmax => {
                var prng = std.Random.DefaultPrng.init(seed ^ 0x5c0);
                const x = try a.alloc(f32, r.x.len);
                // the scores are FP32 products: a nudge of under 2^-9 of each moves it off the BF16 grid
                for (x, r.x) |*o, h| {
                    const v = c.fromBf16(h);
                    o.* = if (std.math.isFinite(v)) v * (1 + (prng.random().float(f32) - 0.5) * 0x1p-9) else v;
                }
                self.scores = x;
                self.start = try a.dupe(u8, std.mem.sliceAsBytes(x));
                self.out = try self.rig.output(self.start.len);
            },
            else => {
                self.start = try a.dupe(u8, std.mem.sliceAsBytes(r.x));
                self.out = try self.rig.output(self.start.len);
            },
        }
        return true;
    }

    pub fn run(self: *Row) !void {
        const r = self.r.?;
        const sh = r.shape;
        var a: cuda.Args = .{};
        var cfg: cuda.launch.Config = .{ .grid = .{ .x = @intCast(sh.rows) }, .block = .{ .x = 256 } };
        if (self.start.len > 0) try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.out, .len = self.start.len }).upload(0, self.start);
        switch (opOf(self.which)) {
            .layer_norm => {
                for ([_]u64{ self.x, self.w, self.b, self.out }) |p| a.add(p);
                a.add(@as(c_int, @intCast(sh.k)));
                a.add(r.eps);
            },
            .softmax => {
                a.add(self.out);
                a.add(@as(c_int, @intCast(sh.k)));
            },
            else => {
                const n = sh.rows * sh.k;
                a.add(self.out);
                a.add(@as(c_longlong, @intCast(n)));
                cfg.grid.x = @intCast((n + 255) / 256);
            },
        }
        try cuda.launch.launch(self.f, cfg, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Row) usize {
        return self.r.?.outputs();
    }

    pub fn reference(self: *Row, want: []f64, mag: []f64) !void {
        const r = self.r.?;
        if (r.op != .softmax) return r.reference(want, mag);
        const k = r.shape.k;
        for (0..r.shape.rows) |row| {
            const x = self.scores[row * k ..][0..k];
            var top: f64 = -std.math.inf(f64);
            for (x) |v| top = if (std.math.isNan(v) or v > top) v else top;
            var sum: f64 = 0;
            for (x) |v| sum += @exp(v - top);
            for (want[row * k ..][0..k], mag[row * k ..][0..k], x) |*w, *m, v| {
                w.* = @exp(v - top) / sum;
                m.* = @abs(w.*);
            }
        }
    }

    /// GELU's erff carries its own slack where 1 + erf cancels; the other rows keep the generic bound.
    pub fn bounds(self: *Row, _: []const f64, _: []const f64, allow: []f64) !void {
        if (opOf(self.which) != .gelu) return;
        for (allow, self.r.?.x) |*l, h| l.* += rows.geluSlack(c.fromBf16(h));
    }

    /// Layer norm's exact stress sums in any order; the softmax's sums as its block adds them (the order aside).
    pub fn simulate(self: *Row, acc: c.Accum, order: g.Order, out: []f64) !void {
        const r = self.r.?;
        if (r.op != .softmax) return r.simulate(acc, order, out);
        const k = r.shape.k;
        const t = try self.rig.gpu.gpa.alloc(f64, k);
        defer self.rig.gpu.gpa.free(t);
        for (0..r.shape.rows) |row| {
            const x = self.scores[row * k ..][0..k];
            var top: f64 = -std.math.inf(f64);
            for (x) |v| top = if (std.math.isNan(v) or v > top) v else top;
            for (t, x) |*v, h| v.* = @as(f32, @floatCast(@exp(h - top)));
            const sum = blockSum(acc, t);
            for (out[row * k ..][0..k], t) |*o, v| o.* = v / sum;
        }
    }
};

fn rnd(acc: c.Accum, v: f64) f64 {
    return switch (acc) {
        .f32 => @as(f32, @floatCast(v)),
        .f16 => @as(f16, @floatCast(v)),
        .bf16 => c.fromBf16(c.toBf16(@floatCast(v))),
        .none => v,
    };
}

/// joint_schema.cu's block_sum in `acc`: 256 strided chains, each wave's xor tree, the eight waves in order.
pub fn blockSum(acc: c.Accum, t: []const f64) f64 {
    var lane: [256]f64 = @splat(0);
    for (t, 0..) |v, i| lane[i % 256] = rnd(acc, lane[i % 256] + v);
    var total: f64 = 0;
    for (0..8) |w| {
        var v: [32]f64 = lane[w * 32 ..][0..32].*;
        var o: usize = 16;
        while (o > 0) : (o >>= 1) {
            var next: [32]f64 = undefined;
            for (&next, 0..) |*n, i| n.* = rnd(acc, v[i] + v[i ^ o]);
            v = next;
        }
        total = rnd(acc, total + v[0]);
    }
    return total;
}
