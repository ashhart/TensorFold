//! Row kernels under the contract on Metal: MLX's precise softmax and RMS norm, and the planted controls.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

pub const Which = enum { softmax, rms, fmax_max, atomic_sum, scribble_sum, tail_sum, half_dot };

fn opOf(w: Which) rows.Op {
    return switch (w) {
        .softmax => .softmax,
        .rms => .rms,
        .fmax_max => .max,
        .half_dot => .dot,
        else => .sum,
    };
}

/// Four inputs a thread in order, then two 32-way simd_sum trees.
fn simdTrees(_: c.Shape) usize {
    return 14;
}

/// A lane's chain over every 32nd input, then one simd_sum tree.
fn laneChain(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5;
}

/// The threads MLX's row kernels take: four inputs a thread, whole simdgroups.
fn rowThreads(k: usize) usize {
    return ((k + 3) / 4 + 31) / 32 * 32;
}

const softmax_why = "a bf16 output of at most 4,096 terms in [0, 1]: no accumulator's error survives its rounding 100x";
const rms_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

pub fn spec(which: Which) c.Spec {
    const ctl = &[_]c.Shape{.{ .rows = 8, .n = 1, .k = 4096 }};
    return switch (which) {
        .softmax => .{ .name = "metal softmax tf_softmax_bf16", .accum = .f32, .depth = simdTrees, .ops = 64, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 128 } }, .served = &.{ .{ .rows = 64, .n = 2048, .k = 2048 }, .{ .rows = 16, .n = 300, .k = 300 } }, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = softmax_why } },
        .rms => .{ .name = "metal RMS norm tf_rms_norm_bf16", .accum = .f32, .depth = simdTrees, .ops = 8, .mids = 1, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 128 } }, .served = &.{ .{ .rows = 32, .n = 2688, .k = 2688 }, .{ .rows = 4, .n = 1024, .k = 1024 } }, .sums = .{ .bf16 = 0, .why = rms_why } },
        .fmax_max => .{ .name = "control: fmax row max", .accum = .none, .depth = laneChain, .out = .f32, .served = ctl },
        .atomic_sum => .{ .name = "control: atomic-add row sum", .accum = .f32, .depth = laneChain, .out = .f32, .served = ctl },
        .scribble_sum => .{ .name = "control: row sum writing its input", .accum = .f32, .depth = laneChain, .out = .f32, .served = ctl },
        .tail_sum => .{ .name = "control: row sum losing a tail input", .accum = .f32, .depth = laneChain, .out = .f32, .tiles = .{ .{}, .{}, .{ .size = 32 } }, .served = ctl },
        .half_dot => .{ .name = "control: dot summed in half", .accum = .f32, .depth = laneChain, .out = .f32, .served = ctl },
    };
}

/// The gate each control must fail.
pub fn control(which: Which) ?g.Gate {
    return switch (which) {
        .softmax, .rms => null,
        .fmax_max => .nonfinite,
        .atomic_sum, .scribble_sum => .repeat,
        .tail_sum => .shapes,
        .half_dot => .sums,
    };
}

pub const Row = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    pipe: mtl.Pipeline,
    r: ?rows.Rows = null,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: Which) !Row {
        const src = switch (which) {
            .softmax => sources.ops_softmax,
            .rms => sources.ops_embed_norm,
            else => sources.core_contract_controls,
        };
        const name = switch (which) {
            .softmax => "tf_softmax_bf16",
            .rms => "tf_rms_norm_bf16",
            inline else => |w| "tf_ctl_" ++ @tagName(w),
        };
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = which, .pipe = try rig_.pipeline(device, src, name) };
    }

    pub fn deinit(self: *Row) void {
        if (self.r) |r| r.deinit();
        self.rig.deinit();
        self.pipe.deinit();
    }

    pub fn kernel(self: *Row) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt, .control = control(self.which) };
    }

    const vt = g.hooks(Row);

    fn outSize(self: *const Row) usize {
        return c.size(spec(self.which).out);
    }

    pub fn prepare(self: *Row, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if ((self.which == .softmax or self.which == .rms) and (shape.k > 4096 or shape.n != shape.k)) return false;
        if (self.r) |r| r.deinit();
        self.r = null;
        self.rig.reset();
        const r = try rows.Rows.init(self.a, opOf(self.which), shape);
        self.r = r;
        r.fill(fill, seed);
        self.x = try self.rig.input(std.mem.sliceAsBytes(r.x));
        if (self.which == .rms or self.which == .half_dot) self.w = try self.rig.input(std.mem.sliceAsBytes(r.w));
        self.out = try self.rig.output(r.outputs() * self.outSize());
        return true;
    }

    pub fn run(self: *Row) !void {
        const r = self.r.?;
        const k: i32 = @intCast(r.shape.k);
        const p = self.rig.begin();
        const e = p.e;
        e.setPipeline(self.pipe);
        e.setBuffer(self.x.buf, self.x.off, 0);
        const groups = mtl.Size.of(r.shape.rows, 1, 1);
        switch (self.which) {
            .softmax => {
                e.setValue(k, 1);
                e.setBuffer(self.out.buf, self.out.off, 2);
                e.dispatchGroups(groups, mtl.Size.of(rowThreads(r.shape.k), 1, 1));
            },
            .rms => {
                e.setBuffer(self.w.buf, self.w.off, 1);
                e.setValue(r.eps, 2);
                e.setValue(k, 3);
                e.setBuffer(self.out.buf, self.out.off, 4);
                e.dispatchGroups(groups, mtl.Size.of(rowThreads(r.shape.k), 1, 1));
            },
            .atomic_sum => {
                @memset(self.rig.bytes(self.out, r.outputs() * 4), 0);
                e.setValue(k, 1);
                e.setBuffer(self.out.buf, self.out.off, 2);
                e.dispatchThreads(mtl.Size.of(r.shape.k, r.shape.rows, 1), mtl.Size.of(32, 1, 1));
            },
            else => {
                e.setValue(k, 1);
                e.setBuffer(self.out.buf, self.out.off, 2);
                if (self.which == .half_dot) e.setBuffer(self.w.buf, self.w.off, 3);
                e.dispatchGroups(groups, mtl.Size.of(32, 1, 1));
            },
        }
        try self.rig.finish(p);
    }

    pub fn count(self: *Row) usize {
        return self.r.?.outputs();
    }

    pub fn reference(self: *Row, want: []f64, mag: []f64) !void {
        return self.r.?.reference(want, mag);
    }

    pub fn simulate(self: *Row, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(acc, order, out);
    }
};
