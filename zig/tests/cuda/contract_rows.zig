//! Row kernels under the contract on CUDA: the RMS norm rows, Nemotron's router logits and top-k, and the controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

pub const Which = enum { rms_vec, rms_tree, rms_f32w, rms_f32w_bad_dtype, fmax_max, atomic_sum, scribble_sum, tail_sum, half_dot };

/// 32 fma a thread over 16-byte loads, then 256 threads' tree.
fn vecDepth(_: c.Shape) usize {
    return 42;
}

/// Eight squared terms per vector load, then the 256-thread reduction tree.
fn f32WeightDepth(shape: c.Shape) usize {
    return (shape.k + 2047) / 2048 * 8 + 8;
}

/// Every 1,024th square a thread, then the 1,024-thread tree.
fn treeDepth(s: c.Shape) usize {
    return (s.k + 1023) / 1024 + 12;
}

fn laneChain(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5;
}

const f32w_why = "BF16 output rounding can hide BF16 reduction error; the FP16 replay must still miss the exact-sum bound 100x";

const vec_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps (the fp32-out form proves it)";

pub fn spec(which: Which) c.Spec {
    const ctl = &[_]c.Shape{.{ .rows = 8, .n = 1, .k = 4096 }};
    return switch (which) {
        .rms_vec => .{ .name = "cuda RMS norm tf_rms_rows_bf16 (16-byte form)", .accum = .f32, .depth = vecDepth, .ops = 8, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 2048, .step = 8, .min = 8 } }, .served = &.{ .{ .rows = 17, .n = 2688, .k = 2688 }, .{ .rows = 3, .n = 7168, .k = 7168 } }, .sums = .{ .bf16 = 0, .why = vec_why } },
        .rms_f32w, .rms_f32w_bad_dtype => .{ .name = if (which == .rms_f32w) "cuda RMS norm tf_add_rms_rows_bf16h_f32w (Qwen GDN scales)" else "control: FP32 GDN scales read as BF16", .accum = .f32, .depth = f32WeightDepth, .ops = 8, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 128, .step = 64, .min = 64 } }, .served = &.{ .{ .rows = 48, .n = 128, .k = 128 }, .{ .rows = 6144, .n = 128, .k = 128 }, .{ .rows = 196608, .n = 128, .k = 128 } }, .sums = .{ .bf16 = 0, .why = f32w_why } },
        .rms_tree => .{ .name = "cuda RMS norm tf_rms_rows (f16 in, fp32 out)", .accum = .f32, .depth = treeDepth, .ops = 8, .out = .f32, .tiles = .{ .{}, .{}, .{ .size = 1024 } }, .served = &.{ .{ .rows = 17, .n = 2688, .k = 2688 }, .{ .rows = 3, .n = 9216, .k = 9216 } } },
        .fmax_max => .{ .name = "control: fmax row max", .accum = .none, .depth = laneChain, .out = .f32, .served = ctl },
        .atomic_sum => .{ .name = "control: atomic-add row sum", .accum = .f32, .depth = laneChain, .out = .f32, .served = ctl },
        .scribble_sum => .{ .name = "control: row sum writing its input", .accum = .f32, .depth = laneChain, .out = .f32, .served = ctl },
        .tail_sum => .{ .name = "control: row sum losing a tail input", .accum = .f32, .depth = laneChain, .out = .f32, .tiles = .{ .{}, .{}, .{ .size = 32 } }, .served = ctl },
        .half_dot => .{ .name = "control: dot summed in half", .accum = .f32, .depth = laneChain, .out = .f32, .served = ctl },
    };
}

pub fn control(which: Which) ?g.Gate {
    return switch (which) {
        .rms_vec, .rms_tree, .rms_f32w => null,
        .rms_f32w_bad_dtype => .shapes,
        .fmax_max => .nonfinite,
        .atomic_sum, .scribble_sum => .repeat,
        .tail_sum => .shapes,
        .half_dot => .sums,
    };
}

fn opOf(w: Which) rows.Op {
    return switch (w) {
        .rms_vec, .rms_tree, .rms_f32w, .rms_f32w_bad_dtype => .rms,
        .fmax_max => .max,
        .half_dot => .dot,
        else => .sum,
    };
}

pub const Row = struct {
    rig: rig_.Rig,
    which: Which,
    norms: *const cuda.norms.Norms,
    ctl: ?cuda.Function = null,
    r: ?rows.Rows = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,
    wf: []f32 = &.{},

    pub fn init(gpu: check.Gpu, s: cuda.Stream, norms: *const cuda.norms.Norms, ctl_mod: ?cuda.Module, which: Which) !Row {
        var row = Row{ .rig = .{ .gpu = gpu, .s = s }, .which = which, .norms = norms };
        if (control(which) != null and which != .rms_f32w_bad_dtype) row.ctl = try ctl_mod.?.function(switch (which) {
            inline else => |w| "tf_ctl_" ++ @tagName(w),
        });
        return row;
    }

    pub fn deinit(self: *Row) void {
        if (self.r) |r| r.deinit();
        if (self.wf.len != 0) self.rig.gpu.gpa.free(self.wf);
        self.rig.deinit();
    }

    pub fn kernel(self: *Row) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt, .control = control(self.which) };
    }

    const vt = g.hooks(Row);

    pub fn prepare(self: *Row, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.which == .rms_vec and (shape.k % 8 != 0 or shape.k > 8192)) return false;
        const f32w = self.which == .rms_f32w or self.which == .rms_f32w_bad_dtype;
        if (f32w and (shape.k % 64 != 0 or shape.k > 8192)) return false;
        if (!f32w and opOf(self.which) == .rms and shape.n != shape.k) return false;
        if (self.r) |r| r.deinit();
        self.r = null;
        self.rig.reset();
        if (self.wf.len != 0) self.rig.gpu.gpa.free(self.wf);
        self.wf = &.{};
        const a = self.rig.gpu.gpa;
        var r = try rows.Rows.init(a, opOf(self.which), shape);
        if (f32w) r.eps = 1e-6;
        self.r = r;
        r.fill(fill, seed);
        // Integer squares scaled by 256 stay exactly summable in FP32 while overflowing FP16 even at GDN width 128.
        if (f32w and fill == .stress) for (r.x) |*value| {
            value.* = c.toBf16(c.fromBf16(value.*) * 16);
        };
        if (self.which == .rms_tree) {
            const h = try a.alloc(f16, r.x.len);
            defer a.free(h);
            for (h, r.x) |*o, b| o.* = @floatCast(c.fromBf16(b));
            self.x = try self.rig.input(std.mem.sliceAsBytes(h));
        } else self.x = try self.rig.input(std.mem.sliceAsBytes(r.x));
        if (opOf(self.which) == .rms) {
            const wf = try a.alloc(f32, r.w.len);
            defer a.free(wf);
            for (wf, r.w, 0..) |*o, b, i| o.* = c.fromBf16(b) + if (f32w) @as(f32, @floatFromInt(i % 7 + 1)) * 0.000731 else 0;
            if (f32w and fill == .nonfinite) {
                wf[0] = std.math.nan(f32);
                wf[1] = std.math.inf(f32);
                wf[2] = -std.math.inf(f32);
            }
            if (f32w) self.wf = try a.dupe(f32, wf);
            self.w = try self.rig.input(std.mem.sliceAsBytes(if (f32w) self.wf else wf));
        } else if (self.which == .half_dot) self.w = try self.rig.input(std.mem.sliceAsBytes(r.w));
        self.out = try self.rig.output(r.outputs() * c.size(spec(self.which).out));
        return true;
    }

    pub fn run(self: *Row) !void {
        const r = self.r.?;
        const s = self.rig.s;
        const k: c_int = @intCast(r.shape.k);
        switch (self.which) {
            .rms_vec => try self.norms.rms(s, self.x, .bf16, self.w, self.out, .bf16, r.shape.rows, r.shape.k, r.eps),
            .rms_f32w, .rms_f32w_bad_dtype => try self.norms.addRmsH(s, self.x, null, self.w, if (self.which == .rms_f32w) .f32 else .bf16, 0, self.out, null, r.shape.rows, r.shape.k, r.eps),
            .rms_tree => try self.norms.rms(s, self.x, .f16, self.w, self.out, .f32, r.shape.rows, r.shape.k, r.eps),
            .atomic_sum => {
                try self.rig.zero(self.out, r.outputs() * 4);
                var a: cuda.Args = .{};
                a.add(self.x);
                a.add(k);
                a.add(self.out);
                try cuda.launch.launch(self.ctl.?, .{ .grid = .{ .x = @intCast((r.shape.k + 255) / 256), .y = @intCast(r.shape.rows), .z = 1 }, .block = .{ .x = 256 } }, s, &a);
            },
            else => {
                var a: cuda.Args = .{};
                a.add(self.x);
                a.add(k);
                a.add(self.out);
                if (self.which == .half_dot) a.add(self.w);
                try cuda.launch.launch(self.ctl.?, .{ .grid = .{ .x = @intCast(r.shape.rows), .y = 1, .z = 1 }, .block = .{ .x = 32 } }, s, &a);
            },
        }
        try s.synchronize();
    }

    pub fn count(self: *Row) usize {
        return self.r.?.outputs();
    }

    pub fn reference(self: *Row, want: []f64, mag: []f64) !void {
        try self.r.?.reference(want, mag);
        self.scaleReference(want);
        self.scaleReference(mag);
    }

    /// RMS is linear in its scale; retain the actual FP32 scale in the shared FP64 formula.
    fn scaleReference(self: *Row, values: []f64) void {
        if (self.wf.len == 0) return;
        for (values, 0..) |*value, i| value.* *= @as(f64, self.wf[i % self.wf.len]) / c.fromBf16(self.r.?.w[i % self.wf.len]);
    }

    pub fn simulate(self: *Row, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.r.?.simulate(acc, order, out);
        self.scaleReference(out);
    }
};

/// Diagnostic subset; both entries also run in the complete registered CUDA contract.
pub fn f32Weights(gpu: check.Gpu) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var norms = try cuda.norms.Norms.load(gpu.d);
    defer norms.unload();
    var cases = [_]Row{
        try Row.init(gpu, stream, &norms, null, .rms_f32w),
        try Row.init(gpu, stream, &norms, null, .rms_f32w_bad_dtype),
    };
    defer for (&cases) |*case| case.deinit();
    const kernels = [_]g.Kernel{ cases[0].kernel(), cases[1].kernel() };
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, &kernels, 0x5135f32, &tally);
    std.debug.print("SUMMARY {d} kernels: {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ tally.kernels, tally.passed, tally.failed, tally.controls, tally.caught, tally.missed });
    try check.expect(tally.problems() == 0, "Qwen FP32 scales: {d} gate failures, {d} controls missed", .{ tally.failed, tally.missed });
}
