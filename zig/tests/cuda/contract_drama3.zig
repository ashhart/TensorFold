//! Drama 3's RMS norm rows (d3_rmsnorm with its plain gamma, the gated-norm form) under the contract on CUDA.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

/// ceil(k / 256) squares a thread, then a warp's tree and the block's tree of 8 warps.
fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 10;
}

const why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

pub fn spec() c.Spec {
    return .{ .name = "cuda drama3 d3_rmsnorm (bf16 rows, plain gamma)", .accum = .f32, .depth = depth, .ops = 8, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = &.{ .{ .rows = 1, .n = 2048, .k = 2048 }, .{ .rows = 168, .n = 2048, .k = 2048 } }, .sums = .{ .bf16 = 0, .why = why } };
}

pub const Norm = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    r: ?rows.Rows = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, mod: cuda.Module) !Norm {
        return .{ .rig = .{ .gpu = gpu, .s = s }, .f = try mod.function("d3_rmsnorm") };
    }

    pub fn deinit(self: *Norm) void {
        if (self.r) |r| r.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Norm) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Norm);

    pub fn prepare(self: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != shape.k) return false;
        if (self.r) |r| r.deinit();
        self.r = null;
        self.rig.reset();
        const r = try rows.Rows.init(self.rig.gpu.gpa, .rms, shape);
        self.r = r;
        r.fill(fill, seed);
        self.x = try self.rig.input(std.mem.sliceAsBytes(r.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(r.w));
        self.out = try self.rig.output(r.outputs() * c.size(spec().out));
        return true;
    }

    pub fn run(self: *Norm) !void {
        const r = self.r.?;
        const k: c_int = @intCast(r.shape.k);
        var a: cuda.Args = .{};
        a.add(self.x);
        a.add(k);
        a.add(self.w);
        a.add(self.out);
        a.add(k);
        a.add(k);
        a.add(r.eps);
        a.add(@as(c_int, 0));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(r.shape.rows), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Norm) usize {
        return self.r.?.outputs();
    }

    pub fn reference(self: *Norm, want: []f64, mag: []f64) !void {
        return self.r.?.reference(want, mag);
    }

    pub fn simulate(self: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(acc, order, out);
    }
};
