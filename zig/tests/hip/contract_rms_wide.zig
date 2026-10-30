//! streams.hpp's split wide norm under the contract: tf_fn_rms_wide's replacement at Flash Next's MTP row.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

const width = hip.Launcher.fn_wide_width;

/// Forty strided squares a thread, its wave's xor tree, then eight waves in order.
fn depth(_: c.Shape) usize {
    return width / 256 + 14;
}

/// Rows 1 to 16, each through rmsWide, so the build's gate picks the launches.
const served = blk: {
    var s: [16]c.Shape = undefined;
    for (&s, 1..) |*x, r| x.* = .{ .rows = r, .n = width, .k = width };
    break :blk s;
};

const why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

pub fn spec() c.Spec {
    return .{ .name = "hip RMS norm tf_fn_rms_wide_waves + tf_fn_rms_wide_apply (bf16)", .accum = .f32, .depth = depth, .ops = 8, .out = .bf16, .replaces = "tf_fn_rms_wide", .served = &served, .sums = .{ .bf16 = 0, .why = why } };
}

pub const Wide = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    r: ?rows.Rows = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,
    waves: u64 = 0,
    base: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) Wide {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l };
    }

    pub fn deinit(self: *Wide) void {
        if (self.r) |r| r.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Wide) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Wide);

    pub fn prepare(self: *Wide, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != width or shape.n != width) return false;
        if (self.r) |r| r.deinit();
        self.r = null;
        self.rig.reset();
        const a = self.rig.a;
        const r = try rows.Rows.init(a, .rms, shape);
        self.r = r;
        r.fill(fill, seed);
        self.x = try self.rig.input(std.mem.sliceAsBytes(r.x));
        const wf = try a.alloc(f32, r.w.len);
        defer a.free(wf);
        for (wf, r.w) |*o, b| o.* = c.fromBf16(b);
        self.w = try self.rig.input(std.mem.sliceAsBytes(wf));
        self.out = try self.rig.output(shape.rows * width * 2);
        self.waves = try self.rig.scratch(shape.rows * hip.Launcher.fn_wide_waves * 4);
        self.base = try self.rig.scratch(shape.rows * width * 2);
        return true;
    }

    /// rmsWide into `y`: with scratch it may take the split launches, without it tf_fn_rms_wide.
    fn norm(self: *Wide, y: u64, waves: u64) !void {
        const r = self.r.?;
        try self.l.fn_rms_wide(@ptrFromInt(self.x), @ptrFromInt(self.w), @ptrFromInt(y), if (waves == 0) null else @ptrFromInt(waves), 2, @intCast(r.shape.rows), width, r.eps, self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Wide) !void {
        try self.norm(self.out, self.waves);
    }

    pub fn replaced(self: *Wide) ![]const u8 {
        try self.norm(self.base, 0);
        return self.rig.bytes(self.base, self.r.?.shape.rows * width * 2);
    }

    pub fn count(self: *Wide) usize {
        return self.r.?.outputs();
    }

    pub fn reference(self: *Wide, want: []f64, mag: []f64) !void {
        return self.r.?.reference(want, mag);
    }

    pub fn simulate(self: *Wide, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(acc, order, out);
    }
};
