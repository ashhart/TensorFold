//! The CPU RMS norm under the contract: Qwen3.5-2B's hidden rows, and four planted controls.
const std = @import("std");
const contract = @import("contract");
const k = @import("cpu_mixers").ops.kernels;
const rig_ = @import("contract_rig.zig");

const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;

pub const Which = enum { norm, sanitize, in_place, shift, bf16 };

/// Sixteen squares a lane in an fma chain, the 32 lanes' tree, then each 512-wide block's sum in turn.
fn depth(s: c.Shape) usize {
    return 16 + 5 + (s.k + 511) / 512;
}

const sums_why = "the output follows the squares' sum as ss^-1/2 into bf16: a bf16 sum's error moves it by a few of its ulps, not 100";
/// The 2B's hidden rows: a 128-row prompt chunk, a 32-row verify window, one row a decode step.
const served = [_]c.Shape{ .{ .rows = 128, .n = 2048, .k = 2048 }, .{ .rows = 32, .n = 2048, .k = 2048 }, .{ .rows = 1, .n = 2048, .k = 2048 } };
const ctl = [_]c.Shape{.{ .rows = 8, .n = 1024, .k = 1024 }};

pub fn spec(w: Which) c.Spec {
    const name: []const u8 = switch (w) {
        .norm => "cpu RMS norm",
        .sanitize => "control: norm taking a non-finite input as zero",
        .in_place => "control: norm writing its output over its input",
        .shift => "control: norm scaling by the next element's weight",
        .bf16 => "control: norm summing its squares in bf16",
    };
    return .{ .name = name, .accum = .f32, .depth = depth, .ops = 5, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 512, .step = 512, .min = 512 } }, .served = if (w == .norm) &served else &ctl, .sums = .{ .bf16 = 0, .why = sums_why } };
}

/// The gate each control must fail.
pub fn control(w: Which) ?g.Gate {
    return switch (w) {
        .norm => null,
        .sanitize => .nonfinite,
        .in_place => .repeat,
        .shift => .shapes,
        .bf16 => .sums,
    };
}

pub const Norm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    runs: usize = 0,
    r: ?rows_.Rows = null,
    x: []u16 = &.{},
    w: []u16 = &.{},
    scratch: []u16 = &.{},
    out: []u16 = &.{},

    pub fn init(a: std.mem.Allocator, which: Which) Norm {
        return .{ .a = a, .rig = .{ .a = a }, .which = which };
    }

    pub fn deinit(n: *Norm) void {
        n.drop();
        n.rig.deinit();
    }

    fn drop(n: *Norm) void {
        if (n.r) |r| r.deinit();
        n.r = null;
        n.rig.reset();
    }

    pub fn kernel(n: *Norm) g.Kernel {
        return .{ .spec = spec(n.which), .ctx = n, .vt = &vt, .control = control(n.which) };
    }

    const vt = g.hooks(Norm);

    pub fn prepare(n: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        n.drop();
        if (shape.rows == 0 or shape.k == 0 or shape.k % 512 != 0) return false;
        const r = try rows_.Rows.init(n.a, .rms, shape);
        n.r = r;
        r.fill(fill, seed);
        n.x = try n.rig.input(u16, r.x);
        n.w = try n.rig.input(u16, r.w);
        n.scratch = try n.rig.scratch(u16, @max(r.x.len, r.w.len));
        n.out = try n.rig.output(u16, r.outputs());
        return true;
    }

    pub fn run(n: *Norm) !void {
        const r = n.r.?;
        const width = r.shape.k;
        const threads = 1 + n.runs % @import("contract_projection.zig").thread_counts;
        n.runs += 1;
        switch (n.which) {
            .norm => try k.rmsnormOn(n.x, n.w, r.eps, r.shape.rows, n.out, threads),
            .sanitize => {
                const clean = n.scratch[0..n.x.len];
                for (clean, n.x) |*v, h| v.* = if (std.math.isFinite(k.f32of(h))) h else 0;
                try k.rmsnormOn(clean, n.w, r.eps, r.shape.rows, n.out, threads);
            },
            .in_place => {
                try k.rmsnormOn(n.x, n.w, r.eps, r.shape.rows, n.out, threads);
                @memcpy(n.x, n.out);
            },
            .shift => {
                const shifted = n.scratch[0..width];
                for (shifted, 0..) |*v, i| v.* = n.w[(i + 1) % width];
                try k.rmsnormOn(n.x, shifted, r.eps, r.shape.rows, n.out, threads);
            },
            .bf16 => for (0..r.shape.rows) |row| {
                const x = n.x[row * width ..][0..width];
                var ss: f32 = 0;
                for (x) |h| ss = k.f32of(k.bf16of(@mulAdd(f32, k.f32of(h), k.f32of(h), ss)));
                const inv = k.rsqrt(ss / @as(f32, @floatFromInt(width)) + r.eps);
                for (n.out[row * width ..][0..width], x, n.w) |*o, h, ww| o.* = k.bf16of(k.f32of(ww) * (k.f32of(h) * inv));
            },
        }
    }

    pub fn count(n: *Norm) usize {
        return n.r.?.outputs();
    }

    pub fn reference(n: *Norm, want: []f64, mag: []f64) !void {
        return n.r.?.reference(want, mag);
    }

    pub fn simulate(n: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        return n.r.?.simulate(acc, order, out);
    }
};

test "each control binds to its gate, and the norm's served rows are the 2B's hidden width" {
    try std.testing.expectEqual(g.Gate.nonfinite, control(.sanitize).?);
    try std.testing.expectEqual(g.Gate.repeat, control(.in_place).?);
    try std.testing.expectEqual(g.Gate.shapes, control(.shift).?);
    try std.testing.expectEqual(g.Gate.sums, control(.bf16).?);
    try std.testing.expectEqual(@as(usize, 25), depth(.{ .rows = 1, .n = 2048, .k = 2048 }));
    for (served) |s| try std.testing.expectEqual(s.n, s.k);
}
