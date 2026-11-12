//! BF16 column broadcasts under the HIP kernel contract, with independent FP64 scalar references.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;

pub const Which = enum { multiply, multiply_inplace, fill, shifted_multiply, shifted_fill, lost_tail, fmax_multiply };

fn depth(_: c.Shape) usize {
    return 0;
}

const mul_shapes = [_]c.Shape{
    .{ .rows = 57600, .n = 1536, .k = 1 },
    .{ .rows = 512, .n = 1536, .k = 1 },
    .{ .rows = 17, .n = 1536, .k = 1 },
    .{ .rows = 1, .n = 1536, .k = 1 },
};
const fill_shapes = [_]c.Shape{
    .{ .rows = 57600, .n = 1536, .k = 1 },
    .{ .rows = 14400, .n = 6144, .k = 1 },
    .{ .rows = 512, .n = 256, .k = 1 },
    .{ .rows = 17, .n = 256, .k = 1 },
};
const control_shapes = [_]c.Shape{.{ .rows = 7, .n = 1536, .k = 1 }};

pub fn spec(which: Which) c.Spec {
    return .{
        .name = switch (which) {
            .multiply => "hip BF16 column multiply",
            .multiply_inplace => "hip BF16 column multiply in place",
            .fill => "hip BF16 column fill",
            .shifted_multiply => "control: BF16 column multiply shifted weight",
            .shifted_fill => "control: BF16 column fill shifted weight",
            .lost_tail => "control: BF16 column fill loses tail",
            .fmax_multiply => "control: BF16 column multiply drops NaN",
        },
        .accum = .none,
        .depth = depth,
        .ops = if (which == .fill or which == .shifted_fill or which == .lost_tail) 0 else 1,
        .out = .bf16,
        .in_place = which == .multiply_inplace,
        .tiles = .{ .{ .size = 4 }, .{ .size = 256 }, .{} },
        .served = switch (which) {
            .multiply, .multiply_inplace => &mul_shapes,
            .fill => &fill_shapes,
            else => &control_shapes,
        },
    };
}

pub const Columns = struct {
    rig: rig_.Rig,
    which: Which,
    launcher: *const hip.Launcher,
    control_fn: ?hip.Function,
    host: ?rows.Rows = null,
    x: u64 = 0,
    weight: u64 = 0,
    out: u64 = 0,

    pub fn init(a: std.mem.Allocator, r: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, ctl: hip.Module, which: Which) !Columns {
        const f: ?hip.Function = switch (which) {
            .multiply, .multiply_inplace, .fill => null,
            inline else => |w| try ctl.function("tf_ctl_columns_" ++ @tagName(w)),
        };
        return .{ .rig = .{ .a = a, .r = r, .s = s }, .which = which, .launcher = l, .control_fn = f };
    }

    pub fn deinit(v: *Columns) void {
        if (v.host) |h| h.deinit();
        v.rig.deinit();
    }

    const vt = g.hooks(Columns);

    pub fn kernel(v: *Columns) g.Kernel {
        return .{
            .spec = spec(v.which),
            .ctx = v,
            .vt = &vt,
            .control = switch (v.which) {
                .multiply, .multiply_inplace, .fill => null,
                .fmax_multiply => .nonfinite,
                else => .shapes,
            },
        };
    }

    fn isMul(v: *Columns) bool {
        return v.which != .fill and v.which != .shifted_fill and v.which != .lost_tail;
    }

    pub fn prepare(v: *Columns, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != 1 or shape.n == 0 or shape.rows == 0 or shape.n > 65536) return false;
        if (v.host) |h| h.deinit();
        v.host = null;
        v.rig.reset();
        const h = try rows.Rows.init(v.rig.a, .rms, .{ .rows = shape.rows, .n = shape.n, .k = shape.n });
        v.host = h;
        h.fill(fill, seed);
        if (fill == .nonfinite) {
            const plants = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), 0 };
            for (plants, 0..) |f, i| if (i < h.w.len) {
                h.w[i] = c.toBf16(f);
            };
        }
        if (v.which == .multiply_inplace) {
            v.x = try v.rig.output(h.x.len * 2);
            try v.rig.put(v.x, std.mem.sliceAsBytes(h.x));
        } else if (v.isMul()) v.x = try v.rig.input(std.mem.sliceAsBytes(h.x));
        v.weight = try v.rig.input(std.mem.sliceAsBytes(h.w));
        v.out = if (v.which == .multiply_inplace) v.x else try v.rig.output(h.x.len * 2);
        return true;
    }

    pub fn run(v: *Columns) !void {
        const h = v.host.?;
        if (v.which == .multiply_inplace) try v.rig.put(v.x, std.mem.sliceAsBytes(h.x));
        if (v.control_fn) |f| {
            var args: hip.launch.Args = .{};
            try args.add(v.x);
            try args.add(v.weight);
            try args.add(v.out);
            try args.add(@as(i64, @intCast(h.x.len)));
            try args.add(@as(c_int, @intCast(h.shape.k)));
            try hip.launch.launch(f, .{ .grid = .{ .x = @intCast((h.x.len + 255) / 256) }, .block = .{ .x = 256 } }, v.rig.s, &args);
        } else {
            var arena = hip.Arena.counting();
            const o: hip.ops.Ops = .{ .l = v.launcher, .bf16 = true, .stream = v.rig.s.handle, .arena = &arena };
            if (v.isMul()) {
                try hip.columns.multiply(o, .{ .ptr = v.x, .kind = .bf16 }, v.weight, .{ .ptr = v.out, .kind = .bf16 }, h.shape.rows, h.shape.k);
            } else try hip.columns.fill(o, v.weight, .{ .ptr = v.out, .kind = .bf16 }, h.shape.rows, h.shape.k);
        }
        try v.rig.s.synchronize();
    }

    pub fn count(v: *Columns) usize {
        return v.host.?.x.len;
    }

    pub fn reference(v: *Columns, want: []f64, mag: []f64) !void {
        const h = v.host.?;
        for (want, mag, 0..) |*out, *bound, i| {
            const weight: f64 = c.fromBf16(h.w[i % h.shape.k]);
            out.* = if (v.isMul()) @as(f64, c.fromBf16(h.x[i])) * weight else weight;
            bound.* = @abs(out.*);
        }
    }

    pub fn simulate(v: *Columns, _: c.Accum, _: g.Order, out: []f64) !void {
        const mag = try v.rig.a.alloc(f64, out.len);
        defer v.rig.a.free(mag);
        try v.reference(out, mag);
    }
};
