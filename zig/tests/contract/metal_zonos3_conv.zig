//! ZONOS3's encoder convolution on Metal under the contract: every layout the front end launches it at
//! (contract_zonos3_conv.zig's CUDA twin), its taps' inputs staged in taps x Cin floats of threadgroup memory.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const ref = @import("metal_zonos3_ref.zig");

const c = contract.spec;
const g = contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;

const Layout = struct { cout: usize, k: usize, cin: usize, taps: usize, stride: usize, pad: usize, dilated: bool = false };

/// The front end's convolutions: the first, a unit's dilated and pointwise ones at 64 and 128, the two strided.
const layouts = [_]Layout{
    .{ .cout = 64, .k = 448, .cin = 64, .taps = 7, .stride = 1, .pad = 6, .dilated = true },
    .{ .cout = 64, .k = 7, .cin = 1, .taps = 7, .stride = 1, .pad = 6 },
    .{ .cout = 64, .k = 64, .cin = 64, .taps = 1, .stride = 1, .pad = 0 },
    .{ .cout = 128, .k = 256, .cin = 64, .taps = 4, .stride = 2, .pad = 2 },
    .{ .cout = 128, .k = 896, .cin = 128, .taps = 7, .stride = 1, .pad = 6, .dilated = true },
    .{ .cout = 128, .k = 128, .cin = 128, .taps = 1, .stride = 1, .pad = 0 },
    .{ .cout = 256, .k = 1024, .cin = 128, .taps = 8, .stride = 4, .pad = 4 },
};

/// A lane's fma chain over every 32nd input, the simdgroup's butterfly, then the bias.
fn convDepth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 6;
}

/// Output rows x output channels x (taps x input channels).
pub fn convSpec(fault: bool) c.Spec {
    const served = comptime blk: {
        var list: [layouts.len]c.Shape = undefined;
        for (&list, layouts) |*x, l| x.* = .{ .rows = if (l.k > 512) 8 else 16, .n = l.cout, .k = l.k };
        const final = list;
        break :blk &final;
    };
    return .{ .name = if (fault) "control: metal zonos3 ze_conv launched with a pad one short" else "metal zonos3 ze_conv", .accum = .f32, .depth = convDepth, .out = .f32, .served = served };
}

/// The encoder's strided, dilated convolution over rows of channels, zero outside the rows.
pub const Conv = struct {
    pipe: mtl.Pipeline,
    /// control: the device's pad is one less
    fault: bool = false,
    l: Layout = undefined,
    dilation: usize = 1,
    T: usize = 0,
    x: []f32 = &.{},
    w: []f32 = &.{},
    b: []f32 = &.{},
    dev: [3]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Conv) c.Spec {
        return convSpec(self.fault);
    }

    pub fn control(self: *const Conv) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *Conv, a: std.mem.Allocator) void {
        a.free(self.x);
        a.free(self.w);
        a.free(self.b);
        self.x = &.{};
        self.w = &.{};
        self.b = &.{};
    }

    pub fn prepare(self: *Conv, a: std.mem.Allocator, rig: *ref.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0) return false;
        const l = for (layouts) |x| {
            if (x.cout == shape.n and x.k == shape.k) break x;
        } else return false;
        self.free(a);
        self.l = l;
        self.T = shape.rows;
        // the units' dilations are 1, 3 and 9; gate 1 takes 9, where each of the first rows feeds one output row
        self.dilation = if (!l.dilated) 1 else if (fill == .nonfinite) 9 else ([_]usize{ 1, 3, 9 })[rng.uintLessThan(usize, 3)];
        const rows = self.T * l.stride;
        self.x = try ref.values(a, rng, rows * l.cin, .random, 1);
        self.w = try ref.values(a, rng, l.cout * l.k, .random, 1);
        self.b = try ref.values(a, rng, l.cout, .random, 1);
        if (fill == .stress) {
            @memset(self.w, 1);
            @memset(self.b, 0);
            for (self.x) |*v| v.* = @floatFromInt(1 + 2 * rng.uintLessThan(u32, 128));
            ref.stress(self.x[0..@min(64, self.x.len)], rng);
        }
        if (fill == .nonfinite) ref.plant(self.x, rows, l.cin);
        self.dev = .{ try ref.input(rig, self.x), try ref.input(rig, self.w), try ref.input(rig, self.b) };
        self.out = try rig.output(self.T * l.cout * 4);
        return true;
    }

    pub fn run(self: *Conv, rig: *ref.Rig) !void {
        const l = self.l;
        const A = extern struct { Cin: i32, T: i32, K: i32, dilation: i32, stride: i32, pad: i32, Cout: i32 };
        const args: A = .{ .Cin = ref.int(l.cin), .T = ref.int(self.T * l.stride), .K = ref.int(l.taps), .dilation = ref.int(self.dilation), .stride = ref.int(l.stride), .pad = ref.int(l.pad * self.dilation - ref.one(self.fault and l.pad > 0)), .Cout = ref.int(l.cout) };
        try rig.go(self.pipe, .{ self.T, (l.cout + 7) / 8, 1 }, 256, &.{ self.dev[0], self.dev[1], self.dev[2], self.out }, args, l.k * 4);
    }

    pub fn count(self: *const Conv) usize {
        return self.T * self.l.cout;
    }

    pub fn solve(self: *const Conv, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const l = self.l;
        const rows: i64 = @intCast(self.T * l.stride);
        const terms = try a.alloc(Iv, l.k);
        defer a.free(terms);
        const scratch = try a.alloc(f64, l.k);
        defer a.free(scratch);
        for (out, 0..) |*o, at| {
            const t = at / l.cout;
            const w = self.w[at % l.cout * l.k ..][0..l.k];
            for (terms, w, 0..) |*term, weight, i| {
                const p = @as(i64, @intCast(t * l.stride + i / l.cin * self.dilation)) - @as(i64, @intCast(l.pad * self.dilation));
                term.* = if (p < 0 or p >= rows) Iv.of(0) else Iv.of(weight).mul(Iv.of(self.x[@as(usize, @intCast(p)) * l.cin + i % l.cin]));
            }
            o.* = s.sum(terms, convDepth(.{ .rows = 1, .n = l.cout, .k = l.k }), scratch).add(Iv.of(self.b[at % l.cout]));
        }
    }
};

/// The convolution and its control.
pub const Set = struct {
    conv: [2]ref.Gated(Conv),

    pub fn init(set: *Set, libs: *ref.Libs, rig: ref.Rig, list: ref.List) !void {
        for (&set.conv, 0..) |*x, j| x.* = .{ .rig = rig, .k = .{ .pipe = try libs.pipe(.encoder, "ze_conv"), .fault = j == 1 } };
        for (&set.conv) |*x| try list.add(x.kernel());
    }

    pub fn deinit(set: *Set) void {
        for (&set.conv) |*x| x.deinit();
    }
};
