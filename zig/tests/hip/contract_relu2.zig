//! The stream tile with Nemotron's relu^2 as its epilogue under the contract on HIP, at each width oQ checkpoints hold.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const Allocator = std.mem.Allocator;
const Launcher = hip.Launcher;
const c = contract.spec;
const g = contract.gates;

const Kernels = @FieldType(hip.Launcher, "affine");
/// launches/affine.zig's Arg, as relu2Go takes it.
pub const Arg = @typeInfo(@TypeOf(Kernels.relu2Go)).@"fn".param_types[2].?;

const group = 64;
/// The widths Nemotron's MLX and oQ checkpoints store the relu^2 products at.
pub const widths = [_]u8{ 4, 5, 6, 8 };

/// A group's codes by dot2 pairs, its scale and bias, then the groups' sum: one fp32 chain over k.
fn depth(s: c.Shape) usize {
    return s.k / 2 + s.k / group + 8;
}

fn name(comptime bits: u8, k: kit.Kind, plant: bool) []const u8 {
    const fmt = "hip affine_dot2_stream_relu2 ({d}-bit, {s})";
    const half = std.fmt.comptimePrint(fmt, .{ bits, "fp16" });
    const kinds = [2][]const u8{ half, std.fmt.comptimePrint(fmt, .{ bits, "bf16" }) };
    const control = std.fmt.comptimePrint("control: the {d}-bit relu2 stream tile reads its codes as 4-bit", .{bits});
    return if (plant) control else kinds[@intFromBool(k == .bf16)];
}

fn nameOf(bits: u8, k: kit.Kind, plant: bool) []const u8 {
    inline for (widths) |b| if (b == bits) return name(b, k, plant);
    unreachable;
}
const bf16_why = "the bound's two roundings to bf16 cap the miss of a sum relu turns to 0 at 85x";

pub fn spec(k: kit.Kind, bits: u8, plant: bool) c.Spec {
    return .{
        .name = nameOf(bits, k, plant),
        .accum = .f32,
        .depth = depth,
        .ops = 1,
        // the sum rounded to the kind, then squared
        .mids = 2,
        .out = kit.elem(k),
        .replaces = "affine_dot2_stream then tf_nemo_relu2",
        .nan = .{ .stated = "the unfused bytes and every float64 output class" },
        // a one-row layer's routed fc1 and shared up, and two rows
        .served = &.{
            .{ .rows = 1, .n = 1856, .k = 2688 },
            .{ .rows = 1, .n = 3712, .k = 2688 },
            .{ .rows = 2, .n = 1856, .k = 2688 },
        },
        .sums = if (k == .bf16) .{ .f16 = 40, .bf16 = 40, .why = bf16_why } else .{},
    };
}

/// x [rows][k] of the kind times the MLX matrix [n][k] at `bits`: relu(round(y))^2 [rows][n] of the kind.
pub const Relu2 = struct {
    rig: rig_.Rig,
    l: *const Launcher,
    k: kit.Kind,
    bits: u8,
    /// The planted control: the tile told 4-bit over codes packed at `bits`, which gate 3 must turn red.
    plant: bool,
    arena: std.heap.ArenaAllocator,
    f: ?contract.affine.Affine = null,
    arg: Arg = undefined,
    sum: u64 = 0,
    base: u64 = 0,

    pub fn init(a: Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const Launcher, bits: u8, plant: bool) Relu2 {
        const rig: rig_.Rig = .{ .a = a, .r = rt, .s = s };
        return .{ .rig = rig, .l = l, .k = kit.act(), .bits = bits, .plant = plant, .arena = .init(a) };
    }

    pub fn deinit(self: *Relu2) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Relu2) g.Kernel {
        const control: ?g.Gate = if (self.plant) .shapes else null;
        return .{ .spec = spec(self.k, self.bits, self.plant), .ctx = self, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Relu2);

    pub fn prepare(self: *Relu2, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k % group != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        const bits = self.bits;
        var f = try contract.affine.Affine.init(a, shape, bits, group);
        try f.fill(fill, seed);
        // gate 4's rows sum to a negative product, which relu would hide: x negated makes it positive
        if (fill == .stress) for (f.x) |*b| {
            b.* ^= 0x8000;
        };
        if (self.k == .f16) for (f.x) |*b| {
            if (@abs(c.fromBf16(b.*)) < 0x1p-14) b.* = 0; // fp16 normals only, so each bf16 value is exact
        };
        self.f = f;
        const xb = try a.alloc(u8, f.x.len * 2);
        for (f.x, 0..) |b, i| kit.put(self.k, xb, i, c.fromBf16(b));
        const w = try a.alloc(u32, shape.n * shape.k * bits / 32);
        f.words(w);
        const x = try self.rig.input(xb);
        const words = try self.rig.input(std.mem.sliceAsBytes(w));
        const scale = try self.rig.input(std.mem.sliceAsBytes(f.scales));
        const bias = try self.rig.input(std.mem.sliceAsBytes(f.biases));
        const out = try self.rig.output(f.outputs() * 2);
        self.sum = try self.rig.scratch(f.outputs() * 4);
        self.base = try self.rig.scratch(f.outputs() * 2);
        self.arg = .{
            .x = x,
            .words = words,
            .scale = .{ .p = scale, .kind = 1 }, // bf16 tables
            .bias = .{ .p = bias, .kind = 1 },
            .out = 0,
            .m = @intCast(shape.rows),
            .n = @intCast(shape.n),
            .k = @intCast(shape.k),
            .bits = if (self.plant) 4 else bits,
            .group = group,
            .fp16 = @intFromBool(self.k == .f16),
            .out16 = out,
        };
        return true;
    }

    pub fn run(self: *Relu2) !void {
        try self.l.affine.relu2Go(self.l.r, self.arg, 1, self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    /// The plain stream tile's fp32 sums at the codes' own width, then tf_nemo_relu2 into the kind.
    pub fn replaced(self: *Relu2) ![]const u8 {
        var plain = self.arg;
        plain.bits = self.bits;
        plain.out = self.sum;
        plain.out16 = 0;
        try self.l.affine.streamGo(self.l.r, plain, .{}, plain.n, 1, self.rig.s.handle);
        const n = self.f.?.outputs();
        var a: hip.launch.Args = .{};
        for ([_]u64{ self.sum, self.base }) |v| try a.add(v);
        try a.add(@backingInt(self.k));
        try a.add(@as(c_longlong, @intCast(n)));
        const cfg: hip.launch.Config = .{ .grid = .{ .x = @intCast((n + 255) / 256) }, .block = .{ .x = 256 } };
        try hip.launch.launch(self.l.nemo.relu2, cfg, self.rig.s, &a);
        return self.rig.bytes(self.base, n * 2);
    }

    /// The original non-finite plants must match every float64 class and the unfused kernels' bytes.
    pub fn check(self: *Relu2, at: c.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const a = self.arena.allocator();
        const out = try a.dupe(u8, try self.rig.outputBytes());
        if (!std.mem.eql(u8, out, try self.replaced())) return "non-finite output differs from unfused kernels' bytes";
        const want = try a.alloc(f64, self.count());
        const mag = try a.alloc(f64, self.count());
        try self.reference(want, mag);
        for (want, 0..) |w, i| {
            if (c.classOf(c.read(kit.elem(self.k), out, i)) != c.classOf(w))
                return "non-finite output class differs from float64";
        }
        return null;
    }

    pub fn count(self: *Relu2) usize {
        return self.f.?.outputs();
    }

    /// relu(y)^2 of the float64 product (NaN kept as torch keeps it), the square's slope on the product's magnitude.
    pub fn reference(self: *Relu2, want: []f64, mag: []f64) !void {
        try self.f.?.reference(want, mag);
        for (want, mag) |*w, *m| {
            const r = if (std.math.isNan(w.*)) w.* else @max(w.*, 0);
            m.* *= 2 * @abs(r);
            w.* = r * r;
        }
    }

    pub fn simulate(self: *Relu2, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.f.?.simulate(acc, order, out);
        for (out) |*w| {
            const r = if (std.math.isNan(w.*)) w.* else @max(w.*, 0);
            w.* = r * r;
        }
    }
};
