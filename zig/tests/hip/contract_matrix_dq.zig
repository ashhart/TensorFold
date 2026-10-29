//! The prompt tile over dequantized weights against its float64 formula, where each weight rounds to bf16 once.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const Kernels = @FieldType(hip.Launcher, "affine");
/// launches/affine.zig's Arg, as blockGo takes it.
pub const Arg = @typeInfo(@TypeOf(Kernels.blockGo)).@"fn".param_types[2].?;

const bits = 4;
/// The tile's K stage is a group, so only this group takes it.
const group = 32;

/// A K stage's matrix-core steps and the epilogue.
fn depth(s: c.Shape) usize {
    return s.k / 16 + 8;
}

/// Each weight rounds to bf16 before its product: at most this much of the terms' magnitude, on ordinary input.
const weight_round: f64 = 0x1p-9;

/// What gate 4 asks instead: a stage's 16 products enter the matrix core's own fp32 order, so a chain is not exact.
const sums_why = "the matrix core adds a stage's 16 products to the accumulator in its own order and rounding, so a " ++
    "chain of its stages is fp32-deep but not exact in every order; the group terms are gone, which is the point";

pub const name_bf16 = "hip affine_wmma_gemm_dq (4-bit, group 32, bf16 rows, fp32 out)";
pub const name_f16 = "hip affine_wmma_gemm_dq (4-bit, group 32, fp16 rows, fp32 out)";

pub fn spec(k: kit.Kind) c.Spec {
    const served = &[_]c.Shape{
        .{ .rows = 16, .n = 2560, .k = 2560 },
        .{ .rows = 130, .n = 640, .k = 2560 },
        .{ .rows = 129, .n = 2561, .k = 2560 },
        .{ .rows = 17, .n = 641, .k = 640 },
    };
    return .{
        .name = if (k == .bf16) name_bf16 else name_f16,
        .accum = .f32,
        .depth = depth,
        .ops = 1,
        .out = .f32,
        // a 128-row block, 128 outputs a column, and the 32-input stages the sums run in
        .tiles = .{
            .{ .size = 128, .step = 128, .min = 1 },
            .{ .size = 128, .step = 128, .min = 1 },
            .{ .size = 32, .step = 32, .min = 32 },
        },
        .served = served,
        .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = sums_why },
    };
}

/// x [rows][k] times the MLX 4-bit matrix [n][k], every weight read as bf16(scale * code + bias), in fp32.
pub const Dq = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    f: ?contract.affine.Affine = null,
    fill: g.Fill = .random,
    arg: Arg = undefined,

    /// `l` is a launcher loaded with the run's dequantizing switch on: only then does the tile resolve.
    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) Dq {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Dq) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Dq) g.Kernel {
        return .{ .spec = spec(self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Dq);

    pub fn prepare(self: *Dq, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k < group or shape.k % group != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var f = try contract.affine.Affine.init(a, shape, bits, group);
        f.form = .dequantized;
        try f.fill(fill, seed);
        if (self.k == .f16) for (f.x) |*b| {
            if (@abs(c.fromBf16(b.*)) < 0x1p-14) b.* = 0; // fp16 normals only, so each bf16 value is exact
        };
        self.f = f;
        self.fill = fill;
        const xb = try a.alloc(u8, f.x.len * 2);
        for (f.x, 0..) |b, i| kit.put(self.k, xb, i, c.fromBf16(b));
        const w = try a.alloc(u32, shape.n * shape.k * bits / 32);
        f.words(w);
        self.arg = .{
            .x = try self.rig.input(xb),
            .words = try self.rig.input(std.mem.sliceAsBytes(w)),
            .scale = .{ .p = try self.rig.input(std.mem.sliceAsBytes(f.scales)), .kind = 1 },
            .bias = .{ .p = try self.rig.input(std.mem.sliceAsBytes(f.biases)), .kind = 1 },
            .out = try self.rig.output(f.outputs() * 4),
            .m = @intCast(shape.rows),
            .n = @intCast(shape.n),
            .k = @intCast(shape.k),
            .bits = bits,
            .group = group,
            .fp16 = @intFromBool(self.k == .f16),
        };
        return true;
    }

    pub fn run(self: *Dq) !void {
        try self.l.affine.blockGo(self.l.r, self.arg, self.rig.s.handle, 1, .matrix_dq);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Dq) usize {
        return self.f.?.outputs();
    }

    pub fn reference(self: *Dq, want: []f64, mag: []f64) !void {
        try self.f.?.reference(want, mag);
    }

    pub fn simulate(self: *Dq, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.f.?.simulate(acc, order, out);
    }

    /// The sums' bound, plus each weight's one bf16 rounding before its product, where the weights are not integers.
    pub fn bounds(self: *Dq, want: []const f64, mag: []const f64, allow: []f64) !void {
        const s = spec(self.k);
        const round: f64 = if (self.fill == .stress) 0.0 else weight_round;
        for (allow, want, mag) |*limit, w, m| limit.* = c.bound(s, self.f.?.shape, w, m) + round * m;
    }
};
