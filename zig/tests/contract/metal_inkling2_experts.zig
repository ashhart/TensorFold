//! Inkling 2 Mini M3 contracts: FP64 decode/dot/activation, ascending-expert combine, and the grid against main.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const bf = contract.bf16_round.bf16;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.experts;
const controls = @import("metal_inkling2_experts_controls.zig");
const src = @import("inkling2_metal");
/// main's expert kernels (5e390519d0): a 32-thread group a route pair (combine: a token), lanes looping over outputs.
const frozen = @embedFile("baselines/inkling2_experts_main.metal.txt");
/// main's Ink2MatrixArgs, as its launcher bound it.
const MainArgs = extern struct { k: u32, n: u32, topk: u32, experts: u32 };

pub const Kind = enum { nv_gu, nv_down, mx_gu, mx_down, nv_decode, mx_decode, combine };

/// A cell: an entry or control against float64, or a grid operation against main's bytes, as is or with a grid fault.
pub const Variant = enum {
    plain,
    control,
    grid,
    grid_block,
    grid_pair,
    unsorted,

    fn isGrid(v: Variant) bool {
        return v == .grid or v == .grid_block or v == .grid_pair or v == .unsorted;
    }
};

/// The operations the output grid changed: the four products and the combine (the decodes keep their launch).
pub const grid_kinds = [_]Kind{ .nv_gu, .nv_down, .mx_gu, .mx_down, .combine };

fn symbol(k: Kind) [:0]const u8 {
    return switch (k) {
        .nv_gu => "tf_ink2_nvfp4_gu",
        .nv_down => "tf_ink2_nvfp4_down",
        .mx_gu => "tf_ink2_mxfp8_gu",
        .mx_down => "tf_ink2_mxfp8_down",
        .nv_decode => "tf_ink2_nvfp4_quantize",
        .mx_decode => "tf_ink2_mxfp8_quantize",
        .combine => "tf_ink2_combine",
    };
}

fn mutation(k: Kind) controls.Mutation {
    return switch (k) {
        .nv_gu => .nv_gu_up,
        .nv_down => .nv_down_phi,
        .mx_gu => .mx_gu_up,
        .mx_down => .mx_down_scale,
        .nv_decode => .nv_decode_nibble,
        .mx_decode => .mx_decode_scale,
        .combine => .combine_slot_order,
    };
}

fn isNv(k: Kind) bool {
    return k == .nv_gu or k == .nv_down or k == .nv_decode;
}
fn isGu(k: Kind) bool {
    return k == .nv_gu or k == .mx_gu;
}
fn isDown(k: Kind) bool {
    return k == .nv_down or k == .mx_down;
}
fn isDecode(k: Kind) bool {
    return k == .nv_decode or k == .mx_decode;
}
fn depth(s: c.Shape) usize {
    return s.k + 3;
}

pub fn spec(k: Kind, control: bool) c.Spec {
    const base = if (control) mutation(k).name() else switch (k) {
        .nv_gu => "metal inkling2 experts tf_ink2_nvfp4_gu",
        .nv_down => "metal inkling2 experts tf_ink2_nvfp4_down",
        .mx_gu => "metal inkling2 experts tf_ink2_mxfp8_gu",
        .mx_down => "metal inkling2 experts tf_ink2_mxfp8_down",
        .nv_decode => "metal inkling2 experts tf_ink2_nvfp4_quantize",
        .mx_decode => "metal inkling2 experts tf_ink2_mxfp8_quantize",
        .combine => "metal inkling2 experts tf_ink2_combine",
    };
    return .{
        .name = base,
        .accum = if (isDecode(k)) .none else .f32,
        .depth = depth,
        .ops = if (isGu(k)) 12 else if (isDecode(k)) 1 else 3,
        .mids = if (isGu(k)) 3 else 0,
        .out = .bf16,
        .served = switch (k) {
            .nv_gu, .mx_gu => &.{ .{ .rows = 2, .n = 4, .k = 32 }, .{ .rows = 4, .n = 8, .k = 64 } },
            .nv_down, .mx_down => &.{ .{ .rows = 2, .n = 8, .k = 32 }, .{ .rows = 4, .n = 12, .k = 64 } },
            .nv_decode, .mx_decode => &.{ .{ .rows = 2, .n = 32, .k = 32 }, .{ .rows = 3, .n = 64, .k = 64 } },
            .combine => &.{ .{ .rows = 1, .n = 8, .k = 8 }, .{ .rows = 2, .n = 8, .k = 8 }, .{ .rows = 4, .n = 12, .k = 8 } },
        },
        .sums = if (k == .combine) .{ .f16 = 0, .bf16 = 100, .why = "bounded cancellation is exact in fp32 but loses its residual in bf16" } else .{ .f16 = 0, .bf16 = 0, .why = "independent FP64 terms and explicit BF16 nodes" },
    };
}

fn gridName(k: Kind, v: Variant) []const u8 {
    if (v == .unsorted) return "metal inkling2 experts tf_ink2_combine unsorted ids";
    const names: [3][]const u8 = switch (k) {
        .nv_gu => .{ "metal inkling2 experts tf_ink2_nvfp4_gu grid against main", "control: metal inkling2 experts tf_ink2_nvfp4_gu output block dropped", "control: metal inkling2 experts tf_ink2_nvfp4_gu pair index halved" },
        .nv_down => .{ "metal inkling2 experts tf_ink2_nvfp4_down grid against main", "control: metal inkling2 experts tf_ink2_nvfp4_down output block dropped", "control: metal inkling2 experts tf_ink2_nvfp4_down pair index halved" },
        .mx_gu => .{ "metal inkling2 experts tf_ink2_mxfp8_gu grid against main", "control: metal inkling2 experts tf_ink2_mxfp8_gu output block dropped", "control: metal inkling2 experts tf_ink2_mxfp8_gu pair index halved" },
        .mx_down => .{ "metal inkling2 experts tf_ink2_mxfp8_down grid against main", "control: metal inkling2 experts tf_ink2_mxfp8_down output block dropped", "control: metal inkling2 experts tf_ink2_mxfp8_down pair index halved" },
        .combine => .{ "metal inkling2 experts tf_ink2_combine grid against main", "control: metal inkling2 experts tf_ink2_combine output block dropped", "control: metal inkling2 experts tf_ink2_combine token index halved" },
        .nv_decode, .mx_decode => unreachable,
    };
    return names[if (v == .grid_block) 1 else if (v == .grid_pair) 2 else 0];
}

/// A grid cell: output blocks with tails, Mini's widths and 64 rows, every output byte against main's kernel.
pub fn gridSpec(k: Kind, v: Variant) c.Spec {
    var s = spec(k, false);
    s.name = gridName(k, v);
    s.replaces = "main's kernel on its one-group-a-pair launch (5e390519d0)";
    // edges in the outputs a row: 1, 31, 32, 33 and the widest served (combine uses 128-wide groups)
    s.tiles = if (k == .combine) .{ .{}, .{ .size = 128 }, .{} } else .{ .{}, .{ .size = 32 }, .{} };
    s.served = if (v == .unsorted) &.{
        .{ .rows = 4, .n = 129, .k = 8 },
        .{ .rows = 8, .n = 2304, .k = 8 },
    } else switch (k) {
        .nv_gu, .mx_gu => &.{ .{ .rows = 2, .n = 65, .k = 64 }, .{ .rows = 1, .n = 1152, .k = 2304 }, .{ .rows = 64, .n = 33, .k = 32 }, .{ .rows = 3, .n = 64, .k = 32 } },
        .nv_down, .mx_down => &.{ .{ .rows = 2, .n = 65, .k = 64 }, .{ .rows = 1, .n = 2304, .k = 1152 }, .{ .rows = 64, .n = 33, .k = 32 }, .{ .rows = 3, .n = 63, .k = 32 } },
        .combine => &.{ .{ .rows = 2, .n = 129, .k = 8 }, .{ .rows = 4, .n = 2304, .k = 8 }, .{ .rows = 64, .n = 129, .k = 8 } },
        .nv_decode, .mx_decode => unreachable,
    };
    return s;
}

const Sim = struct { acc: c.Accum, order: g.Order };
/// The longest dot a cell's float64 reference holds: Mini's hidden width, GU's reduction.
const max_terms = 2304;
const Sample = struct { value: f64, magnitude: f64 };
const Dot = struct { value: f64, magnitude: f64 };

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipes: host.Pipelines,
    kind: Kind,
    ctl: controls.Mutation,
    variant: Variant = .plain,
    /// main's kernel, for a grid cell's baseline, and the buffer main's launch writes.
    base: ?mtl.Pipeline = null,
    base_out: rig_.Ref = undefined,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u8 = &.{},
    scales: []u8 = &.{},
    amax: []f32 = &.{},
    denominator: []f32 = &.{},
    ids: []i32 = &.{},
    weights: []f32 = &.{},
    refs: [7]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, control: bool) !Probe {
        const text = try controls.apply(a, if (control) mutation(kind) else .none);
        defer a.free(text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipes = try host.Pipelines.loadSource(device, text), .kind = kind, .ctl = if (control) mutation(kind) else .none, .variant = if (control) .control else .plain };
    }

    /// A grid cell: this tree's kernels (with a planted grid fault for a control) and main's frozen ones.
    pub fn initGrid(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, v: Variant) !Probe {
        const m: controls.Mutation = switch (v) {
            .grid_block => .grid_block_drop,
            .grid_pair => .grid_pair_index,
            else => .none,
        };
        const text = try controls.apply(a, m);
        defer a.free(text);
        var pipes = try host.Pipelines.loadSource(device, text);
        errdefer pipes.deinit();
        const lib = try mtl.Library.fromSource(device, frozen, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const base = try mtl.Pipeline.init(device, lib, symbol(kind), false);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipes = pipes, .kind = kind, .ctl = m, .variant = v, .base = base };
    }

    fn clear(p: *Probe) void {
        p.a.free(p.x);
        p.a.free(p.w);
        p.a.free(p.scales);
        p.a.free(p.amax);
        p.a.free(p.denominator);
        p.a.free(p.ids);
        p.a.free(p.weights);
        p.x = &.{};
        p.w = &.{};
        p.scales = &.{};
        p.amax = &.{};
        p.denominator = &.{};
        p.ids = &.{};
        p.weights = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.clear();
        p.pipes.deinit();
        if (p.base) |base| base.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        const s = if (p.variant.isGrid()) gridSpec(p.kind, p.variant) else spec(p.kind, p.ctl != .none);
        const exact = p.kind == .combine and (p.ctl == .combine_slot_order or p.variant == .unsorted);
        return .{ .spec = s, .ctx = p, .vt = if (exact) &vt_exact else &vt, .control = p.ctl.gate() };
    }
    const vt = blk: {
        var table = g.hooks(Probe);
        table.check = null;
        break :blk table;
    };
    const vt_exact = g.hooks(Probe);

    fn call(p: *const Probe) host.Call {
        return .{ .rows = @intCast(p.shape.rows), .k = @intCast(if (p.kind == .combine) 32 else p.shape.k), .n = @intCast(p.shape.n), .experts = if (p.kind == .combine) 8 else 4, .topk = if (p.kind == .combine) 8 else 2, .format = if (isNv(p.kind)) .nvfp4 else .mxfp8 };
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const k = p.kind;
        // a grid cell reaches Mini's widths and 64 rows; the float64 cells stay small
        const grid = p.variant.isGrid();
        if (shape.rows == 0 or shape.rows > @as(usize, if (grid) 64 else 8)) return false;
        const max_k: usize = if (grid) max_terms else 64;
        if (isGu(k) and (shape.n > @as(usize, if (grid) 1152 else 8) or shape.k > max_k or shape.k % 32 != 0)) return false;
        if (isDown(k) and (shape.n > @as(usize, if (grid) 2304 else 12) or shape.k > max_k or shape.k % 32 != 0)) return false;
        if (isDecode(k) and (shape.n != shape.k or shape.k > 64 or shape.k % 32 != 0)) return false;
        if (k == .combine and (shape.n > @as(usize, if (grid) 2304 else 12) or shape.k != 8)) return false;
        p.clear();
        p.shape = shape;
        const nv = isNv(k);
        const topk: usize = if (k == .combine) 8 else 2;
        const routes = shape.rows * topk;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (!isDecode(k)) {
            const nx = if (isGu(k)) shape.rows * shape.k else if (isDown(k)) routes * shape.k else routes * shape.n;
            p.x = try p.a.alloc(u16, nx);
            for (p.x, 0..) |*v, i| v.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (i % 7 == 0) @as(f32, 3) else 1));
        }
        p.ids = try p.a.alloc(i32, routes);
        const combine_ids = [_]i32{ 2, 0, 1, 7, 3, 6, 4, 5 };
        const sorted_ids = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7 };
        const rev_ids = [_]i32{ 7, 6, 5, 4, 3, 2, 1, 0 };
        const stress_alt_ids = [_]i32{ 5, 1, 6, 0, 3, 7, 2, 4 };
        const dup_bad_ids = [_]i32{ 2, 0, -1, 1, 0, 8, 2, 0 };
        for (p.ids, 0..) |*id, i| {
            if (k != .combine) {
                id.* = @intCast(if (i % 2 == 0) 2 else (i / 2) % 2);
                continue;
            }
            const token = i / 8;
            const slot = i % 8;
            if (p.ctl == .combine_slot_order) {
                id.* = combine_ids[slot];
            } else if (p.variant == .unsorted) {
                id.* = switch (token % 3) {
                    0 => combine_ids[slot],
                    1 => rev_ids[slot],
                    else => if (fill == .stress) stress_alt_ids[slot] else dup_bad_ids[slot],
                };
            } else {
                id.* = switch (token % 4) {
                    0 => combine_ids[slot],
                    1 => sorted_ids[slot],
                    2 => rev_ids[slot],
                    else => if (fill == .stress) stress_alt_ids[slot] else dup_bad_ids[slot],
                };
            }
        }
        const matrices: usize = if (isGu(k)) 8 * shape.n else if (isDown(k)) 4 * shape.n else shape.rows;
        if (k != .combine) {
            p.w = try p.a.alloc(u8, matrices * (if (nv) shape.k / 2 else shape.k));
            p.scales = try p.a.alloc(u8, matrices * (shape.k / @as(usize, if (nv) 16 else 32)));
            for (p.w) |*v| v.* = rng.int(u8);
            for (p.scales, 0..) |*v, i| v.* = if (nv) @intCast(0x30 + (i % 17)) else @intCast(125 + (i % 5));
            if (!nv) for (p.w) |*v| if (v.* & 0x7f == 0x7f) {
                v.* = (v.* & 0x80) | 0x7e;
            };
            if (k == .mx_decode) {
                p.w[0] = 0x30;
                p.scales[0] = 255; // Decodes to 2^127 without first forming an overflowing 2^128.
            }
        }
        if (nv) {
            p.amax = try p.a.alloc(f32, if (isDecode(k)) 1 else 4);
            for (p.amax, 0..) |*v, i| v.* = 2688.125 + @as(f32, @floatFromInt(i)) * 0.375;
            p.denominator = try p.a.alloc(f32, 1);
            p.denominator[0] = 2695.75;
        }
        if (k == .combine) p.weights = try p.a.alloc(f32, routes);
        if (fill == .stress and !isDecode(k)) {
            for (p.x, 0..) |*v, i| v.* = c.toBf16(@as(f32, @floatFromInt(128 * (1 + i % 13))) * (if (i >= p.x.len / 2) @as(f32, -1) else 1));
            if (k != .combine) @memset(p.scales, if (nv) 0x38 else 127);
        }
        if (k == .combine) {
            const values = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
            for (p.x, 0..) |*v, i| {
                const token = (i / p.shape.n) / 8;
                v.* = if (fill == .stress or (p.variant == .unsorted and fill == .random and token % 3 != 0))
                    c.toBf16(1)
                else
                    c.toBf16(values[i % 8]);
            }
            if (fill == .stress) {
                const terms = [_]f32{ 1024, 1, 1, 1, 1, 1, 1, -1029 };
                // Map expert indices back to route slots; the ascending sum is 1024 + six 1s - 1029.
                for (0..p.shape.rows) |token| for (0..8) |expert| for (0..8) |slot| if (p.ids[token * 8 + slot] == expert) {
                    p.weights[token * 8 + slot] = terms[expert];
                };
            } else if (p.ctl == .combine_slot_order) {
                const control_weights = [_]f32{ 0x1p127, 0x1p126, -0x1p127 / 3.0, 0.25, -0.2, 1.0 / 3.0, -2.0 / 7.0, 0.375 };
                for (p.weights, 0..) |*v, i| v.* = control_weights[i % 8];
                // Only this planted control overflows route order; ascending experts remain finite.
            } else if (p.variant == .unsorted and fill == .random) {
                const ord = [_]f32{ 0.125, -0.25, 0.375, -0.5, 0.625, -0.75, 0.875, -1 };
                const rev_w = [_]f32{ 0x1p127, 0x1p127, 0.25, -0.5, 0.375, -0.25, 0.125, -0x1p127 };
                const dup_w = [_]f32{ 0.25, 0x1p25, 999.0, 0.5, -0x1p25, -999.0, -0.125, 1.0 };
                for (p.weights, 0..) |*v, i| {
                    const token = i / 8;
                    const slot = i % 8;
                    v.* = switch (token % 3) {
                        0 => ord[slot],
                        1 => rev_w[slot],
                        else => dup_w[slot],
                    };
                }
            } else {
                const ordinary_weights = [_]f32{ 0.125, -0.25, 0.375, -0.5, 0.625, -0.75, 0.875, -1 };
                for (p.weights, 0..) |*v, i| v.* = ordinary_weights[i % 8];
            }
        }
        if (fill == .nonfinite) {
            if (!isDecode(k) and p.x.len >= 3) {
                p.x[0] = c.toBf16(std.math.nan(f32));
                p.x[p.x.len / 2] = c.toBf16(std.math.inf(f32));
                p.x[p.x.len - 1] = c.toBf16(-std.math.inf(f32));
            } else if (isDecode(k) and p.scales.len > 1) {
                if (nv) p.scales[0] = 0x7f else p.w[0] = 0x7f;
            }
        }
        const out = try p.rig.output(p.count() * 2);
        if (k == .combine) {
            p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
            p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
            p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.weights));
            p.refs[3] = out;
        } else if (isDecode(k)) {
            p.refs[0] = try p.rig.input(p.w);
            p.refs[1] = try p.rig.input(p.scales);
            if (nv) {
                p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.amax));
                p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.denominator));
                p.refs[4] = out;
            } else p.refs[2] = out;
        } else {
            p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
            p.refs[1] = try p.rig.input(p.w);
            p.refs[2] = try p.rig.input(p.scales);
            if (nv) {
                p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.amax));
                p.refs[4] = try p.rig.input(std.mem.sliceAsBytes(p.denominator));
                p.refs[5] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
                p.refs[6] = out;
            } else {
                p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
                p.refs[4] = out;
            }
        }
        if (grid) p.base_out = try p.rig.scratch(p.count() * 2);
        return true;
    }

    /// main's bytes on the prepared inputs: its frozen kernel on its own launch, into a buffer poisoned as the output.
    pub fn replaced(p: *Probe) ![]const u8 {
        const base = p.base orelse return error.NoBaseline;
        const len = p.count() * 2;
        @memset(p.rig.bytes(p.base_out, len), 0xff);
        const bound: usize = if (p.kind == .combine) 4 else if (isNv(p.kind)) 7 else 5;
        const c_ = p.call();
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(base);
        for (p.refs[0 .. bound - 1], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setBuffer(p.base_out.buf, p.base_out.off, bound - 1);
        e.setValue(MainArgs{ .k = c_.k, .n = c_.n, .topk = c_.topk, .experts = c_.experts }, bound);
        const groups = if (p.kind == .combine) c_.rows else c_.rows * c_.topk;
        e.dispatchGroups(mtl.Size.of(groups, 1, 1), mtl.Size.of(32, 1, 1));
        try p.rig.finish(pass);
        return p.rig.bytes(p.base_out, len);
    }

    fn hr(b: rig_.Ref) host.Ref {
        return .{ .buf = b.buf, .off = b.off };
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const nv = isNv(p.kind);
        if (isDecode(p.kind)) {
            if (nv) try host.nvfp4Quantize(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[4]), @intCast(p.shape.rows), @intCast(p.shape.k)) else try host.mxfp8Quantize(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), @intCast(p.shape.rows), @intCast(p.shape.k));
        } else if (isGu(p.kind)) {
            if (nv) try host.nvfp4Gu(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[4]), hr(p.refs[5]), hr(p.refs[6]), p.call()) else try host.mxfp8Gu(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[4]), p.call());
        } else if (isDown(p.kind)) {
            if (nv) try host.nvfp4Down(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[4]), hr(p.refs[5]), hr(p.refs[6]), p.call()) else try host.mxfp8Down(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[4]), p.call());
        } else try host.combine(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), p.call());
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return if (isGu(p.kind) or isDown(p.kind)) p.shape.rows * 2 * p.shape.n else p.shape.rows * p.shape.n;
    }

    fn e2(code: u8) f64 {
        const v = ([_]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 })[code & 7];
        return if (code & 8 != 0) -v else v;
    }
    fn e4(code: u8) f64 {
        const exponent = (code >> 3) & 15;
        const mantissa = code & 7;
        if (exponent == 15 and mantissa == 7) return std.math.nan(f64);
        const v = if (exponent == 0) @as(f64, @floatFromInt(mantissa)) / 512 else std.math.ldexp(@as(f64, 8 + @as(u16, mantissa)), @as(i32, exponent) - 10);
        return if (code & 0x80 != 0) -v else v;
    }
    fn weight(p: *const Probe, expert: usize, row: usize, col: usize) f64 {
        const nv = isNv(p.kind);
        const k = p.shape.k;
        const rows_per_expert: usize = if (isGu(p.kind)) 2 * p.shape.n else p.shape.n;
        const matrix_row = if (isDecode(p.kind)) row else expert * rows_per_expert + row;
        if (!nv) {
            const value = p.w[matrix_row * k + col];
            const scale = p.scales[matrix_row * (k / 32) + col / 32];
            return bf(std.math.ldexp(e4(value), @as(i32, scale) - 127));
        }
        const codes = p.w[matrix_row * (k / 2) + col / 2];
        const code = if (col % 2 == 0) codes & 15 else codes >> 4;
        const scale = p.scales[matrix_row * (k / 16) + col / 16];
        const factor = bf(p.amax[if (isDecode(p.kind)) 0 else expert]) / bf(p.denominator[0]);
        return bf(@as(f32, @floatCast(e2(code) * e4(scale) * factor)));
    }

    fn dot(p: *const Probe, expert: usize, row: usize, x: []const u16, sim: ?Sim) Dot {
        var terms: [max_terms]f64 = undefined;
        std.debug.assert(p.shape.k <= max_terms);
        var magnitude: f64 = 0;
        for (0..p.shape.k) |col| {
            const term = p.weight(expert, row, col) * @as(f64, c.fromBf16(x[col]));
            terms[col] = term;
            magnitude += @abs(term);
        }
        return .{ .value = if (sim) |s| rows.sumIn(s.acc, s.order, terms[0..p.shape.k]) else blk: {
            var sum: f64 = 0;
            for (terms[0..p.shape.k]) |term| sum += term;
            break :blk sum;
        }, .magnitude = magnitude };
    }

    fn evaluate(p: *const Probe, at: usize, sim: ?Sim) Sample {
        if (isDecode(p.kind)) return .{ .value = p.weight(0, at / p.shape.k, at % p.shape.k), .magnitude = 0 };
        if (p.kind == .combine) {
            const token = at / p.shape.n;
            const col = at % p.shape.n;
            var terms: [8]f64 = undefined;
            var magnitude: f64 = 0;
            var made: usize = 0;
            for (0..8) |expert| for (0..8) |slot| if (p.ids[token * 8 + slot] == @as(i32, @intCast(expert))) {
                const value = p.weights[token * 8 + slot] * @as(f64, c.fromBf16(p.x[(token * 8 + slot) * p.shape.n + col]));
                terms[made] = value;
                made += 1;
                magnitude += @abs(value);
            };
            const sum = if (sim) |s| rows.sumIn(s.acc, s.order, terms[0..made]) else blk: {
                var total: f64 = 0;
                for (terms[0..made]) |term| total += term;
                break :blk total;
            };
            return .{ .value = bf(sum), .magnitude = magnitude };
        }
        const pair = at / p.shape.n;
        const n = at % p.shape.n;
        const token = pair / 2;
        const expert: usize = @intCast(p.ids[pair]);
        const x = if (isGu(p.kind)) p.x[token * p.shape.k ..][0..p.shape.k] else p.x[pair * p.shape.k ..][0..p.shape.k];
        if (isDown(p.kind)) {
            const d = p.dot(expert, n, x, sim);
            return .{ .value = bf(d.value), .magnitude = d.magnitude };
        }
        const gd = p.dot(expert, 2 * n, x, sim);
        const ud = p.dot(expert, 2 * n + 1, x, sim);
        const gate = bf(gd.value);
        const up = bf(ud.value);
        const left = 4 * std.math.tanh(gate / 4) * (1 / (1 + @exp(-gate)));
        return .{ .value = bf(left * (25 * std.math.tanh(up / 25))), .magnitude = gd.magnitude + ud.magnitude };
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*v, *m, i| {
            const sample = p.evaluate(i, null);
            v.* = sample.value;
            m.* = sample.magnitude;
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*v, i| v.* = p.evaluate(i, .{ .acc = acc, .order = order }).value;
    }
    pub fn check(p: *Probe, gate: g.Gate) !?[]const u8 {
        if (gate != .shapes) return null;
        if (p.kind == .mx_decode and p.ctl == .none) if (try @import("metal_inkling2_decode_exact.zig").check(p.a, p.rig.device, p.rig.queue, &p.pipes)) |why| return why;
        const bytes = try p.rig.outputBytes();
        const want = try p.a.alloc(f64, p.count());
        defer p.a.free(want);
        const mag = try p.a.alloc(f64, p.count());
        defer p.a.free(mag);
        try p.reference(want, mag);
        for (want, 0..) |w, i| {
            const got = c.read(.bf16, bytes, i);
            if (std.math.isNan(w) and std.math.isNan(got)) continue;
            if (got != w) return "route-slot output differs from the ascending-expert reference";
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: [std.enums.values(Kind).len * 2 + grid_kinds.len * 3 + 1]Probe = undefined,
    made: usize = 0,
    grouped: @import("metal_inkling2_experts_grouped.zig").Suite = .{},

    pub fn deinit(s: *Suite) void {
        for (s.probes[0..s.made]) |*probe| probe.deinit();
        s.grouped.deinit();
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(Kind)) |kind| for ([_]bool{ false, true }) |control| {
            const label = if (control) mutation(kind).name() else spec(kind, false).name;
            if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
            s.probes[s.made] = try Probe.init(s.a, s.device, s.queue, kind, control);
            s.made += 1;
            try s.kernels.append(s.a, s.probes[s.made - 1].kernel());
        };
        for (grid_kinds) |kind| for ([_]Variant{ .grid, .grid_block, .grid_pair }) |v| {
            if (only) |needle| if (std.mem.indexOf(u8, gridSpec(kind, v).name, needle) == null) continue;
            s.probes[s.made] = try Probe.initGrid(s.a, s.device, s.queue, kind, v);
            s.made += 1;
            try s.kernels.append(s.a, s.probes[s.made - 1].kernel());
        };
        const unsorted_name = gridSpec(.combine, .unsorted).name;
        if (only == null or std.mem.indexOf(u8, unsorted_name, only.?) != null) {
            s.probes[s.made] = try Probe.initGrid(s.a, s.device, s.queue, .combine, .unsorted);
            s.made += 1;
            try s.kernels.append(s.a, s.probes[s.made - 1].kernel());
        }
        try s.grouped.collect(s.a, s.device, s.queue, s.kernels, only);
    }
};

test "grid cells: the five changed operations, each against main, with two grid controls, under the experts filter" {
    var names: [grid_kinds.len * 3][]const u8 = undefined;
    var made: usize = 0;
    for (grid_kinds) |kind| for ([_]Variant{ .grid, .grid_block, .grid_pair }) |v| {
        const s = gridSpec(kind, v);
        try std.testing.expect(s.replaces != null);
        try std.testing.expect(std.mem.indexOf(u8, s.name, "metal inkling2 experts ") != null);
        try std.testing.expect(std.mem.indexOf(u8, s.name, symbol(kind)) != null);
        // several output blocks at a served shape, and Mini's own width
        var widest: usize = 0;
        for (s.served) |x| widest = @max(widest, x.n);
        try std.testing.expect(s.served[0].n > 64 and widest >= 1152);
        for (names[0..made]) |other| try std.testing.expect(!std.mem.eql(u8, other, s.name));
        names[made] = s.name;
        made += 1;
    };
    try std.testing.expect(std.mem.indexOf(u8, frozen, "for (uint n = lane; n < a.n; n += 32u)") != null);
    try std.testing.expect(std.mem.indexOf(u8, frozen, "tg.x") == null);
}

test "contract labels resolve to the seven shipped entries" {
    for (std.enums.values(Kind)) |kind| {
        const name = spec(kind, false).name;
        try std.testing.expect(std.mem.startsWith(u8, name, "metal inkling2 experts "));
        try std.testing.expect(std.mem.indexOf(u8, src.experts, symbol(kind)) != null);
    }
}
