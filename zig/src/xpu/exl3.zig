// SPDX-License-Identifier: Apache-2.0

//! EXL3 projections on the Intel GPU: zig/kernels/xpu/exl3.cl over Level Zero.

//! Same steps as core/exl3_projection.zig: rotate in, matvec, rotate out.

//! Routed experts use exl3_route. A fused MLP uses encodeMoe. Prompt rows use exl3_gemm.

const std = @import("std");
const Context = @import("context.zig").Context;
const Error = @import("context.zig").Error;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const Stream = @import("stream.zig").Stream;
const mod = @import("module.zig");
const launch = @import("launch.zig").launch;
const kernels = @import("kernels.zig");
pub const variants = @import("exl3_variants.zig");

pub const Output = enum(u32) { bf16 = 0, f32 = 1, f16 = 2 };
pub const Input = enum(u32) { bf16 = 0, f16 = 1 };

pub const Matrix = struct {
    trellis: u64,
    suh: u64,
    svh: u64,
    k: u32,
    n: u32,
    halves: u32,

    pub fn validate(m: Matrix) error{ UnsupportedExl3Shape, UnsupportedExl3Width }!void {
        if (m.k == 0 or m.n == 0 or m.k % 128 != 0 or m.n % 128 != 0) return error.UnsupportedExl3Shape;
        if (m.halves % 2 != 0 or m.halves < 4 or m.halves > 16) return error.UnsupportedExl3Width;
    }
};

pub const Group = struct {
    m: Matrix,
    rows: u32,
    x: u64,
    x_stride: u32,
    y: u64,
    y_stride: u32,
    output: Output = .bf16,
    input: Input = .bf16,
    scratch: u64,
};

pub const Desc = extern struct {
    trellis: u64,
    suh: u64,
    svh: u64,
    x: u64,
    rot: u64,
    sums: u64,
    out: u64,
    k: u32,
    n: u32,
    halves: u32,
    rows: u32,
    x_stride: u32,
    out_stride: u32,
    out_type: u32,
    x_f16: u32 = 0,
};

comptime {
    std.debug.assert(@sizeOf(Desc) == 88);
}

fn rotBytes(k: usize, rows: usize) usize {
    return std.mem.alignForward(usize, k * rows * 4, 256);
}

pub fn splitBound(n: usize) usize {
    const strips = n / 16;
    return @min(variants.ks_max, std.math.ceilPowerOfTwoAssert(usize, std.math.divCeil(usize, variants.sub_group_target, strips) catch unreachable));
}

pub fn scratchBytes(k: usize, n: usize, rows: usize) usize {
    return rotBytes(k, rows) + splitBound(n) * n * rows * 4;
}

pub fn promptScratchBytes(k: usize, n: usize, rows: usize) usize {
    return rotBytes(k, rows) + n * rows * 4;
}

pub fn descriptor(g: Group) !Desc {
    try g.m.validate();
    if (g.rows == 0 or g.rows > variants.moe.prompt_rows_max) return error.InvalidExl3Rows;
    if (g.x_stride < g.m.k or g.y_stride < g.m.n) return error.InvalidExl3Stride;
    return .{
        .trellis = g.m.trellis,
        .suh = g.m.suh,
        .svh = g.m.svh,
        .x = g.x,
        .rot = g.scratch,
        .sums = g.scratch + rotBytes(g.m.k, g.rows),
        .out = g.y,
        .k = g.m.k,
        .n = g.m.n,
        .halves = g.m.halves,
        .rows = g.rows,
        .x_stride = g.x_stride,
        .out_stride = g.y_stride,
        .out_type = @backingInt(g.output),
        .x_f16 = @backingInt(g.input),
    };
}

const local_rotate = [3]u32{ 16, 1, 1 };
const local_rotate_out = [3]u32{ 16, 8, 1 }; // exl3.cl RO_PARTS sub-groups a block
const local_matvec = [3]u32{ 128, 1, 1 };

const MoeKernels = struct {
    gate_up: mod.Kernel,
    act: mod.Kernel,
    down: mod.Kernel,
    rotate_in_h: mod.Kernel,
    gemm: mod.Kernel,
    rotate_out_rows: mod.Kernel,
    rot: mod.Kernel,
    tile: mod.Kernel,
    out_rows: mod.Kernel,
    sort: mod.Kernel,

    const names = [_][:0]const u8{ "exl3_moe_gate_up", "exl3_moe_act", "exl3_moe_down", "exl3_rotate_in_h", "exl3_gemm", "exl3_rotate_out_rows", "exl3_moe_rot", "exl3_moe_tile", "exl3_moe_out_rows", "exl3_moe_sort" };
    const locals = [_][3]u32{ .{ 128, 1, 1 }, .{ 32, 1, 1 }, .{ 128 / variants.moe.ds * variants.moe.pd, 1, 1 }, local_rotate, .{ 128, 1, 1 }, local_rotate, local_rotate, .{ 128, 1, 1 }, local_rotate, .{ 256, 1, 1 } };

    fn load(m: mod.Module) Error!MoeKernels {
        var ks: [names.len]mod.Kernel = undefined;
        var done: usize = 0;
        errdefer for (ks[0..done]) |*k| k.deinit();
        for (&ks, names, locals) |*k, n, l| {
            k.* = try m.kernel(n, l);
            done += 1;
            try k.indirect();
        }
        return .{ .gate_up = ks[0], .act = ks[1], .down = ks[2], .rotate_in_h = ks[3], .gemm = ks[4], .rotate_out_rows = ks[5], .rot = ks[6], .tile = ks[7], .out_rows = ks[8], .sort = ks[9] };
    }

    fn deinit(mk: *MoeKernels) void {
        inline for (.{ &mk.gate_up, &mk.act, &mk.down, &mk.rotate_in_h, &mk.gemm, &mk.rotate_out_rows, &mk.rot, &mk.tile, &mk.out_rows, &mk.sort }) |k| k.deinit();
    }
};

pub const Pipelines = struct {
    base: mod.Module,
    rotate_in: mod.Kernel,
    rotate_out: mod.Kernel,
    route: mod.Kernel,
    matvec_modules: [variants.all.len]mod.Module,
    matvec: [variants.all.len]mod.Kernel,
    moe_modules: [variants.moe.bits.len]mod.Module,
    moe: [variants.moe.bits.len]MoeKernels,

    pub fn load(ctx: *const Context) Error!Pipelines {
        var p: Pipelines = undefined;
        p.base = try mod.Module.load(ctx, kernels.exl3);
        errdefer p.base.unload();
        p.rotate_in = try p.base.kernel("exl3_rotate_in", local_rotate);
        errdefer p.rotate_in.deinit();
        p.rotate_out = try p.base.kernel("exl3_rotate_out", local_rotate_out);
        errdefer p.rotate_out.deinit();
        p.route = try p.base.kernel("exl3_route", .{ 1, 1, 1 }); // one work-item a slot, no bound check in the kernel
        errdefer p.route.deinit();
        for ([_]mod.Kernel{ p.rotate_in, p.rotate_out, p.route }) |k| try k.indirect();
        var done: usize = 0;
        errdefer for (0..done) |i| {
            p.matvec[i].deinit();
            p.matvec_modules[i].unload();
        };
        for (kernels.exl3_matvec, 0..) |image, i| {
            p.matvec_modules[i] = try mod.Module.load(ctx, image);
            p.matvec[i] = p.matvec_modules[i].kernel("exl3_matvec", local_matvec) catch |e| {
                p.matvec_modules[i].unload();
                return e;
            };
            p.matvec[i].indirect() catch |e| {
                p.matvec[i].deinit();
                p.matvec_modules[i].unload();
                return e;
            };
            done += 1;
        }
        var moe_done: usize = 0;
        errdefer for (0..moe_done) |i| {
            p.moe[i].deinit();
            p.moe_modules[i].unload();
        };
        for (kernels.exl3_moe, 0..) |image, i| {
            p.moe_modules[i] = try mod.Module.load(ctx, image);
            p.moe[i] = MoeKernels.load(p.moe_modules[i]) catch |e| {
                p.moe_modules[i].unload();
                return e;
            };
            moe_done += 1;
        }
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        for (&p.moe, &p.moe_modules) |*k, *m| {
            k.deinit();
            m.unload();
        }
        for (&p.matvec, &p.matvec_modules) |*k, *m| {
            k.deinit();
            m.unload();
        }
        p.route.deinit();
        p.rotate_out.deinit();
        p.rotate_in.deinit();
        p.base.unload();
        p.* = undefined;
    }
};

pub const Plan = struct {
    table: DeviceBuffer,
    count: u32,
    bits: u32,
    rows: u32,
    k_max: u32,
    n_max: u32,
    strips: variants.Strips,
    k_tiles_gcd: u32,

    pub fn init(ctx: *const Context, s: Stream, gpa: std.mem.Allocator, groups: []const Group) !Plan {
        if (groups.len == 0 or groups.len > std.math.maxInt(u16)) return error.InvalidExl3Groups;
        const descs = try gpa.alloc(Desc, groups.len);
        defer gpa.free(descs);
        var p: Plan = .{ .table = undefined, .count = @intCast(groups.len), .bits = groups[0].m.halves / 2, .rows = 0, .k_max = 0, .n_max = 0, .strips = .{ .min = std.math.maxInt(u32), .gcd = 0, .total = 0 }, .k_tiles_gcd = 0 };
        for (groups, descs) |g, *d| {
            d.* = try descriptor(g);
            if (g.m.halves / 2 != p.bits) return error.MixedExl3Widths; // the host sorts groups by width
            p.rows = @max(p.rows, g.rows);
            p.k_max = @max(p.k_max, g.m.k);
            p.n_max = @max(p.n_max, g.m.n);
            p.strips.min = @min(p.strips.min, g.m.n / 16);
            p.strips.gcd = std.math.gcd(p.strips.gcd, g.m.n / 16);
            p.strips.total += g.m.n / 16;
            p.k_tiles_gcd = std.math.gcd(p.k_tiles_gcd, g.m.k / 16);
        }
        if (p.rows > variants.rows_max) { // prompt rows: exl3_gemm of the fused MLP image
            const m = variants.moe;
            if (std.mem.indexOfScalar(u32, &m.bits, p.bits) == null) return error.UnsupportedExl3Width;
            if (p.strips.gcd % m.gemm_s != 0 or p.k_tiles_gcd % m.gemm_kc != 0) return error.UnsupportedExl3Shape;
        } else if (variants.pick(p.bits, p.rows, p.strips) == null) return error.UnsupportedExl3Width;
        p.table = try DeviceBuffer.fromHost(ctx, s, std.mem.sliceAsBytes(descs));
        return p;
    }

    pub fn experts(ctx: *const Context, s: Stream, gpa: std.mem.Allocator, ms: []const Matrix) !Plan {
        if (ms.len == 0) return error.InvalidExl3Groups;
        const descs = try gpa.alloc(Desc, ms.len);
        defer gpa.free(descs);
        for (ms, descs) |m, *d| {
            try m.validate();
            if (m.k != ms[0].k or m.n != ms[0].n or m.halves != ms[0].halves) return error.MixedExl3Experts;
            d.* = .{ .trellis = m.trellis, .suh = m.suh, .svh = m.svh, .x = 0, .rot = 0, .sums = 0, .out = 0, .k = m.k, .n = m.n, .halves = m.halves, .rows = 1, .x_stride = m.k, .out_stride = m.n, .out_type = 0 };
        }
        const strips = ms[0].n / 16;
        return .{ .table = try DeviceBuffer.fromHost(ctx, s, std.mem.sliceAsBytes(descs)), .count = @intCast(ms.len), .bits = ms[0].halves / 2, .rows = 1, .k_max = ms[0].k, .n_max = ms[0].n, .strips = .{ .min = strips, .gcd = strips, .total = strips }, .k_tiles_gcd = ms[0].k / 16 };
    }

    pub fn deinit(p: *Plan) void {
        p.table.free();
        p.* = undefined;
    }
};

fn setU64(k: mod.Kernel, i: u32, v: u64) Error!void {
    try k.setBytes(i, std.mem.asBytes(&v));
}

const Shape = struct { variant: usize, ks: u32 };

fn shapeOf(plan: *const Plan, rows: u32, strips_total: u32) Error!Shape {
    const strips: variants.Strips = .{ .min = plan.strips.min, .gcd = plan.strips.gcd, .total = strips_total };
    const vi = variants.pick(plan.bits, rows, strips) orelse return error.Invalid;
    return .{ .variant = vi, .ks = variants.split(plan.k_tiles_gcd) };
}

const fuse_n_max = 65536;

fn fused(rows: u32, n_max: u32, k_tiles_gcd: u32, ks: u32) bool {
    return rows <= 4 and n_max < fuse_n_max and (k_tiles_gcd / ks) % 8 == 0;
}

fn stages(p: *const Pipelines, s: Stream, table: ?*anyopaque, count: u32, rows: u32, k_max: u32, n_max: u32, k_tiles_gcd: u32, sh: Shape) Error!void {
    const mv = p.matvec[sh.variant];
    const ns = variants.all[sh.variant].ns;
    const fuse = fused(rows, n_max, k_tiles_gcd, sh.ks);
    for ([_]mod.Kernel{ p.rotate_in, mv, p.rotate_out }) |k| {
        try k.setPtr(0, null); // base: unused with -DEXL3_ABS
        try k.setPtr(1, table);
    }
    try mv.setU32(2, sh.ks);
    try mv.setU32(3, @intFromBool(fuse));
    try p.rotate_out.setU32(2, sh.ks);
    if (!fuse) try launch(p.rotate_in, s, .{ k_max / 128, rows, count });
    try launch(mv, s, .{ std.math.divCeil(u32, n_max / 16 / ns, local_matvec[0] / 16) catch unreachable, count, sh.ks });
    try launch(p.rotate_out, s, .{ n_max / 128, rows, count });
}

pub fn encode(p: *const Pipelines, s: Stream, plan: *const Plan) Error!void {
    if (plan.rows > variants.rows_max) return encodeRows(p, s, plan, plan.rows);
    try stages(p, s, plan.table.ptr, plan.count, plan.rows, plan.k_max, plan.n_max, plan.k_tiles_gcd, try shapeOf(plan, plan.rows, plan.strips.total));
}

pub const Route = struct {
    ids: u64,
    tokens: u32,
    topk: u32,
    x: u64,
    x_stride: u32,
    x_per_slot: bool,
    y: u64,
    y_stride: u32,
    output: Output = .bf16,
    input: Input = .bf16,
    scratch: u64,
    slots: u64,
};

pub fn slotTableBytes(slots: usize) usize {
    return slots * @sizeOf(Desc);
}

pub fn encodeRouted(p: *const Pipelines, s: Stream, experts: *const Plan, r: Route) Error!void {
    const slots = r.tokens * r.topk;
    if (slots == 0) return error.Invalid;
    const k = p.route;
    const sh = try shapeOf(experts, 1, experts.strips.min * slots); // one expert's strips a slot
    try k.setPtr(0, experts.table.ptr);
    try k.setPtr(1, @ptrFromInt(r.ids));
    try k.setPtr(2, @ptrFromInt(r.slots));
    try k.setU32(3, r.topk);
    try setU64(k, 4, r.x);
    try k.setU32(5, r.x_stride);
    try k.setU32(6, @intFromBool(r.x_per_slot));
    try setU64(k, 7, r.scratch);
    try setU64(k, 8, r.scratch + rotBytes(experts.k_max, slots));
    try setU64(k, 9, r.y);
    try k.setU32(10, r.y_stride);
    try k.setU32(11, @backingInt(r.output));
    try k.setU32(12, sh.ks);
    try k.setU32(13, @backingInt(r.input));
    try launch(k, s, .{ slots, 1, 1 });
    try stages(p, s, @ptrFromInt(r.slots), slots, 1, experts.k_max, experts.n_max, experts.k_tiles_gcd, sh);
}

pub const MoeRoute = struct {
    ids: u64,
    tokens: u32,
    topk: u32,
    shared: ?u32 = null,
    x: u64,
    x_stride: u32,
    y: u64,
    y_stride: u32,
    output: Output = .bf16,
    input: Input = .bf16,
    scratch: u64,

    pub fn slots(r: MoeRoute) u32 {
        return r.tokens * r.topk + if (r.shared != null) r.tokens else 0;
    }
};

const MoeScratch = struct { gu: usize, act: usize, size: usize };

fn moeScratch(hidden: usize, ff: usize, slots: usize) MoeScratch {
    _ = hidden;
    const gu = std.mem.alignForward(usize, slots * variants.moe.p * 2 * ff * 4, 256);
    return .{ .gu = 0, .act = gu, .size = gu + slots * ff * 4 };
}

pub fn moeScratchBytes(hidden: usize, ff: usize, slots: usize) usize {
    return moeScratch(hidden, ff, slots).size;
}

pub fn encodeMoe(p: *const Pipelines, s: Stream, gate: *const Plan, up: *const Plan, down: *const Plan, r: MoeRoute) Error!void {
    const hidden = gate.k_max;
    const ff = gate.n_max;
    const m = variants.moe;
    if (up.k_max != hidden or up.n_max != ff or down.k_max != ff or down.n_max != hidden) return error.Invalid;
    if (gate.count != up.count or gate.count != down.count or gate.bits != up.bits or gate.bits != down.bits) return error.Invalid;
    if (hidden % (128 * m.p) != 0 or ff % (128 * m.pd) != 0 or hidden > m.k_max or ff > m.k_max) return error.Invalid;
    if (r.shared) |e| if (e >= gate.count) return error.Invalid;
    if (r.x_stride < hidden or r.y_stride < hidden) return error.Invalid;
    const mi = std.mem.indexOfScalar(u32, &m.bits, gate.bits) orelse return error.Invalid;
    const k = p.moe[mi];
    const slots = r.slots();
    if (slots == 0) return error.Invalid;
    const pairs = r.tokens * r.topk;
    const shared = r.shared orelse 0;
    const sc = moeScratch(hidden, ff, slots);
    const ids: ?*anyopaque = @ptrFromInt(r.ids);

    try k.gate_up.setPtr(0, null); // base: unused with -DEXL3_ABS
    try k.gate_up.setPtr(1, gate.table.ptr);
    try k.gate_up.setPtr(2, up.table.ptr);
    try k.gate_up.setPtr(3, ids);
    try k.gate_up.setU32(4, r.topk);
    try k.gate_up.setU32(5, pairs);
    try k.gate_up.setU32(6, shared);
    try setU64(k.gate_up, 7, r.x);
    try k.gate_up.setU32(8, r.x_stride);
    try setU64(k.gate_up, 9, r.scratch + sc.gu);
    try k.gate_up.setU32(10, @backingInt(r.input));
    try launch(k.gate_up, s, .{ ff / 128 * m.p, slots, 1 });

    try k.act.setPtr(0, null);
    try k.act.setPtr(1, gate.table.ptr);
    try k.act.setPtr(2, up.table.ptr);
    try k.act.setPtr(3, down.table.ptr);
    try k.act.setPtr(4, ids);
    try k.act.setU32(5, pairs);
    try k.act.setU32(6, shared);
    try setU64(k.act, 7, r.scratch + sc.gu);
    try setU64(k.act, 8, r.scratch + sc.act);
    try k.act.setU32(9, m.p);
    try launch(k.act, s, .{ ff / 128, slots, 1 });

    try k.down.setPtr(0, null);
    try k.down.setPtr(1, down.table.ptr);
    try k.down.setPtr(2, ids);
    try k.down.setU32(3, pairs);
    try k.down.setU32(4, shared);
    try setU64(k.down, 5, r.scratch + sc.act);
    try setU64(k.down, 6, r.y);
    try k.down.setU32(7, r.y_stride);
    try k.down.setU32(8, @backingInt(r.output));
    try launch(k.down, s, .{ hidden / 128, slots, 1 });
}

pub fn encodeRows(p: *const Pipelines, s: Stream, plan: *const Plan, rows: u32) Error!void {
    if (rows == 0 or rows > plan.rows or plan.rows <= variants.rows_max) return error.Invalid;
    const m = variants.moe;
    const mi = std.mem.indexOfScalar(u32, &m.bits, plan.bits) orelse return error.Invalid;
    const k = p.moe[mi];
    for ([_]mod.Kernel{ k.rotate_in_h, k.gemm, k.rotate_out_rows }) |kk| {
        try kk.setPtr(0, null); // base: unused with -DEXL3_ABS
        try kk.setPtr(1, plan.table.ptr);
        try kk.setU32(2, rows);
    }
    try launch(k.rotate_in_h, s, .{ plan.k_max / 128, rows, plan.count });
    try launch(k.gemm, s, .{ plan.n_max / 16 / m.gemm_s, std.math.divCeil(u32, rows, 8 * m.gemm_m) catch unreachable, plan.count });
    try launch(k.rotate_out_rows, s, .{ plan.n_max / 128, rows, plan.count });
}

pub const MoeTiles = @import("exl3_tiles.zig").MoeTiles;
pub const tileBound = @import("exl3_tiles.zig").tileBound;
pub const moeTileScratchBytes = @import("exl3_tiles.zig").moeTileScratchBytes;
pub const encodeMoeTiles = @import("exl3_tiles.zig").encodeMoeTiles;

test {
    _ = variants;
    _ = @import("exl3_scratch_test.zig");
}
