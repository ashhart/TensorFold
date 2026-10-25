// SPDX-License-Identifier: Apache-2.0
//! EXL3 projections on the Intel GPU: zig/kernels/xpu/exl3.cl over Level Zero.
//! The same steps as core/exl3_projection.zig (Metal) and cuda/exl3.zig: rotate in, matvec, rotate out.
//! Model code gives device addresses and shapes only; it never sees a Level Zero type.
//! A launch covers many groups (projections) of one width through a descriptor table on the device, built once
//! at load time and run every step. Routed experts build their slot table on the device (exl3_route).
//! A whole routed MLP (gate, up, SiLU, down for every pick, plus the shared expert) runs fused: encodeMoe.
//! Prompt rows (more than variants.rows_max) run on XMX: exl3_gemm for groups, expert tiles for the MLP (encodeMoeTiles).

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

/// One stored EXL3 weight W[K,N] (codebook mul1, fp16 suh and svh): device addresses of the checkpoint tensors.
pub const Matrix = struct {
    trellis: u64,
    suh: u64,
    svh: u64,
    k: u32,
    n: u32,
    /// Twice the bits: 8 for 4 bits, 12 for 6 bits.
    halves: u32,

    pub fn validate(m: Matrix) error{ UnsupportedExl3Shape, UnsupportedExl3Width }!void {
        if (m.k == 0 or m.n == 0 or m.k % 128 != 0 or m.n % 128 != 0) return error.UnsupportedExl3Shape;
        if (m.halves % 2 != 0 or m.halves < 4 or m.halves > 16) return error.UnsupportedExl3Width;
    }
};

/// One projection of a launch. x is [rows][x_stride] in `input`, y is [rows][y_stride] in `output`, and scratch holds
/// scratchBytes(k, n, rows) bytes (promptScratchBytes above variants.rows_max rows) that no other group of the launch uses.
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

/// The table row of exl3.cl (exl3_desc).
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

/// The most K splits a launch holding a group of width n can pick: a launch has at least that group's strips.
fn splitBound(n: usize) usize {
    const strips = n / 16;
    return @min(variants.ks_max, std.math.ceilPowerOfTwoAssert(usize, std.math.divCeil(usize, variants.sub_group_target, strips) catch unreachable));
}

/// fp32 rotated input [rows][k], then fp32 sums [splits][rows][n] (see splitBound).
pub fn scratchBytes(k: usize, n: usize, rows: usize) usize {
    return rotBytes(k, rows) + splitBound(n) * n * rows * 4;
}

/// A prompt group (more than variants.rows_max rows): fp16 hi and lo planes of the rotated input (the size of rotBytes),
/// then fp32 sums [rows][n], one split.
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
        .out_type = @intFromEnum(g.output),
        .x_f16 = @intFromEnum(g.input),
    };
}

const local_rotate = [3]u32{ 16, 1, 1 };
const local_rotate_out = [3]u32{ 16, 8, 1 }; // exl3.cl RO_PARTS sub-groups a block
const local_matvec = [3]u32{ 128, 1, 1 };

/// The kernels of a fused routed MLP image (exl3_variants.moe): the decode MLP, then the prompt kernels (groups and
/// expert tiles on XMX).
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
        // every EXL3 kernel reads weights, activations and scratch through device addresses in a table
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

/// A resident descriptor table: groups of one width and at most variants.rows_max rows, run as one launch.
pub const Plan = struct {
    table: DeviceBuffer,
    count: u32,
    bits: u32,
    rows: u32,
    k_max: u32,
    n_max: u32,
    strips: variants.Strips,
    k_tiles_gcd: u32,

    /// Checks every group and uploads the table (waits for the copy).
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

    /// Expert tables for `encodeRouted`: only trellis, suh, svh, k, n and halves are read; all experts share K and N.
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

/// NS image and K splits for a launch over `plan`'s widths with `strips_total` strips in all.
const Shape = struct { variant: usize, ks: u32 };

fn shapeOf(plan: *const Plan, rows: u32, strips_total: u32) Error!Shape {
    const strips: variants.Strips = .{ .min = plan.strips.min, .gcd = plan.strips.gcd, .total = strips_total };
    const vi = variants.pick(plan.bits, rows, strips) orelse return error.Invalid;
    return .{ .variant = vi, .ks = variants.split(strips.total / variants.all[vi].ns, plan.k_tiles_gcd) };
}

/// The matvec rotates x itself (exl3.cl RI) at 1 to 4 rows when every K split is whole 128 blocks, on launches
/// below fuse_n_max columns. Each sub-group repeats the rotation: on lm_head (N 248320, 3880 sub-groups) that took
/// 797 to 815 us at 1 row and 1339 to 1477 us at 4 rows. On one layer's experts it saves a launch (B70, 2026-10-25).
const fuse_n_max = 65536;

fn fused(rows: u32, n_max: u32, k_tiles_gcd: u32, ks: u32) bool {
    return rows <= 4 and n_max < fuse_n_max and (k_tiles_gcd / ks) % 8 == 0;
}

/// Queues the stages over a table of `count` groups (no host wait): rotate_in (unless fused), matvec, rotate_out.
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

/// y = x @ W for every group of the plan.
pub fn encode(p: *const Pipelines, s: Stream, plan: *const Plan) Error!void {
    if (plan.rows > variants.rows_max) return encodeRows(p, s, plan, plan.rows);
    try stages(p, s, plan.table.ptr, plan.count, plan.rows, plan.k_max, plan.n_max, plan.k_tiles_gcd, try shapeOf(plan, plan.rows, plan.strips.total));
}

/// Routed experts, one row a (token, pick) slot. ids are u32 [tokens][topk] on the device. Slot t * topk + j reads
/// x row t (gate and up) or row t * topk + j (down, x_per_slot) and writes y row t * topk + j.
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
    /// scratchBytes(k, n, tokens * topk) bytes.
    scratch: u64,
    /// slotTableBytes(tokens * topk) bytes for the descriptors exl3_route writes.
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
    try k.setU32(11, @intFromEnum(r.output));
    try k.setU32(12, sh.ks);
    try k.setU32(13, @intFromEnum(r.input));
    try launch(k, s, .{ slots, 1, 1 });
    try stages(p, s, @ptrFromInt(r.slots), slots, 1, experts.k_max, experts.n_max, experts.k_tiles_gcd, sh);
}

/// A fused routed MLP: for each (token, pick) slot y = down(silu(gate x) * up x), and with `shared` one more slot a
/// token on the shared expert. gate, up and down are Plan.experts tables in the same expert order (gate and up
/// W[hidden, ff], down W[ff, hidden]); the shared expert is an entry of them. ids are u32 [tokens][topk] on the device.
/// y row t * topk + j is token t's pick j; rows from tokens * topk on are the shared expert's, one a token.
pub const MoeRoute = struct {
    ids: u64,
    tokens: u32,
    topk: u32,
    /// Table index of the shared expert, or null for none.
    shared: ?u32 = null,
    x: u64,
    x_stride: u32,
    y: u64,
    y_stride: u32,
    output: Output = .bf16,
    input: Input = .bf16,
    /// moeScratchBytes(hidden, ff, slots()) bytes.
    scratch: u64,

    pub fn slots(r: MoeRoute) u32 {
        return r.tokens * r.topk + if (r.shared != null) r.tokens else 0;
    }
};

/// Scratch of a fused MLP: gate|up split sums, then the down input (fp32), 256-byte aligned.
const MoeScratch = struct { gu: usize, act: usize, size: usize };

fn moeScratch(hidden: usize, ff: usize, slots: usize) MoeScratch {
    _ = hidden;
    const gu = std.mem.alignForward(usize, slots * variants.moe.p * 2 * ff * 4, 256);
    return .{ .gu = 0, .act = gu, .size = gu + slots * ff * 4 };
}

pub fn moeScratchBytes(hidden: usize, ff: usize, slots: usize) usize {
    return moeScratch(hidden, ff, slots).size;
}

/// Three launches, no host wait. The splits are added in a fixed order: the result does not depend on the schedule.
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
    try k.gate_up.setU32(10, @intFromEnum(r.input));
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
    try k.down.setU32(8, @intFromEnum(r.output));
    try launch(k.down, s, .{ hidden / 128, slots, 1 });
}

/// The first `rows` rows of a prompt plan (built with more than variants.rows_max rows): rotate_in to fp16 hi and lo
/// planes, exl3_gemm on XMX, rotate_out. One plan of the chunk size serves every chunk length.
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

/// A routed MLP over prompt rows on XMX, as encodeMoe (the same y and slot numbering). exl3_moe_sort groups the slots
/// by expert into tiles of at most variants.moe.tile_rows slots, in buffers of the caller: order u32 [slots], tiles
/// uint4 [tileBound], ntiles u32.
pub const MoeTiles = struct {
    route: MoeRoute,
    order: u64,
    tiles: u64,
    ntiles: u64,
};

/// The most tiles of a prompt MLP over `tables` expert table entries (the shared expert included): one partial tile an
/// entry over the full ones.
pub fn tileBound(slots: u32, tables: u32) u32 {
    return slots / variants.moe.tile_rows + tables;
}

/// Scratch of the prompt MLP: gate|up rotated input [2][slots][hidden] fp32 (then the down sums [slots][hidden]),
/// gate|up sums [slots][2][ff], the down input [slots][ff].
const TileScratch = struct { rot: usize, gu: usize, act: usize, size: usize };

fn tileScratch(hidden: usize, ff: usize, slots: usize) TileScratch {
    const rot = std.mem.alignForward(usize, 2 * slots * hidden * 4, 256);
    const gu = std.mem.alignForward(usize, slots * 2 * ff * 4, 256);
    return .{ .rot = 0, .gu = rot, .act = rot + gu, .size = rot + gu + slots * ff * 4 };
}

pub fn moeTileScratchBytes(hidden: usize, ff: usize, slots: usize) usize {
    return tileScratch(hidden, ff, slots).size;
}

/// Six launches, no host wait: sort, rotate gate|up input a slot, gate|up tiles, act (one split), down tiles, rotate out.
pub fn encodeMoeTiles(p: *const Pipelines, s: Stream, gate: *const Plan, up: *const Plan, down: *const Plan, t: MoeTiles) Error!void {
    const r = t.route;
    const hidden = gate.k_max;
    const ff = gate.n_max;
    const m = variants.moe;
    if (up.k_max != hidden or up.n_max != ff or down.k_max != ff or down.n_max != hidden) return error.Invalid;
    if (gate.count != up.count or gate.count != down.count or gate.bits != up.bits or gate.bits != down.bits) return error.Invalid;
    if (hidden % 128 != 0 or ff % 128 != 0 or ff % (16 * m.tile_ns) != 0 or hidden % (16 * m.tile_ns) != 0) return error.Invalid;
    if (r.shared) |e| if (e >= gate.count) return error.Invalid;
    if (r.x_stride < hidden or r.y_stride < hidden) return error.Invalid;
    const mi = std.mem.indexOfScalar(u32, &m.bits, gate.bits) orelse return error.Invalid;
    const k = p.moe[mi];
    const slots = r.slots();
    if (slots == 0) return error.Invalid;
    const pairs = r.tokens * r.topk;
    const shared = r.shared orelse 0;
    const sc = tileScratch(hidden, ff, slots);
    const ids: ?*anyopaque = @ptrFromInt(r.ids);
    const order: ?*anyopaque = @ptrFromInt(t.order);
    const tiles: ?*anyopaque = @ptrFromInt(t.tiles);
    const ntiles: ?*anyopaque = @ptrFromInt(t.ntiles);
    const tile_bound = tileBound(slots, gate.count);

    try k.sort.setPtr(0, ids);
    try k.sort.setPtr(1, order);
    try k.sort.setPtr(2, tiles);
    try k.sort.setPtr(3, ntiles);
    try k.sort.setU32(4, if (r.shared) |e| e else gate.count); // no shared expert: an empty bucket past the table
    try k.sort.setU32(5, pairs);
    try k.sort.setU32(6, if (r.shared != null) r.tokens else 0);
    try launch(k.sort, s, .{ 1, 1, 1 });

    const tile_x = struct {
        fn f(n: u32) u32 {
            return std.math.divCeil(u32, n / 16 / m.tile_ns, 8) catch unreachable; // work-groups of 8 sub-groups
        }
    }.f;

    try k.rot.setPtr(0, null);
    try k.rot.setPtr(1, gate.table.ptr);
    try k.rot.setPtr(2, up.table.ptr);
    try k.rot.setPtr(3, ids);
    try k.rot.setU32(4, r.topk);
    try k.rot.setU32(5, pairs);
    try k.rot.setU32(6, shared);
    try setU64(k.rot, 7, r.x);
    try k.rot.setU32(8, r.x_stride);
    try k.rot.setU32(9, @intFromEnum(r.input));
    try setU64(k.rot, 10, r.scratch + sc.rot);
    try k.rot.setU32(11, slots);
    try launch(k.rot, s, .{ hidden / 128, slots, 2 });

    try k.tile.setPtr(0, null);
    try k.tile.setPtr(1, gate.table.ptr);
    try k.tile.setPtr(2, up.table.ptr);
    try k.tile.setPtr(3, order);
    try k.tile.setPtr(4, tiles);
    try k.tile.setPtr(5, ntiles);
    try setU64(k.tile, 6, r.scratch + sc.rot);
    try setU64(k.tile, 7, @as(u64, slots) * hidden);
    try setU64(k.tile, 8, r.scratch + sc.gu);
    try k.tile.setU32(9, 2 * ff);
    try k.tile.setU32(10, ff);
    try launch(k.tile, s, .{ tile_x(ff), tile_bound, 2 });

    try k.act.setPtr(0, null);
    try k.act.setPtr(1, gate.table.ptr);
    try k.act.setPtr(2, up.table.ptr);
    try k.act.setPtr(3, down.table.ptr);
    try k.act.setPtr(4, ids);
    try k.act.setU32(5, pairs);
    try k.act.setU32(6, shared);
    try setU64(k.act, 7, r.scratch + sc.gu);
    try setU64(k.act, 8, r.scratch + sc.act);
    try k.act.setU32(9, 1);
    try launch(k.act, s, .{ ff / 128, slots, 1 });

    // down sums reuse the rotated input's space (its gate|up use is done)
    try k.tile.setPtr(1, down.table.ptr);
    try k.tile.setPtr(2, down.table.ptr);
    try setU64(k.tile, 6, r.scratch + sc.act);
    try setU64(k.tile, 7, 0);
    try setU64(k.tile, 8, r.scratch + sc.rot);
    try k.tile.setU32(9, hidden);
    try k.tile.setU32(10, 0);
    try launch(k.tile, s, .{ tile_x(hidden), tile_bound, 1 });

    try k.out_rows.setPtr(0, null);
    try k.out_rows.setPtr(1, down.table.ptr);
    try k.out_rows.setPtr(2, ids);
    try k.out_rows.setU32(3, pairs);
    try k.out_rows.setU32(4, shared);
    try setU64(k.out_rows, 5, r.scratch + sc.rot);
    try setU64(k.out_rows, 6, r.y);
    try k.out_rows.setU32(7, r.y_stride);
    try k.out_rows.setU32(8, @intFromEnum(r.output));
    try launch(k.out_rows, s, .{ hidden / 128, slots, 1 });
}

test "descriptor layout and scratch" {
    const m = Matrix{ .trellis = 0x1000, .suh = 0x2000, .svh = 0x3000, .k = 2048, .n = 512, .halves = 8 };
    const d = try descriptor(.{ .m = m, .rows = 3, .x = 0x4000, .x_stride = 2048, .y = 0x5000, .y_stride = 512, .output = .f32, .scratch = 0x10000 });
    try std.testing.expectEqual(@as(u64, 0x10000 + 2048 * 3 * 4), d.sums);
    try std.testing.expectEqual(@as(u32, 1), d.out_type);
    try std.testing.expectEqual(@as(usize, 2048 * 3 * 4 + 16 * 512 * 3 * 4), scratchBytes(2048, 512, 3));
    try std.testing.expectEqual(@as(usize, 1), splitBound(248320));
    try std.testing.expectError(error.InvalidExl3Rows, descriptor(.{ .m = m, .rows = 1025, .x = 0, .x_stride = 2048, .y = 0, .y_stride = 512, .scratch = 0 }));
    // prompt rows: one split
    try std.testing.expectEqual(@as(usize, 2048 * 512 * 4 + 512 * 512 * 4), promptScratchBytes(2048, 512, 512));
    try std.testing.expectError(error.UnsupportedExl3Shape, (Matrix{ .trellis = 0, .suh = 0, .svh = 0, .k = 100, .n = 512, .halves = 8 }).validate());
    // Sage: hidden 2048, ff 512, 8 picks and the shared expert
    try std.testing.expectEqual(@as(usize, 9 * 4 * 2 * 512 * 4 + 9 * 512 * 4), moeScratchBytes(2048, 512, 9));
}

test {
    _ = variants;
}
