//! Real Metal dispatch for Inkling 2 Mini's M3 NVFP4/MXFP8 routed experts.
const std = @import("std");
const mtl = @import("metal");
const src = @import("inkling2_metal");
const plan = @import("experts_plan.zig");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Format = plan.Format;
pub const Call = plan.Call;
pub const Operation = enum { nvfp4_gu, nvfp4_down, mxfp8_gu, mxfp8_down, nvfp4_quantize, mxfp8_quantize, combine, group, nvfp4_gu_grouped, nvfp4_down_grouped, mxfp8_gu_grouped, mxfp8_down_grouped };

pub const function_names = [_][:0]const u8{
    "tf_ink2_nvfp4_gu",
    "tf_ink2_nvfp4_down",
    "tf_ink2_mxfp8_gu",
    "tf_ink2_mxfp8_down",
    "tf_ink2_nvfp4_quantize",
    "tf_ink2_mxfp8_quantize",
    "tf_ink2_combine",
    "tf_ink2_group",
    "tf_ink2_nvfp4_gu_grouped",
    "tf_ink2_nvfp4_down_grouped",
    "tf_ink2_mxfp8_gu_grouped",
    "tf_ink2_mxfp8_down_grouped",
};

/// The columns the units kernels read a row: experts_units.metal's INK2_GU_K and INK2_DOWN_K.
const units_gu_k = 2304;
const units_down_k = 1152;
const units_names = [_][:0]const u8{ "tf_ink2_nvfp4_gu_units", "tf_ink2_nvfp4_down_units", "tf_ink2_mxfp8_gu_units", "tf_ink2_mxfp8_down_units", "tf_ink2_logits_units", "tf_ink2_dense_units" };

/// A chunk of at least this many rows runs grouped by expert; a row's sums are the same either way.
pub const grouped_rows: u32 = 8;
const group_cap: u32 = 256;

pub const Pipelines = struct {
    entries: [function_names.len]mtl.Pipeline,
    /// Each expert's count, then its route pairs: launches in one encoder run in order, so they share it.
    groups: mtl.Buffer,
    /// The gate-up and down products on the tensor units (NVFP4 then MXFP8), where the GPU has them.
    units: ?[units_names.len]mtl.Pipeline = null,

    pub fn load(device: mtl.Device) !Pipelines {
        var out = try loadSource(device, src.experts);
        errdefer out.deinit();
        if (device.tensorUnits()) try out.loadUnits(device, src.experts_units);
        return out;
    }

    /// The tensor-unit kernels from `text`; the contract's controls pass an edited one. The GPU must have the units.
    pub fn loadUnits(p: *Pipelines, device: mtl.Device, text: []const u8) !void {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [units_names.len]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |pipe| pipe.deinit();
        for (units_names, &pipes) |name, *pipe| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        p.units = pipes;
    }

    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var out: Pipelines = undefined;
        var made: usize = 0;
        errdefer for (out.entries[0..made]) |pipe| pipe.deinit();
        for (function_names, 0..) |name, i| {
            out.entries[i] = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
            const scalar = std.mem.endsWith(u8, name, "quantize") or std.mem.endsWith(u8, name, "group");
            if (out.entries[i].maxThreads() < @as(usize, if (scalar) 32 else 256)) return error.ExpertThreads;
        }
        out.groups = try device.buffer((plan.expert_count + plan.expert_count * group_cap) * @sizeOf(u32), mtl.ResourceOptions.shared);
        out.units = null;
        return out;
    }

    pub fn deinit(p: *Pipelines) void {
        if (p.units) |pipes| for (pipes) |pipe| pipe.deinit();
        p.groups.deinit();
        for (p.entries) |pipe| pipe.deinit();
    }

    fn get(p: *const Pipelines, op: Operation) mtl.Pipeline {
        return p.entries[@backingInt(op)];
    }
};

fn args(c: Call) plan.MatrixArgs {
    return .{ .k = c.k, .n = c.n, .topk = c.topk, .experts = c.experts };
}

fn launch(p: *const Pipelines, e: mtl.ComputeEncoder, op: Operation, groups: u32) void {
    e.setPipeline(p.get(op));
    e.dispatchGroups(mtl.Size.of(groups, 1, 1), mtl.Size.of(32, 1, 1));
}

fn launchGrid(p: *const Pipelines, e: mtl.ComputeEncoder, op: Operation, g: plan.Geometry) void {
    e.setPipeline(p.get(op));
    e.dispatchGroups(mtl.Size.of(g.outputs, g.pairs, 1), mtl.Size.of(g.threads, 1, 1));
}

/// Packed NVFP4 GU: X [rows,k], ids [rows,topk], Y [rows*topk,n].
pub fn nvfp4Gu(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, scales: Ref, amax: Ref, denominator: Ref, ids: Ref, out: Ref, c: Call) !void {
    if (c.format != .nvfp4) return error.WrongExpertFormat;
    try plan.check(c);
    for ([_]Ref{ x, w, scales, amax, denominator, ids, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(args(c), 7);
    launchGrid(p, e, .nvfp4_gu, plan.geometry(c));
}

/// Packed MXFP8 GU with UE8M0 scales per row/block32.
pub fn mxfp8Gu(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, scales: Ref, ids: Ref, out: Ref, c: Call) !void {
    if (c.format != .mxfp8) return error.WrongExpertFormat;
    try plan.check(c);
    for ([_]Ref{ x, w, scales, ids, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(args(c), 5);
    launchGrid(p, e, .mxfp8_gu, plan.geometry(c));
}

pub fn nvfp4Down(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, w: Ref, scales: Ref, amax: Ref, denominator: Ref, ids: Ref, out: Ref, c: Call) !void {
    if (c.format != .nvfp4) return error.WrongExpertFormat;
    try plan.check(c);
    for ([_]Ref{ act, w, scales, amax, denominator, ids, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(args(c), 7);
    launchGrid(p, e, .nvfp4_down, plan.geometry(c));
}

pub fn mxfp8Down(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, w: Ref, scales: Ref, ids: Ref, out: Ref, c: Call) !void {
    if (c.format != .mxfp8) return error.WrongExpertFormat;
    try plan.check(c);
    for ([_]Ref{ act, w, scales, ids, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(args(c), 5);
    launchGrid(p, e, .mxfp8_down, plan.geometry(c));
}

/// Decode one expert matrix to BF16, not an activation quantization operation.
pub fn nvfp4Quantize(p: *const Pipelines, e: mtl.ComputeEncoder, w: Ref, scales: Ref, amax: Ref, denominator: Ref, out: Ref, rows: u32, k: u32) !void {
    if (k == 0 or k % 16 != 0 or rows == 0) return error.BadExpertBlock;
    for ([_]Ref{ w, scales, amax, denominator, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(plan.DecodeArgs{ .rows = rows, .k = k }, 5);
    launch(p, e, .nvfp4_quantize, (rows * k + 31) / 32);
}

pub fn mxfp8Quantize(p: *const Pipelines, e: mtl.ComputeEncoder, w: Ref, scales: Ref, out: Ref, rows: u32, k: u32) !void {
    if (k == 0 or k % 32 != 0 or rows == 0) return error.BadExpertBlock;
    for ([_]Ref{ w, scales, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(plan.DecodeArgs{ .rows = rows, .k = k }, 3);
    launch(p, e, .mxfp8_quantize, (rows * k + 31) / 32);
}

/// Sum per-token BF16 down outputs in ascending expert index and round once to BF16.
pub fn combine(p: *const Pipelines, e: mtl.ComputeEncoder, down: Ref, ids: Ref, weights: Ref, out: Ref, c: Call) !void {
    try plan.check(c);
    for ([_]Ref{ down, ids, weights, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(args(c), 4);
    launchGrid(p, e, .combine, plan.combineGeometry(c));
}

const GroupArgs = extern struct { pairs: u32, experts: u32, cap: u32 };
const GroupedArgs = extern struct { k: u32, n: u32, topk: u32, cap: u32 };

fn groupedArgs(c: Call) GroupedArgs {
    return .{ .k = c.k, .n = c.n, .topk = c.topk, .cap = group_cap };
}

fn counts(p: *const Pipelines) Ref {
    return .{ .buf = p.groups };
}

fn members(p: *const Pipelines) Ref {
    return .{ .buf = p.groups, .off = plan.expert_count * @sizeOf(u32) };
}

fn launchGrouped(p: *const Pipelines, e: mtl.ComputeEncoder, op: Operation, c: Call) void {
    e.setPipeline(p.get(op));
    e.dispatchGroups(mtl.Size.of((c.n + 7) / 8, c.experts, 1), mtl.Size.of(256, 1, 1));
}

/// Each expert's route pairs for the grouped launches that follow; at most 256 rows.
pub fn group(p: *const Pipelines, e: mtl.ComputeEncoder, ids: Ref, c: Call) !void {
    try plan.check(c);
    if (c.rows > group_cap or c.experts != plan.expert_count) return error.BadExpertShape;
    for ([_]Ref{ ids, counts(p), members(p) }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(GroupArgs{ .pairs = c.rows * c.topk, .experts = c.experts, .cap = group_cap }, 3);
    e.setPipeline(p.get(.group));
    e.dispatchThreads(mtl.Size.of(c.experts, 1, 1), mtl.Size.of(32, 1, 1));
}

/// gu after group: the same outputs, each expert's weight blocks decoded once a pass of four rows.
pub fn guGrouped(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, scales: Ref, nv_globals: ?struct { amax: Ref, denominator: Ref }, out: Ref, c: Call) !void {
    try plan.check(c);
    switch (c.format) {
        .nvfp4 => {
            const g = nv_globals orelse return error.MissingNvfp4Globals;
            for ([_]Ref{ x, w, scales, g.amax, g.denominator, counts(p), members(p), out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
            e.setValue(groupedArgs(c), 8);
            launchGrouped(p, e, .nvfp4_gu_grouped, c);
        },
        .mxfp8 => {
            for ([_]Ref{ x, w, scales, counts(p), members(p), out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
            e.setValue(groupedArgs(c), 6);
            launchGrouped(p, e, .mxfp8_gu_grouped, c);
        },
    }
}

/// gu, or the down products (`down`), after group on the tensor units: an op is 16 of an expert's pairs by 32 rows.
pub fn units(p: *const Pipelines, e: mtl.ComputeEncoder, down: bool, x: Ref, w: Ref, scales: Ref, nv_globals: ?struct { amax: Ref, denominator: Ref }, out: Ref, c: Call) !void {
    try plan.check(c);
    const pipes = p.units orelse return error.NoTensorUnits;
    // Gate and up rows interleave, so gu has two weight rows an output; the kernels fix the columns of each product.
    const weight_rows = if (down) c.n else 2 * c.n;
    if (c.k != @as(u32, if (down) units_down_k else units_gu_k) or weight_rows % 32 != 0) return error.BadExpertShape;
    switch (c.format) {
        .nvfp4 => {
            const g = nv_globals orelse return error.MissingNvfp4Globals;
            for ([_]Ref{ x, w, scales, g.amax, g.denominator, counts(p), members(p), out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
            e.setValue(groupedArgs(c), 8);
            e.setPipeline(pipes[@intFromBool(down)]);
        },
        .mxfp8 => {
            for ([_]Ref{ x, w, scales, counts(p), members(p), out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
            e.setValue(groupedArgs(c), 6);
            e.setPipeline(pipes[2 + @as(usize, @intFromBool(down))]);
        },
    }
    e.dispatchGroups(mtl.Size.of((weight_rows + 31) / 32, c.experts, 1), mtl.Size.of(256, 1, 1));
}

const LogitArgs = extern struct { rows: u32, vocab: u32 };
/// Rows of h against the vocabulary on the tensor units: fp32 out[row][v], bf16-rounded and capped as the glue's.
pub fn logitsUnits(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, unembed: Ref, out: Ref, rows: u32, vocab: u32) !void {
    const pipes = p.units orelse return error.NoTensorUnits;
    for ([_]Ref{ h, unembed, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(LogitArgs{ .rows = rows, .vocab = vocab }, 3);
    e.setPipeline(pipes[4]);
    e.dispatchGroups(mtl.Size.of((vocab + 31) / 32, (rows + 15) / 16, 1), mtl.Size.of(256, 1, 1));
}

const DenseArgs = extern struct { rows: u32, n: u32, k: u32 };
/// A prompt's rows of x [rows][k] against a bf16 weight [n][k] on the tensor units; k is 2,304 or 3,072.
pub fn denseUnits(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, out: Ref, rows: u32, n: u32, k: u32) !void {
    const pipes = p.units orelse return error.NoTensorUnits;
    if ((k != 2304 and k != 3072) or rows == 0 or n == 0) return error.BadExpertShape;
    for ([_]Ref{ x, w, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(DenseArgs{ .rows = rows, .n = n, .k = k }, 3);
    e.setPipeline(pipes[5]);
    e.dispatchGroups(mtl.Size.of((n + 31) / 32, (rows + 15) / 16, 1), mtl.Size.of(256, 1, 1));
}

/// The down products after group, as guGrouped.
pub fn downGrouped(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, w: Ref, scales: Ref, nv_globals: ?struct { amax: Ref, denominator: Ref }, out: Ref, c: Call) !void {
    try plan.check(c);
    switch (c.format) {
        .nvfp4 => {
            const g = nv_globals orelse return error.MissingNvfp4Globals;
            for ([_]Ref{ act, w, scales, g.amax, g.denominator, counts(p), members(p), out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
            e.setValue(groupedArgs(c), 8);
            launchGrouped(p, e, .nvfp4_down_grouped, c);
        },
        .mxfp8 => {
            for ([_]Ref{ act, w, scales, counts(p), members(p), out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
            e.setValue(groupedArgs(c), 6);
            launchGrouped(p, e, .mxfp8_down_grouped, c);
        },
    }
}

pub fn gu(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, scales: Ref, nv_globals: ?struct { amax: Ref, denominator: Ref }, ids: Ref, out: Ref, c: Call) !void {
    switch (c.format) {
        .nvfp4 => if (nv_globals) |g| return nvfp4Gu(p, e, x, w, scales, g.amax, g.denominator, ids, out, c) else return error.MissingNvfp4Globals,
        .mxfp8 => return mxfp8Gu(p, e, x, w, scales, ids, out, c),
    }
}
