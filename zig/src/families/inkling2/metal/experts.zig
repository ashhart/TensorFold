//! Real Metal dispatch for Inkling 2 Mini's M3 NVFP4/MXFP8 routed experts.
const std = @import("std");
const mtl = @import("metal");
const src = @import("inkling2_metal");
const plan = @import("experts_plan.zig");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Format = plan.Format;
pub const Call = plan.Call;
pub const Operation = enum { nvfp4_gu, nvfp4_down, mxfp8_gu, mxfp8_down, nvfp4_quantize, mxfp8_quantize, combine };

pub const function_names = [_][:0]const u8{
    "tf_ink2_nvfp4_gu",
    "tf_ink2_nvfp4_down",
    "tf_ink2_mxfp8_gu",
    "tf_ink2_mxfp8_down",
    "tf_ink2_nvfp4_quantize",
    "tf_ink2_mxfp8_quantize",
    "tf_ink2_combine",
};

pub const Pipelines = struct {
    entries: [function_names.len]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, src.experts);
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
            if (out.entries[i].maxThreads() < @as(usize, if (std.mem.endsWith(u8, name, "quantize")) 32 else 256)) return error.ExpertThreads;
        }
        return out;
    }

    pub fn deinit(p: *Pipelines) void {
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

pub fn gu(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, scales: Ref, nv_globals: ?struct { amax: Ref, denominator: Ref }, ids: Ref, out: Ref, c: Call) !void {
    switch (c.format) {
        .nvfp4 => if (nv_globals) |g| return nvfp4Gu(p, e, x, w, scales, g.amax, g.denominator, ids, out, c) else return error.MissingNvfp4Globals,
        .mxfp8 => return mxfp8Gu(p, e, x, w, scales, ids, out, c),
    }
}
