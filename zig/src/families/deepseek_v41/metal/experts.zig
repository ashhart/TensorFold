//! DeepSeek V4.1 routed experts on Metal: Kimi's plan, chains and combine, the V4.1 act_quant, gate/up and down.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const hidden = 5120;
pub const width = 2304;
pub const experts = 384;
pub const topk = 6;
pub const swiglu_limit: f32 = 10.0;
/// Combine's canonical groups: eight groups of 48 experts, summed in a tree over the groups.
pub const per_group = experts / 8;
pub const up_rows = [_]u32{ 1, 2, 4 };
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

/// An expert's six checkpoint tensors by GPU address (Kimi's layout: w1, w3 [width, hidden], w2 [hidden, width]).
pub const ExpertPtrs = extern struct { w1p: u64, w1s: u64, w3p: u64, w3s: u64, w2p: u64, w2s: u64 };
const PlanArgs = extern struct { pairs: u32, experts: u32, first: u32, last: u32 };
const XpArgs = extern struct { K: u32, N: u32, topk: u32, rows_max: u32, beta: f32 = 0, lin: f32 = 0 };
const UpArgs = extern struct { K: u32, N: u32, topk: u32, rows_max: u32, limit: f32 };

/// The V4.1 kernels' source: `head` before it (a TF_CTL define), `edits` applied to Kimi's text (planted controls).
pub fn source(a: std.mem.Allocator, head: []const u8, edits: []const [2][]const u8) ![]u8 {
    const common = try std.mem.replaceOwned(u8, a, sources.kimi_common, "#pragma once\n", "");
    defer a.free(common);
    var kimi = try std.mem.replaceOwned(u8, a, sources.kimi_experts, "#include \"kimi_common.h\"\n", common);
    defer a.free(kimi);
    for (edits) |e| {
        const next = try std.mem.replaceOwned(u8, a, kimi, e[0], e[1]);
        a.free(kimi);
        kimi = next;
    }
    const body = try std.mem.replaceOwned(u8, a, sources.deepseek_v41_experts, "#include \"../kimi/experts.metal\"\n", kimi);
    defer a.free(body);
    return std.mem.concat(a, u8, &.{ head, body });
}

pub const Pipelines = struct {
    plan: mtl.Pipeline,
    up: [3]mtl.Pipeline,
    /// act_quant of the rows once, and the gate/up over its values (the decode layers'): slots of one row, of more
    quant: mtl.Pipeline,
    upc: [2]mtl.Pipeline,
    /// Kimi's down for the slots of one row, and of more (four rows a pass)
    down: [2]mtl.Pipeline,
    combine: mtl.Pipeline,
    combine_f32: mtl.Pipeline,

    pub fn load(a: std.mem.Allocator, device: mtl.Device) !Pipelines {
        const text = try source(a, "", &.{});
        defer a.free(text);
        return loadSource(device, text);
    }

    /// The same kernels from other text: the contract's planted controls edit it.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        p.plan = try mtl.Pipeline.init(device, lib, "k3_route_plan", false);
        p.combine = try mtl.Pipeline.init(device, lib, "k3_xp_combine", false);
        p.combine_f32 = try mtl.Pipeline.init(device, lib, "k3_xp_combine_f32", false);
        inline for (up_rows, 0..) |r, i| p.up[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_dsv41_xp_up_r{d}", .{r}), false);
        p.quant = try mtl.Pipeline.init(device, lib, "tf_dsv41_xp_quant", false);
        inline for (.{ "one", "multi" }, 0..) |k, i| p.upc[i] = try mtl.Pipeline.init(device, lib, "tf_dsv41_xp_upc_" ++ k, false);
        p.down = .{ try mtl.Pipeline.init(device, lib, "tf_dsv41_xp_down_one", false), try mtl.Pipeline.init(device, lib, "tf_dsv41_xp_down_multi", false) };
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        p.plan.deinit();
        p.combine.deinit();
        p.combine_f32.deinit();
        for (p.up) |x| x.deinit();
        p.quant.deinit();
        for (p.upc) |x| x.deinit();
        for (p.down) |x| x.deinit();
    }
};

/// The routing buffers of one layer's call, all on the GPU: ids and weights [rows][topk], then the plan's outputs.
pub const Route = struct { ids: Ref, weights: Ref, slots: Ref, nslots: Ref, pairs: Ref, where: Ref };

/// `topk` and `experts` are the layer's (a DSpark block's differ); the combine's groups are eighths of the ids.
pub const Call = struct { rows: u32, k: u32 = hidden, n: u32 = width, experts: u32 = experts, topk: u32 = topk };

pub fn perGroup(c: Call) u32 {
    return c.experts / 8;
}

pub fn check(c: Call) !void {
    if (c.rows == 0 or c.rows > 16 or c.k % 32 != 0 or c.k == 0 or c.k > 5120 or c.n % 32 != 0 or c.n == 0) return error.BadExpertShape;
    if (c.topk == 0 or c.topk > topk or c.experts % 8 != 0 or c.experts < c.topk or c.experts > 1024) return error.BadExpertShape;
}

/// Bytes of scratch a call needs, by buffer: slots, nslots, pairs, where, act (bf16), out (bf16).
pub fn bytes(c: Call) [6]usize {
    const pairs = @as(usize, c.rows) * c.topk;
    return .{ 12 * pairs, 4, 4 * pairs, 4 * pairs, 2 * pairs * c.n, 2 * pairs * c.k };
}

fn bind(e: mtl.ComputeEncoder, refs: []const Ref) void {
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}

/// Declare every buffer an expert table points into: the kernels reach them by address.
pub fn use(e: mtl.ComputeEncoder, backing: []const mtl.Buffer) void {
    for (backing) |b| e.useResource(b, 1);
}

pub fn plan(p: *const Pipelines, e: mtl.ComputeEncoder, r: Route, c: Call) void {
    e.setPipeline(p.plan);
    bind(e, &.{ r.ids, r.slots, r.nslots, r.pairs, r.where });
    e.setValue(PlanArgs{ .pairs = c.rows * c.topk, .experts = c.experts, .first = 0, .last = c.experts }, 5);
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
}

fn groups(c: Call, n: u32) mtl.Size {
    return mtl.Size.of(n / 32, @min(c.experts, c.rows * c.topk), 1);
}

/// Gate/up with the clamped SwiGLU into act [pair position][width] (bf16, act_quant already applied).
pub fn up(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, table: Ref, r: Route, act: Ref, c: Call) void {
    e.setPipeline(p.up[if (c.rows == 1) 0 else if (c.rows == 2) 1 else 2]);
    bind(e, &.{ x, table, r.slots, r.nslots, r.pairs, act });
    e.setValue(UpArgs{ .K = c.k, .N = c.n, .topk = c.topk, .rows_max = c.rows, .limit = swiglu_limit }, 6);
    e.setBuffer(r.weights.buf, r.weights.off, 7);
    e.dispatchGroups(groups(c, c.n), mtl.Size.of(256, 1, 1));
}

/// act_quant of x [rows][k] bf16 once: E4M3 values as fp16 [rows][k], fp32 block scales [rows][k / 32], for `upCoded`.
pub fn quant(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, codes: Ref, scales: Ref, c: Call) void {
    e.setPipeline(p.quant);
    bind(e, &.{ x, codes, scales });
    e.setValue(UpArgs{ .K = c.k, .N = c.n, .topk = c.topk, .rows_max = c.rows, .limit = swiglu_limit }, 3);
    e.dispatchThreads(mtl.Size.of(c.k / 32, c.rows, 1), mtl.Size.of(32, 1, 1));
}

/// `up` over `quant`'s output (after a barrier): slots of one row, then of more; a row's bits are the same in either.
pub fn upCoded(p: *const Pipelines, e: mtl.ComputeEncoder, codes: Ref, scales: Ref, table: Ref, r: Route, act: Ref, c: Call) void {
    for (p.upc[0..if (c.rows > 1) 2 else 1]) |pipe| {
        e.setPipeline(pipe);
        bind(e, &.{ codes, table, r.slots, r.nslots, r.pairs, act });
        e.setValue(UpArgs{ .K = c.k, .N = c.n, .topk = c.topk, .rows_max = c.rows, .limit = swiglu_limit }, 6);
        e.setBuffer(r.weights.buf, r.weights.off, 7);
        e.setBuffer(scales.buf, scales.off, 8);
        e.dispatchGroups(groups(c, c.n), mtl.Size.of(256, 1, 1));
    }
}

/// Kimi's down: act [pair position][width] to out [pair position][hidden] (bf16); the slots of one row, then of more.
pub fn down(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, table: Ref, r: Route, out: Ref, c: Call) void {
    for (p.down[0..if (c.rows > 1) 2 else 1]) |pipe| {
        e.setPipeline(pipe);
        bind(e, &.{ act, table, r.slots, r.nslots, r.pairs, out });
        e.setValue(XpArgs{ .K = c.n, .N = c.k, .topk = c.topk, .rows_max = c.rows }, 6);
        e.dispatchGroups(groups(c, c.k), mtl.Size.of(256, 1, 1));
    }
}

/// Kimi's combine over canonical groups of `per_group` experts; the up already applied the route weights: pass ones.
pub fn combine(p: *const Pipelines, e: mtl.ComputeEncoder, out: Ref, r: Route, weights: Ref, y: Ref, f32_out: bool, c: Call) void {
    e.setPipeline(if (f32_out) p.combine_f32 else p.combine);
    bind(e, &.{ out, r.where, weights, y, r.ids });
    e.setValue(XpArgs{ .K = c.n, .N = c.k, .topk = c.topk, .rows_max = c.rows }, 5);
    e.setValue([3]u32{ perGroup(c), 0, 8 }, 6);
    e.dispatchThreads(mtl.Size.of(c.k, c.rows, 1), mtl.Size.of(256, 1, 1));
}
