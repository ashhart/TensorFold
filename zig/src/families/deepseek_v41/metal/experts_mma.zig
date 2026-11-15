//! DeepSeek V4.1 routed experts for prompt chunks on simdgroup MMA or M5 tensor units: tiles, act_quant, gate+up, down.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ex = @import("experts.zig");

pub const Ref = ex.Ref;
pub const Route = ex.Route;
pub const ExpertPtrs = ex.ExpertPtrs;
pub const use = ex.use;
pub const topk = ex.topk;
/// Pair positions of one expert a simdgroup carries (the source's XM_RT).
pub const row_tile = 16;

const XmArgs = extern struct { K: u32, N: u32, topk: u32, pairs: u32, limit: f32 };

pub const Call = struct { rows: u32, k: u32 = ex.hidden, n: u32 = ex.width, experts: u32 = ex.experts };

/// Whether this GPU takes the tensor-unit kernels: generation 17 (M5, the one checked) and Metal 4.1 (macOS 27).
pub fn units(device: mtl.Device) bool {
    return device.tensorUnits() and device.generation() == 17 and mtl.objc.osVersion().major >= 27;
}

/// Most tiles a chunk can make: full tiles, plus one partial per active expert.
pub fn maxTiles(c: Call) u32 {
    const pairs = c.rows * topk;
    return pairs / row_tile + @min(c.experts, pairs);
}

/// Scratch by buffer: tiles, act_quant codes and scales, act, out, and on tensor units (`u`) down's halves and scales.
pub fn bytes(c: Call, u: bool) [6]usize {
    const pairs = @as(usize, c.rows) * topk;
    const codes = @as(usize, c.rows) * c.k * @as(usize, if (u) 2 else 1);
    const dq = if (u) 2 * pairs * c.n + 4 * pairs * c.n / 32 else 0;
    return .{ 16 * (1 + maxTiles(c)), codes, 4 * @as(usize, c.rows) * c.k / 32, 2 * (pairs * c.n + 1), 2 * (pairs * c.k + 1), dq };
}

pub fn check(c: Call) !void {
    if (c.rows == 0 or c.k % 32 != 0 or c.k == 0 or c.k > 8192 or c.n % 32 != 0 or c.n == 0) return error.BadExpertShape;
}

/// The decode experts source (with `head`, `edits`), then the MMA kernels, and with `u` the tensor-unit kernels.
pub fn source(a: std.mem.Allocator, head: []const u8, edits: []const [2][]const u8, u: bool) ![]u8 {
    const base = try ex.source(a, head, edits);
    defer a.free(base);
    const mma = try std.mem.replaceOwned(u8, a, sources.dsv41.xmma, "#include \"experts.metal\"\n", base);
    if (!u) return mma;
    defer a.free(mma);
    return std.mem.concat(a, u8, &.{ mma, sources.dsv41.xunits });
}

pub const Pipelines = struct {
    tiles: mtl.Pipeline,
    quant: mtl.Pipeline,
    up: mtl.Pipeline,
    down: mtl.Pipeline,
    /// The tensor-unit kernels when the source has them: act_quant to halves, gate+up, down's input halves, down.
    xu: ?[4]mtl.Pipeline = null,

    /// The tensor-unit kernels where units() allows them and they build, else the simdgroup ones.
    pub fn load(a: std.mem.Allocator, device: mtl.Device) !Pipelines {
        if (units(device)) {
            const text = try source(a, "", &.{}, true);
            defer a.free(text);
            if (loadSource(device, text, true)) |p| return p else |err| std.log.warn("dsv41 xmma: tensor-unit kernels did not build ({t}); simdgroup kernels run", .{err});
        }
        const text = try source(a, "", &.{}, false);
        defer a.free(text);
        return loadSource(device, text, false);
    }

    pub fn loadSource(device: mtl.Device, text: []const u8, u: bool) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = .{
            .tiles = try mtl.Pipeline.init(device, lib, "tf_dsv41_xm_tiles", false),
            .quant = try mtl.Pipeline.init(device, lib, "tf_dsv41_xm_quant", false),
            .up = try mtl.Pipeline.init(device, lib, "tf_dsv41_xm_up", false),
            .down = try mtl.Pipeline.init(device, lib, "tf_dsv41_xm_down", false),
        };
        errdefer p.deinit();
        if (u) {
            var xu: [4]mtl.Pipeline = undefined;
            var n: usize = 0;
            errdefer for (xu[0..n]) |*q| q.deinit();
            for (&xu, [_][]const u8{ "tf_dsv41_xu_quant", "tf_dsv41_xu_up", "tf_dsv41_xu_dprep", "tf_dsv41_xu_down" }) |*q, name| {
                q.* = try mtl.Pipeline.init(device, lib, name, false);
                n += 1;
            }
            p.xu = xu;
        }
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        p.tiles.deinit();
        p.quant.deinit();
        p.up.deinit();
        p.down.deinit();
        if (p.xu) |*xu| for (xu) |*q| q.deinit();
        p.xu = null;
    }
};

fn bind(e: mtl.ComputeEncoder, refs: []const Ref) void {
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}

/// The tile table from a plan's slots (`ex.plan`'s layout): entry 0 is the count, then {expert, first, rows}.
pub fn tiles(p: *const Pipelines, e: mtl.ComputeEncoder, r: Route, table: Ref) void {
    e.setPipeline(p.tiles);
    bind(e, &.{ r.slots, r.nslots, table });
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
}

/// act_quant of the source rows x [rows][k] into codes [rows][k] (lane order; halves on units), scales [rows][k / 32].
pub fn quant(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, codes: Ref, scales: Ref, c: Call) void {
    e.setPipeline(if (p.xu) |xu| xu[0] else p.quant);
    bind(e, &.{ x, codes, scales });
    e.setValue([2]u32{ c.k, c.rows }, 3);
    e.dispatchThreads(mtl.Size.of(c.k / 32, c.rows, 1), mtl.Size.of(32, 1, 1));
}

/// Gate and up with the V4.1 epilogue into act [pair position][width] (bf16, act_quant applied).
pub fn up(p: *const Pipelines, e: mtl.ComputeEncoder, codes: Ref, scales: Ref, table: Ref, r: Route, tile_table: Ref, act: Ref, c: Call) void {
    e.setPipeline(if (p.xu) |xu| xu[1] else p.up);
    bind(e, &.{ codes, scales, table, tile_table, r.pairs, act });
    e.setValue(XmArgs{ .K = c.k, .N = c.n, .topk = topk, .pairs = c.rows * topk, .limit = ex.swiglu_limit }, 6);
    e.setBuffer(r.weights.buf, r.weights.off, 7);
    e.dispatchGroups(mtl.Size.of(c.n / 32, maxTiles(c), 1), mtl.Size.of(if (p.xu != null) 512 else 64, 1, 1));
}

/// Down: act [pair position][width] to out [pair position][hidden]; `c.k` is hidden, `c.n` the width; dq: bytes()[5].
pub fn down(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, dq: Ref, table: Ref, tile_table: Ref, out: Ref, c: Call) void {
    const pairs = c.rows * topk;
    const args = XmArgs{ .K = c.n, .N = c.k, .topk = topk, .pairs = pairs, .limit = 0 };
    if (p.xu) |xu| {
        const ds: Ref = .{ .buf = dq.buf, .off = dq.off + 2 * @as(usize, pairs) * c.n };
        e.setPipeline(xu[2]);
        bind(e, &.{ act, dq, ds });
        e.setValue([2]u32{ c.n, pairs }, 3);
        e.dispatchThreads(mtl.Size.of(c.n / 4, pairs, 1), mtl.Size.of(32, 1, 1));
        e.setPipeline(xu[3]);
        bind(e, &.{ act, table, tile_table, out });
        e.setValue(args, 4);
        e.setBuffer(dq.buf, dq.off, 5);
        e.setBuffer(ds.buf, ds.off, 6);
        e.dispatchGroups(mtl.Size.of((c.k + 63) / 64, maxTiles(c), 1), mtl.Size.of(512, 1, 1));
        return;
    }
    e.setPipeline(p.down);
    bind(e, &.{ act, table, tile_table, out });
    e.setValue(args, 4);
    e.dispatchGroups(mtl.Size.of(c.k / 32, maxTiles(c), 1), mtl.Size.of(32, 1, 1));
}
