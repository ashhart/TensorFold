//! DeepSeek V4.1 routed experts for prompt chunks on simdgroup MMA: tiles, act_quant, gate+up, down; equal to decode.
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

/// Most tiles a chunk can make: full tiles, plus one partial per active expert.
pub fn maxTiles(c: Call) u32 {
    const pairs = c.rows * topk;
    return pairs / row_tile + @min(c.experts, pairs);
}

/// Scratch a call needs by buffer: tiles, act_quant codes, act_quant scales, act (bf16), out (bf16).
pub fn bytes(c: Call) [5]usize {
    const pairs = @as(usize, c.rows) * topk;
    return .{ 16 * (1 + maxTiles(c)), @as(usize, c.rows) * c.k, 4 * @as(usize, c.rows) * c.k / 32, 2 * (pairs * c.n + 1), 2 * (pairs * c.k + 1) };
}

pub fn check(c: Call) !void {
    if (c.rows == 0 or c.k % 32 != 0 or c.k == 0 or c.k > 8192 or c.n % 32 != 0 or c.n == 0) return error.BadExpertShape;
}

/// The kernels' source: the decode experts source (with `head`, `edits`) followed by the MMA kernels.
pub fn source(a: std.mem.Allocator, head: []const u8, edits: []const [2][]const u8) ![]u8 {
    const base = try ex.source(a, head, edits);
    defer a.free(base);
    return std.mem.replaceOwned(u8, a, sources.deepseek_v41_xmma, "#include \"experts.metal\"\n", base);
}

pub const Pipelines = struct {
    tiles: mtl.Pipeline,
    quant: mtl.Pipeline,
    up: mtl.Pipeline,
    down: mtl.Pipeline,

    pub fn load(a: std.mem.Allocator, device: mtl.Device) !Pipelines {
        const text = try source(a, "", &.{});
        defer a.free(text);
        return loadSource(device, text);
    }

    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        return .{
            .tiles = try mtl.Pipeline.init(device, lib, "tf_dsv41_xm_tiles", false),
            .quant = try mtl.Pipeline.init(device, lib, "tf_dsv41_xm_quant", false),
            .up = try mtl.Pipeline.init(device, lib, "tf_dsv41_xm_up", false),
            .down = try mtl.Pipeline.init(device, lib, "tf_dsv41_xm_down", false),
        };
    }

    pub fn deinit(p: *Pipelines) void {
        p.tiles.deinit();
        p.quant.deinit();
        p.up.deinit();
        p.down.deinit();
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

/// act_quant of the chunk's source rows x [rows][k] bf16 into codes [rows][k] (lane order) and scales [rows][k / 32].
pub fn quant(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, codes: Ref, scales: Ref, c: Call) void {
    e.setPipeline(p.quant);
    bind(e, &.{ x, codes, scales });
    e.setValue([2]u32{ c.k, c.rows }, 3);
    e.dispatchThreads(mtl.Size.of(c.k / 32, c.rows, 1), mtl.Size.of(32, 1, 1));
}

/// Gate and up (one simdgroup each) with the V4.1 epilogue into act [pair position][width] (bf16, act_quant applied).
pub fn up(p: *const Pipelines, e: mtl.ComputeEncoder, codes: Ref, scales: Ref, table: Ref, r: Route, tile_table: Ref, act: Ref, c: Call) void {
    e.setPipeline(p.up);
    bind(e, &.{ codes, scales, table, tile_table, r.pairs, act });
    e.setValue(XmArgs{ .K = c.k, .N = c.n, .topk = topk, .pairs = c.rows * topk, .limit = ex.swiglu_limit }, 6);
    e.setBuffer(r.weights.buf, r.weights.off, 7);
    e.dispatchGroups(mtl.Size.of(c.n / 32, maxTiles(c), 1), mtl.Size.of(64, 1, 1));
}

/// Down: act [pair position][width] to out [pair position][hidden] (bf16); `c.k` is hidden, `c.n` the width.
pub fn down(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, table: Ref, tile_table: Ref, out: Ref, c: Call) void {
    e.setPipeline(p.down);
    bind(e, &.{ act, table, tile_table, out });
    e.setValue(XmArgs{ .K = c.n, .N = c.k, .topk = topk, .pairs = c.rows * topk, .limit = 0 }, 4);
    e.dispatchGroups(mtl.Size.of(c.k / 32, maxTiles(c), 1), mtl.Size.of(32, 1, 1));
}
