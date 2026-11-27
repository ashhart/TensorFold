//! A control vector on Inkling 2 Mini: one direction per decoder visit, and the launch that writes a steered visit.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const Ref = @import("inkling2_metal_ref").Ref;

/// The width of a visit's output.
pub const hidden = 2304;

/// The steered visit output, and the add the separately ordered reference launches after main's mlp_partial.
pub const symbol = "tf_ink2_mlp_partial_steer";
pub const add_symbol = "tf_ink2_steer_add";
/// The steered visit output fused with the next reader's mix, as main's tf_inkling2_mlp_partial_ar fuses it.
pub const ar_symbol = "tf_ink2_mlp_partial_ar_steer";
pub const norm_symbol = "tf_ink2_mlp_partial_ar_norm_steer";

/// The steer kernels' arguments: the elements and the strength.
pub const Args = extern struct { count: u32, alpha: f32 };
/// The fused launch's arguments: main's rows, sources and queries, then the strength.
pub const ArArgs = extern struct { rows: u32, sources: u32, queries: u32, alpha: f32 };
pub const NormArgs = extern struct { rows: u32, sources: u32, alpha: f32 };

/// One visit's direction: its zero-based visit id and the vector to add.
pub const Visit = struct { id: u32, dir: []const f32 };

/// What a plan is built from: the strength, the visits a token makes and the directions.
pub const Source = struct {
    alpha: f32,
    visits_total: u32,
    visits: []const Visit,
};

/// Why a vector's directions do not fit the visit plan.
pub const Problem = error{ BadSteerShape, BadSteerLayer, BadSteerWidth };

/// A vector fits when it holds a direction, every id names a visit from 1 to the last, each direction 2304 wide.
pub fn check(src: Source) Problem!void {
    if (src.visits_total == 0 or src.visits.len == 0) return error.BadSteerShape;
    for (src.visits) |v| {
        // visit 0's output is the embedding layer's: the format carries no direction for it
        if (v.id == 0 or v.id >= src.visits_total) return error.BadSteerLayer;
        if (v.dir.len != hidden) return error.BadSteerWidth;
    }
}

/// The directions on the device, one 2304-float row per covered visit, and the steered launch's pipeline.
pub const Plan = struct {
    gpa: std.mem.Allocator,
    alpha: f32,
    ids: []u32,
    /// The row of `dirs` that steers each visit; null where the vector does not cover it.
    cover: []?u32,
    dirs: mtl.Buffer,
    pipe: mtl.Pipeline,
    ar_pipe: mtl.Pipeline,
    norm_pipe: mtl.Pipeline,

    /// The direction that steers `visit`, as a buffer reference; null when the vector does not cover it.
    pub fn dir(p: *const Plan, visit: u32) ?Ref {
        if (visit >= p.cover.len) return null;
        const row = p.cover[visit] orelse return null;
        return .{ .buf = p.dirs, .off = @as(usize, row) * hidden * 4 };
    }

    /// The steered visit output: p = bf16(bf16(t + x) + alpha * d), over `count` elements of whole rows.
    pub fn partial(p: *const Plan, e: mtl.ComputeEncoder, d: Ref, t: Ref, x: Ref, out: Ref, count: usize) void {
        std.debug.assert(count > 0 and count % hidden == 0);
        e.setPipeline(p.pipe);
        inline for (.{ t, x, out, d }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(Args{ .count = @intCast(count), .alpha = p.alpha }, 4);
        e.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// The steered output and the next mix in one launch: main's mlpPartialAr arguments, with p steered by `d`.
    pub fn partialAr(p: *const Plan, e: mtl.ComputeEncoder, d: Ref, t: Ref, x: Ref, blocks: Ref, inv_rms: Ref, queries: Ref, p_out: Ref, partial_out: Ref, ar_out: Ref, rows: u32, sources_: u32, queries_: u32) void {
        std.debug.assert(rows > 0 and rows <= 256 and sources_ > 0 and sources_ <= 5 and queries_ > 0 and queries_ <= 3);
        e.setPipeline(p.ar_pipe);
        inline for (.{ t, x, blocks, inv_rms, queries, p_out, partial_out, ar_out, d }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(ArArgs{ .rows = rows, .sources = sources_, .queries = queries_, .alpha = p.alpha }, 9);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// The steered output, the next mix and its norm in one launch: main's mlpPartialArNorm, p steered by `d`.
    pub fn partialArNorm(p: *const Plan, e: mtl.ComputeEncoder, d: Ref, t: Ref, x: Ref, blocks: Ref, inv_rms: Ref, queries: Ref, norm_w: Ref, p_out: Ref, partial_out: Ref, norm_out: Ref, rows: u32, sources_: u32) void {
        std.debug.assert(rows > 0 and rows <= 256 and sources_ > 0 and sources_ <= 5);
        e.setPipeline(p.norm_pipe);
        inline for (.{ t, x, blocks, inv_rms, queries, norm_w, d, p_out, partial_out, norm_out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(NormArgs{ .rows = rows, .sources = sources_, .alpha = p.alpha }, 10);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn deinit(p: *Plan) void {
        p.norm_pipe.deinit();
        p.ar_pipe.deinit();
        p.pipe.deinit();
        p.dirs.deinit();
        p.gpa.free(p.ids);
        p.gpa.free(p.cover);
        p.gpa.destroy(p);
    }
};

/// The plan for a vector: its directions in one device buffer and the steered pipeline, compiled here alone.
pub fn init(gpa: std.mem.Allocator, device: mtl.Device, src: Source) !*Plan {
    try check(src);
    const plan = try gpa.create(Plan);
    errdefer gpa.destroy(plan);
    const ids = try gpa.alloc(u32, src.visits.len);
    errdefer gpa.free(ids);
    const cover = try gpa.alloc(?u32, src.visits_total);
    errdefer gpa.free(cover);
    @memset(cover, null);
    const dirs = try device.buffer(src.visits.len * hidden * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer dirs.deinit();
    const rows = dirs.slice(f32, src.visits.len * hidden);
    for (src.visits, 0..) |v, i| {
        @memcpy(rows[i * hidden ..][0..hidden], v.dir);
        ids[i] = v.id;
        cover[v.id] = @intCast(i);
    }
    const lib = try mtl.Library.fromSource(device, sources.inkling2_steer, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, symbol, false);
    errdefer pipe.deinit();
    if (pipe.maxThreads() < 256) return error.Inkling2Threads;
    const ar_pipe = try mtl.Pipeline.init(device, lib, ar_symbol, false);
    errdefer ar_pipe.deinit();
    if (ar_pipe.maxThreads() < 256) return error.Inkling2Threads;
    const norm_pipe = try mtl.Pipeline.init(device, lib, norm_symbol, false);
    errdefer norm_pipe.deinit();
    if (norm_pipe.maxThreads() < 256) return error.Inkling2Threads;
    plan.* = .{ .gpa = gpa, .alpha = src.alpha, .ids = ids, .cover = cover, .dirs = dirs, .pipe = pipe, .ar_pipe = ar_pipe, .norm_pipe = norm_pipe };
    return plan;
}
