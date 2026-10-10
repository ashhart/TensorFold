//! The products on a projection of any format, dispatched on its tag; `affine*` are the MLX affine path behind them.

const t = @import("types.zig");
const quant = @import("core").quant;
const launches = @import("../launches.zig");
const affine_launch = @import("../launches/affine.zig");
const registry = @import("core").registry;
const Ops = @import("ops.zig").Ops;
const Error = t.Error;
const Tensor = t.Tensor;
const Affine = t.Affine;
const Projection = quant.Projection;
const mlx = quant.mlx;
const p = t.p;
const f = t.f;
const i = t.i;
const int = t.int;

/// The MLX matrix of a projection as the MLX launches take it.
fn affineOf(proj: Projection, h: mlx.Matrix) Affine {
    const tables: t.Kind = switch (h.tables) {
        .f32 => .f32,
        .bf16 => .bf16,
        .f16 => .f16,
    };
    return .{ .words = h.words, .scale = h.scale, .bias = h.bias, .tables = tables, .n = proj.n, .k = proj.k, .bits = h.bits, .group = h.group, .partial = proj.partial };
}

/// matmul(x, proj) for a projection of any format: fp32 when `f32_out`, else x's kind (the MLX rules of `affine`).
pub fn project(o: Ops, x: Tensor, proj: Projection, m: usize, f32_out: bool) Error!Tensor {
    switch (proj.handle) {
        .mlx => |h| return affine(o, x, affineOf(proj, h), m, f32_out),
        .dense => |h| {
            if (f32_out or m == 0 or x.kind == .f32) return error.BadShape;
            const out: Tensor = .{ .ptr = try o.arena.take(m * proj.n * x.kind.size()), .kind = x.kind };
            try denseRows(o, x, h.weight, out, m, proj.n, proj.k);
            return out;
        },
    }
}

/// Up to four products of the same `m` rows in one launch; false, nothing launched, when no grouped tile takes them.
pub fn projectGroup(o: Ops, x: Tensor, ps: []const Projection, m: usize, outs: []Tensor) Error!bool {
    var ws: [4]Affine = undefined;
    if (ps.len > ws.len) return false;
    for (ps, ws[0..ps.len]) |pr, *w| switch (pr.handle) {
        .mlx => |h| w.* = affineOf(pr, h),
        .dense => return false,
    };
    return affineGroup(o, x, ws[0..ps.len], m, outs);
}

/// The routed gate and up with the activation as the epilogue; null when the format or the tile does not take it.
pub fn projectRoutedAct(o: Ops, x: Tensor, proj: Projection, items: u64, count: usize, members: u64, pairs: usize, x_div: usize, rows: usize, limit: f32) Error!?Tensor {
    switch (proj.handle) {
        .mlx => |h| return affineRoutedAct(o, x, affineOf(proj, h), items, count, members, pairs, x_div, rows, limit),
        .dense => return null,
    }
}

/// matmul_routed on a stack of experts: every item in one launch; out (pairs, N) fp32.
pub fn projectRouted(o: Ops, x: Tensor, proj: Projection, items: u64, count: usize, members: u64, pairs: usize, x_div: usize, rows: usize) Error!u64 {
    switch (proj.handle) {
        .mlx => |h| return affineRouted(o, x, affineOf(proj, h), items, count, members, pairs, x_div, rows),
        .dense => return error.BadShape,
    }
}

/// gather_rows: `n` embedding rows of a table of any format with a row gather, by device ids, dequantized into `out`.
pub fn embedRows(o: Ops, table: Projection, ids: u64, n: usize, out: Tensor) Error!void {
    switch (table.handle) {
        .mlx => |h| return embedAffine(o, affineOf(table, h), ids, n, out),
        .dense => |h| try o.l.tf_embed_dense(f(h.weight), @ptrFromInt(ids), int(n), int(table.k), p(out.ptr), @backingInt(out.kind), o.stream),
    }
}

/// The product of `m` rows of x with `w` as the registry sees it; a lane round keeps its decode tile at any row count.
fn shapeOf(o: Ops, x: Tensor, w: Affine, m: usize) registry.Shape {
    return .{
        .m = @intCast(m),
        .n = w.n,
        .k = w.k,
        .bits = w.bits,
        .group = w.group,
        .fp16 = x.kind == .f16,
        .tables = switch (w.tables) {
            .f32 => .f32,
            .bf16 => .bf16,
            .f16 => .f16,
        },
        .x_aligned = x.ptr % 16 == 0,
        .words_aligned = w.words % affine_launch.Kernels.gemmAlign(w.bits) == 0,
        .round = o.window,
    };
}

/// matmul(x, ...) as the Python wrapper runs it on the auto schedule: fp32 (`f32_out`), else the input dtype.
pub fn affine(o: Ops, x: Tensor, w: Affine, m: usize, f32_out: bool) Error!Tensor {
    try w.check();
    if (m == 0) return error.BadShape;
    const fp16 = x.kind == .f16;
    if (x.kind == .f32 or (fp16 and o.bf16) or (!fp16 and !o.bf16)) return error.BadShape;
    const n: usize = w.n;
    const kind = w.tables.table();
    var half = false;
    const z = o.l;
    const shape = shapeOf(o, x, w, m);
    const tile = z.affine.choose(.project, if (o.prefill) .prefill else .decode, shape) orelse return error.BadShape;
    half = !f32_out and !o.prefill and tile.roundsAct(shape);
    const out = try o.arena.take(m * n * @as(usize, if (half) 2 else 4));
    {
        const arg: affine_launch.Arg = .{
            .x = x.ptr,
            .words = w.words,
            .scale = .{ .p = w.scale, .kind = kind },
            .bias = .{ .p = w.bias, .kind = kind },
            .out = if (half) 0 else out,
            .m = int(m),
            .n = int(n),
            .k = int(w.k),
            .bits = w.bits,
            .group = w.group,
            .fp16 = @intFromBool(fp16),
            .out16 = if (half) out else 0,
        };
        try tile.launch(&z.affine, .{ .r = z.r, .s = o.stream, .arg = arg });
    }
    if (half) return .{ .ptr = out, .kind = x.kind };
    if (f32_out) return .{ .ptr = out, .kind = .f32 };
    const narrow = try o.arena.take(m * n * 2);
    try o.cast(.{ .ptr = out, .kind = .f32 }, .{ .ptr = narrow, .kind = x.kind }, m * n);
    return .{ .ptr = narrow, .kind = x.kind };
}

/// Up to four products of the same rows in one stream-tile launch, rounded to x's kind; false if unsupported.
pub fn affineGroup(o: Ops, x: Tensor, ws: []const Affine, m: usize, outs: []Tensor) Error!bool {
    const z = o.l;
    if (o.prefill or !o.fused() or ws.len < 2 or ws.len > 4 or m == 0 or x.kind == .f32) return false;
    var widest: u32 = 0;
    for (ws) |w| {
        try w.check();
        if (w.k != ws[0].k or w.bits != ws[0].bits or w.group != ws[0].group or w.tables != ws[0].tables or w.partial) return false;
        widest = @max(widest, w.n);
    }
    const kind = ws[0].tables.table();
    var widest_w = ws[0];
    widest_w.n = widest;
    if (z.affine.choose(.group, .decode, shapeOf(o, x, widest_w, m)) == null) return false;
    var sides: [4]affine_launch.Side = undefined;
    for (ws, outs[0..ws.len], sides[0..ws.len]) |w, *out, *side| {
        out.* = .{ .ptr = try o.arena.take(m * w.n * 2), .kind = x.kind };
        side.* = .{ .words = w.words, .scale = w.scale, .bias = w.bias, .n = int(w.n), .out = out.ptr };
    }
    const arg: affine_launch.Arg = .{
        .x = x.ptr,
        .words = 0,
        .scale = .{ .p = 0, .kind = kind },
        .bias = .{ .p = 0, .kind = kind },
        .out = 0,
        .m = int(m),
        .n = int(widest),
        .k = int(ws[0].k),
        .bits = int(ws[0].bits),
        .group = int(ws[0].group),
        .fp16 = @intFromBool(x.kind == .f16),
    };
    try z.affine.groupRun(z.r, arg, sides[0..ws.len], true, o.stream);
    return true;
}

/// Routed gate and up in one launch, silu(gate) * up as epilogue into x's kind; null when the tile cannot.
pub fn affineRoutedAct(o: Ops, x: Tensor, w: Affine, items: u64, count: usize, members: u64, pairs: usize, x_div: usize, rows: usize, limit: f32) Error!?Tensor {
    const z = o.l;
    if (o.prefill or !o.fused() or x.kind == .f32 or w.n % 2 != 0) return null;
    try w.check();
    const kind = w.tables.table();
    var shape = shapeOf(o, x, w, rows);
    shape.pairs = true;
    if (z.affine.choose(.routed_act, .decode, shape) == null) return null;
    const out = try o.arena.take(pairs * (w.n / 2) * 2);
    const arg: affine_launch.Arg = .{
        .x = x.ptr,
        .words = w.words,
        .scale = .{ .p = w.scale, .kind = kind },
        .bias = .{ .p = w.bias, .kind = kind },
        .out = 0,
        .m = int(rows),
        .n = int(w.n),
        .k = int(w.k),
        .bits = int(w.bits),
        .group = int(w.group),
        .fp16 = @intFromBool(x.kind == .f16),
        .route = .{ .items = items, .members = members, .x_div = int(x_div) },
        .out16 = out,
    };
    if (!try z.affine.pairRun(z.r, arg, limit, int(count), o.stream)) return null;
    return .{ .ptr = out, .kind = x.kind };
}

/// matmul_routed: every item (expert, first, count) in one launch over stacked weights; out (pairs, N) fp32.
pub fn affineRouted(o: Ops, x: Tensor, w: Affine, items: u64, count: usize, members: u64, pairs: usize, x_div: usize, rows: usize) Error!u64 {
    try w.check();
    const out = try o.arena.of(f32, pairs * w.n);
    if (o.prefill) {
        const z = o.l;
        const kind = w.tables.table();
        try z.affine.prefillLaunch(z.r, .{ .x = x.ptr, .words = w.words, .scale = .{ .p = w.scale, .kind = kind }, .bias = .{ .p = w.bias, .kind = kind }, .out = out, .m = int(rows), .n = int(w.n), .k = int(w.k), .bits = int(w.bits), .group = int(w.group), .fp16 = @intFromBool(x.kind == .f16), .route = .{ .items = items, .members = members, .x_div = int(x_div) } }, o.stream, int(count));
        return out;
    }
    try o.l.tf_affine_routed(p(x.ptr), p(w.words), p(w.scale), p(w.bias), w.tables.table(), p(out), i(items), int(count), i(members), int(x_div), int(rows), int(w.n), int(w.k), w.bits, w.group, @intFromBool(x.kind == .f16), o.stream);
    return out;
}

/// The MLX affine gather: `n` embedding rows of width `table.k` by device ids, dequantized into `out`.
pub fn embedAffine(o: Ops, table: Affine, ids: u64, n: usize, out: Tensor) Error!void {
    try table.check();
    try o.l.tf_embed_rows(p(table.words), p(table.scale), p(table.bias), @backingInt(table.tables), @ptrFromInt(ids), int(n), table.bits, table.group, int(table.k), p(out.ptr), @backingInt(out.kind), o.stream);
}

/// x (rows, k) of x's kind times fp32 weights (n, k), out (rows, n) in x's kind: an unquantized draft projection.
pub fn denseRows(o: Ops, x: Tensor, w: u64, out: Tensor, rows: usize, n: usize, k: usize) Error!void {
    try o.l.tf_dense_rows(p(x.ptr), @backingInt(x.kind), f(w), p(out.ptr), int(rows), int(n), int(k), o.stream);
}
