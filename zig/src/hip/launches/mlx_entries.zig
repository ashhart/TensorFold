//! The registry's MLX affine entries: each tile with its needs, the shapes it has code for and its launch.

const std = @import("std");
const registry = @import("core").registry;
const tuning = @import("../tuning/tuning.zig");
const affine = @import("affine.zig");

const Entry = affine.Entry;
const Shape = registry.Shape;
const Env = registry.Env;
const Call = affine.Call;
const Op = registry.Op;
const Kernels = affine.Kernels;
const Error = @import("../runtime.zig").Error;
const Launch = affine.Launch;

// ---- what the GPU has and the run switched on

fn always(_: Env) bool {
    return true;
}

fn rdna2(e: Env) bool {
    return !e.bf16;
}

fn matrixCores(e: Env) bool {
    return e.matrix;
}

fn streamOn(e: Env) bool {
    return e.stream_on;
}

fn gemmOn(e: Env) bool {
    return e.gemm_on;
}

fn matrixOn(e: Env) bool {
    return e.matrix and e.matrix_on and e.gemm_on;
}

// ---- the shapes the tiles have code for

fn groupOk(s: Shape) bool {
    return s.group == 32 or s.group == 64 or s.group == 128;
}

fn widthOk(s: Shape) bool {
    return switch (s.bits) {
        2, 3, 4, 5, 6, 8 => true,
        else => false,
    };
}

/// The words the GEMM tiles load wide, and tables of one kind.
fn gemmFits(s: Shape) bool {
    return s.words_aligned and s.tables_alike;
}

/// The stream tile: whole groups of a multiple of 32, x on 16 bytes, 16-bit tables of one kind.
fn streamShape(s: Shape) bool {
    return s.n >= 1 and s.group != 0 and s.group % 32 == 0 and s.group <= affine.lane_group_max and s.k % s.group == 0 and
        s.x_aligned and s.tables != .f32 and s.tables_alike and widthOk(s);
}

fn lanesOk(s: Shape) bool {
    return s.m >= 1 and s.m <= affine.lane_rows and s.n >= 1 and s.group != 0 and s.group % 32 == 0 and
        s.group <= affine.lane_group_max and s.k % s.group == 0;
}

fn rowOk(s: Shape) bool {
    return s.m == 1 and lanesOk(s) and Kernels.rowLanes(@intCast(s.k / s.group)) != 0;
}

fn blockOk(s: Shape) bool {
    return s.m >= 1 and s.n >= 1 and s.group != 0 and s.group % 32 == 0 and s.k % s.group == 0 and s.k % 16 == 0;
}

fn prefillOk(s: Shape) bool {
    return s.m >= 1 and s.n >= 1 and groupOk(s) and s.k % s.group == 0 and s.k % 16 == 0;
}

/// What the activation type's dot2 kernels take of a product: any group of an even width.
fn dot2Shape(s: Shape) bool {
    return s.m >= 1 and s.n >= 1 and s.k >= 1 and s.group != 0 and s.group % 2 == 0 and s.k % s.group == 0;
}

/// What a decode launch takes of a product on this build: the checks every tile of it shares.
fn valid(s: Shape) bool {
    if (s.fp16) return dot2Shape(s);
    return s.m >= 1 and s.n >= 1 and groupOk(s) and s.k % s.group == 0 and s.k % 16 == 0;
}

fn routedOk(s: Shape) bool {
    return groupOk(s) and s.items >= 1;
}

// ---- the decode tiles of a plain product

fn fitsSplit(s: Shape) bool {
    return s.parts > 1 and s.fp16 and valid(s) and groupOk(s) and (s.k / s.group) % s.parts == 0;
}

fn fitsStream(s: Shape) bool {
    return streamShape(s) and (s.round or valid(s));
}

fn fitsRow(s: Shape) bool {
    return valid(s) and groupOk(s) and rowOk(s);
}

fn fitsLanes(s: Shape) bool {
    return valid(s) and groupOk(s) and lanesOk(s);
}

fn fitsWide(s: Shape) bool {
    return valid(s) and groupOk(s) and s.m > 8 and (s.m < 64 or s.k % 16 != 0);
}

fn fitsMatrix(s: Shape) bool {
    return valid(s) and blockOk(s) and !s.fp16 and gemmFits(s);
}

fn fitsGemm(s: Shape) bool {
    return valid(s) and groupOk(s) and blockOk(s) and gemmFits(s);
}

fn fitsBlock(s: Shape) bool {
    return valid(s) and groupOk(s) and blockOk(s);
}

fn fitsTiled(s: Shape) bool {
    return dot2Shape(s) and !groupOk(s) and s.m > 1 and s.group <= 128;
}

fn fitsReference(s: Shape) bool {
    return dot2Shape(s);
}

// ---- the decode tiles of a plan over stacked experts, a group of products and a plan's gate and up

fn fitsRoutedStream(s: Shape) bool {
    return routedOk(s) and streamShape(s);
}

fn fitsRoutedRow(s: Shape) bool {
    return routedOk(s) and rowOk(s);
}

fn fitsRoutedLanes(s: Shape) bool {
    return routedOk(s) and lanesOk(s);
}

fn fitsRoutedMatrix(s: Shape) bool {
    return routedOk(s) and blockOk(s) and !s.fp16 and gemmFits(s);
}

fn fitsRoutedGemm(s: Shape) bool {
    return routedOk(s) and blockOk(s) and gemmFits(s);
}

fn fitsRoutedBlock(s: Shape) bool {
    return routedOk(s) and blockOk(s);
}

fn fitsGroup(s: Shape) bool {
    return streamShape(s);
}

fn fitsPair(s: Shape) bool {
    return s.pairs and s.n >= 2 and s.n % 2 == 0 and streamShape(s);
}

// ---- the prefill tiles

fn kpFits(comptime tier: usize) *const fn (Shape) bool {
    return struct {
        fn f(s: Shape) bool {
            return prefillOk(s) and gemmFits(s) and Kernels.tierTakes(tier, @intCast(s.m));
        }
    }.f;
}

fn fitsPrefillMatrix(s: Shape) bool {
    return prefillOk(s) and !s.fp16 and gemmFits(s);
}

fn fitsPrefillGemm(s: Shape) bool {
    return prefillOk(s) and gemmFits(s);
}

fn fitsPrefillBlock(s: Shape) bool {
    return prefillOk(s);
}

// ---- the launches

fn launchStream(k: *const Kernels, c: Call) Error!void {
    try k.streamGo(c.r, c.arg, .{}, c.arg.n, c.items, c.s);
}

fn launchPair(k: *const Kernels, c: Call) Error!void {
    try k.streamGo(c.r, c.arg, .{ .pair_cols = @divExact(c.arg.n, 2), .limit = c.limit }, @divExact(c.arg.n, 2), c.items, c.s);
}

fn launchGroup(k: *const Kernels, c: Call) Error!void {
    try k.groupGo(c.r, c.arg, c.sides, c.out_half, c.s);
}

fn launchRow(k: *const Kernels, c: Call) Error!void {
    try k.rowGo(c.r, c.arg, c.s, c.items);
}

fn launchLanes(k: *const Kernels, c: Call) Error!void {
    try k.lanesGo(c.r, c.arg, c.s, c.items);
}

fn launchWide(k: *const Kernels, c: Call) Error!void {
    try k.wideGo(c.r, c.arg, c.s);
}

fn launchTiled(k: *const Kernels, c: Call) Error!void {
    try k.tiledGo(c.r, c.arg, c.s);
}

fn launchReference(k: *const Kernels, c: Call) Error!void {
    try k.referenceGo(c.r, c.arg, c.s);
}

fn launchSplit(k: *const Kernels, c: Call) Error!void {
    try k.splitGo(c.r, c.arg, c.partial, c.parts, c.s);
}

fn launchBlock(comptime kind: affine.BlockKind) Launch {
    return struct {
        fn f(k: *const Kernels, c: Call) Error!void {
            try k.blockGo(c.r, c.arg, c.s, c.items, kind);
        }
    }.f;
}

fn launchKp(comptime tier: usize) Launch {
    return struct {
        fn f(k: *const Kernels, c: Call) Error!void {
            try k.kpGo(c.r, c.arg, c.s, c.items, tier);
        }
    }.f;
}

// ---- the entries

/// What an entry is besides its op, path and name.
const Spec = struct {
    family: registry.Family,
    fits: *const fn (Shape) bool,
    launch: Launch,
    caps: *const fn (Env) bool = &always,
    policy: *const fn (Env) bool = &always,
    rounds: registry.Rounds = .never,
    round_any: bool = false,
};

fn entry(comptime op: Op, comptime path: registry.Path, comptime name: []const u8, comptime spec: Spec) Entry {
    return .{
        .id = "mlx." ++ @tagName(op) ++ "." ++ @tagName(path) ++ "." ++ name,
        .format = .mlx,
        .op = op,
        .path = path,
        .family = spec.family,
        .caps = spec.caps,
        .policy = spec.policy,
        .fits = spec.fits,
        .rounds = spec.rounds,
        .round_any = spec.round_any,
        .launch = spec.launch,
    };
}

const project_decode = [_]Entry{
    entry(.project, .decode, "split", .{ .family = .split, .fits = &fitsSplit, .launch = &launchSplit, .caps = &rdna2 }),
    entry(.project, .decode, "stream", .{ .family = .stream, .fits = &fitsStream, .launch = &launchStream, .policy = &streamOn, .rounds = .always, .round_any = true }),
    entry(.project, .decode, "row", .{ .family = .row, .fits = &fitsRow, .launch = &launchRow, .rounds = .fp16 }),
    entry(.project, .decode, "lanes", .{ .family = .lanes, .fits = &fitsLanes, .launch = &launchLanes, .rounds = .fp16 }),
    entry(.project, .decode, "wide", .{ .family = .wide, .fits = &fitsWide, .launch = &launchWide, .caps = &rdna2 }),
    entry(.project, .decode, "matrix", .{ .family = .gemm, .fits = &fitsMatrix, .launch = launchBlock(.matrix), .caps = &matrixCores, .policy = &matrixOn }),
    entry(.project, .decode, "gemm", .{ .family = .gemm, .fits = &fitsGemm, .launch = launchBlock(.gemm), .policy = &gemmOn }),
    entry(.project, .decode, "block", .{ .family = .gemm, .fits = &fitsBlock, .launch = launchBlock(.block) }),
    entry(.project, .decode, "tiled", .{ .family = .tiled, .fits = &fitsTiled, .launch = &launchTiled, .caps = &rdna2 }),
    entry(.project, .decode, "reference", .{ .family = .reference, .fits = &fitsReference, .launch = &launchReference, .caps = &rdna2 }),
};

const routed_decode = [_]Entry{
    entry(.routed, .decode, "stream", .{ .family = .stream, .fits = &fitsRoutedStream, .launch = &launchStream, .policy = &streamOn }),
    entry(.routed, .decode, "row", .{ .family = .row, .fits = &fitsRoutedRow, .launch = &launchRow }),
    entry(.routed, .decode, "lanes", .{ .family = .lanes, .fits = &fitsRoutedLanes, .launch = &launchLanes }),
    entry(.routed, .decode, "matrix", .{ .family = .gemm, .fits = &fitsRoutedMatrix, .launch = launchBlock(.matrix), .caps = &matrixCores, .policy = &matrixOn }),
    entry(.routed, .decode, "gemm", .{ .family = .gemm, .fits = &fitsRoutedGemm, .launch = launchBlock(.gemm), .policy = &gemmOn }),
    entry(.routed, .decode, "block", .{ .family = .gemm, .fits = &fitsRoutedBlock, .launch = launchBlock(.block) }),
};

const other_decode = [_]Entry{
    entry(.group, .decode, "stream", .{ .family = .stream, .fits = &fitsGroup, .launch = &launchGroup, .policy = &streamOn, .round_any = true }),
    entry(.routed_act, .decode, "stream", .{ .family = .stream, .fits = &fitsPair, .launch = &launchPair, .policy = &streamOn }),
};

fn prefill(comptime op: Op) [7]Entry {
    return .{
        entry(op, .prefill, "kp0", .{ .family = .gemm, .fits = kpFits(0), .launch = launchKp(0), .policy = &gemmOn }),
        entry(op, .prefill, "kp1", .{ .family = .gemm, .fits = kpFits(1), .launch = launchKp(1), .policy = &gemmOn }),
        entry(op, .prefill, "kp2", .{ .family = .gemm, .fits = kpFits(2), .launch = launchKp(2), .policy = &gemmOn }),
        entry(op, .prefill, "kp3", .{ .family = .gemm, .fits = kpFits(3), .launch = launchKp(3), .policy = &gemmOn }),
        entry(op, .prefill, "matrix", .{ .family = .gemm, .fits = &fitsPrefillMatrix, .launch = launchBlock(.matrix), .caps = &matrixCores, .policy = &matrixOn }),
        entry(op, .prefill, "gemm", .{ .family = .gemm, .fits = &fitsPrefillGemm, .launch = launchBlock(.gemm), .policy = &gemmOn }),
        entry(op, .prefill, "block", .{ .family = .gemm, .fits = &fitsPrefillBlock, .launch = launchBlock(.block) }),
    };
}

pub const all = project_decode ++ routed_decode ++ other_decode ++ prefill(.project) ++ prefill(.routed);

// ---- the choices on the GPUs' tables

const Caps = @import("../caps.zig").Caps;

fn kernelsOn(gfx: []const u8) affine.Registry {
    const caps = Caps.of(gfx).?;
    return affine.Registry.init(if (caps.generation == .rdna2) &tuning.gfx1030 else &tuning.gfx1100, &all);
}

fn envOn(gfx: []const u8, stream_on: bool, gemm_on: bool, matrix_on: bool) Env {
    const caps = Caps.of(gfx).?;
    const matrix = caps.act == .bf16 and caps.matrix == .wmma11;
    return .{ .bf16 = caps.act == .bf16, .matrix = matrix, .matrix_on = matrix and matrix_on, .stream_on = stream_on, .gemm_on = gemm_on };
}

test "a lane round and a prompt each keep one family at every row count, on every GPU and policy" {
    for ([_][]const u8{ "gfx1030", "gfx1100", "gfx1151" }) |gfx| {
        const r = kernelsOn(gfx);
        for ([_]bool{ true, false }) |matrix_on| try r.verify(envOn(gfx, true, true, matrix_on), .mlx);
    }
}

test "the previous decode tiles take over where the stream tile stops" {
    const r = kernelsOn("gfx1030");
    var shape: Shape = .{ .m = 1, .n = 4096, .k = 4096, .bits = 4, .group = 64, .fp16 = true, .tables = .f16 };
    const env = envOn("gfx1030", true, true, false);
    // a call outside a lane round: the stream tile to 16 rows, then the wide tile, then the GEMM tile from 64
    const want = [_]struct { u32, []const u8 }{ .{ 1, "stream" }, .{ 16, "stream" }, .{ 17, "wide" }, .{ 63, "wide" }, .{ 64, "gemm" }, .{ 200, "gemm" } };
    for (want) |w| {
        shape.m = w[0];
        const e = r.select(env, .mlx, .project, .decode, shape).?;
        try std.testing.expect(std.mem.endsWith(u8, e.id, w[1]));
    }
    // a lane round keeps the stream tile
    shape.round = true;
    for ([_]u32{ 1, 16, 17, 63, 64, 200 }) |m| {
        shape.m = m;
        try std.testing.expectEqual(registry.Family.stream, r.select(env, .mlx, .project, .decode, shape).?.family);
    }
    // the same with the stream tile switched off: the reference tiles by rows
    const off = envOn("gfx1030", false, true, false);
    shape.m = 4;
    try std.testing.expectEqual(registry.Family.lanes, r.select(off, .mlx, .project, .decode, shape).?.family);
    shape.m = 1;
    try std.testing.expectEqual(registry.Family.row, r.select(off, .mlx, .project, .decode, shape).?.family);
}

test "the matrix cores take 16 rows and more of a call outside a lane round" {
    const r = kernelsOn("gfx1100");
    var shape: Shape = .{ .m = 15, .n = 4096, .k = 4096, .bits = 4, .group = 64, .fp16 = false };
    const on = envOn("gfx1100", true, true, true);
    try std.testing.expect(std.mem.endsWith(u8, r.select(on, .mlx, .project, .decode, shape).?.id, "stream"));
    shape.m = 16;
    try std.testing.expect(std.mem.endsWith(u8, r.select(on, .mlx, .project, .decode, shape).?.id, "matrix"));
    const off = envOn("gfx1100", true, true, false);
    try std.testing.expect(std.mem.endsWith(u8, r.select(off, .mlx, .project, .decode, shape).?.id, "stream"));
    shape.m = 17;
    try std.testing.expect(std.mem.endsWith(u8, r.select(off, .mlx, .project, .decode, shape).?.id, "gemm"));
    // words the GEMM tile cannot load wide keep the reference tile
    shape.words_aligned = false;
    try std.testing.expect(std.mem.endsWith(u8, r.select(off, .mlx, .project, .decode, shape).?.id, "block"));
}

test "a prompt takes the K-parallel tiles by rows and by items, then the 128-row tile" {
    const r = kernelsOn("gfx1100");
    const env = envOn("gfx1100", true, true, true);
    var shape: Shape = .{ .m = 1, .n = 4096, .k = 4096, .bits = 4, .group = 64, .fp16 = false };
    const plain = [_]struct { u32, []const u8 }{ .{ 1, "kp0" }, .{ 2, "kp0" }, .{ 3, "kp1" }, .{ 4, "kp1" }, .{ 5, "kp2" }, .{ 32, "kp2" }, .{ 33, "matrix" }, .{ 2048, "matrix" } };
    for (plain) |p| {
        shape.m = p[0];
        try std.testing.expect(std.mem.endsWith(u8, r.select(env, .mlx, .project, .prefill, shape).?.id, p[1]));
    }
    shape.items = 100;
    shape.m = 30;
    try std.testing.expect(std.mem.endsWith(u8, r.select(env, .mlx, .routed, .prefill, shape).?.id, "kp3"));
    shape.items = 301;
    try std.testing.expect(std.mem.endsWith(u8, r.select(env, .mlx, .routed, .prefill, shape).?.id, "matrix"));
    // without the matrix cores the dot2 GEMM tile
    shape.items = 0;
    shape.m = 64;
    try std.testing.expect(std.mem.endsWith(u8, r.select(envOn("gfx1100", true, true, false), .mlx, .project, .prefill, shape).?.id, "gemm"));
    // RDNA2 takes a plan of up to 410 items on the second K-parallel tile
    const v = kernelsOn("gfx1030");
    shape.fp16 = true;
    shape.items = 350;
    shape.m = 10;
    try std.testing.expect(std.mem.endsWith(u8, v.select(envOn("gfx1030", true, true, false), .mlx, .routed, .prefill, shape).?.id, "kp1"));
}
