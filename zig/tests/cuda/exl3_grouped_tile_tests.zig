//! Grouped EXL3: maxm 16 and the old maxm 64 are bit-equal; maxm 16 on a tile-64 plan drops members.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const f64t = @import("exl3_experts_f64_tests.zig");

const Rig = quant_ref_tests.RigT;
const E = f64t.E;
const D = f64t.D;
const I = f64t.I;
const SLOTS = f64t.SLOTS;

const Plan = struct { p: cuda.grouped.Plan, items: usize, pairs: usize };
const Out = struct { dev: u64, bytes: []u8 };

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(61);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try cuda.grouped.Router.resolve(mod);
    var exp = try cuda.exl3_experts.Experts.load(gpu.d);
    defer exp.unload();
    const b = try f64t.buildLayer(&r, &.{ 3, 4, 5, 6, 3, 4, 5, 6, 3 });
    const l: cuda.experts.Layer = .{ .format = .exl3, .exl3 = b.layer, .up = 0, .down = 0, .width = I, .dims = D, .experts = E };
    var g: cuda.exl3_grouped.Grouped = .{ .d = gpu.d, .kernels = &exp };
    defer g.deinit();
    const grp: cuda.experts.Grouped = .{ .exl3 = &g };

    for ([_]usize{ 1, 4, 16 }) |rows| {
        const pairs = rows * SLOTS;
        const p16 = try plan(&r, router, b.dpick, pairs, 16);
        const p64 = try plan(&r, router, b.dpick, pairs, 64);
        const up16 = try upBytes(&r, grp, l, b.dx, p16, 16);
        const upOld = try upBytes(&r, grp, l, b.dx, p16, 64);
        try check.sameBytes("up: maxm 16 vs the old maxm 64 on a tile-16 plan", upOld.bytes, up16.bytes);
        const upT64 = try upBytes(&r, grp, l, b.dx, p64, 64);
        try check.sameBytes("up: a tile-64 plan through decode()", upT64.bytes, up16.bytes);
        const down16 = try downBytes(&r, grp, l, up16.dev, p16, 16);
        const downOld = try downBytes(&r, grp, l, up16.dev, p16, 64);
        try check.sameBytes("down: maxm 16 vs the old maxm 64 on a tile-16 plan", downOld, down16);
        const downT64 = try downBytes(&r, grp, l, upT64.dev, p64, 64);
        try check.sameBytes("down: a tile-64 plan through decode()", downT64, down16);
        check.pass("exl3-grouped-tile {d} rows: maxm 16 and the old 64 bit-equal, tile-64 plan bit-equal", .{rows});
    }

    // 16 rows' 128 pairs all pick expert 0: two 64-member items, so the maxm-16 mutant drops members 16..63.
    const skew_host = try r.a.alloc(i32, 16 * SLOTS);
    @memset(skew_host, 0);
    const skew = try r.dev(std.mem.sliceAsBytes(skew_host));
    const mp = try plan(&r, router, skew, 16 * SLOTS, 64);
    const poison_x = try r.bfs(16 * SLOTS * D, 0.5);
    const poison_x_dev = try r.dev(std.mem.sliceAsBytes(poison_x));
    var gm: cuda.exl3_grouped.Grouped = .{ .d = gpu.d, .kernels = &exp };
    defer gm.deinit();
    const gmgrp: cuda.experts.Grouped = .{ .exl3 = &gm };
    const poisoned = try upBytes(&r, gmgrp, l, poison_x_dev, mp, 64);
    const mutant = try upBytes(&r, gmgrp, l, b.dx, mp, 16);
    var gc: cuda.exl3_grouped.Grouped = .{ .d = gpu.d, .kernels = &exp };
    defer gc.deinit();
    const real = try upBytes(&r, .{ .exl3 = &gc }, l, b.dx, mp, 64);
    const row: usize = 16 * I * 2;
    try check.sameBytes("planted mutant keeps the poisoned members 16..31", mutant.bytes[row..][0..row], poisoned.bytes[row..][0..row]);
    try check.expect(!std.mem.eql(u8, mutant.bytes, real.bytes), "planted maxm 16 on a tile-64 plan dropped members and must differ from maxm 64", .{});
    check.pass("exl3-grouped-tile planted mutant (maxm 16 on a tile-64 plan) drops members 16..63", .{});
}

fn plan(r: *Rig, router: cuda.grouped.Router, pick: u64, pairs: usize, tile: usize) !Plan {
    const items = cuda.grouped.maxItems(pairs, E, tile);
    const p: cuda.grouped.Plan = .{
        .members = try r.zeros(pairs * 4 + 16),
        .items = try r.zeros(items * 3 * 4 + 16),
        .counts = try r.zeros(16),
        .rank = try r.zeros(pairs * 4 + 16),
        .hist = try r.zeros((E + 1) * 4 * ((pairs + 1023) / 1024 + 1)),
    };
    try router.route(r.s, pick, pairs, E, tile, p);
    return .{ .p = p, .items = items, .pairs = pairs };
}

fn upBytes(r: *Rig, grp: cuda.experts.Grouped, l: cuda.experts.Layer, x: u64, p: Plan, tile: usize) !Out {
    const out = try r.zeros(p.pairs * I * 2);
    try grp.decode(r.s, .up, .{ .x = x, .stride = D, .slots = SLOTS }, l, p.p, out, p.items, p.pairs, tile, -1);
    return .{ .dev = out, .bytes = try r.back(out, p.pairs * I * 2) };
}

fn downBytes(r: *Rig, grp: cuda.experts.Grouped, l: cuda.experts.Layer, act: u64, p: Plan, tile: usize) ![]u8 {
    const out = try r.zeros(p.pairs * D * 4);
    try grp.decode(r.s, .down_f32, .{ .x = act, .stride = I }, l, p.p, out, p.items, p.pairs, tile, -1);
    return r.back(out, p.pairs * D * 4);
}
