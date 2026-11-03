//! GLM's EXL3 decode on the 16-pair item plan gives the plain decode's slot-row bytes, case by case.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const Gpu = check.Gpu;
const Rig = qr.RigT;
const ex = cuda.exl3_experts;

const E = 12; // routed experts (the shared one is id E, skipped)
const D = 512; // dims
const top = glm.exl3.top_k;
const slots = top + 1;
const limit: f32 = 7.0;

fn signs(r: *Rig, n: usize, scale: f16) ![]f16 {
    const out = try r.a.alloc(f16, n);
    for (out) |*v| v.* = if (r.rng.boolean()) scale else -scale;
    return out;
}

/// `E` K x N trellises back to back as the loader lays them out: the pointer table, with suh and svh.
fn matrix(r: *Rig, k: usize, n: usize, shared_suh: ?[]const f16, suh: *u64, svh: *u64) !u64 {
    const words = k / 16 * (n / 16) * 64; // i16 an expert
    const tr = try r.a.alloc(i16, E * words);
    for (tr) |*v| v.* = r.rng.int(i16);
    const su = try r.a.alloc(f16, E * k);
    for (0..E) |e| @memcpy(su[e * k ..][0..k], if (shared_suh) |s| s else try signs(r, k, 1.0));
    suh.* = try r.dev(std.mem.sliceAsBytes(su));
    svh.* = try r.dev(std.mem.sliceAsBytes(try signs(r, E * n, 0.02)));
    const base = try r.dev(std.mem.sliceAsBytes(tr));
    const ptrs = try r.a.alloc(u64, E);
    for (ptrs, 0..) |*ptr, e| ptr.* = base + e * words * 2;
    return r.dev(std.mem.sliceAsBytes(ptrs));
}

/// A layer of width `n`; `one`: every gate and up input shares one rotation (the loader's shx).
fn weights(r: *Rig, n: usize, one: bool) !glm.exl3.Experts {
    var w: glm.exl3.Experts = .{ .gt = 0, .ut = 0, .dt = 0, .k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .shx = one };
    const k2: [E]i32 = @splat(8);
    w.k2 = try r.dev(std.mem.sliceAsBytes(&k2));
    const rot = if (one) try signs(r, D, 1.0) else null;
    w.gt = try matrix(r, D, n, rot, &w.suh_g, &w.svh_g);
    w.ut = try matrix(r, D, n, rot, &w.suh_u, &w.svh_u);
    w.dt = try matrix(r, n, D, null, &w.suh_d, &w.svh_d);
    return w;
}

/// The common module's layer GLM's cuda_exl3.layer builds from `w` (the stage checks call the module directly).
fn commonLayer(w: glm.exl3.Experts, n: usize) ex.Layer {
    return .{ .gate = w.gt, .up = w.ut, .down = w.dt, .gate_k2 = w.k2, .up_k2 = w.k2, .down_k2 = w.k2, .suh_g = w.suh_g, .suh_u = w.suh_u, .svh_g = w.svh_g, .svh_u = w.svh_u, .suh_d = w.suh_d, .svh_d = w.svh_d, .experts = E, .dims = D, .width = n, .codebook = .mcg, .limit = limit, .act_mode = 0, .uniform_k2 = 8, .shared_input = w.shx };
}

/// Each row's 8 distinct routed picks (lockstep: every row the same picks), then the shared expert's id.
fn picks(r: *Rig, rows: usize, lockstep: bool) ![]i32 {
    const out = try r.a.alloc(i32, rows * slots);
    var first: [top]i32 = undefined; // row 0's routed picks, reused every row under lockstep
    for (0..rows) |row| {
        if (row == 0 or !lockstep) {
            var seen: [E]bool = @splat(false);
            for (0..top) |j| {
                var e = r.rng.uintLessThan(usize, E);
                while (seen[e]) e = r.rng.uintLessThan(usize, E);
                seen[e] = true;
                first[j] = @intCast(e);
            }
        }
        for (0..top) |j| out[row * slots + j] = first[j];
        out[row * slots + slots - 1] = E;
    }
    return out;
}

/// The routed slot rows of `ey` [rows][slots][D] packed as [rows][top][D] (the plain path's y layout).
fn packSlots(r: *Rig, ey: u64, rows: usize) ![]u8 {
    const raw = try r.back(ey, rows * slots * D * 4);
    const out = try r.a.alloc(u8, rows * top * D * 4);
    for (0..rows) |row| @memcpy(out[row * top * D * 4 ..][0 .. top * D * 4], raw[row * slots * D * 4 ..][0 .. top * D * 4]);
    return out;
}

/// The plain decode, as the old launch shape: the 8 compacted picks through the common module's runSlots.
fn plain(r: *Rig, k: *const glm.exl3.Kernels, l: ex.Layer, x: []const u16, pk: []const i32, rows: usize) ![]u8 {
    const p8 = try r.a.alloc(i32, rows * top);
    for (0..rows) |row| @memcpy(p8[row * top ..][0..top], pk[row * slots ..][0..top]);
    const sc = try ex.scratch(try r.zeros(try ex.scratchBytes(l, rows, top)), l, rows, top);
    try k.runSlots(r.s, l, try r.dev(std.mem.sliceAsBytes(x)), D, .bf16, try r.dev(std.mem.sliceAsBytes(p8)), rows, sc);
    try r.s.synchronize();
    return r.back(sc.y, rows * top * D * 4);
}

/// The family's decode dispatch, whose decode windows run on the item plan: ey [rows][slots][D] fp32 bytes.
fn items(r: *Rig, k: *const glm.exl3.Kernels, ek: *const glm.experts.Kernels, w: glm.exl3.Experts, n: usize, x: []const u16, pk: []const i32, rows: usize) ![]u8 {
    const P = rows * slots; // the item plan groups all 9 slots
    const plan: cuda.grouped.Plan = .{
        .members = try r.zeros(P * 4),
        .items = try r.zeros(cuda.grouped.maxItems(P, E + 1, glm.experts.tile) * 12),
        .counts = try r.zeros(8),
        .rank = try r.zeros(P * 4),
        .hist = try r.zeros(4),
    };
    const sz = glm.exl3.scratchBytes(rows, D, n, false, w.shx);
    const sc: glm.exl3.Scratch = .{ .picks = try r.zeros(sz[0]), .xg = try r.zeros(sz[1]), .xu = try r.zeros(sz[2]), .z = try r.zeros(sz[3]) };
    const ey = try r.zeros(rows * slots * D * 4);
    const xd = try r.zeros(P * n * 2);
    try glm.exl3.run(k, ek, r.s, w, sc, plan, try r.dev(std.mem.sliceAsBytes(x)), D, try r.dev(std.mem.sliceAsBytes(pk)), slots, rows, E, D, n, limit, false, ey, xd);
    return packSlots(r, ey, rows);
}

const Tally = struct { cases: usize = 0, failed: usize = 0 };

fn value(b: []const u8) f64 {
    if (b.len == 4) return @as(f32, @bitCast(std.mem.readInt(u32, b[0..4], .little)));
    return @as(f16, @bitCast(std.mem.readInt(u16, b[0..2], .little)));
}

fn bitsOf(b: []const u8) u32 {
    return if (b.len == 4) std.mem.readInt(u32, b[0..4], .little) else std.mem.readInt(u16, b[0..2], .little);
}

/// Byte-equal, else one FAIL line: the count, the first (row of `per_row`, column), got/want, canaries, NaNs, most.
fn same(t: *Tally, name: []const u8, got: []const u8, want: []const u8, elem: usize, per_row: usize) void {
    t.cases += 1;
    if (got.len == want.len and std.mem.eql(u8, got, want)) return;
    t.failed += 1;
    if (got.len != want.len) {
        std.debug.print("FAIL {s}: {d} bytes, expected {d}\n", .{ name, got.len, want.len });
        return;
    }
    var first: usize = 0;
    var differ: usize = 0;
    var canary: usize = 0;
    var nan: usize = 0;
    var most: f64 = 0;
    var scale: f64 = 0;
    for (0..got.len / elem) |i| {
        const g = got[i * elem ..][0..elem];
        const w = want[i * elem ..][0..elem];
        if (std.mem.allEqual(u8, g, 0xff)) canary += 1;
        const gv = value(g);
        const wv = value(w);
        if (std.math.isNan(gv)) nan += 1;
        if (std.math.isFinite(wv)) scale = @max(scale, @abs(wv));
        if (std.mem.eql(u8, g, w)) continue;
        if (differ == 0) first = i;
        differ += 1;
        if (std.math.isFinite(gv) and std.math.isFinite(wv)) most = @max(most, @abs(gv - wv));
    }
    const g = got[first * elem ..][0..elem];
    const w = want[first * elem ..][0..elem];
    std.debug.print("FAIL {s}: {d} of {d} values differ, first at row {d} column {d}: got {e} (0x{x}), want {e} (0x{x}); {d} canary, {d} NaN; largest finite difference {e} of largest |want| {e}\n", .{ name, differ, got.len / elem, first / per_row, first % per_row, value(g), bitsOf(g), value(w), bitsOf(w), canary, nan, most, scale });
}

pub fn run_(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(29);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = stream, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }

    var pm = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer pm.unload();
    var k = try glm.exl3.Kernels.load(gpu.d);
    defer k.unload();
    var ek = try glm.experts.Kernels.load(gpu.d, pm, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    defer ek.unload();
    var t: Tally = .{};
    for ([_]usize{ 256, 640 }) |n| { // 640: a TP=3 rank's 5 blocks, 10 down steps
        const cases = [_]struct { rows: usize, lock: bool }{
            .{ .rows = 1, .lock = false }, .{ .rows = 2, .lock = false }, .{ .rows = 3, .lock = false },
            .{ .rows = 8, .lock = false }, .{ .rows = 17, .lock = false }, // distinct windows; 17 rows: two tiles
            .{ .rows = 16, .lock = true }, .{ .rows = 32, .lock = true }, // lockstep: every row the same picks,
            .{ .rows = 48, .lock = true }, .{ .rows = 64, .lock = true }, // so an expert spans many 16-pair items
        };
        for (cases) |cs| { // distinct and lockstep decode windows
            const rows = cs.rows;
            const x = try r.bfs(rows * D, 1.0);
            const pk = try picks(&r, rows, cs.lock);
            const case = try std.fmt.allocPrint(r.a, "width {d}, {d} rows {s}", .{ n, rows, if (cs.lock) "lockstep" else "distinct" });
            const w = try weights(&r, n, false);
            const l = commonLayer(w, n);
            const want = try plain(&r, &k, l, x, pk, rows);
            same(&t, try std.fmt.allocPrint(r.a, "glm-exl3-items {s}: plain twice (control)", .{case}), try plain(&r, &k, l, x, pk, rows), want, 4, D);
            var nan: usize = 0;
            for (0..rows) |row| for (0..top) |j| for (0..D) |c| {
                nan += @intFromBool(std.math.isNan(value(want[((row * top + j) * D + c) * 4 ..][0..4])));
            };
            t.cases += 1;
            if (nan != 0) {
                t.failed += 1;
                std.debug.print("FAIL glm-exl3-items {s}: the plain decode left {d} routed values NaN\n", .{ case, nan });
            }
            // A wrong plan (a members offset, a bad first/count, the shared item unskipped) misroutes a slot.
            same(&t, try std.fmt.allocPrint(r.a, "glm-exl3-items {s}, items: ey", .{case}), try items(&r, &k, &ek, w, n, x, pk, rows), want, 4, D);
        }
    }
    std.debug.print("RESULT glm-exl3-items: {d} of {d} comparisons differ\n", .{ t.failed, t.cases });
    if (t.failed != 0) {
        std.debug.print("FAIL glm-exl3-items: {d} of {d} comparisons differ\n", .{ t.failed, t.cases });
        return error.TestFailed;
    }
    check.pass("glm-exl3-items: {d} comparisons equal the plain decode byte for byte", .{t.cases});
}
