//! Checkpoint prompt tiles against the decode MMA chains and the glue after them, partial row and column tiles too.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Rig = @import("quant_ref_tests.zig").RigT;

fn bytes(r: *Rig, n: usize, scales: bool) !u64 {
    const host = try r.a.alloc(u8, n);
    for (host) |*v| v.* = if (scales) 0x20 + r.rng.uintLessThan(u8, 16) else r.rng.int(u8);
    return r.dev(host);
}

/// Pair p to expert ids[p] of `count`, each expert's pairs in order, in items of `tile` (the plan kernels' layout).
fn plan(r: *Rig, ids: []const i32, count: usize, tile: usize) !struct { p: cuda.grouped.Plan, items: usize } {
    const members = try r.a.alloc(i32, ids.len);
    var items: std.ArrayList(i32) = .empty;
    var first: usize = 0;
    for (0..count) |e| {
        const begin = first;
        for (ids, 0..) |id, pair| if (id == e) {
            members[first] = @intCast(pair);
            first += 1;
        };
        var at = begin;
        while (at < first) : (at += tile) try items.appendSlice(r.a, &.{ @intCast(e), @intCast(at), @intCast(@min(tile, first - at)) });
    }
    const n: i32 = @intCast(items.items.len / 3);
    return .{ .p = .{ .items = try r.dev(std.mem.sliceAsBytes(items.items)), .members = try r.dev(std.mem.sliceAsBytes(members)), .counts = try r.dev(std.mem.asBytes(&n)), .rank = 0, .hist = 0 }, .items = @intCast(n) };
}

fn floats(r: *Rig, n: usize, lo: f32, hi: f32) !u64 {
    const host = try r.a.alloc(f32, n);
    for (host) |*v| v.* = lo + r.rng.float(f32) * (hi - lo);
    return r.dev(std.mem.sliceAsBytes(host));
}

fn launch(r: *Rig, f: cuda.Function, grid: [2]usize, block: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]) }, .block = .{ .x = block } }, r.s, a);
}

/// `m` rows through `count` experts (k inputs, n outputs): prompt tiles against decode warps (up then quant4, down).
fn experts(r: *Rig, q: *const cuda.ck.Ck, m: usize, k: usize, n: usize, count: usize) !void {
    const pairs = m * count;
    const al = try r.a.alloc(f32, count);
    for (al, 0..) |*v, e| v.* = 0.125 * @as(f32, @floatFromInt(1 + e % 3));
    const alpha = try r.dev(std.mem.sliceAsBytes(al));
    const l: cuda.ck.Layer = .{
        .up = try bytes(r, count * n * k / 2, false),
        .up_s = try bytes(r, count * n * k / 16, true),
        .down = try bytes(r, count * n * k / 2, false),
        .down_s = try bytes(r, count * n * k / 16, true),
        .up_alpha = alpha,
        .down_alpha = alpha,
        .up_act = 1,
        .down_act = 0.375,
        .experts = count,
        .width = n,
        .dims = k,
    };
    const x = try bytes(r, cuda.ck.rowsBytes(m, k), true);
    const act = try bytes(r, cuda.ck.rowsBytes(pairs, n), true);
    const ids = try r.a.alloc(i32, pairs);
    for (ids, 0..) |*id, p| id.* = @intCast(p % count);
    const narrow = try plan(r, ids, count, 16);
    const wide = try plan(r, ids, count, cuda.ck.prompt_tile);
    const up0 = try r.zeros(pairs * n * 2);
    const rows0 = try r.zeros(cuda.ck.rowsBytes(pairs, n));
    const rows1 = try r.zeros(cuda.ck.rowsBytes(pairs, n));
    const down0 = try r.zeros(pairs * k * 4);
    const down1 = try r.zeros(pairs * k * 4);
    try q.up(r.s, l, x, m, count, narrow.p, up0, narrow.items);
    try q.quant4(r.s, up0, n, pairs, n, l.down_act, rows0);
    try q.promptUp(r.s, l, x, m, count, wide.p, pairs, rows1, wide.items);
    try q.down(r.s, l, act, pairs, m, narrow.p, down0, narrow.items);
    try q.promptDown(r.s, l, act, pairs, wide.p, down1, wide.items);
    try check.sameBytes("checkpoint prompt up rows", try r.back(rows1, cuda.ck.rowsBytes(pairs, n)), try r.back(rows0, cuda.ck.rowsBytes(pairs, n)));
    try check.sameBytes("checkpoint prompt down", try r.back(down1, pairs * k * 4), try r.back(down0, pairs * k * 4));
    check.pass("checkpoint experts: {d} pairs over {d} experts [{d}, {d}], prompt NVFP4 up rows == decode up then quant4, prompt down == decode down", .{ pairs, count, n, k });
}

/// Prompt down summing shares (pair g * shares + s, expert s * count + e), combined and added, against decode's fp32.
fn shareSums(r: *Rig, q: *const cuda.ck.Ck, latent: cuda.Module, rows: usize, picks: usize, count: usize, shares: usize, k: usize, n: usize) !void {
    const groups = rows * picks;
    const pairs = groups * shares;
    const ids = try r.a.alloc(i32, pairs);
    for (0..groups) |g| for (0..shares) |s| {
        ids[g * shares + s] = @intCast(s * count + g * 7 % count);
    };
    const al = try r.a.alloc(f32, shares * count);
    for (al, 0..) |*v, e| v.* = 0.125 * @as(f32, @floatFromInt(1 + e % 5));
    const l: cuda.ck.Layer = .{
        .up = 0,
        .up_s = 0,
        .down = try bytes(r, shares * count * n * k / 2, false),
        .down_s = try bytes(r, shares * count * n * k / 16, true),
        .up_alpha = 0,
        .down_alpha = try r.dev(std.mem.sliceAsBytes(al)),
        .up_act = 1,
        .down_act = 1,
        .experts = shares * count,
        .width = k,
        .dims = n,
    };
    const act = try bytes(r, cuda.ck.rowsBytes(pairs, k), true);
    const narrow = try plan(r, ids, shares * count, 16);
    const wide = try plan(r, ids, shares * count, cuda.ck.prompt_tile);
    const parts = try r.zeros(pairs * n * 4);
    const sums = try r.zeros(groups * n * 2);
    try q.down(r.s, l, act, pairs, rows, narrow.p, parts, narrow.items);
    try q.promptSum(r.s, l, act, pairs, wide.p, sums, wide.items, shares);
    const wts = try floats(r, groups, 0, 1);
    const out0 = try r.zeros(rows * n * 2);
    const out1 = try r.zeros(rows * n * 2);
    var a: cuda.Args = .{};
    a.add(parts);
    for ([_]usize{ n, shares * n }) |v| a.add(@as(i64, @intCast(v)));
    a.add(@as(c_int, @intCast(shares)));
    a.add(wts);
    a.add(@as(c_int, @intCast(picks)));
    a.add(out0);
    a.add(@as(c_int, @intCast(n)));
    try launch(r, try latent.function("tf_super_combine"), .{ rows, (n + 255) / 256 }, 256, &a);
    var b: cuda.Args = .{};
    for ([_]u64{ sums, wts }) |v| b.add(v);
    b.add(@as(c_int, @intCast(picks)));
    b.add(out1);
    b.add(@as(c_int, @intCast(n)));
    try launch(r, try latent.function("tf_super_combine_sums"), .{ rows, (n + 1023) / 1024 }, 128, &b);
    try check.sameBytes("checkpoint share sums through the combine", try r.back(out1, rows * n * 2), try r.back(out0, rows * n * 2));
    // the same sums as the shared expert's: each group a row of the shared add
    const y = try r.a.alloc(u8, groups * n * 2);
    for (std.mem.bytesAsSlice(u16, y)) |*v| v.* = 0x3f00 + r.rng.uintLessThan(u16, 0x0200) | @as(u16, r.rng.int(u1)) << 15;
    const y0 = try r.dev(y);
    const y1 = try r.dev(y);
    var c: cuda.Args = .{};
    for ([_]u64{ y0, parts }) |v| c.add(v);
    for ([_]usize{ n, shares * n }) |v| c.add(@as(i64, @intCast(v)));
    c.add(@as(c_int, @intCast(shares)));
    c.add(@as(c_int, @intCast(n)));
    try launch(r, try latent.function("tf_super_moe_out"), .{ groups, (n + 255) / 256 }, 256, &c);
    var d: cuda.Args = .{};
    for ([_]u64{ y1, sums }) |v| d.add(v);
    d.add(@as(c_int, @intCast(n)));
    try launch(r, try latent.function("tf_super_moe_out_sums"), .{ groups, (n + 2047) / 2048 }, 256, &d);
    try check.sameBytes("checkpoint share sums through the shared add", try r.back(y1, groups * n * 2), try r.back(y0, groups * n * 2));
    check.pass("checkpoint share sums: {d} picks over {d} experts in {d} shares [{d}, {d}], combine and shared add byte-equal to decode's fp32 shares", .{ groups, count, shares, n, k });
}

fn lane(r: *Rig, q: *const cuda.ck.Ck, n: usize, k: usize) !void {
    const m = 129;
    const host = try r.a.alloc(u8, m * k);
    for (host) |*v| v.* = 0x20 + r.rng.uintLessThan(u8, 32);
    const x = try r.dev(host);
    const w: cuda.qmmf.Weight = .{ .mode = .fp8, .codes = try bytes(r, ((n + 63) / 64) * 64 * k, true), .scales = 0, .scale = 0.125, .n = @intCast(n), .k = @intCast(k), .npad = @intCast((n + 63) / 64 * 64) };
    const sk = cuda.qmmf.splitK(n, k);
    const part = try r.zeros(sk * m * n * 4);
    const wide = try r.zeros(m * n * 2);
    const narrow = try r.zeros(m * n * 2);
    try q.lane8(r.s, x, m, w, 0.5, wide, part);
    for (0..m) |row| try q.lane8(r.s, x + row * k, 1, w, 0.5, narrow + row * n * 2, part);
    try check.sameBytes("checkpoint FP8 prompt rows", try r.back(wide, m * n * 2), try r.back(narrow, m * n * 2));
    check.pass("checkpoint FP8: {d} rows x [{d}, {d}], {d} K slices, prompt byte-equal to rows alone", .{ m, n, k, sk });
}

pub fn run(gpu: check.Gpu) !void {
    const major: u32 = @intCast(try gpu.ctx.attribute(.compute_capability_major));
    if (major < 12) {
        std.debug.print("SKIP checkpoint prompt tiles: need Blackwell FP4 MMA\n", .{});
        return;
    }
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x5eed_f4f8);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var q = try cuda.ck.Ck.load(gpu.d, major, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    defer q.unload();
    try experts(&r, &q, 129, 256, 192, 2); // partial row and column tiles
    try experts(&r, &q, 300, 1024, 1344, 96); // Super's routed widths, more units than resident blocks
    var latent = try cuda.Module.load(gpu.d, cuda.kernels.nemotron_latent);
    defer latent.unload();
    try shareSums(&r, &q, latent, 64, 22, 8, 2, 1344, 1024); // Super's routed shares, two items an expert
    try shareSums(&r, &q, latent, 300, 1, 1, 2, 2688, 4096); // its shared expert: one expert, a pick a row
    try lane(&r, &q, 192, 256);
    try lane(&r, &q, 192, 2048);
    try lane(&r, &q, 4096, 8192);
}
