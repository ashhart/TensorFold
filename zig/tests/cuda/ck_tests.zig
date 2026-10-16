//! Checkpoint prompt tiles against the decode MMA chains, including partial row and column tiles.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Rig = @import("quant_ref_tests.zig").RigT;

fn bytes(r: *Rig, n: usize, scales: bool) !u64 {
    const host = try r.a.alloc(u8, n);
    for (host) |*v| v.* = if (scales) 0x20 + r.rng.uintLessThan(u8, 16) else r.rng.int(u8);
    return r.dev(host);
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
    try lane(&r, &q, 192, 256);
    try lane(&r, &q, 192, 2048);
    try lane(&r, &q, 4096, 8192);
}
