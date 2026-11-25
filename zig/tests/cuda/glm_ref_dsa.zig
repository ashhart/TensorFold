//! Full GLM-5.3's bf16 projections at TP=3 shares on glm_mm vs float64: prompt and decode rows apart, worst columns.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const ref = @import("glm_ref_tests.zig");
const glm = @import("glm_cuda");
const Gpu = check.Gpu;
const Rig = qr.RigT;

/// A projection class at rank 0's and rank 2's shares (22 and 21 heads; 704 and 640 shared rows; vocabulary in 128s).
const Class = struct { name: []const u8, n: [2]usize, k: [2]usize, wide: bool = false, prompt: usize = 160, decode: []const usize = &.{ 1, 4, 17, 64 } };

const classes = [_]Class{
    .{ .name = "q_a and kv_a (one stack)", .n = .{ 2048 + 576, 2048 + 576 }, .k = .{ 6144, 6144 } },
    .{ .name = "q_b", .n = .{ 22 * 256, 21 * 256 }, .k = .{ 2048, 2048 } },
    .{ .name = "o", .n = .{ 6144, 6144 }, .k = .{ 22 * 256, 21 * 256 }, .wide = true },
    .{ .name = "indexer keys and weights", .n = .{ 128 + 32, 128 + 32 }, .k = .{ 6144, 6144 } },
    .{ .name = "indexer q_b", .n = .{ 32 * 128, 32 * 128 }, .k = .{ 2048, 2048 } },
    .{ .name = "dense gate and up", .n = .{ 2 * 4096, 2 * 4096 }, .k = .{ 6144, 6144 } },
    .{ .name = "dense down", .n = .{ 6144, 6144 }, .k = .{ 4096, 4096 }, .wide = true },
    .{ .name = "shared expert gate and up", .n = .{ 2 * 704, 2 * 640 }, .k = .{ 6144, 6144 } },
    .{ .name = "shared expert down", .n = .{ 6144, 6144 }, .k = .{ 704, 640 }, .wide = true },
    .{ .name = "MTP eh", .n = .{ 6144, 6144 }, .k = .{ 2 * 6144, 2 * 6144 } },
    .{ .name = "head", .n = .{ 404 * 128, 403 * 128 }, .k = .{ 6144, 6144 }, .prompt = 64, .decode = &.{ 1, 4 } },
};

/// One class at one share: a prompt call's rows and each decode window's (its first rows) against the same float64.
fn class(r: *Rig, mk: *const glm.mm.Kernels, c: Class, share: usize) !void {
    const n = c.n[share];
    const k = c.k[share];
    const p = c.prompt;
    const x = try r.bfs(p * k, 1.0);
    const w = try r.bfs(n * k, 0.05);
    const want = try ref.dots(r, x, w, p, n, k, c.wide);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const tiled = try r.zeros(n * k * 2);
    try glm.mm.tile(mk, r.s, try r.dev(std.mem.sliceAsBytes(w)), tiled, n, k);
    const tw: glm.mm.Tiled = .{ .w = tiled, .n = n, .k = k };
    const part = try r.zeros(glm.mm.partBytes(64, tw));
    const eb: usize = if (c.wide) 4 else 2;
    const rank: usize = if (share == 0) 0 else 2;
    var name: [160]u8 = undefined;
    const out = try r.zeros(p * n * eb);
    try glm.mm.matmul(mk, r.s, dx, k, tw, out, part, p, c.wide, true);
    try ref.closeCols(r, try std.fmt.bufPrint(&name, "{s}, rank {d} [{d}, {d}]: a {d}-row prompt call", .{ c.name, rank, n, k, p }), out, c.wide, want.want, want.tol, n);
    for (c.decode) |dm| {
        const o = try r.zeros(dm * n * eb);
        try glm.mm.matmul(mk, r.s, dx, k, tw, o, part, dm, c.wide, false);
        try ref.closeCols(r, try std.fmt.bufPrint(&name, "{s}, rank {d} [{d}, {d}]: a {d}-row window ({d} K slices)", .{ c.name, rank, n, k, dm, glm.mm.splitK(n, k) }), o, c.wide, want.want[0 .. dm * n], want.tol[0 .. dm * n], n);
    }
}

/// `glm-ref-dsa`: the classes above at both shares, then the router at full GLM-5.3's hidden size; synthetic, seeded.
pub fn run(gpu: Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(593);
    var r: Rig = .{ .gpu = gpu, .a = undefined, .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var mm_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_mm);
    defer mm_mod.unload();
    const mk = try glm.mm.Kernels.load(mm_mod);
    for (classes) |c| for (0..2) |share| {
        var arena: std.heap.ArenaAllocator = .init(gpu.gpa); // a class's float64 rows go before the next class's
        defer arena.deinit();
        r.a = arena.allocator();
        try class(&r, &mk, c, share);
        try s.synchronize();
        for (r.bufs.items) |*b| b.free();
        r.bufs.clearRetainingCapacity();
    };
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    r.a = arena.allocator();
    var glue_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer glue_mod.unload();
    const ks = try glm.glue.Kernels.load(glue_mod);
    try ref.routers(&r, .{ .glue = &ks }, 6144);
    try @import("glm_own_residual.zig").run(&r, .{ .glue = &ks });
    check.pass("glm-ref-dsa: {d} projection classes at both TP=3 shares, prompt and decode rows, and the router at 6144, within {e:.3} of float64 tolerance; the own-partial residual byte for byte", .{ classes.len, r.worst });
}
