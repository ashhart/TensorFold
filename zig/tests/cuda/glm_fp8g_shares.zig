//! `glm-fp8g-shares`: block-FP8 TP shares packed as GLM's loader packs them, on the lane, against FP64 of the cut.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const reference = @import("quant_ref_tests.zig");
const Rig = reference.RigT;
const bf = reference.helpers.bfOf;
const fp8pb = glm.weights.formats.fp8pb;
const e4m3 = core.quant.fp8.e4m3;

/// A random FP8_PB_WO matrix [n, k]: finite e4m3 codes and its 128 x 128 block scales.
fn matrix(r: *Rig, n: usize, k: usize) !fp8pb.Fp8pb {
    const codes = try r.a.alloc(u8, n * k);
    for (codes) |*v| v.* = reference.helpers.fp8CodeOf(r);
    const inv = try r.a.alloc(f32, ((n + 127) / 128) * ((k + 127) / 128));
    for (inv) |*v| v.* = r.rng.float(f32) * 0.01 + 0.001;
    return .{ .n = n, .k = k, .codes = codes, .scales = inv };
}

/// Weight (row, col) of the stored matrix, read without the packing.
fn at(p: fp8pb.Fp8pb, row: usize, col: usize) f64 {
    return e4m3(p.codes[row * p.k + col]) * p.scales[(row / 128) * ((p.k + 127) / 128) + col / 128];
}

/// `parts` packed for the lane, x [m][k] times them in fp32, against FP64 of the parts' rows and inputs as stored.
fn share(r: *Rig, lane: *const cuda.qmmf.Lane, label: []const u8, parts: []const fp8pb.Share) !void {
    const z = try fp8pb.sizes(parts);
    const codes = try r.a.alloc(u8, z.codes);
    const tiles = try r.a.alloc(f32, z.scales);
    try fp8pb.pack(r.a, parts, codes, tiles);
    const w = cuda.fp8.weight(try r.dev(codes), try r.dev(std.mem.sliceAsBytes(tiles)), z.n, z.k);
    const m = 17; // two 16-row tiles, the second partial
    const x = try r.bfs(m * z.k, 1.0);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const want = try r.a.alloc(f64, m * z.n);
    const tol = try r.a.alloc(f64, m * z.n);
    const wrow = try r.a.alloc(f64, z.k);
    var col: usize = 0;
    for (parts) |q| for (q.lo..q.hi) |src| {
        for (wrow, 0..) |*v, i| v.* = at(q.p.*, src, q.klo + i); // output col's weights, decoded once
        for (0..m) |row| {
            var sum: f64 = 0;
            var magnitude: f64 = 0;
            for (wrow, 0..) |wv, i| {
                const v = bf(x[row * z.k + i]) * wv;
                sum += v;
                magnitude += @abs(v);
            }
            want[row * z.n + col] = sum;
            tol[row * z.n + col] = magnitude * 2e-6 + @abs(sum) * 1e-6 + 1e-30;
        }
        col += 1;
    };
    var out = try cuda.DeviceBuffer.alloc(r.gpu.d, m * z.n * 4);
    defer {
        cuda.completion.drain(r.gpu.d, r.s, "GLM FP8 share output released before completion");
        out.free();
    }
    try out.fill8(0xff, r.s.handle);
    if (lane.partBytes(m, w) != 0) return error.SlicePartialsNotHeld;
    try lane.matmulOut(r.s, dx, z.k, m, w, out.ptr, .f32, 0);
    try r.close(label, out.ptr, true, want, tol);
}

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0xf8_5ba7e);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        cuda.completion.drain(gpu.d, s, "GLM FP8 share inputs released before completion");
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    // the shared expert as the loader cuts it (shard.Share's spans of 2,048 rows): [gate | up] by rows, down by inputs
    const gate = try matrix(&r, 2048, 4096);
    const up = try matrix(&r, 2048, 4096);
    const down = try matrix(&r, 4096, 2048);
    const spans = [_][2]usize{ .{ 0, 2048 }, .{ 0, 1024 }, .{ 1024, 2048 }, .{ 0, 704 }, .{ 704, 1408 }, .{ 1408, 2048 } };
    var label: [96]u8 = undefined;
    for (spans) |sp| {
        const gu = [_]fp8pb.Share{ .{ .p = &gate, .lo = sp[0], .hi = sp[1], .klo = 0, .khi = 4096 }, .{ .p = &up, .lo = sp[0], .hi = sp[1], .klo = 0, .khi = 4096 } };
        try share(&r, &lane, try std.fmt.bufPrint(&label, "[gate | up] rows {d}..{d}", .{ sp[0], sp[1] }), &gu);
        const d = [_]fp8pb.Share{.{ .p = &down, .lo = 0, .hi = 4096, .klo = sp[0], .khi = sp[1] }};
        try share(&r, &lane, try std.fmt.bufPrint(&label, "down inputs {d}..{d}", .{ sp[0], sp[1] }), &d);
    }
    check.pass("glm-fp8g-shares: TP 1, 2 and 3 shares of block-FP8 [gate | up] and down within {d:.3} of FP64", .{r.worst});
}
