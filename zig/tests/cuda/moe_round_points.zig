//! Serving's BF16 edges in the actual adjoint kernels, independently of the broad layer tolerance.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig = @import("train_back_rig.zig");
const R = 3;
const D = 2048;
const W = 512;
const E = 256;
const K = 8;
const S = K + 1;
fn uncertain(g: f64) bool {
    const si = rig.silu(rig.rb(g));
    if (si == 0) return false;
    const bits = rig.bf16(si);
    const rounded = rig.bf(bits);
    const gap = @min(@abs(si - (rounded + rig.bf(bits - 1)) / 2), @abs(si - (rounded + rig.bf(bits + 1)) / 2));
    return gap <= 32 * std.math.floatEps(f32) * @abs(si);
}
pub fn run(gpu: check.Gpu, k: *const cuda.train_q4.Kernels, rounded: bool) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x35bed6e5);
    var r: rig.Rig = .{ .gpu = gpu, .a = arena.allocator(), .rng = prng.random(), .s = try cuda.Stream.init(gpu.d, true) };
    defer r.s.deinit();
    defer {
        cuda.completion.drain(gpu.d, r.s, "MoE rounding qualification did not drain");
        r.release();
        r.bufs.deinit(gpu.gpa);
    }
    const gu = try r.f64s(R * S * 2 * W, -2, 2, .f);
    // Avoid a BF16 tie whose expf uncertainty could legitimately pick either neighboring SiLU value.
    for (0..R * S) |s| for (gu[s * 2 * W ..][0..W]) |*v| {
        while (uncertain(v.*)) v.* = rig.rf(r.uni(-2, 2));
    };
    const da = try r.f64s(R * S * W, -1, 1, .f);
    const weights = try r.f64s(R * S, 0.02, 0.9, .bf);
    var order: [R * S]i32 = undefined;
    for (&order, 0..) |*p, i| p.* = @intCast(i * 7 % order.len);
    const want_a = try r.a.alloc(f64, R * S);
    const want_gu = try r.a.alloc(f64, gu.len);
    for (order, 0..) |p, s| {
        const pair: usize = @intCast(p);
        var sum: f64 = 0;
        for (0..W) |i| {
            const gv = rig.rb(gu[s * 2 * W + i]);
            const uv = rig.rb(gu[s * 2 * W + W + i]);
            const si = rig.rb(rig.silu(gv));
            const d = da[s * W + i];
            sum += d * rig.rb(si * uv);
            want_gu[s * 2 * W + i] = weights[pair] * d * uv * rig.siluGrad(gv);
            want_gu[s * 2 * W + W + i] = weights[pair] * d * si;
        }
        want_a[pair] = sum;
    }
    const output_a = try r.zeros(R * S * 4);
    const output_gu = try r.zeros(gu.len * 4);
    var args: cuda.Args = .{};
    for ([_]u64{ try r.dev(f32, try rig.toF32(r.a, gu)), try r.dev(f32, try rig.toF32(r.a, da)), try r.dev(i32, &order), try r.dev(f32, try rig.toF32(r.a, weights)), output_a, output_gu }) |v| args.add(v);
    args.add(@as(c_int, W));
    args.add(@as(c_int, @intFromBool(rounded)));
    try cuda.launch.launch(k.fq[5], .{ .grid = .{ .x = R * S }, .block = .{ .x = 128 } }, r.s, &args);
    r.compare("served MoE BF16 gate/up/SiLU/activation edges", try r.back(f32, output_gu, gu.len), want_gu, 2 * W, 2e-5, 2e-6);
    r.compare("served MoE rounded activation slot scores", try r.back(f32, output_a, R * S), want_a, S, 2e-5, 2e-6);
    const logits = try r.f64s(R * (E + 1), -4, 4, .f);
    const router = try r.f64s((E + 1) * D, -0.1, 0.1, .bf);
    var ids: [R * S]i32 = undefined;
    const want_dl = try r.a.alloc(f64, R * S);
    const want_dx = try r.a.alloc(f64, R * D);
    @memset(want_dx, 0);
    for (0..R) |row| {
        for (0..K) |j| ids[row * S + j] = @intCast((row * 53 + j * 31) % E);
        ids[row * S + K] = E;
        var sum: f64 = 0;
        var mix: f64 = 0;
        for (ids[row * S ..][0..K]) |id| sum += @exp(logits[row * (E + 1) + @as(usize, @intCast(id))]);
        for (ids[row * S ..][0..K], 0..) |id, j| mix += @exp(logits[row * (E + 1) + @as(usize, @intCast(id))]) / sum * weights[row * S + j];
        for (ids[row * S ..][0..K], 0..) |id, j| want_dl[row * S + j] = @exp(logits[row * (E + 1) + @as(usize, @intCast(id))]) / sum * (weights[row * S + j] - mix);
        const sg = rig.sig(rig.rb(logits[row * (E + 1) + E]));
        want_dl[row * S + K] = weights[row * S + K] * sg * (1 - sg);
        for (ids[row * S ..][0..S], 0..) |id, j| for (0..D) |d| {
            want_dx[row * D + d] += want_dl[row * S + j] * router[@as(usize, @intCast(id)) * D + d];
        };
    }
    const dx = try r.zeros(R * D * 4);
    const dl = try r.zeros(R * S * 4);
    const scratch: cuda.train_q4.MoeScratch = .{ .logits = try r.dev(f32, try rig.toF32(r.a, logits)), .a = try r.dev(f32, try rig.toF32(r.a, weights)), .dlog = dl, .order = 0, .inv = 0, .offsets = 0, .tiles = 0, .gu = 0, .dact = 0, .dgu = 0, .dxp = 0 };
    const ops: cuda.train_q4.Ops = .{ .k = k, .s = r.s };
    try ops.routeBack(try r.dev(u16, try rig.toBf16(r.a, router)), .{ .x = 0, .wts = 0, .ids = try r.dev(i32, &ids), .served_rounding = rounded }, dx, scratch, R, K, E, D);
    r.compare("served MoE rounded shared logit", try r.back(f32, dl, R * S), want_dl, S, 2e-5, 2e-6);
    r.compare("served MoE router dx", try r.back(f32, dx, R * D), want_dx, D, 2e-5, 2e-6);
    std.debug.print("SUMMARY MoE rounding points: {d} checks, {d} failed, serving rounding {any}\n", .{ r.checks, r.bad, rounded });
    if (r.bad != 0) return error.TestFailed;
}
