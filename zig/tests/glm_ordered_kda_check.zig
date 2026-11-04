//! The real KDA encoder at every cut, its FP32 state and convolution tail, and independent stage references.
const std = @import("std");
const mtl = @import("metal");
const glm = @import("tensorfold").glm;
const util = @import("glm_ordered_check.zig");
const oracle = @import("glm_ordered_kda_ref.zig");
const a = std.heap.page_allocator;
const R = 33;
const Ref = glm.affine.Ref;
fn u16s(r: Ref, n: usize) []u16 {
    return @as([*]u16, @ptrCast(@alignCast(r.addr())))[0..n];
}
fn f32s(r: Ref, n: usize) []f32 {
    return @as([*]f32, @ptrCast(@alignCast(r.addr())))[0..n];
}
fn copy(r: Ref, bytes: []const u8) void {
    @memcpy(r.addr()[0..bytes.len], bytes);
}
const Ctx = struct { c: *const glm.config.Config, k: *const glm.kernels.Kernels, s: *glm.state.State, pskip: u32 = 0 };
pub fn check(k: *const glm.kernels.Kernels, d: mtl.Device, queue: mtl.Queue) !void {
    const c: glm.config.Config = .{ .run = 1, .layers = 1, .mtp = 0 };
    const W = c.kdaWidth();
    const D = c.kda_dim;
    const H = c.kda_heads;
    const P = c.kdaProj();
    const state_len = H * D * D;
    const conv_len = 3 * 3 * W;
    var owned: std.ArrayList(mtl.Buffer) = .empty;
    defer {
        for (owned.items) |b| b.deinit();
        owned.deinit(a);
    }
    const Alloc = struct {
        d: mtl.Device,
        list: *std.ArrayList(mtl.Buffer),
        fn get(x: @This(), bytes: usize) !Ref {
            const b = try util.buffer(x.d, bytes);
            errdefer b.deinit();
            try x.list.append(a, b);
            return util.rb(b);
        }
    };
    const alloc: Alloc = .{ .d = d, .list = &owned };
    var rng = std.Random.DefaultPrng.init(743128);
    const random = rng.random();
    const proj = try alloc.get(R * P * 2);
    for (u16s(proj, R * P)) |*v| v.* = util.bf((random.float(f32) - 0.5) * 2);
    var state: glm.state.State = .{ .cap = R };
    state.kda[0] = .{ .st = .{ try alloc.get(state_len * 4), try alloc.get(state_len * 4) }, .cs = .{ try alloc.get(conv_len * 2), try alloc.get(conv_len * 2) }, .proj = proj };
    const L = &state.kda[0];
    for (f32s(L.st[0], state_len)) |*v| v.* = (random.float(f32) - 0.5) * 0.1;
    for (u16s(L.cs[0], conv_len)) |*v| v.* = util.bf((random.float(f32) - 0.5) * 2);
    const initial = try a.dupe(f32, f32s(L.st[0], state_len));
    defer a.free(initial);
    const history = try a.dupe(u16, u16s(L.cs[0], conv_len));
    defer a.free(history);
    const fq: glm.affine.Affine = .{ .w = try alloc.get(W * D), .s = try alloc.get(W * (D / 64) * 2), .b = try alloc.get(W * (D / 64) * 2), .n = W, .k = D, .bits = 8, .group = 64 };
    var gq = fq;
    gq.w = try alloc.get(W * D);
    random.bytes(fq.w.addr()[0 .. W * D]);
    random.bytes(gq.w.addr()[0 .. W * D]);
    for (u16s(fq.s, W * (D / 64))) |*v| v.* = util.bf(0.001);
    for (u16s(fq.b, W * (D / 64))) |*v| v.* = util.bf(-0.1);
    const w: glm.weights.Kda = .{ .in_proj = fq, .f_b = fq, .g_b = gq, .o_proj = fq, .conv_w = try alloc.get(4 * 3 * W * 4), .a = try alloc.get(H * 4), .a_log = try alloc.get(H * 4), .dt_bias = try alloc.get(W * 4), .o_norm = try alloc.get(D * 4) };
    for (f32s(w.conv_w, 4 * 3 * W)) |*v| v.* = (random.float(f32) - 0.5) * 0.5;
    @memset(f32s(w.a, H), 1);
    @memset(f32s(w.dt_bias, W), 0.1);
    @memset(f32s(w.o_norm, D), 1);
    const scratch = try alloc.get(R * (14 * W + H * 4));
    const y = try alloc.get(R * W * 2);
    const ctx: Ctx = .{ .c = &c, .k = k, .s = &state };
    var cb = queue.commandBuffer();
    var e = cb.compute(.serial);
    glm.ordered_kda.encode(&ctx, e, 0, &w, proj, P, y, scratch, R, R);
    e.end();
    try util.finish(cb);
    const plane = R * W * 2;
    const sy = scratch.at(3 * plane);
    const gate = scratch.at(6 * plane);
    try oracle.verify(a, .{ .rows = R, .heads = H, .dim = D, .initial = initial, .q = u16s(scratch, R * W), .k = u16s(scratch.at(plane), R * W), .v = u16s(scratch.at(2 * plane), R * W), .decay = f32s(scratch.at(4 * plane), R * W), .beta = f32s(scratch.at(7 * plane), R * H), .sy = u16s(sy, R * W), .final = f32s(L.st[1], state_len) });
    // The post-pass reference consumes SY and GATE at their explicit BF16 points.
    for (0..R) |r| for (0..H) |h| {
        const at = (r * H + h) * D;
        var sum: f64 = 0;
        for (u16s(sy, R * W)[at..][0..D]) |v| sum += oracle.bf(v) * oracle.bf(v);
        const inv = 1 / @sqrt(sum / @as(f64, @floatFromInt(D)) + @as(f64, c.eps));
        for (0..D) |j| {
            const want = oracle.bf(u16s(sy, R * W)[at + j]) * inv / (1 + @exp(-oracle.bf(u16s(gate, R * W)[at + j])));
            const bound = (0x1p-8 + @as(f64, @floatFromInt(2 * D + 32)) * 0x1p-24) * @abs(want) + 0x1p-126;
            if (@abs(oracle.bf(u16s(y, R * W)[at + j]) - want) > bound) return error.KdaPostFp64Mismatch;
        }
    };
    const full_y = try a.dupe(u16, u16s(y, R * W));
    defer a.free(full_y);
    const full_st = try a.dupe(f32, f32s(L.st[1], state_len));
    defer a.free(full_st);
    const full_cs = try a.dupe(u16, u16s(L.cs[1], conv_len));
    defer a.free(full_cs);
    for ([_]u32{ 1, 2, 3, 4, 8, 16, 17, 33 }) |cut| {
        copy(L.st[0], std.mem.sliceAsBytes(initial));
        copy(L.cs[0], std.mem.sliceAsBytes(history));
        L.cur = 0;
        @memset(y.addr()[0 .. R * W * 2], 0xff);
        cb = queue.commandBuffer();
        e = cb.compute(.serial);
        var r: u32 = 0;
        while (r < R) : (r += cut) {
            const n = @min(cut, R - r);
            glm.ordered_kda.encode(&ctx, e, 0, &w, proj.at(@as(usize, r) * P * 2), P, y.at(@as(usize, r) * W * 2), scratch, n, @min(n, 16));
            L.cur = 1 - L.cur;
        }
        e.end();
        try util.finish(cb);
        if (!std.mem.eql(u16, full_y, u16s(y, R * W))) return error.KdaCutOutputDiffers;
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(full_st), std.mem.sliceAsBytes(f32s(L.st[L.cur], state_len)))) return error.KdaCutStateDiffers;
        if (!std.mem.eql(u16, full_cs, u16s(L.cs[L.cur], conv_len))) return error.KdaCutConvDiffers;
    }
    // Partial acceptance replays from the original slot, overwriting the rejected full window's destination.
    copy(L.st[0], std.mem.sliceAsBytes(initial));
    copy(L.cs[0], std.mem.sliceAsBytes(history));
    L.cur = 0;
    cb = queue.commandBuffer();
    e = cb.compute(.serial);
    glm.ordered_kda.encode(&ctx, e, 0, &w, proj, P, y, scratch, 3, 16);
    e.end();
    try util.finish(cb);
    const prefix_st = try a.dupe(f32, f32s(L.st[1], state_len));
    defer a.free(prefix_st);
    const prefix_cs = try a.dupe(u16, u16s(L.cs[1], conv_len));
    defer a.free(prefix_cs);
    const prefix_y = try a.dupe(u16, u16s(y, 3 * W));
    defer a.free(prefix_y);
    cb = queue.commandBuffer();
    e = cb.compute(.serial);
    glm.ordered_kda.encode(&ctx, e, 0, &w, proj, P, y, scratch, 16, 16);
    glm.ordered_kda.encode(&ctx, e, 0, &w, proj, P, y, scratch, 3, 16);
    e.end();
    try util.finish(cb);
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(prefix_st), std.mem.sliceAsBytes(f32s(L.st[1], state_len)))) return error.KdaReplayStateDiffers;
    if (!std.mem.eql(u16, prefix_cs, u16s(L.cs[1], conv_len))) return error.KdaReplayConvDiffers;
    if (!std.mem.eql(u16, prefix_y, u16s(y, 3 * W))) return error.KdaReplayOutputDiffers;
    std.debug.print("KDA: cuts 1/2/3/4/8/16/17/33 have identical Y, FP32 state and convolution tail; partial replay and FP64 scan/post pass\\n", .{});
}
