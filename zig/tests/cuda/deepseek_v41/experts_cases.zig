//! Bit checks of the pointwise, glue, prefill glue, top-k keys and x3gm plan kernels against torch.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");
const fixture = @import("../fixture.zig");
const Fixture = fixture.Fixture;
const experts = @import("experts.zig");

const exl3 = dsv41.exl3;
const Gpu = check.Gpu;
const Bufs = experts.Bufs;
const size = experts.size;
const same = experts.same;

/// pointwise.cu against torch: the shared expert's SwiGLU with the limit, and the partial adds (fp32, into bf16).
pub fn pointwiseCases(gpu: Gpu, k: *const dsv41.Kernels, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var b: Bufs = .{ .gpu = gpu, .fx = &fx };
    defer b.deinit();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads and memsets on the default stream
    defer stream.deinit();
    const o = k.others(stream);
    const R = try size(fx, "R");
    const n = try size(fx, "n");
    const D = try size(fx, "D");
    const g = try b.up("g");
    const u = try b.up("u");
    const act = try b.zeros(R * n * 4, 0xee);
    try o.swiglu(g.ptr, n, u.ptr, n, act.ptr, n, R, n, @floatCast(try fx.float("limit")));
    try stream.synchronize();
    try same(gpu, fx, act, "act", "swiglu vs torch clamp_ / silu / mul_");
    const a = try b.up("a");
    const bb = try b.up("b");
    const y16 = try b.zeros(R * D * 2, 0xee);
    try o.addBf16(a.ptr, bb.ptr, y16.ptr, R * D);
    try o.addInto(a.ptr, bb.ptr, R * D);
    try stream.synchronize();
    try same(gpu, fx, a, "sum32", "out += shared");
    try same(gpu, fx, y16, "sum16", "torch.add(out=bf16)");
    check.pass("BITEXACT pointwise.cu == torch: swiglu with limit ({d} x {d}, NaN / inf / -0.0), adds into fp32 and bf16", .{ R, n });
}

/// glue.cu against torch: the window's embedding send / sum, the casts, positions, gather_cols and the carry.
pub fn glueCases(gpu: Gpu, k: *const dsv41.Kernels, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var b: Bufs = .{ .gpu = gpu, .fx = &fx };
    defer b.deinit();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads on the default stream
    defer stream.deinit();
    const o = k.others(stream);
    const V = try size(fx, "V");
    const D = try size(fx, "D");
    const W = try size(fx, "W");
    const n = try size(fx, "n");
    const c = try size(fx, "c");

    const embed = try b.up("embed");
    const ids = try b.up("ids");
    const send = try b.zeros(n * D * 2, 0xee);
    try o.embedSend(ids.ptr, n, embed.ptr, try fx.int("lo"), V, D, send.ptr);
    const recv = try b.up("recv");
    const summed = try b.zeros(n * 4 * D * 2, 0xee);
    try o.embedSum(recv.ptr, W, n, D, summed.ptr);
    const x = try b.up("x");
    const cast = try b.zeros(n * D * 2, 0xee);
    try o.castBf16(x.ptr, cast.ptr, n * D);
    const kw = try b.zeros(n * D * 4, 0xee);
    try o.kitWeights(x.ptr, kw.ptr, n * D);
    const kl = try b.up("x");
    try o.kitLogits(kl.ptr, n * D);
    const gc = try b.up("gc");
    const cols = try b.zeros(n * W * c * 2, 0xee);
    try o.gatherCols(gc.ptr, W, n, c, cols.ptr);
    const src = try b.up("src");
    const row: usize = @intCast(try fx.int("row"));
    const carry = try b.zeros(1024 * 4, 0xee);
    try o.carry(src.ptr, @intCast(try fx.int("ld")), 0, row, 1024, carry.ptr);
    // the device-held row (graph-safe form): the same bits
    const acc = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&@as(i32, @intCast(row))));
    try b.list.append(gpu.gpa, acc);
    const carry2 = try b.zeros(1024 * 4, 0xee);
    try o.carry(src.ptr, @intCast(try fx.int("ld")), acc.ptr, 0, 1024, carry2.ptr);
    const start: i32 = @intCast(try fx.int("start"));
    const pos = try b.zeros(4, 0xee);
    const pos64 = try b.zeros(n * 8, 0xee);
    const lo = try b.zeros(n * 4, 0xee);
    try o.positions(start, n, pos.ptr, pos64.ptr, lo.ptr);
    // the graphed window's form: start read from the device, the same values
    const start_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&start));
    try b.list.append(gpu.gpa, start_dev);
    const pos_d = try b.zeros(4, 0xee);
    const pos64_d = try b.zeros(n * 8, 0xee);
    const lo_d = try b.zeros(n * 4, 0xee);
    try o.positionsDev(start_dev.ptr, n, pos_d.ptr, pos64_d.ptr, lo_d.ptr);
    // widen_f32: the recv rows (bf16, NaN / inf / -0 among them) to fp32, against the bits shifted up on the host
    const wide = try b.zeros(W * n * D * 4, 0xee);
    try o.widenF32(recv.ptr, wide.ptr, W * n * D);
    try stream.synchronize();
    {
        const recv_host = try fx.bytes("recv");
        defer gpu.gpa.free(recv_host);
        const want = try gpu.gpa.alloc(u32, recv_host.len / 2);
        defer gpu.gpa.free(want);
        for (want, 0..) |*w, i| w.* = @as(u32, std.mem.readInt(u16, recv_host[2 * i ..][0..2], .little)) << 16;
        const got = try gpu.gpa.alloc(u8, 4 * want.len);
        defer gpu.gpa.free(got);
        try wide.download(0, got);
        try check.sameBytes("glue widen_f32", got, std.mem.sliceAsBytes(want));
    }

    try same(gpu, fx, send, "send", "glue embed_send");
    try same(gpu, fx, summed, "summed", "glue embed_sum");
    try same(gpu, fx, cast, "cast", "glue cast_bf16");
    try same(gpu, fx, kw, "kw", "glue kit_weights");
    try same(gpu, fx, kl, "kl", "glue kit_logits");
    try same(gpu, fx, cols, "cols", "glue gather_cols");
    try same(gpu, fx, carry, "carry", "glue carry (host row)");
    try same(gpu, fx, carry2, "carry", "glue carry (device row)");
    // positions: no torch op to record; Win.make's values
    var want_pos: [1]i32 = .{start};
    const want64 = try gpu.gpa.alloc(i64, n);
    defer gpu.gpa.free(want64);
    const want_lo = try gpu.gpa.alloc(i32, n);
    defer gpu.gpa.free(want_lo);
    for (want64, want_lo, 0..) |*p, *l, i| {
        p.* = start + @as(i64, @intCast(i));
        l.* = 0;
    }
    for ([_]struct { cuda.DeviceBuffer, []const u8 }{ .{ pos, std.mem.sliceAsBytes(&want_pos) }, .{ pos64, std.mem.sliceAsBytes(want64) }, .{ lo, std.mem.sliceAsBytes(want_lo) }, .{ pos_d, std.mem.sliceAsBytes(&want_pos) }, .{ pos64_d, std.mem.sliceAsBytes(want64) }, .{ lo_d, std.mem.sliceAsBytes(want_lo) } }) |pw| {
        const got = try gpu.gpa.alloc(u8, pw[1].len);
        defer gpu.gpa.free(got);
        try pw[0].download(0, got);
        try check.sameBytes("glue positions", got, pw[1]);
    }
    check.pass("BITEXACT glue.cu == torch: embed send / sum (W {d}), cast_bf16, kit_weights, kit_logits, gather_cols, carry, positions (host and device start), widen_f32", .{W});
}

/// prefill_glue.cu against torch: block_prefill.zig's glue steps (picks, masks, casts, projections, ring copies).
pub fn pfglueCases(gpu: Gpu, k: *const dsv41.Kernels, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var b: Bufs = .{ .gpu = gpu, .fx = &fx };
    defer b.deinit();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads on the default stream
    defer stream.deinit();
    const o = k.others(stream);
    var nb: [64]u8 = undefined;
    var nb2: [64]u8 = undefined;

    const ntp = try size(fx, "tp");
    for (0..ntp) |i| {
        const R = try size(fx, try std.fmt.bufPrint(&nb, "tp{d}_R", .{i}));
        const n = try size(fx, try std.fmt.bufPrint(&nb, "tp{d}_n", .{i}));
        const count = try size(fx, try std.fmt.bufPrint(&nb, "tp{d}_count", .{i}));
        const keys = try b.up(try std.fmt.bufPrint(&nb, "tp{d}_keys", .{i}));
        const got = try b.zeros(R * count * 4, 0xee);
        try o.topPositions(keys.ptr, n, R, n, count, got.ptr);
        try stream.synchronize();
        try same(gpu, fx, got, try std.fmt.bufPrint(&nb, "tp{d}_want", .{i}), try std.fmt.bufPrint(&nb2, "pfglue top_positions {d}", .{i}));
    }
    {
        const R = try size(fx, "ck_R");
        const cnb = try size(fx, "ck_nb");
        const bs = try size(fx, "ck_bs");
        const blocks = try b.up("ck_blocks");
        const got = try b.zeros(R * cnb * bs * 4, 0xee);
        try o.candKeys(blocks.ptr, cnb, R, cnb, bs, got.ptr);
        try stream.synchronize();
        try same(gpu, fx, got, "ck_want", "pfglue candidate_keys");
    }
    {
        const sel = try b.up("tp0_want");
        const pos = try b.up("vc_pos");
        const R = try size(fx, "tp0_R");
        const K = try size(fx, "tp0_count");
        for ([_]u32{ 4, 128 }) |ratio| {
            const got = try b.zeros(R * 4, 0xee);
            try o.visibleCounts(sel.ptr, K, pos.ptr, ratio, R, got.ptr);
            try stream.synchronize();
            try same(gpu, fx, got, try std.fmt.bufPrint(&nb, "vc{d}_want", .{ratio}), "pfglue visible_counts");
        }
    }
    {
        const n = try size(fx, "gp_n");
        const ld = try size(fx, "gp_slots");
        const topk = try size(fx, "gp_topk");
        const pick = try b.up("gp_pick");
        const wts = try b.up("gp_wts");
        const pk = try b.zeros(n * topk * 4, 0xee);
        const w6 = try b.zeros(n * topk * 4, 0xee);
        try o.gmPicks(pick.ptr, wts.ptr, ld, topk, n, pk.ptr, w6.ptr);
        try stream.synchronize();
        try same(gpu, fx, pk, "gp_pk", "pfglue gm_picks picks");
        try same(gpu, fx, w6, "gp_w6", "pfglue gm_picks kit weights");
    }
    {
        const E = try size(fx, "wm_E");
        const P = try size(fx, "wm_P");
        const pick = try b.up("wm_pick");
        const tab = try b.up("wm_tab");
        for ([_]u32{ 4, 10 }) |k2| {
            const got = try b.zeros(P * 4, 0xee);
            try o.widthMask(pick.ptr, P, tab.ptr, E, k2, got.ptr);
            try stream.synchronize();
            try same(gpu, fx, got, try std.fmt.bufPrint(&nb, "wm{d}_want", .{k2}), "pfglue width mask");
        }
    }
    {
        const x = try b.up("f64_in");
        const n = x.len / 8;
        const got = try b.zeros(n * 4, 0xee);
        try o.f64Bf16F32(x.ptr, got.ptr, n);
        try stream.synchronize();
        try same(gpu, fx, got, "f64_want", "pfglue fp64 -> bf16 -> fp32");
    }
    {
        const n = try size(fx, "pj_n");
        const hd = try size(fx, "pj_hd");
        const kv = try b.up("pj_kv");
        const gate = try b.up("pj_gate");
        const two = try b.zeros(n * 2 * hd * 4, 0xee);
        try o.widenCat(kv.ptr, gate.ptr, n, hd, two.ptr);
        const one = try b.zeros(n * hd * 4, 0xee);
        try o.widenCat(kv.ptr, 0, n, hd, one.ptr);
        try stream.synchronize();
        try same(gpu, fx, two, "pj_two", "pfglue projection [kv | gate]");
        try same(gpu, fx, one, "pj_one", "pfglue projection kv");
    }
    for (0..2) |i| {
        const sr = try size(fx, try std.fmt.bufPrint(&nb, "rc{d}_sr", .{i}));
        const dr = try size(fx, try std.fmt.bufPrint(&nb, "rc{d}_dr", .{i}));
        const rb = try size(fx, try std.fmt.bufPrint(&nb, "rc{d}_rb", .{i}));
        const lo: u64 = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "rc{d}_lo", .{i})));
        const hi: u64 = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "rc{d}_hi", .{i})));
        const src = try b.up(try std.fmt.bufPrint(&nb, "rc{d}_src", .{i}));
        const dst = try b.up(try std.fmt.bufPrint(&nb, "rc{d}_dst", .{i}));
        try o.ringCopy(src.ptr, sr, dst.ptr, dr, rb, lo, hi);
        try stream.synchronize();
        try same(gpu, fx, dst, try std.fmt.bufPrint(&nb, "rc{d}_want", .{i}), "pfglue staging ring copy");
    }
    check.pass("BITEXACT prefill_glue.cu == torch: top_positions ({d} shapes, counts 512 / 2,048), candidate_keys, visible_counts, gm_picks, width mask, fp64 casts, projection, ring copies", .{ntp});
}

/// topk_keys.cu against pick.top (torch.topk on pick.keys) at k 1 / 5 / 64 / 1,024 over 64,640 columns.
pub fn topkCases(gpu: Gpu, k: *const dsv41.Kernels, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var b: Bufs = .{ .gpu = gpu, .fx = &fx };
    defer b.deinit();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads and memsets on the default stream
    defer stream.deinit();
    const o = k.others(stream);
    const R = try size(fx, "R");
    const C = try size(fx, "C");
    const lg = try b.up("lg");
    var name_buf: [32]u8 = undefined;
    for (0..try size(fx, "ks")) |i| {
        const kk = try size(fx, try std.fmt.bufPrint(&name_buf, "k{d}", .{i}));
        const vals = try b.zeros(R * kk * 4, 0xee);
        const cols = try b.zeros(R * kk * 8, 0xee);
        try o.topKeys(lg.ptr, C, R, C, kk, vals.ptr, cols.ptr);
        try stream.synchronize();
        try same(gpu, fx, cols, try std.fmt.bufPrint(&name_buf, "cols{d}", .{i}), "topk_keys cols");
        try same(gpu, fx, vals, try std.fmt.bufPrint(&name_buf, "vals{d}", .{i}), "topk_keys vals");
    }
    check.pass("BITEXACT topk_keys.cu == pick.top on {d} rows x {d} columns, 4 k", .{ R, C });
}

/// x3gm_plan.cu against torch's x3gm.plan on full-sized picks (the "plan" case).
pub fn planCases(gpu: Gpu, k: *const dsv41.Kernels, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var b: Bufs = .{ .gpu = gpu, .fx = &fx };
    defer b.deinit();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads and memsets on the default stream
    defer stream.deinit();
    const o = k.experts(stream);
    var name_buf: [64]u8 = undefined;
    const n = try size(fx, "cases");
    for (0..n) |i| {
        const P = try size(fx, try std.fmt.bufPrint(&name_buf, "c{d}_P", .{i}));
        const E = try size(fx, try std.fmt.bufPrint(&name_buf, "c{d}_E", .{i}));
        const bm = try size(fx, try std.fmt.bufPrint(&name_buf, "c{d}_bm", .{i}));
        const T = exl3.planPasses(P, E, bm);
        const pk = try b.up(try std.fmt.bufPrint(&name_buf, "c{d}_pk", .{i}));
        const order = try b.zeros(P * 4, 0xee);
        const pe = try b.zeros(T * 4, 0xee);
        const poff = try b.zeros(T * 4, 0xee);
        const pcnt = try b.zeros(T * 4, 0xee);
        const npass = try b.zeros(4, 0xee);
        try o.gmPlan(pk.ptr, P, E, bm, order.ptr, pe.ptr, poff.ptr, pcnt.ptr, npass.ptr);
        try stream.synchronize();
        for ([_][]const u8{ "order", "pe", "poff", "pcnt", "npass" }, [_]cuda.DeviceBuffer{ order, pe, poff, pcnt, npass }) |part, buf| {
            const name = try std.fmt.bufPrint(&name_buf, "c{d}_{s}", .{ i, part });
            try same(gpu, fx, buf, name, name);
        }
    }
    check.pass("BITEXACT x3gm_plan.cu == torch x3gm.plan on {d} cases (up to 24,576 pairs, 384 experts)", .{n});
}
