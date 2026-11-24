//! Unit tests for GLM MTP prep and 4-bit/64 fused head-top1 kernel dispatch rules.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const affine4 = @import("../../core/affine4.zig");
const wts = @import("weights.zig");
const kernels = @import("kernels.zig");

test "tryHeadTop1 admission requires TP=1, m_row=0, 4-bit/64 head, aligned K and n, and single row" {
    const ok: wts.Affine = .{ .w = undefined, .s = undefined, .b = undefined, .n = 154880, .k = 4096, .bits = 4, .group = 64 };
    try std.testing.expect(kernels.canHeadTop1(true, 1, 0, 154880, ok, 1));
    try std.testing.expect(!kernels.canHeadTop1(true, 1, 0, 154880, ok, 2));
    try std.testing.expect(!kernels.canHeadTop1(true, 1, 0, 154880, ok, 3));
    try std.testing.expect(!kernels.canHeadTop1(true, 1, 0, 154880, ok, kernels.max_rows));
    try std.testing.expect(!kernels.canHeadTop1(false, 1, 0, 154880, ok, 1));
    try std.testing.expect(!kernels.canHeadTop1(true, 2, 0, 154880, ok, 1));
    try std.testing.expect(!kernels.canHeadTop1(true, 1, 1, 154880, ok, 1));
    try std.testing.expect(!kernels.canHeadTop1(true, 1, 0, 154882, ok, 1));
    try std.testing.expect(!kernels.canHeadTop1(true, 1, 0, 154880, ok, 0));
    try std.testing.expect(!kernels.canHeadTop1(true, 1, 0, 154880, ok, kernels.max_rows + 1));
    var bad = ok;
    bad.bits = 8;
    try std.testing.expect(!kernels.canHeadTop1(true, 1, 0, 154880, bad, 1));
    bad = ok;
    bad.group = 32;
    try std.testing.expect(!kernels.canHeadTop1(true, 1, 0, 154880, bad, 1));
    bad = ok;
    bad.more = 1;
    try std.testing.expect(!kernels.canHeadTop1(true, 1, 0, 154880, bad, 1));
    bad = ok;
    bad.k = 256;
    try std.testing.expect(!kernels.canHeadTop1(true, 1, 0, 154880, bad, 1));
}

test "glm_mtp_prep_4b and glm_head_top1_4b_one match exact host reference on Metal" {
    const dev = mtl.Device.init() catch return error.SkipZigTest;
    defer dev.deinit();
    const lib = try mtl.Library.fromSource(dev, sources.glm_mtp_opt, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const p_prep = try mtl.Pipeline.init(dev, lib, "glm_mtp_prep_4b", false);
    defer p_prep.deinit();
    const p_one = try mtl.Pipeline.init(dev, lib, "glm_head_top1_4b_one", false);
    defer p_one.deinit();
    const p_red = try mtl.Pipeline.init(dev, lib, "glm_reduce_top1", false);
    defer p_red.deinit();

    const width: usize = 512;
    const rows: usize = 2;
    const vocab: usize = 64;
    const groups: usize = vocab * (width / 64);
    const b_ids = try dev.buffer(rows * 4, mtl.ResourceOptions.shared);
    defer b_ids.deinit();
    const b_w = try dev.buffer(groups * 32, mtl.ResourceOptions.shared);
    defer b_w.deinit();
    const b_s = try dev.buffer(groups * 2, mtl.ResourceOptions.shared);
    defer b_s.deinit();
    const b_b = try dev.buffer(groups * 2, mtl.ResourceOptions.shared);
    defer b_b.deinit();
    const b_we = try dev.buffer(width * 2, mtl.ResourceOptions.shared);
    defer b_we.deinit();
    const b_wh = try dev.buffer(width * 2, mtl.ResourceOptions.shared);
    defer b_wh.deinit();
    const b_h = try dev.buffer(rows * width * 2, mtl.ResourceOptions.shared);
    defer b_h.deinit();
    const b_out = try dev.buffer(rows * 2 * width * 2, mtl.ResourceOptions.shared);
    defer b_out.deinit();
    const b_head_in = try dev.buffer(rows * width * 2, mtl.ResourceOptions.shared);
    defer b_head_in.deinit();
    const b_pairs = try dev.buffer(((vocab + 7) / 8) * 8, mtl.ResourceOptions.shared);
    defer b_pairs.deinit();
    const b_pick1 = try dev.buffer(4, mtl.ResourceOptions.shared);
    defer b_pick1.deinit();

    const ids: [*]u32 = @ptrCast(@alignCast(b_ids.contents()));
    ids[0] = 3;
    ids[1] = 17;
    const ww: [*]u32 = @ptrCast(@alignCast(b_w.contents()));
    const ws: [*]u16 = @ptrCast(@alignCast(b_s.contents()));
    const wb: [*]u16 = @ptrCast(@alignCast(b_b.contents()));
    const we: [*]u16 = @ptrCast(@alignCast(b_we.contents()));
    const wh: [*]u16 = @ptrCast(@alignCast(b_wh.contents()));
    const hh: [*]u16 = @ptrCast(@alignCast(b_h.contents()));
    const hin: [*]u16 = @ptrCast(@alignCast(b_head_in.contents()));
    var prng = std.Random.DefaultPrng.init(11);
    for (0..groups) |g| {
        ws[g] = affine4.asBf16(0.02);
        wb[g] = affine4.asBf16(-0.15);
        for (0..8) |j| ww[g * 8 + j] = prng.random().int(u32);
    }
    for (0..width) |j| {
        we[j] = affine4.asBf16(1.0 + 0.001 * @as(f32, @floatFromInt(j % 13)));
        wh[j] = affine4.asBf16(0.8 + 0.002 * @as(f32, @floatFromInt(j % 7)));
    }
    for (0..rows * width) |j| {
        hh[j] = affine4.asBf16(@sin(@as(f32, @floatFromInt(j)) * 0.05));
        hin[j] = affine4.asBf16(@cos(@as(f32, @floatFromInt(j)) * 0.03));
    }

    const q = try dev.queue();
    defer q.deinit();
    const cb = q.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(p_prep);
    inline for (.{ b_ids, b_w, b_s, b_b, b_we, b_wh, b_h, b_out }, 0..) |b, i| enc.setBuffer(b, 0, i);
    enc.setValue(kernels.MtpPrepArgs{ .rows = rows, .width = width, .eps = 1e-5 }, 8);
    enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(((width + 3) / 4 + 31) / 32 * 32, 1, 1));
    const one_pairs: u32 = (vocab + 7) / 8;
    const args_one = kernels.HeadTop1Args{ .k = width, .n = vocab, .pairs = one_pairs };
    enc.setPipeline(p_one);
    inline for (.{ b_head_in, b_w, b_s, b_b }, 0..) |b, i| enc.setBuffer(b, 0, i);
    enc.setValue(args_one, 4);
    enc.setBuffer(b_pairs, 0, 5);
    enc.dispatchGroups(mtl.Size.of(one_pairs, 1, 1), mtl.Size.of(128, 1, 1));
    enc.setPipeline(p_red);
    enc.setBuffer(b_pairs, 0, 0);
    enc.setBuffer(b_pick1, 0, 1);
    enc.setValue(args_one, 2);
    enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    try std.testing.expect(cb.failure() == null);

    const pick_one: [*]const u32 = @ptrCast(@alignCast(b_pick1.contents()));
    const prepped: [*]const u16 = @ptrCast(@alignCast(b_out.contents()));
    for (0..rows) |r| {
        const ng = width / 64;
        var e_ss: f64 = 0;
        var h_ss: f64 = 0;
        const tok_row = @as(usize, ids[r]) * ng;
        for (0..width) |j| {
            const g = tok_row + j / 64;
            const code = (ww[g * 8 + (j % 64) / 8] >> @as(u5, @intCast(4 * (j % 8)))) & 0xf;
            const ev = affine4.asFloat(ws[g]) * @as(f32, @floatFromInt(code)) + affine4.asFloat(wb[g]);
            const hv = affine4.asFloat(hh[r * width + j]);
            e_ss += @as(f64, ev) * ev;
            h_ss += @as(f64, hv) * hv;
        }
        const ei: f32 = @floatCast(1.0 / @sqrt(e_ss / @as(f64, width) + 1e-5));
        const hi: f32 = @floatCast(1.0 / @sqrt(h_ss / @as(f64, width) + 1e-5));
        for (0..width) |j| {
            const g = tok_row + j / 64;
            const code = (ww[g * 8 + (j % 64) / 8] >> @as(u5, @intCast(4 * (j % 8)))) & 0xf;
            const ev = affine4.asFloat(ws[g]) * @as(f32, @floatFromInt(code)) + affine4.asFloat(wb[g]);
            const hv = affine4.asFloat(hh[r * width + j]);
            const ne = (ev * ei) * affine4.asFloat(we[j]);
            const nh = (hv * hi) * affine4.asFloat(wh[j]);
            const got_e = affine4.asFloat(prepped[r * 2 * width + j]);
            const got_h = affine4.asFloat(prepped[r * 2 * width + width + j]);
            try std.testing.expect(@abs(ne - got_e) < 0.02);
            try std.testing.expect(@abs(nh - got_h) < 0.02);
        }
        if (r == 0) {
            var best_idx: u32 = 0;
            var best_val: f32 = -std.math.inf(f32);
            for (0..vocab) |v| {
                var acc: f32 = 0;
                for (0..width) |j| {
                    const g = v * ng + j / 64;
                    const code = (ww[g * 8 + (j % 64) / 8] >> @as(u5, @intCast(4 * (j % 8)))) & 0xf;
                    const wv = affine4.asFloat(ws[g]) * @as(f32, @floatFromInt(code)) + affine4.asFloat(wb[g]);
                    acc += affine4.asFloat(hin[j]) * wv;
                }
                if (acc > best_val) {
                    best_val = acc;
                    best_idx = @intCast(v);
                }
            }
            try std.testing.expectEqual(best_idx, pick_one[0]);
        }
    }
}
