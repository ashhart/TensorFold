//! Shared draft-loop and verify-window coverage over recorded CUDA outputs.
const std = @import("std");
const cuda = @import("cuda");
const mtp = @import("cuda_mtp.zig");
const ops = @import("cuda_mtp_ops.zig");
const draw = @import("cuda_mtp_draw.zig");
const wt = @import("cuda_prompt_wt.zig");
const prompt = @import("cuda_prompt.zig");
const f = @import("cuda_mtp_nonfinite_fixture.zig");
const D = f.Device;
const t = std.testing;
const a = t.allocator;

fn joinedBeforeWait(want_best: bool) !void {
    var best = false;
    var topk = false;
    for (D.events[0..D.events_n]) |e| switch (e) {
        .best => best = true,
        .topk => topk = true,
        .wait => {
            try t.expect(best == want_best and topk);
            return;
        },
        else => {},
    };
    return error.MissingCompletionWait;
}

const Flow = struct {
    e: *prompt.Engine,
    h: *mtp.Head,
    st: cuda.Stream,
    calls: *usize,
    lanes: *usize,
    pub fn stream(o: @This()) cuda.Stream {
        return o.st;
    }
    pub fn forwardHeadMany(o: @This(), ds: []const *mtp.Draft, nexts: []const []const u32, rows: []const u64, scores: bool) !void {
        try t.expect(scores);
        try t.expectEqual(ds.len, nexts.len);
        try t.expectEqual(ds.len, rows.len);
        o.calls.* += 1;
        o.lanes.* = ds.len;
        for (ds, nexts) |d, ids| d.len += ids.len;
    }
};

test "Flash Next nonfinite: draftMany drives two lanes through the actual shared reduction and guarded map" {
    var driver = D.driver();
    defer cuda.download_stage.closeDriver(&driver);
    var h = try f.head(&driver);
    h.draw_batch.on = false;
    defer h.artifact.deinit();
    var host0: [3]u16 = undefined;
    var host1: [3]u16 = undefined;
    var x = f.draft(&driver, &host0);
    var y = f.draft(&driver, &host1);
    y.label = "second-stream";
    y.dlogits.ptr = 0x3400;
    const st: cuda.Stream = .{ .d = &driver, .handle = null };
    var e: prompt.Engine = undefined;
    e.gpa = a;
    var calls: usize = 0;
    var lanes: usize = 0;
    const flow: Flow = .{ .e = &e, .h = &h, .st = st, .calls = &calls, .lanes = &lanes };
    var reduced = [_][4]f32{ f.reduction(2, 3, 0), f.reduction(2, 3, 2) };
    D.reductions = &reduced;
    var first: mtp.Chain = .{};
    var second: mtp.Chain = .{};
    try ops.draftMany(flow, &.{ &x, &y }, &.{ &.{11}, &.{12} }, &.{ 9, 10 }, &.{ 1, 1 }, &.{ 8, 8 }, &.{ null, null }, &.{ 0, 0 }, &.{ &first, &second });
    try t.expectEqual(@as(usize, 1), calls);
    try t.expectEqual(@as(usize, 2), lanes);
    try t.expectEqualSlices(u64, &.{ 0x3000, 0x3400 }, D.sources[0..D.best_calls]);
    try t.expectEqual(@as(usize, 1), first.n);
    try t.expectEqual(@as(usize, 1), second.n);
    try t.expectEqual(h.ids[0], first.tokens[0]);
    try t.expectEqual(h.ids[2], second.tokens[0]);
    D.reset();
    reduced[1] = f.reduction(-std.math.inf(f32), std.math.nan(f32), std.math.maxInt(i32));
    try t.expectError(error.ProposalNonfinite, ops.draftMany(flow, &.{ &x, &y }, &.{ &.{11}, &.{12} }, &.{ 9, 10 }, &.{ 1, 1 }, &.{ 8, 8 }, &.{ null, null }, &.{ 0, 0 }, &.{ &first, &second }));
    try t.expectEqual(@as(usize, 2), calls);
    try t.expectEqual(@as(usize, 2), D.best_calls);
    try t.expectEqual(@as(usize, 0), second.n);
}

test "Flash Next nonfinite: a later bad verify reduction publishes no earlier token and shares one wait" {
    var driver = D.driver();
    defer cuda.download_stage.closeDriver(&driver);
    var h = try f.head(&driver);
    defer h.artifact.deinit();
    const host = try a.alloc(u16, 2 * wt.vocab);
    defer a.free(host);
    const scores = try a.alloc(u16, 2 * wt.vocab);
    defer a.free(scores);
    @memset(scores, 0);
    scores[wt.vocab + 3] = 0xffc1;
    D.verify_rows = scores;
    var reductions = [_][4]f32{ f.reduction(1, 2, 17), f.reduction(1, 2, 5) };
    D.reductions = &reductions;
    D.vals = &.{ 0x3f80, 0, 0x3f80, 0 };
    D.cols = &.{ 17, 5, 17, 5 };
    var x = f.draft(&driver, host);
    x.rows = 2;
    const st: cuda.Stream = .{ .d = &driver, .handle = null };
    var good: [2]u32 = undefined;
    try draw.draws(&h, st, a, &x, null, &.{}, &good);
    try t.expectEqualSlices(u32, &.{ 17, 5 }, &good);
    try t.expectEqual(@as(usize, 32), D.best_bytes);
    reductions[1] = f.reduction(1, std.math.nan(f32), 17);
    for ([_]?mtp.Sampling{ null, .{ .seed = 7, .temperature = 1, .top_k = 2 }, .{ .seed = 7, .temperature = 1, .top_k = 0 } }) |sm| {
        D.reset();
        D.cols = &.{ 17, 5, -1, 5 };
        var out = [_]u32{ 42, 43 };
        try t.expectError(error.ProposalNonfinite, draw.draws(&h, st, a, &x, sm, &.{}, &out));
        try t.expectEqualSlices(u32, &.{ 42, 43 }, &out);
        try t.expectEqual(@as(usize, 32), D.best_bytes);
        try t.expectEqual(@as(usize, if (sm != null and sm.?.top_k == 2) 2 else 1), D.waits);
        if (sm != null and sm.?.top_k == 2) {
            try joinedBeforeWait(false);
        }
    }
}

test "Flash Next nonfinite: a later bad candidate leaves the entire verify output unchanged" {
    var driver = D.driver();
    defer cuda.download_stage.closeDriver(&driver);
    var h = try f.head(&driver);
    defer h.artifact.deinit();
    const host = try a.alloc(u16, 2 * wt.vocab);
    defer a.free(host);
    var x = f.draft(&driver, host);
    x.rows = 2;
    D.reductions = &.{ f.reduction(1, 2, 17), f.reduction(1, 2, 17) };
    D.vals = &.{ 0x3f80, 0, 0x3f80, 0 };
    D.cols = &.{ 17, 5, 17, -1 };
    var out = [_]u32{ 42, 43 };
    const st: cuda.Stream = .{ .d = &driver, .handle = null };
    try t.expectError(error.ProposalNonfinite, draw.draws(&h, st, a, &x, .{ .seed = 7, .temperature = 1, .top_k = 2 }, &.{}, &out));
    try t.expectEqualSlices(u32, &.{ 42, 43 }, &out);
    try t.expectEqual(@as(usize, 2), D.waits);
}

test "Flash Next nonfinite: finite head sampled candidates and full rows match the original draw and probability" {
    var driver = D.driver();
    defer cuda.download_stage.closeDriver(&driver);
    var h = try f.head(&driver);
    defer h.artifact.deinit();
    const host = try a.alloc(u16, mtp.max_rows * draw.topk_max * 3);
    defer a.free(host);
    var x = f.draft(&driver, host);
    const st: cuda.Stream = .{ .d = &driver, .handle = null };
    const row = [_]u16{ 0x3f80, 0, 0xbf80 };
    const best = f.reduction(1, 2, 0);
    for ([_]u32{ 0, 2 }) |k| for (0..20) |position| {
        const sm: mtp.Sampling = .{ .seed = 743, .temperature = 0.8, .top_k = k };
        const expected = try mtp.drawPick(a, &row, h.ids, best[1], position, sm);
        for ([_]bool{ false, true }) |batched| {
            D.reset();
            const got = try draw.pick(&h, st, a, &x, best, position, sm, batched);
            try t.expectEqualDeep(expected, got);
            try t.expectEqual(@as(usize, 1), D.waits);
        }
    };
}

test "Flash Next nonfinite: finite verify top-k and full-row draws match keyed original rows with one wait" {
    var driver = D.driver();
    defer cuda.download_stage.closeDriver(&driver);
    var h = try f.head(&driver);
    defer h.artifact.deinit();
    const host = try a.alloc(u16, wt.vocab);
    defer a.free(host);
    const row = try a.alloc(u16, wt.vocab);
    defer a.free(row);
    @memset(row, 0xbf80);
    row[5] = 0;
    row[17] = 0x3f80;
    D.verify_rows = row;
    D.best = f.reduction(1, 2, 17);
    D.cols = &.{ 17, 5 };
    var x = f.draft(&driver, host);
    const st: cuda.Stream = .{ .d = &driver, .handle = null };
    for ([_]u32{ 0, 2 }) |k| for (0..10) |position| {
        const sm: mtp.Sampling = .{ .seed = 43, .temperature = 0.7, .top_k = k };
        x.fed0 = 2050 + position;
        const expected = try mtp.drawRow(a, row, null, x.fed0 + 1, sm);
        D.reset();
        var out: [1]u32 = undefined;
        try draw.draws(&h, st, a, &x, sm, &.{}, &out);
        try t.expectEqual(expected, out[0]);
        try t.expectEqual(@as(usize, 1), D.waits);
        if (k == 2) {
            try joinedBeforeWait(false);
            try t.expectEqual(@as(usize, 1), D.launches);
            try t.expectEqual(@as(usize, 0), D.best_calls);
            try t.expectEqual(@as(u32, 1), D.grids[0]);
        }
    };
}

test "Flash Next nonfinite: masks preserve single-column and empty draws and ignore excluded NaNs and infinity" {
    var driver = D.driver();
    defer cuda.download_stage.closeDriver(&driver);
    var h = try f.head(&driver);
    defer h.artifact.deinit();
    const host = try a.alloc(u16, wt.vocab);
    defer a.free(host);
    const original = try a.alloc(u16, wt.vocab);
    defer a.free(original);
    const masked = try a.alloc(u16, wt.vocab);
    defer a.free(masked);
    @memset(original, 0);
    original[17] = 0x3f80;
    original[5] = 0x7fc1;
    original[7] = 0x7f80;
    D.verify_rows = original;
    D.best = f.reduction(std.math.inf(f32), std.math.nan(f32), 7);
    var x = f.draft(&driver, host);
    const st: cuda.Stream = .{ .d = &driver, .handle = null };
    for ([_]u32{ 0, 1 << 17 }) |bits| for ([_]?mtp.Sampling{ null, .{ .seed = 19, .temperature = 1, .top_k = 2 }, .{ .seed = 19, .temperature = 1, .top_k = 0 } }) |sm| {
        @memcpy(masked, original);
        mtp.maskRow(masked, &.{bits});
        const expected = mtp.drawRow(a, masked, null, x.fed0 + 1, sm);
        D.reset();
        var out = [_]u32{42};
        const actual = draw.draws(&h, st, a, &x, sm, &.{bits}, &out);
        if (expected) |token| {
            try actual;
            try t.expectEqual(token, out[0]);
        } else |err| try t.expectError(err, actual);
        try t.expectEqual(@as(usize, 1), D.waits);
        try t.expectEqual(@as(usize, 0), D.launches);
    };
    var out = [_]u32{42};
    try t.expectError(error.ProposalNonfinite, draw.draws(&h, st, a, &x, null, &.{1 << 5}, &out));
    try t.expectEqual(@as(u32, 42), out[0]);
}

test "Flash Next nonfinite: candidate shape refuses empty mismatched and oversized rows" {
    const sm: mtp.Sampling = .{ .seed = 7, .temperature = 1, .top_k = 2 };
    try t.expectError(error.ProposalScoreShape, draw.drawTopk(a, &.{}, &.{}, null, 1, sm));
    try t.expectError(error.ProposalScoreShape, draw.drawTopk(a, &.{0x3f80}, &.{}, null, 1, sm));
    const vals: [draw.topk_max + 1]u16 = @splat(0x3f80);
    const cols: [draw.topk_max + 1]i32 = @splat(0);
    try t.expectError(error.ProposalScoreShape, draw.drawTopk(a, &vals, &cols, null, 1, sm));
}

test "Flash Next nonfinite: folded NaN and infinity sentinels fetch best only after failure" {
    var driver = D.driver();
    defer cuda.download_stage.closeDriver(&driver);
    var h = try f.head(&driver);
    defer h.artifact.deinit();
    const host = try a.alloc(u16, wt.vocab);
    defer a.free(host);
    const row = try a.alloc(u16, wt.vocab);
    defer a.free(row);
    @memset(row, 0);
    var x = f.draft(&driver, host);
    const st: cuda.Stream = .{ .d = &driver, .handle = null };
    for ([_]u16{ 0xffc1, 0x7fc1, 0x7f80 }) |bits| {
        row[7] = bits;
        D.verify_rows = row;
        D.vals = &.{ 0x3f80, 0 };
        D.cols = &.{ -1, 5 };
        D.best = f.reduction(1, std.math.nan(f32), 17);
        D.reset();
        var out = [_]u32{42};
        try t.expectError(error.ProposalNonfinite, draw.draws(&h, st, a, &x, .{ .seed = 71, .temperature = 1, .top_k = 2 }, &.{}, &out));
        try t.expectEqual(@as(u32, 42), out[0]);
        try t.expectEqualSlices(usize, &.{ 2, 1 }, D.functions[0..D.launches]);
        try t.expectEqualSlices(u32, &.{ 1, 1 }, D.grids[0..D.launches]);
        try joinedBeforeWait(false);
        try t.expectEqual(@as(usize, 2), D.waits);
    }
}

test "Flash Next batch draws: four draft steps preserve eight serial chains and every probability" {
    var driver = D.driver();
    defer cuda.download_stage.closeDriver(&driver);
    var ids: [32]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i * 3);
    var h = try f.headIds(&driver, &ids);
    defer h.artifact.deinit();
    var vals: [8 * 20]u16 = undefined;
    var cols: [8 * 20]i32 = undefined;
    for (0..8) |r| for (0..20) |i| {
        vals[r * 20 + i] = @intCast(@as(u32, @bitCast(@as(f32, @floatFromInt(i)) / 10)) >> 16);
        cols[r * 20 + i] = @intCast(i);
    };
    D.vals = &vals;
    D.cols = &cols;
    D.best = f.reduction(mtp.promote(vals[19]), 4, 19);
    var host: [8][mtp.max_rows * draw.topk_max * 3]u16 = undefined;
    var ds: [8]mtp.Draft = undefined;
    var refs: [8]*mtp.Draft = undefined;
    var sms: [8]?mtp.Sampling = undefined;
    const nexts: [8][]const u32 = @splat(&.{11});
    const rows: [8]u64 = @splat(0);
    var positions: [8]u64 = undefined;
    var baseline: [8]mtp.Chain = @splat(.{});
    var got: [8]mtp.Chain = @splat(.{});
    var outs: [8]*mtp.Chain = undefined;
    const st: cuda.Stream = .{ .d = &driver, .handle = null };
    var e: prompt.Engine = undefined;
    e.gpa = a;
    var calls: usize = 0;
    var lanes: usize = 0;
    const flow: Flow = .{ .e = &e, .h = &h, .st = st, .calls = &calls, .lanes = &lanes };
    for (0..2) |side| {
        for (&ds, &refs, &sms, &positions, &outs, 0..) |*d, *ref, *sm, *pos, *out, i| {
            d.* = f.draft(&driver, &host[i]);
            d.dlogits.ptr += i * 0x100;
            d.dlogits.len = ids.len * 2;
            ref.* = d;
            sm.* = if (i < 6) .{ .seed = 91 + i * 11, .temperature = 0.7, .top_p = 0.95, .top_k = 20 } else null;
            pos.* = 100 + i * 31;
            out.* = if (side == 0) &baseline[i] else &got[i];
        }
        h.draw_batch.on = side == 1;
        D.reset();
        calls = 0;
        try ops.draftMany(flow, &refs, &nexts, &rows, &@as([8]usize, @splat(4)), &positions, &sms, &@as([8]f64, @splat(0)), &outs);
        try t.expectEqual(@as(usize, 4), calls);
        for (outs) |out| try t.expectEqual(@as(usize, 4), out.n);
        if (side == 1) {
            try t.expectEqual(@as(usize, 8), D.launches);
            try t.expectEqual(@as(usize, 4), D.waits);
            try t.expectEqual(@as(usize, 4), D.reads);
            try t.expectEqual(@as(usize, 32), D.copies);
            for (0..4) |step| {
                try t.expectEqualSlices(usize, &.{ 1, 2 }, D.functions[step * 2 ..][0..2]);
                try t.expectEqualSlices(u32, &.{ 8, 8 }, D.grids[step * 2 ..][0..2]);
                for (0..8) |r| {
                    const i = step * 8 + r;
                    try t.expectEqual(ds[r].dlogits.ptr, D.copy_src[i]);
                    try t.expectEqual(h.draw_batch.scores.ptr + r * ids.len * 2, D.copy_dst[i]);
                    try t.expectEqual(ids.len * 2, D.copy_bytes[i]);
                }
            }
        }
    }
    for (baseline, got) |old, new| {
        try t.expectEqualSlices(u32, old.tokens[0..old.n], new.tokens[0..new.n]);
        try t.expectEqualSlices(f64, old.probs[0..old.n], new.probs[0..new.n]);
    }
}

test "Flash Next batch draws: lone sampled pick joins both launches before one read, fallback keeps two waits" {
    var driver = D.driver();
    defer cuda.download_stage.closeDriver(&driver);
    var h = try f.head(&driver);
    defer h.artifact.deinit();
    var host: [mtp.max_rows * draw.topk_max * 3]u16 = undefined;
    var d = f.draft(&driver, &host);
    const st: cuda.Stream = .{ .d = &driver, .handle = null };
    const sm: mtp.Sampling = .{ .seed = 73, .temperature = 0.7, .top_k = 2 };
    for (0..20) |pos| {
        const expected = try mtp.drawPick(a, &.{ 0x3f80, 0, 0xbf80 }, h.ids, D.best[1], pos, sm);
        for ([_]bool{ false, true }) |on| {
            D.reset();
            const got = try @import("cuda_draft_draw.zig").single(&h, st, a, &d, pos, sm, on);
            try t.expectEqualDeep(expected, got);
            try t.expectEqual(@as(usize, if (on) 1 else 2), D.waits);
            try t.expectEqual(@as(usize, if (on) 1 else 3), D.reads);
            if (on) try joinedBeforeWait(true);
        }
    }
}
