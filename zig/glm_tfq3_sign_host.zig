//! Record the real TFQ3 planner, run metadata and Metal dispatch without a device.
const std = @import("std");
const tfq3 = @import("src/families/glm/tfq3.zig");
const affine = @import("src/families/glm/affine.zig");
const wide = @import("src/families/glm/wide.zig");
const mtl = @import("metal");
const t = std.testing;

fn planned(sizes: []const u32, k: u32) !affine.Affine {
    var metas: [affine.max_runs]tfq3.NonMoeTensorMeta = undefined;
    var pws: [affine.max_runs]?*const tfq3.NonMoeTensorMeta = @splat(null);
    const widths: [affine.max_runs]affine.Width = @splat(.{ .bits = 4, .group = 512 });
    var words: u64 = 0;
    var scales: u32 = 0;
    var total: u32 = 0;
    for (sizes, 0..) |n, i| {
        metas[i] = std.mem.zeroes(tfq3.NonMoeTensorMeta);
        metas[i].R = n;
        metas[i].C = k;
        metas[i].b_bits = 4;
        metas[i].word_off = words;
        metas[i].word_count = @as(u64, n) * k / 8;
        metas[i].sc_off = scales;
        words += metas[i].word_count;
        scales += (n / 128) * (k / 128);
        total += n;
        pws[i] = &metas[i];
    }
    var starts: [affine.max_runs + 1]usize = undefined;
    const count = tfq3.planRuns(sizes.len, pws[0..sizes.len], widths[0..sizes.len], true, false, &starts);
    try t.expectEqual(@as(usize, 1), count);
    const nm: tfq3.NonMoe = .{ .tables = .{ .id = 3 }, .scales = .{ .id = 2 }, .weights = .{ .id = 1 }, .metas = &.{}, .bytes = 0 };
    var out: affine.Affine = undefined;
    var row: usize = 0;
    tfq3.emitRun(&nm, pws[starts[0]..starts[1]], &out, 0, &row, total, k);
    out.n = total;
    out.more = 0;
    return out;
}

fn pipes() tfq3.Pipes {
    var p: tfq3.Pipes = undefined;
    for (&p.list, 0..) |*v, i| v.* = @intCast(i);
    p.xrot_buf = .{ .id = 4 };
    p.yr_buf = .{ .id = 5 };
    return p;
}

test "TFQ3 signs: real planner and emitter keep standalone absolute and every fused boundary" {
    for ([_][]const u32{ &.{154880}, &.{20480}, &.{4096}, &.{ 8192, 8192, 8192 }, &.{ 16384, 4096 }, &.{ 2048, 2048 } }) |sizes| {
        const q = try planned(sizes, if (sizes[0] == 16384 or sizes[0] == 20480) 1536 else 4096);
        var want: [affine.max_runs - 1]u32 = @splat(0);
        var end: u32 = 0;
        for (sizes[0 .. sizes.len - 1], 0..) |n, i| {
            end += n;
            want[i] = end;
        }
        try t.expectEqualSlices(u32, &want, &q.run(0).tfq3_ends);
        const p = pipes();
        for (1..17) |rows| for ([_]bool{ false, true }) |prompt| {
            var rec: mtl.Record = .{};
            tfq3.dispatch.dense(&p, .{ .r = &rec }, q.w, q.s, q.b, q.bits, .{ .buf = .{ .id = 6 } }, .{ .buf = .{ .id = 7 } }, q.n, q.k, @intCast(rows), q.k, q.n, prompt, q.run(0).tfq3_ends);
            try t.expect(rec.n >= 2);
            for (rec.passes[0..rec.n]) |pass| for (pass.lengths, 0..) |len, at| {
                if (len == 0) continue;
                try t.expectEqual(@sizeOf(tfq3.DenseDims), len);
                const d = std.mem.bytesToValue(tfq3.DenseDims, pass.values[at][0..@sizeOf(tfq3.DenseDims)]);
                try t.expectEqualSlices(u32, &want, &d.sr_ends);
            };
        };
    }
}

test "TFQ3 signs: specialized KDA forwards the actual qkv boundaries at every decode width" {
    var q = try planned(&.{ 8192, 8192, 8192 }, 4096);
    q.n += 320;
    q.more = 1;
    q.runs[0] = .{ .w = q.w, .s = q.s, .b = q.b, .row = 24576, .n = 320, .bits = 8, .group = 64 };
    const p = pipes();
    for (1..17) |rows| {
        var rec: mtl.Record = .{};
        try t.expect(tfq3.dispatch.tryKdaIn(&p, .{ .r = &rec }, q, .{ .buf = .{ .id = 6 } }, .{ .buf = .{ .id = 7 } }, @intCast(rows), 4096, q.n));
        const last = rec.passes[rec.n - 1];
        const d = std.mem.bytesToValue(tfq3.DenseDims, last.values[2][0..@sizeOf(tfq3.DenseDims)]);
        try t.expectEqualSlices(u32, &.{ 8192, 16384, 0, 0, 0 }, &d.sr_ends);
    }
}

test "TFQ3 signs: shader uses explicit boundaries in inverse, fused decode and prefill" {
    const common = @embedFile("kernels/metal/glm_tfq3/common.metal");
    try t.expect(std.mem.indexOf(u8, common, "r0 & 8191u") == null);
    try t.expect(std.mem.indexOf(u8, common, "r0 - start") != null);
    try t.expect(std.mem.indexOf(u8, common, "uint  sr_ends[5]") != null);
    for ([_][]const u8{ common, @embedFile("kernels/metal/glm_tfq3/dense_decode.metal"), @embedFile("kernels/metal/glm_tfq3/prefill.metal") }) |source| {
        try t.expect(std.mem.indexOf(u8, source, "pw_sr0(r0, d)") != null);
        try t.expect(std.mem.indexOf(u8, source, "pw_sr0(r0, d.n, d.k)") == null);
    }
    try t.expectEqual(@as(usize, 52), @sizeOf(tfq3.DenseDims));
    try t.expectEqual(@as(usize, 32), @offsetOf(tfq3.DenseDims, "sr_ends"));
}

fn localRow(ends: [5]u32, row: u32) u32 {
    var start: u32 = 0;
    for (ends) |end| {
        if (end == 0 or row < end) break;
        start = end;
    }
    return row - start;
}

test "TFQ3 signs: actual denseMm prompt fallback retains each run's decode signs at every width" {
    for ([_][]const u32{ &.{154880}, &.{20480}, &.{4096}, &.{ 8192, 8192, 8192 }, &.{ 16384, 4096 }, &.{ 2048, 2048 } }) |sizes| {
        const fused = try planned(sizes, if (sizes[0] == 16384 or sizes[0] == 20480) 1536 else 4096);
        for ([_]bool{ false, true }) |second| {
            var q = fused;
            const offset: u32 = if (second) 128 else 0;
            if (second) {
                q.more = 1;
                q.n += offset;
                q.tfq3_ends = @splat(0);
                q.runs[0] = fused.run(0);
                q.runs[0].row = offset;
            }
            var p = std.mem.zeroes(wide.Pipes);
            p.tfq3 = pipes();
            try t.expect(wide.mm(&p, fused) == null);
            for (1..17) |rows| {
                const io: wide.Io = .{ .in = .{ .buf = .{ .id = 6 } }, .out = .{ .buf = .{ .id = 7 }, .off = 512 }, .ldx = q.k, .ldo = q.n, .rows = @intCast(rows) };
                var decode: mtl.Record = .{};
                wide.dense(&p, .{ .r = &decode }, q, io);
                var prompt: mtl.Record = .{};
                wide.denseMm(&p, .{ .r = &prompt }, q, io, .{ .buf = .{ .id = 8 } });
                for ([_]*const mtl.Record{ &decode, &prompt }) |rec| {
                    var found: usize = 0;
                    for (rec.passes[0..rec.n]) |pass| for (pass.lengths, 0..) |len, at| {
                        if (len != @sizeOf(tfq3.DenseDims)) continue;
                        const d = std.mem.bytesToValue(tfq3.DenseDims, pass.values[at][0..@sizeOf(tfq3.DenseDims)]);
                        if (d.n != fused.n) continue;
                        found += 1;
                        try t.expectEqualSlices(u32, &fused.tfq3_ends, &d.sr_ends);
                        try t.expectEqual(q.n, d.ldo);
                        for ([_]u32{ 2047, 2048, 8191, 8192, 16383, 16384, fused.n - 1 }) |row| {
                            if (row >= fused.n) continue;
                            const expected = localRow(fused.tfq3_ends, row);
                            const got = localRow(d.sr_ends, row);
                            const want_sign = (0x9e3779b9 ^ expected) *% @as(u32, 2654435761);
                            const got_sign = (0x9e3779b9 ^ got) *% @as(u32, 2654435761);
                            try t.expectEqual(want_sign >> 31, got_sign >> 31);
                        }
                        const out_index: usize = if (pass.pipe == p.tfq3.?.get(.fwht_y_gen)) 1 else 4;
                        if (pass.pipe != p.tfq3.?.get(.fwht_x_gen) and pass.buffers[out_index].id == 7) try t.expectEqual(@as(usize, 512 + offset * 2), pass.buffers[out_index].off);
                    };
                    try t.expect(found > 0);
                }
            }
        }
    }
}
