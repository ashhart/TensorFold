//! Actual shared encoders against independent release ABI and launch constants, with inert Metal recording.
const std = @import("std");
const mtl = @import("metal");
const am = @import("src/core/affine_mm.zig");
const lane = @import("src/core/lane_projection.zig");
const ks = @import("kernel_sources");
const R = struct { buf: mtl.Buffer, off: usize };
fn ref(id: usize) R {
    return .{ .buf = .{ .id = id }, .off = id * 256 };
}
const Q = struct { w: R, s: R, b: R, n: u32 = 128, k: u32 = 256 };
fn weights() Q {
    return .{ .w = ref(2), .s = ref(3), .b = ref(4) };
}
fn pipes() am.Pipes {
    var p: am.Pipes = undefined;
    for (&p, 0..) |*v, i| v.* = @intCast(100 + i);
    return p;
}
fn val(p: mtl.Pass, comptime T: type, i: usize) !T {
    try std.testing.expectEqual(@sizeOf(T), p.lengths[i]);
    return std.mem.bytesToValue(T, p.values[i][0..@sizeOf(T)]);
}
fn bound(p: mtl.Pass, slot: usize, id: usize) !void {
    try std.testing.expectEqual(mtl.Bound{ .id = id, .off = id * 256 }, p.buffers[slot]);
}
test "shared affine release ABI and entry ordering" {
    const expected = [_][]const u8{ "tf_affine_row_sums", "tf_affine_mm", "tf_affine_gather_32", "tf_affine_gather_64", "tf_affine_gather_scatter_32", "tf_affine_gather_scatter_64", "tf_affine_gather_glu_32", "tf_affine_gather_glu_64" };
    try std.testing.expectEqual(@as(usize, 8), am.names.len);
    for (am.names, 0..) |name, i| try std.testing.expectEqualStrings(expected[i], name);
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(am.Args));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(am.Strides));
}
test "shared affine row sums dense pitches batches and bindings match release" {
    try std.testing.expectEqual(@as(usize, 8), am.names.len);
    if (comptime am.names.len == 8) {
        var r: mtl.Record = .{};
        const e = mtl.ComputeEncoder{ .r = &r };
        am.rowSumsAt(e, pipes(), 64, ref(1), ref(5), 70, 256, 768);
        const s = am.Strides{ .x_row = 768, .y_row = 384, .sums_row = 3, .x_batch = 256, .y_batch = 128, .sums_batch = 1, .w_batch = 128 };
        am.denseBatch(e, pipes(), ref(1), ref(5), weights(), ref(6), 70, 3, s);
        try std.testing.expectEqual(@as(usize, 2), r.n);
        const a = r.passes[0];
        const d = r.passes[1];
        try std.testing.expectEqual(@as(u32, 100), a.pipe);
        try bound(a, 0, 1);
        try bound(a, 6, 5);
        try std.testing.expectEqual(am.Args{ .rows = 70, .n = 768, .k = 256, .experts = 0 }, try val(a, am.Args, 5));
        try std.testing.expectEqual(mtl.Size.of(4, 70, 1), a.grid);
        try std.testing.expectEqual(mtl.Size.of(4, 1, 1), a.threads);
        try std.testing.expect(a.direct);
        try std.testing.expectEqual(@as(u32, 101), d.pipe);
        for ([_]usize{ 1, 2, 3, 4, 5 }, 0..) |id, i| try bound(d, i, id);
        try bound(d, 6, 6);
        try std.testing.expectEqual(am.Args{ .rows = 70, .n = 128, .k = 256, .experts = 0 }, try val(d, am.Args, 5));
        try std.testing.expectEqual(s, try val(d, am.Strides, 7));
        try std.testing.expectEqual(mtl.Size.of(2, 2, 3), d.grid);
        try std.testing.expectEqual(mtl.Size.of(32, 2, 2), d.threads);
        try std.testing.expect(!d.direct);
        r = .{};
        var q = weights();
        q.n = 65;
        am.dense(e, pipes(), ref(1), ref(5), q, ref(6), 1);
        try std.testing.expectEqual(am.Strides{ .x_row = 256, .y_row = 128, .sums_row = 1, .x_batch = 0, .y_batch = 0, .sums_batch = 0, .w_batch = 0, .pad = 0 }, try val(r.passes[0], am.Strides, 7));
        try std.testing.expectEqual(am.Args{ .rows = 1, .n = 128, .k = 256, .experts = 0 }, try val(r.passes[0], am.Args, 5));
        try bound(r.passes[0], 4, 5);
    }
}
test "shared affine expert scatter and GLU thresholds match release" {
    try std.testing.expectEqual(@as(usize, 8), am.names.len);
    if (comptime am.names.len == 8) {
        for ([_]u32{ 63, 64, 199, 200 }) |average| {
            const rows = average * 2;
            var r: mtl.Record = .{};
            const e = mtl.ComputeEncoder{ .r = &r };
            am.gatherTo(e, pipes(), ref(1), ref(5), weights(), ref(7), ref(6), ref(8), rows, 2);
            am.gather(e, pipes(), ref(1), ref(5), weights(), ref(7), ref(6), rows, 2);
            am.gatherGlu(e, pipes(), ref(1), ref(5), weights(), weights(), ref(7), ref(8), ref(6), rows, 2, 4, 7.0);
            const bm: u32 = if (average < 64) 32 else 64;
            for (r.passes[0..2], 0..) |p, i| {
                try std.testing.expectEqual(@as(u32, if (average < 64) 102 else 103) + @as(u32, if (i == 0) 2 else 0), p.pipe);
                try bound(p, 4, 5);
                try bound(p, 7, 7);
                if (i == 0) try bound(p, 8, 8);
                try std.testing.expectEqual(am.Args{ .rows = @intCast(rows), .n = 128, .k = 256, .experts = 2 }, try val(p, am.Args, 5));
                try std.testing.expectEqual(mtl.Size.of(2, @min(rows, (rows + bm - 1) / bm + 1), 1), p.grid);
            }
            const p = r.passes[2];
            const gb: u32 = if (average < 200) 32 else 64;
            try std.testing.expectEqual(@as(u32, if (average < 200) 106 else 107), p.pipe);
            try bound(p, 4, 5);
            try bound(p, 7, 7);
            try bound(p, 8, 8);
            try bound(p, 9, 2);
            try bound(p, 10, 3);
            try bound(p, 11, 4);
            try std.testing.expectEqual([2]f32{ 4, 7 }, try val(p, [2]f32, 12));
            try std.testing.expectEqual(mtl.Size.of(4, @min(rows, (rows + gb - 1) / gb + 1), 1), p.grid);
        }
    }
}
test "landed Metal source bytes are pinned without shader compilation" {
    {
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(ks.core_affine_mm, &hash, .{});
        try std.testing.expectEqualStrings("d8f47cbbddf24502ca2c16cfdef0672731cc2dbe59ad027f47a3e662bcf8f627", &std.fmt.bytesToHex(hash, .lower));
    }
    {
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(ks.core_lane_projection, &hash, .{});
        try std.testing.expectEqualStrings("d96f3c86926177d9fe6df6c82beea7e4f379b16654aa835200d50afe182f3c99", &std.fmt.bytesToHex(hash, .lower));
    }
}
test "released lane parameters and conservative scratch size" {
    const l = lane.Layout{ .n = 128, .k = 256, .format = .{ .bits = 4, .group = 64 }, .sk = 4, .reg = true, .precompute_sums = true };
    try std.testing.expectEqual(@as(usize, 14336), l.threadgroupBytes());
    const text = try lane.source(std.testing.allocator, l);
    defer std.testing.allocator.free(text);
    try std.testing.expect(std.mem.endsWith(u8, text, ks.core_lane_projection));
    try std.testing.expect(std.mem.indexOf(u8, text, "TF_FUSED") == null);
}
test "shared lane encode and supplied sums dispatch match released parameters" {
    var r: mtl.Record = .{};
    const e = mtl.ComputeEncoder{ .r = &r };
    const p = lane.Projection{ .pipe = 211, .n = 128, .k = 256, .group_count = 4, .weight_bytes = 16384, .metadata_bytes = 2048, .output_bytes = 2, .y_stride = 192, .tile_count = 4, .sk = 4, .sums_pipe = 210, .sums_buffer = .{ .id = 9 }, .reg = true };
    try p.encode(e, .{ .buf = .{ .id = 1 } }, .{ .buf = .{ .id = 2 } }, .{ .buf = .{ .id = 3 } }, .{ .buf = .{ .id = 4 } }, 70);
    try std.testing.expectEqual(@as(usize, 2), r.n);
    const sums = r.passes[0];
    const dense = r.passes[1];
    try std.testing.expectEqual(@as(u32, 210), sums.pipe);
    try std.testing.expectEqual(mtl.Size.of(320, 1, 1), sums.grid);
    try std.testing.expectEqual(mtl.Size.of(256, 1, 1), sums.threads);
    try std.testing.expectEqual(@as(i32, 70), try val(sums, i32, 1));
    try std.testing.expectEqual(@as(i32, 80), try val(sums, i32, 3));
    try std.testing.expectEqual(@as(u32, 211), dense.pipe);
    try std.testing.expectEqual([2]i32{ 70, 80 }, try val(dense, [2]i32, 3));
    try std.testing.expectEqual(mtl.Size.of(4, 5, 1), dense.grid);
    try std.testing.expectEqual(mtl.Size.of(128, 1, 1), dense.threads);
    try std.testing.expectEqual(@as(usize, 9), dense.buffers[5].id);
    r = .{};
    try p.encodeSums(e, .{ .buf = .{ .id = 1 } }, .{ .buf = .{ .id = 2 } }, .{ .buf = .{ .id = 3 } }, .{ .ref = .{ .buf = .{ .id = 10 }, .off = 256 }, .stride = 96 }, .{ .buf = .{ .id = 4 } }, 70);
    try std.testing.expectEqual(@as(usize, 1), r.n);
    try std.testing.expectEqual([2]i32{ 70, 96 }, try val(r.passes[0], [2]i32, 3));
    try std.testing.expectEqual(mtl.Bound{ .id = 10, .off = 256 }, r.passes[0].buffers[5]);
}
test "shared source format output GLU and fragment prefixes remain released" {
    const a = std.testing.allocator;
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]u16{ 32, 64 }) |group| for ([_]bool{ false, true }) |out| for ([_]bool{ false, true }) |plain| for ([_]bool{ false, true }) |tensor| {
        const text = try am.source(.{ .tensor = tensor }, a, .{ .bits = bits, .group = group, .out_f32 = out, .glu = if (plain) .plain else .clipped });
        defer a.free(text);
        const head = try std.fmt.allocPrint(a, "#define TF_BITS {d}\n#define TF_GROUP {d}\n#define TF_OUT_T {s}\n{s}{s}", .{ bits, group, if (out) "float" else "bfloat", if (plain) "#define TF_GLU_PLAIN 1\n" else "", if (tensor) "" else "#define TF_SIMD_FRAGS 1\n" });
        defer a.free(head);
        try std.testing.expect(std.mem.startsWith(u8, text, head));
        try std.testing.expect(std.mem.indexOf(u8, text, "tf_affine_row_sums") != null);
    };
}
test "Gemma rounded dense and fused source are separate from shared libraries" {
    const own = @import("src/families/gemma4/affine_mm.zig");
    const fused = @import("src/families/gemma4/fused_lane.zig");
    try std.testing.expectEqual(@as(usize, 1), own.names.len);
    const text = try own.source(.{}, std.testing.allocator, .{});
    defer std.testing.allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "tf_affine_row_sums") == null);
    const l = fused.Layout{ .projection = .{ .n = 256, .k = 256, .format = .{ .bits = 4, .group = 64 }, .reg = true, .precompute_sums = true, .sk = 4 }, .source = ks.gemma4_fused, .defines = "#define TF_FUSE 1\n" };
    const f = try fused.source(std.testing.allocator, l);
    defer std.testing.allocator.free(f);
    try std.testing.expect(std.mem.endsWith(u8, f, ks.gemma4_lane_projection));
    try std.testing.expect(std.mem.indexOf(u8, f, "#define TF_FUSED 1\n") != null);
    var record: mtl.Record = .{};
    const proj = fused.Projection{ .pipe = 300, .layout = l };
    const rr = lane.Ref{ .buf = .{ .id = 1 } };
    try proj.encodeFused(.{ .r = &record }, rr, rr, rr, rr, .{ .a = @splat(rr), .o = @splat(rr) }, 1);
    try std.testing.expectEqual(mtl.Size.of(4, 1, 1), record.passes[0].grid);
    try std.testing.expectEqual(mtl.Size.of(256, 1, 1), record.passes[0].threads);
}

test "gemma4 widths: real Gemma row encoder keeps mixed pitches across 16 row chunks" {
    const family_weights = @import("src/families/gemma4/weights.zig");
    const cfg = @import("src/families/gemma4/config.zig");
    for (cfg.widths.bit_widths) |bits| {
        const x = family_weights.Linear{ .w = .{ .id = 2 }, .sb = .{ .id = 3 }, .n = 32, .k = 256, .format = .{ .bits = bits, .group = 64 }, .col = 32, .y_stride = 128 };
        const l = family_weights.layout(cfg.unified12b, x);
        try std.testing.expectEqual(@as(?usize, 128), l.y_stride);
        const p = lane.Projection{ .pipe = 101, .n = 32, .k = 256, .group_count = 4, .weight_bytes = l.weightWords() * 4, .metadata_bytes = l.metadataElements() * 2, .output_bytes = 2, .y_stride = l.y_stride.?, .tile_count = 1, .sk = l.sk, .reg = l.reg, .sums_pipe = if (l.reg) 100 else null, .sums_buffer = if (l.reg) .{ .id = 5 } else null };
        var r: mtl.Record = .{};
        try x.encodeRows(p, .{ .r = &r }, .{ .buf = .{ .id = 1 }, .off = 16 }, .{ .buf = .{ .id = 4 }, .off = 64 + x.col * 2 }, 33);
        const step: usize = if (l.reg) 2 else 1;
        try std.testing.expectEqual(@as(usize, 3) * step, r.n);
        for (0..3) |part| {
            const pass = r.passes[(part + 1) * step - 1];
            try std.testing.expectEqual(mtl.Bound{ .id = 1, .off = 16 + part * 16 * 256 * 2 }, pass.buffers[0]);
            try std.testing.expectEqual(mtl.Bound{ .id = 4, .off = 64 + 32 * 2 + part * 16 * 128 * 2 }, pass.buffers[4]);
            const rows = try val(pass, [2]i32, 3);
            try std.testing.expectEqual(@as(i32, if (part == 2) 1 else 16), rows[0]);
        }
        var bad = x;
        bad.y_stride = 16;
        try std.testing.expectError(error.InvalidOutputStride, family_weights.layout(cfg.unified12b, bad).validate());
    }
}
