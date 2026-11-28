//! Host tests for W8A16 CUDA weight packing: qmmf BF16/FP8G shares, horizontal concat, and TP=1/2/4 FP8 experts.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const weights = @import("fp8_weights.zig");
const partition = @import("fp8_partition.zig");
const pack_mod = @import("fp8_pack.zig");

const testing = std.testing;
const Tensor = core.checkpoint.Tensor;
const fp8 = cuda.fp8;
const fx8 = cuda.experts.Fp8Experts;

fn tensor(dtype: core.safetensors.DType, shape: []const usize, bytes: []const u8) Tensor {
    var t: Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = bytes };
    @memcpy(t.shape[0..shape.len], shape);
    return t;
}

fn fill(bytes: []u8, seed: u8) void {
    for (bytes, 0..) |*b, i| b.* = @truncate(i *% 131 +% seed);
}

fn bf16Bits(v: f32) u16 {
    return core.quant.convert.bf16(v);
}

fn bf16Val(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

test "bf16 projection shares stack rows and slice 64-element input groups before qmmf packing" {
    const a = testing.allocator;
    const n = 256;
    const k = 192;
    var g: [n * k]u16 = undefined;
    var u: [n * k]u16 = undefined;
    for (&g, &u, 0..) |*x, *y, i| {
        x.* = @truncate(i * 3 + 1);
        y.* = @truncate(i * 5 + 2);
    }
    const gt = tensor(.bf16, &.{ n, k }, std.mem.sliceAsBytes(&g));
    const ut = tensor(.bf16, &.{ n, k }, std.mem.sliceAsBytes(&u));
    const rows: partition.Share = .{ .lo = 128, .len = 128 };
    const cols: partition.Share = .{ .lo = 64, .len = 128 };
    var p = try pack_mod.pack(a, &.{ .{ .proj = .{ .bf16 = gt }, .rows = rows }, .{ .proj = .{ .bf16 = ut }, .rows = rows } }, cols);
    defer p.deinit(a);
    try testing.expect(p.mode == .bf16 and p.n == 256 and p.k == 128 and p.scales.len == 0);
    var want_rows: [256 * 128]u16 = undefined;
    for (0..256) |r| for (0..128) |c| {
        const src = if (r < 128) &g else &u;
        want_rows[r * 128 + c] = src[(rows.lo + r % 128) * k + cols.lo + c];
    };
    const want = try a.alloc(u16, 256 * 128);
    defer a.free(want);
    cuda.qmmf.packBf16(want, &want_rows, 256, 128);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), p.codes);
}

test "block-FP8 projection shares repack codes and widen bf16 block scales for single, padded, and stacked parts" {
    const a = testing.allocator;
    const n = 384;
    const k = 256;
    var codes: [2][n * k]u8 = undefined;
    fill(&codes[0], 3);
    fill(&codes[1], 77);
    var inv: [2][3 * 2]u16 = undefined;
    for (&inv, 0..) |*s, p| for (s, 0..) |*v, i| {
        v.* = bf16Bits(@as(f32, @floatFromInt(i + 1 + 10 * p)) * 0.25);
    };
    var parts: [2]weights.Proj = undefined;
    for (&parts, 0..) |*q, p| {
        q.* = .{ .fp8 = .{
            .weight = tensor(.f8_e4m3, &.{ n, k }, &codes[p]),
            .scale_inv = tensor(.bf16, &.{ 3, 2 }, std.mem.sliceAsBytes(&inv[p])),
        } };
    }
    const rows: partition.Share = .{ .lo = 128, .len = 130 };
    const cols: partition.Share = .{ .lo = 128, .len = 128 };
    // Single non-block-aligned slice pads trailing code rows with 0 and trailing scale rows with 1.0.
    var padded_one = try pack_mod.pack(a, &.{.{ .proj = parts[0], .rows = rows }}, cols);
    defer padded_one.deinit(a);
    try testing.expectEqual(@as(usize, 130), padded_one.n);
    try testing.expectEqual(@as(usize, fp8.padded(130) * 128), padded_one.codes.len);
    var sub: [130 * 128]u8 = undefined;
    for (0..130) |r| @memcpy(sub[r * 128 ..][0..128], codes[0][(rows.lo + r) * k + cols.lo ..][0..128]);
    var want_codes: [256 * 128]u8 = undefined;
    fp8.packCodes(&want_codes, &sub, 130, 128);
    try testing.expectEqualSlices(u8, &want_codes, padded_one.codes);
    var whole: [6]f32 = undefined;
    for (&whole, inv[0]) |*w, b| w.* = bf16Val(b);
    var want_scales: [256 * 2]f32 = undefined;
    fp8.packScalesAt(&want_scales, &whole, n, k, rows.lo, cols.lo, 130, 128);
    try testing.expectEqualSlices(f32, &want_scales, padded_one.scales);
    try testing.expectEqual(@as(f32, 4 * 0.25), padded_one.scales[0]);
    try testing.expectEqual(@as(f32, 1.0), padded_one.scales[(2 * 2 + 0) * 64 + 2]);

    // Stacked parts: each row's 64-input group reads its own part's 128x128 block scale.
    const r0: partition.Share = .{ .lo = 128, .len = 128 };
    const r1: partition.Share = .{ .lo = 256, .len = 128 };
    var two = try pack_mod.pack(a, &.{ .{ .proj = parts[0], .rows = r0 }, .{ .proj = parts[1], .rows = r1 } }, cols);
    defer two.deinit(a);
    try testing.expectEqual(@as(usize, 256), two.n);
    for (0..256) |r| for (0..2) |gi| {
        const p = r / 128;
        const src_row = if (p == 0) r0.lo + r else r1.lo + (r - 128);
        const want_s = bf16Val(inv[p][(src_row / 128) * 2 + (cols.lo + gi * 64) / 128]);
        try testing.expectEqual(want_s, two.scales[(r / 64 * 2 + gi) * 64 + r % 64]);
    };
}

test "stacked block-FP8 slices crossing 128x128 scale blocks in both dimensions hit every block transition" {
    const a = testing.allocator;
    const n = 256;
    const k = 256;
    var codes: [2][n * k]u8 = undefined;
    fill(&codes[0], 11);
    fill(&codes[1], 43);
    const inv: [2][4]u16 = .{
        .{ bf16Bits(1.0), bf16Bits(2.0), bf16Bits(3.0), bf16Bits(4.0) },
        .{ bf16Bits(5.0), bf16Bits(6.0), bf16Bits(7.0), bf16Bits(8.0) },
    };
    const p0: weights.Proj = .{ .fp8 = .{
        .weight = tensor(.f8_e4m3, &.{ n, k }, &codes[0]),
        .scale_inv = tensor(.bf16, &.{ 2, 2 }, std.mem.sliceAsBytes(&inv[0])),
    } };
    const p1: weights.Proj = .{ .fp8 = .{
        .weight = tensor(.f8_e4m3, &.{ n, k }, &codes[1]),
        .scale_inv = tensor(.bf16, &.{ 2, 2 }, std.mem.sliceAsBytes(&inv[1])),
    } };
    // Two 64-row slices starting at rows 100 and 127 with columns [64, 192) cross row=128 and col=128 boundaries.
    const r0: partition.Share = .{ .lo = 100, .len = 64 };
    const r1: partition.Share = .{ .lo = 127, .len = 64 };
    const cols: partition.Share = .{ .lo = 64, .len = 128 };
    var packed_w = try pack_mod.pack(a, &.{ .{ .proj = p0, .rows = r0 }, .{ .proj = p1, .rows = r1 } }, cols);
    defer packed_w.deinit(a);
    try testing.expectEqual(@as(usize, 128), packed_w.n);
    try testing.expectEqual(@as(usize, 128), packed_w.k);
    try testing.expectEqual(@as(usize, 256), packed_w.scales.len);

    // Explicit transitions: tile 0 crosses row 128 at index 27->28; tile 1 crosses row 128 at index 0->1.
    try testing.expectEqual(@as(f32, 1.0), packed_w.scales[0 * 64 + 27]);
    try testing.expectEqual(@as(f32, 3.0), packed_w.scales[0 * 64 + 28]);
    try testing.expectEqual(@as(f32, 2.0), packed_w.scales[1 * 64 + 27]);
    try testing.expectEqual(@as(f32, 4.0), packed_w.scales[1 * 64 + 28]);
    try testing.expectEqual(@as(f32, 5.0), packed_w.scales[2 * 64 + 0]);
    try testing.expectEqual(@as(f32, 7.0), packed_w.scales[2 * 64 + 1]);
    try testing.expectEqual(@as(f32, 6.0), packed_w.scales[3 * 64 + 0]);
    try testing.expectEqual(@as(f32, 8.0), packed_w.scales[3 * 64 + 1]);

    var want_scales: [256]f32 = undefined;
    for (0..64) |i| {
        want_scales[0 * 64 + i] = if (i < 28) 1.0 else 3.0;
        want_scales[1 * 64 + i] = if (i < 28) 2.0 else 4.0;
        want_scales[2 * 64 + i] = if (i < 1) 5.0 else 7.0;
        want_scales[3 * 64 + i] = if (i < 1) 6.0 else 8.0;
    }
    try testing.expectEqualSlices(f32, &want_scales, packed_w.scales);

    var raw: [128 * 128]u8 = undefined;
    for (0..64) |i| {
        @memcpy(raw[i * 128 ..][0..128], codes[0][(r0.lo + i) * k + cols.lo ..][0..128]);
        @memcpy(raw[(64 + i) * 128 ..][0..128], codes[1][(r1.lo + i) * k + cols.lo ..][0..128]);
    }
    var want_codes: [128 * 128]u8 = undefined;
    fp8.packCodes(&want_codes, &raw, 128, 128);
    try testing.expectEqualSlices(u8, &want_codes, packed_w.codes);
}

test "packHConcat concatenates two bf16 column slices and packs for qmmf" {
    const a = testing.allocator;
    var left: [128 * 96]u16 = undefined;
    var right: [128 * 96]u16 = undefined;
    for (&left, &right, 0..) |*x, *y, i| {
        x.* = @truncate(i * 7 + 3);
        y.* = @truncate(i * 11 + 5);
    }
    const lt = tensor(.bf16, &.{ 128, 96 }, std.mem.sliceAsBytes(&left));
    const rt = tensor(.bf16, &.{ 128, 96 }, std.mem.sliceAsBytes(&right));
    const rows: partition.Share = .{ .lo = 32, .len = 64 };
    const lc: partition.Share = .{ .lo = 16, .len = 64 };
    const rc: partition.Share = .{ .lo = 32, .len = 64 };
    var p = try pack_mod.packHConcat(a, lt, lc, rt, rc, rows);
    defer p.deinit(a);
    try testing.expect(p.mode == .bf16 and p.n == 64 and p.k == 128 and p.scales.len == 0);
    var merged: [64 * 128]u16 = undefined;
    for (0..64) |i| {
        const r = rows.lo + i;
        @memcpy(merged[i * 128 ..][0..64], left[r * 96 + lc.lo ..][0..64]);
        @memcpy(merged[i * 128 + 64 ..][0..64], right[r * 96 + rc.lo ..][0..64]);
    }
    var want: [128 * 128]u16 = undefined;
    cuda.qmmf.packBf16(&want, &merged, 64, 128);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&want), p.codes);
    try testing.expectError(error.SplitNotOnBlocks, pack_mod.packHConcat(a, lt, .{ .lo = 0, .len = 32 }, rt, .{ .lo = 0, .len = 16 }, rows));
    try testing.expectError(error.SplitOutOfRange, pack_mod.packHConcat(a, lt, .{ .lo = 64, .len = 64 }, rt, rc, rows));
}

test "projection packing refuses empty parts, unaligned slices, mixed formats, and non-finite scales" {
    const a = testing.allocator;
    var w: [256 * 128]u16 = @splat(0);
    var c: [256 * 128]u8 = @splat(0);
    var s: [2]u16 = @splat(0x3f80);
    const b: weights.Proj = .{ .bf16 = tensor(.bf16, &.{ 256, 128 }, std.mem.sliceAsBytes(&w)) };
    const f: weights.Proj = .{ .fp8 = .{ .weight = tensor(.f8_e4m3, &.{ 256, 128 }, &c), .scale_inv = tensor(.bf16, &.{ 2, 1 }, std.mem.sliceAsBytes(&s)) } };
    const all: partition.Share = .{ .lo = 0, .len = 128 };
    try testing.expectError(error.UnexpectedTensor, pack_mod.pack(a, &.{}, all));
    try testing.expectError(error.SplitNotOnBlocks, pack_mod.pack(a, &.{.{ .proj = b, .rows = all }}, .{ .lo = 32, .len = 64 }));
    try testing.expectError(error.SplitOutOfRange, pack_mod.pack(a, &.{.{ .proj = b, .rows = .{ .lo = 200, .len = 128 } }}, all));
    try testing.expectError(error.UnexpectedTensor, pack_mod.pack(a, &.{ .{ .proj = b, .rows = all }, .{ .proj = f, .rows = all } }, all));
    try testing.expectError(error.SplitNotOnBlocks, pack_mod.pack(a, &.{ .{ .proj = f, .rows = .{ .lo = 0, .len = 96 } }, .{ .proj = f, .rows = all } }, all));
    s[1] = 0x7fc0; // BF16 quiet NaN
    try testing.expectError(error.UnsupportedQuantization, pack_mod.pack(a, &.{.{ .proj = f, .rows = all }}, all));
    s[1] = 0x7f80; // BF16 +Inf
    try testing.expectError(error.UnsupportedQuantization, pack_mod.pack(a, &.{.{ .proj = f, .rows = all }}, all));
}

test "hostExperts slices stacked FP8 gate_up and down tensors across TP=1, TP=2, and TP=4 with exact scales" {
    const a = testing.allocator;
    const e = 2;
    const full = 512; // four 128-row blocks: evenly splits across TP=1, TP=2, and TP=4
    const l = 128;
    const gu = try a.alloc(u8, e * 2 * full * l);
    defer a.free(gu);
    const dn = try a.alloc(u8, e * l * full);
    defer a.free(dn);
    fill(gu, 5);
    fill(dn, 91);
    const fb = full / 128;
    var gus: [e * 2 * fb * 1]u16 = undefined;
    var dns: [e * 1 * fb]u16 = undefined;
    for (&gus, 0..) |*v, i| v.* = bf16Bits(@floatFromInt(i + 1));
    for (&dns, 0..) |*v, i| v.* = bf16Bits(@floatFromInt(100 + i));
    var m: weights.Moe = undefined;
    m.gate_up = tensor(.f8_e4m3, &.{ e, 2 * full, l }, gu);
    m.gate_up_s = tensor(.bf16, &.{ e, 2 * fb, 1 }, std.mem.sliceAsBytes(&gus));
    m.down = tensor(.f8_e4m3, &.{ e, l, full }, dn);
    m.down_s = tensor(.bf16, &.{ e, 1, fb }, std.mem.sliceAsBytes(&dns));

    for ([_]usize{ 1, 2, 4 }) |world| {
        for (0..world) |rank| {
            const share = try partition.even(full, 128, rank, world);
            var h = try pack_mod.hostExperts(a, m, share);
            defer h.deinit(a);
            const w = share.len;
            const wb = w / 128;
            const sb = share.lo / 128;
            const want_up = try a.alloc(u32, w * l / 2);
            defer a.free(want_up);
            const cols = try a.alloc(u8, l * w);
            defer a.free(cols);
            const want_dn = try a.alloc(u32, w * l / 4);
            defer a.free(want_dn);
            for (0..e) |x| {
                const base = x * 2 * full * l;
                try fx8.packGateUp(a, want_up, gu[base + share.lo * l ..][0 .. w * l], gu[base + (full + share.lo) * l ..][0 .. w * l], w, l);
                try testing.expectEqualSlices(u32, want_up, h.up[x * w * l / 2 ..][0 .. w * l / 2]);
                for (0..l) |r| @memcpy(cols[r * w ..][0..w], dn[x * l * full + r * full + share.lo ..][0..w]);
                fx8.packOne(want_dn, cols, l, w);
                try testing.expectEqualSlices(u32, want_dn, h.down[x * w * l / 4 ..][0 .. w * l / 4]);
                for (0..wb) |rb| {
                    try testing.expectEqual(@as(f32, @floatFromInt(x * 2 * fb + sb + rb + 1)), h.up_scale[x * 2 * wb + rb]);
                    try testing.expectEqual(@as(f32, @floatFromInt(x * 2 * fb + fb + sb + rb + 1)), h.up_scale[x * 2 * wb + wb + rb]);
                    try testing.expectEqual(@as(f32, @floatFromInt(100 + x * fb + sb + rb)), h.down_scale[x * wb + rb]);
                }
            }
        }
    }

    // Refusal controls: off-block share, out-of-range share, and non-finite scale_inv.
    try testing.expectError(error.SplitNotOnBlocks, pack_mod.hostExperts(a, m, .{ .lo = 64, .len = 128 }));
    try testing.expectError(error.SplitOutOfRange, pack_mod.hostExperts(a, m, .{ .lo = 512, .len = 128 }));
    dns[0] = 0x7fc0; // BF16 quiet NaN
    try testing.expectError(error.UnsupportedQuantization, pack_mod.hostExperts(a, m, .{ .lo = 0, .len = 128 }));
}
