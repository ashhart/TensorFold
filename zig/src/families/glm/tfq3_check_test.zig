//! TFQ3 file checks against valid controls and named refusals, and the fusion and KDA predicates (no GPU).
const std = @import("std");
const check = @import("tfq3_check.zig");
const t = std.testing;

const nm_scales: usize = 16384;
const nm_weights: usize = 32768;

fn nmMeta(name: []const u8, r: u32, c: u32, bits: u32) check.NonMoeTensorMeta {
    var m = std.mem.zeroes(check.NonMoeTensorMeta);
    @memcpy(m.name[0..name.len], name);
    m.R = r;
    m.C = c;
    m.b_bits = bits;
    m.word_count = @as(u64, r) * c * bits / 32;
    return m;
}

/// A valid nonmoe.tffpw: one 128 x 128 4-bit tensor, records at 128, scales and weights on 16 KiB boundaries.
fn nonMoeFile(gpa: std.mem.Allocator, h: *check.NonMoeHeader, m: *check.NonMoeTensorMeta) ![]u8 {
    h.* = .{ .magic = check.magic_nonmoe, .version = check.format_version, .num_tensors = 1, .pad0 = 0, .off_meta = 128, .off_scales = nm_scales, .off_weights = nm_weights, .total_scale_halfs = 1, .total_weight_words = 2048, .pad1 = 0 };
    m.* = nmMeta("lm_head", 128, 128, 4);
    return gpa.alloc(u8, nm_weights + 2048 * 4);
}

fn nonMoeBytes(bytes: []u8, h: check.NonMoeHeader, m: check.NonMoeTensorMeta) []u8 {
    @memset(bytes, 0);
    @memcpy(bytes[0..@sizeOf(check.NonMoeHeader)], std.mem.asBytes(&h));
    @memcpy(bytes[128..][0..@sizeOf(check.NonMoeTensorMeta)], std.mem.asBytes(&m));
    return bytes;
}

test "TFQ3 check: nonmoe.tffpw's valid control passes and each bad header, section or record is refused by name" {
    const gpa = t.allocator;
    var h: check.NonMoeHeader = undefined;
    var m: check.NonMoeTensorMeta = undefined;
    const buf = try nonMoeFile(gpa, &h, &m);
    defer gpa.free(buf);
    const ok = try check.nonMoe(nonMoeBytes(buf, h, m));
    try t.expectEqual(@as(usize, 1), ok.count);
    try t.expectEqual(nm_scales, ok.scales.off);
    try t.expectError(error.Tfq3Truncated, check.nonMoe(buf[0..32]));
    // E326's witness: a 64-byte file whose one record sits at 16384
    var short = h;
    short.off_meta = 16384;
    try t.expectError(error.Tfq3Truncated, check.nonMoe(nonMoeBytes(buf, short, m)[0..64]));
    const Case = struct { anyerror, check.NonMoeHeader, check.NonMoeTensorMeta };
    var cases: [14]Case = undefined;
    for (&cases) |*c| c.* = .{ error.Tfq3Truncated, h, m };
    cases[0][1].magic = 0xDEADBEEF;
    cases[0][0] = error.UnsupportedQuantization;
    cases[1][1].off_meta = std.math.maxInt(u64);
    cases[1][0] = error.Tfq3Overflow;
    cases[2][1].off_meta = 129;
    cases[2][0] = error.Tfq3Misaligned;
    cases[3][1].off_scales = nm_scales - 384;
    cases[3][0] = error.Tfq3Misaligned;
    cases[4][1].off_weights = nm_scales;
    cases[4][0] = error.Tfq3Overlap;
    cases[5][1].total_weight_words = 4096; // past the file
    cases[6][1].total_scale_halfs = std.math.maxInt(u64) / 2 + 1;
    cases[6][0] = error.Tfq3Overflow;
    cases[7][2].b_bits = 3;
    cases[7][0] = error.Tfq3Bits;
    cases[8][2] = nmMeta("lm_head", 100, 128, 4);
    cases[8][0] = error.Tfq3Geometry;
    cases[9][2].word_count = 2047;
    cases[9][0] = error.Tfq3Geometry;
    cases[10][2].word_off = 1; // its 2048 words run one past the stream
    cases[11][2].sc_off = 1;
    cases[12][1].num_tensors = 0;
    cases[12][0] = error.Tfq3Geometry;
    cases[13][1].num_tensors = 200; // records running into the scales
    cases[13][0] = error.Tfq3Overlap;
    for (cases) |c| try t.expectError(c[0], check.nonMoe(nonMoeBytes(buf, c[1], c[2])));
}

const lw: check.LayerWant = .{ .layer = 3, .experts = 1, .hidden = 4096, .inter = 2048 };
const l_meta: usize = @sizeOf(check.LayerHeader);
const l_scales: usize = 32768;
const l_weights: usize = 49152;

/// A valid layer file of `want.experts` experts at `bits`, gate, up and down back to back, scales in the same order.
fn layerFile(gpa: std.mem.Allocator, bits: u32, h: *check.LayerHeader, e: *check.ExpertMeta) ![]u8 {
    const words: usize = 2048 * 4096 * @as(usize, bits) / 32;
    h.* = std.mem.zeroes(check.LayerHeader);
    h.magic = check.magic_layer;
    h.version = check.format_version;
    h.layer_idx = 3;
    h.num_experts = 1;
    h.num_tensors = 3;
    h.hidden_size = 4096;
    h.moe_inter = 2048;
    h.stream_words = 3 * words;
    h.scale_halfs = 3 * 512;
    h.off_meta = l_meta;
    h.off_scales = l_scales;
    h.off_weights = l_weights;
    e.* = std.mem.zeroes(check.ExpertMeta);
    e.b_bits = bits;
    e.up_word_off = words;
    e.down_word_off = 2 * words;
    e.up_sc_off = 512;
    e.down_sc_off = 1024;
    return gpa.alloc(u8, l_weights + 3 * words * 4);
}

fn layerBytes(bytes: []u8, h: check.LayerHeader, e: check.ExpertMeta) []u8 {
    @memset(bytes[0..l_weights], 0);
    @memcpy(bytes[0..@sizeOf(check.LayerHeader)], std.mem.asBytes(&h));
    @memcpy(bytes[l_meta..][0..@sizeOf(check.ExpertMeta)], std.mem.asBytes(&e));
    return bytes;
}

test "TFQ3 check: a layer file's valid control passes and each bad index, width, offset or extent is refused" {
    const gpa = t.allocator;
    var h: check.LayerHeader = undefined;
    var e: check.ExpertMeta = undefined;
    const buf = try layerFile(gpa, 4, &h, &e);
    defer gpa.free(buf);
    try t.expect(!(try check.layer(layerBytes(buf, h, e), lw)).has_5b);
    var five = e;
    five.b_bits = 5; // 5-bit extents overrun a 4-bit stream: refused, not read
    try t.expectError(error.Tfq3Truncated, check.layer(layerBytes(buf, h, five), lw));
    var want7 = lw;
    want7.layer = 7; // E326: layer_07.tffpw holding layer 3's header
    try t.expectError(error.Tfq3Layer, check.layer(layerBytes(buf, h, e), want7));
    var more = lw;
    more.experts = 2;
    try t.expectError(error.Tfq3Layer, check.layer(layerBytes(buf, h, e), more));
    // E326's witness: a 144-byte layer file with one expert's record at 16384
    try t.expectError(error.Tfq3Truncated, check.layer(layerBytes(buf, h, e)[0..144], lw));
    const Case = struct { anyerror, check.LayerHeader, check.ExpertMeta };
    var cases: [10]Case = undefined;
    for (&cases) |*c| c.* = .{ error.Tfq3Truncated, h, e };
    cases[0][1].version = 2;
    cases[0][0] = error.UnsupportedQuantization;
    cases[1][1].num_tensors = 4;
    cases[1][0] = error.Tfq3Geometry;
    cases[2][1].off_meta = l_meta + 1;
    cases[2][0] = error.Tfq3Misaligned;
    cases[3][1].off_weights = l_scales;
    cases[3][0] = error.Tfq3Overlap;
    cases[4][1].stream_words = std.math.maxInt(u64) / 2;
    cases[4][0] = error.Tfq3Overflow;
    cases[5][2].b_bits = 6;
    cases[5][0] = error.Tfq3Bits;
    cases[6][2].down_word_off += 1; // down's last word past the stream
    cases[7][2].down_sc_off += 1;
    cases[8][1].scale_halfs = 3 * 512 - 1;
    cases[9][1].off_meta = 0;
    cases[9][0] = error.Tfq3Overlap;
    for (cases) |c| try t.expectError(c[0], check.layer(layerBytes(buf, c[1], c[2]), lw));
}

test "TFQ3 check: a 5-bit expert sets the routed specialization" {
    const gpa = t.allocator;
    var h: check.LayerHeader = undefined;
    var e: check.ExpertMeta = undefined;
    const buf = try layerFile(gpa, 5, &h, &e);
    defer gpa.free(buf);
    try t.expect((try check.layer(layerBytes(buf, h, e), lw)).has_5b);
}

test "TFQ3 check: the table file needs the kernels' whole table, finite" {
    var tbl: [check.tables_bytes]u8 = @splat(0);
    std.mem.writeInt(u32, tbl[0..4], check.magic_v3_tables, .little);
    std.mem.writeInt(u32, tbl[4..8], check.format_version, .little);
    try check.tables(&tbl);
    try t.expectError(error.Tfq3Table, check.tables(tbl[0 .. check.tables_bytes - 2]));
    try t.expectError(error.Tfq3Truncated, check.tables(tbl[0..4]));
    var bad = tbl;
    std.mem.writeInt(u16, bad[1088..][0..2], 0x7e00, .little); // a NaN in lut4
    try t.expectError(error.Tfq3Table, check.tables(&bad));
    bad = tbl;
    std.mem.writeInt(u32, bad[8..12], @bitCast(std.math.inf(f32)), .little);
    try t.expectError(error.Tfq3Table, check.tables(&bad));
    bad = tbl;
    std.mem.writeInt(u32, bad[0..4], 0xDEADBEEF, .little);
    try t.expectError(error.UnsupportedQuantization, check.tables(&bad));
}

test "TFQ3 check: shared fusion only at the routed width for both projections, the kernels' shapes and every stage" {
    const p = struct {
        fn gu(bits: u8) check.Proj {
            return .{ .runs = 1, .bits = bits, .group = 512, .n = 4096, .k = 4096 };
        }
        fn down(bits: u8) check.Proj {
            return .{ .runs = 1, .bits = bits, .group = 512, .n = 4096, .k = 2048 };
        }
    };
    // the eight width cases: fused only where gate/up, down and the routed specialization agree
    for ([_]bool{ false, true }) |has_5b| for ([_]u8{ 4, 5 }) |gb| for ([_]u8{ 4, 5 }) |db| {
        const want = gb == db and gb == @as(u8, if (has_5b) 5 else 4);
        try t.expectEqual(want, check.sharedFusion(p.gu(gb), p.down(db), has_5b, 4096, 2048, true, true));
    };
    try t.expect(!check.sharedFusion(p.gu(4), p.down(4), false, 4096, 2048, false, true)); // a stage off
    try t.expect(!check.sharedFusion(p.gu(4), p.down(4), false, 4096, 2048, true, false)); // combine unfused
    var two = p.gu(4);
    two.runs = 2;
    try t.expect(!check.sharedFusion(two, p.down(4), false, 4096, 2048, true, true));
    var g64 = p.down(4);
    g64.group = 64;
    try t.expect(!check.sharedFusion(p.gu(4), g64, false, 4096, 2048, true, true));
    var wide = p.gu(4);
    wide.n = 2 * 3072;
    try t.expect(!check.sharedFusion(wide, p.down(4), false, 4096, 2048, true, true));
    try t.expect(!check.sharedFusion(p.gu(4), p.down(4), false, 4096, 3072, true, true));
}

test "TFQ3 check: dense GLU fuses only when up's words, scales and table follow gate's" {
    const n: usize = 12288;
    const k: usize = 4096;
    const gate: check.RunAt = .{ .w_buf = 1, .w_off = 0, .s_buf = 2, .s_off = 0, .t_buf = 3, .t_off = 0, .row = 0, .n = n, .bits = 4, .group = 512 };
    var up = gate;
    up.row = n;
    up.w_off = n * k * 4 / 8;
    up.s_off = (n / 128) * (k / 128) * 2;
    try t.expect(check.gluContiguous(gate, up, n, k));
    var gap = up;
    gap.w_off += 4096 * 4; // E326's witness: a 4096-word gap after gate
    try t.expect(!check.gluContiguous(gate, gap, n, k));
    var scale = up;
    scale.s_off += 2;
    try t.expect(!check.gluContiguous(gate, scale, n, k));
    var other = up;
    other.w_buf = 9;
    try t.expect(!check.gluContiguous(gate, other, n, k));
    var table = up;
    table.t_off = 16384;
    try t.expect(!check.gluContiguous(gate, table, n, k));
    var bits = up;
    bits.bits = 5;
    try t.expect(!check.gluContiguous(gate, bits, n, k));
}

test "TFQ3 check: without a TFQ3 layer the KDA keeps main's pipeline in every case" {
    for ([_]bool{ false, true }) |q8| for ([_]bool{ false, true }) |built| for ([_]bool{ false, true }) |split| for ([_]u32{ 1, 2 }) |tp| for (1..9) |rows| {
        try t.expectEqual(check.KdaQ8.none, check.kdaQ8(false, q8, built, split, tp, @intCast(rows)));
        const got = check.kdaQ8(true, q8, built, split, tp, @intCast(rows));
        const want: check.KdaQ8 = if (!q8 or !built or tp != 1) .none else if (split) .parts else if (rows > 1) .window else .none;
        try t.expectEqual(want, got);
    };
}

test "TFQ3 check: a record serves a tensor only whole and at the model's shape" {
    try check.recordFits(154880, 4096, 0, 154880, 154880, 4096);
    try t.expectError(error.Tfq3RecordShape, check.recordFits(128, 128, 0, 154880, 154880, 4096)); // E329b's witness
    try t.expectError(error.Tfq3RecordShape, check.recordFits(154880, 2048, 0, 154880, 154880, 4096));
    try t.expectError(error.Tfq3RecordSlice, check.recordFits(154880, 4096, 77440, 77440, 154880, 4096)); // TP2 rank 1's head
    try t.expectError(error.Tfq3RecordSlice, check.recordFits(154880, 4096, 0, 77440, 154880, 4096));
}
