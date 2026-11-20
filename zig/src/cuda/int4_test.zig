//! int4.zig's host tests: the packer, the scales, the stored codes, their binary64 reference, the symbols; no device.
const std = @import("std");
const int4 = @import("int4.zig");

/// The INT4-AutoRound checkpoint this kernel is for: hidden 2,560, experts of width 640, a head of 248,320 rows.
const checkpoint = struct {
    const dims = 2560;
    const width = 640;
    const vocab = 248320;
    const experts = 512;
};

/// A synthetic matrix: `n` columns of `k` inputs, every code and scale drawn from `seed`.
const Synth = struct {
    qw: []u8,
    sc: []u8,
    src: int4.Source,

    fn init(a: std.mem.Allocator, n: usize, k: usize, seed: u64) !Synth {
        var s = Synth{ .qw = try a.alloc(u8, k / 8 * n * 4), .sc = try a.alloc(u8, k / 128 * n * 2), .src = undefined };
        var prng = std.Random.DefaultPrng.init(seed);
        prng.random().bytes(s.qw);
        for (0..s.sc.len / 2) |i| std.mem.writeInt(u16, s.sc[2 * i ..][0..2], @intCast((i * 977 + seed) % 4096), .little);
        s.src = .{ .qweight = s.qw, .scales = s.sc, .n_full = n, .k_full = k };
        return s;
    }

    fn deinit(s: *const Synth, a: std.mem.Allocator) void {
        a.free(s.qw);
        a.free(s.sc);
    }
};

test "int4: shuffle puts input 2i at nibble i and 2i + 1 at nibble i + 4" {
    var w: u32 = 0;
    for (0..8) |j| w |= @as(u32, @intCast(j + 1)) << @intCast(4 * j);
    const s = int4.shuffle(w);
    for (0..4) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(2 * i + 1)), (s >> @intCast(4 * i)) & 0xF);
        try std.testing.expectEqual(@as(u32, @intCast(2 * i + 2)), (s >> @intCast(4 * (i + 4))) & 0xF);
    }
}

test "int4: shuffle carries every code to its pair and reverses exactly" {
    var prng = std.Random.DefaultPrng.init(11);
    for (0..64) |_| {
        const w = prng.random().int(u32);
        const s = int4.shuffle(w);
        for (0..4) |i| {
            try std.testing.expectEqual((w >> @intCast(8 * i)) & 0xF, (s >> @intCast(4 * i)) & 0xF);
            try std.testing.expectEqual((w >> @intCast(8 * i + 4)) & 0xF, (s >> @intCast(4 * (i + 4))) & 0xF);
        }
    }
}

test "int4: packWords keeps each lane's word at its column and inputs" {
    const gpa = std.testing.allocator;
    const s = try Synth.init(gpa, 16, 512, 7);
    defer s.deinit(gpa);
    const sl: int4.Slice = .{ .n0 = 8, .n = 8, .k0 = 128, .k = 256, .gs = 128 };
    const out = try gpa.alloc(u32, int4.wordsOf(sl.n, sl.k));
    defer gpa.free(out);
    try int4.packWords(s.src, sl, out);
    for (0..sl.k / sl.gs) |g| for (0..32) |lane| for (0..sl.gs / 32) |b| {
        const word = out[(g * 32 + lane) * (sl.gs / 32) + b];
        for (0..8) |j| {
            const nib: usize = if (j % 2 == 0) j / 2 else j / 2 + 4;
            const got: u4 = @intCast((word >> @intCast(4 * nib)) & 0xF);
            const kin = sl.k0 + g * sl.gs + b * 32 + 8 * (lane % 4) + j;
            try std.testing.expectEqual(int4.code(s.src, sl.n0 + lane / 4, kin), got);
        }
    };
}

test "int4: a whole matrix packs once, and packScales carries each column's own group" {
    const gpa = std.testing.allocator;
    const s = try Synth.init(gpa, checkpoint.width, checkpoint.dims, 3);
    defer s.deinit(gpa);
    const sl: int4.Slice = .{ .n0 = 0, .n = checkpoint.width, .k0 = 0, .k = checkpoint.dims, .gs = int4.checkpoint_group };
    const w = try gpa.alloc(u32, int4.wordsOf(sl.n, sl.k));
    defer gpa.free(w);
    try int4.packWords(s.src, sl, w);
    const sc = try gpa.alloc(u16, int4.scalesOf(sl.n, sl.k));
    defer gpa.free(sc);
    try int4.packScales(s.src, sl, sc);
    try std.testing.expectEqual(@as(usize, 204800), w.len);
    try std.testing.expectEqual(@as(usize, 12800), sc.len);
    const kg = sl.k / sl.gs;
    for ([_]usize{ 0, 1, 639 }) |col| for ([_]usize{ 0, 127, 128, 2559 }) |k| {
        const want: u16 = @intCast((((k / 128) * s.src.n_full + col) * 977 + 3) % 4096);
        try std.testing.expectEqual(want, sc[(col / 8 * kg + k / 128) * 8 + col % 8]);
    };
    // one word of the last column and group, read back through code()
    const col = 637;
    const kin = 2559;
    const lane = 4 * (col % 8) + (kin % 32) / 8;
    const word = w[(col / 8 * kg + kin / 128) * 32 * 4 + lane * 4 + (kin % int4.checkpoint_group) / 32];
    const j = kin % 8;
    const nib: usize = if (j % 2 == 0) j / 2 else j / 2 + 4;
    try std.testing.expectEqual(int4.code(s.src, col, kin), @as(u4, @intCast((word >> @intCast(4 * nib)) & 0xF)));
}

test "int4: packScales copies the group's own fp16 bits, and both packers refuse a short buffer" {
    const gpa = std.testing.allocator;
    const s = try Synth.init(gpa, 16, 512, 5);
    defer s.deinit(gpa);
    const sl: int4.Slice = .{ .n0 = 8, .n = 8, .k0 = 256, .k = 128, .gs = 128 };
    var sc: [8]u16 = undefined;
    try int4.packScales(s.src, sl, &sc);
    for (0..8) |c| {
        const want: u16 = @intCast((((sl.k0 / 128) * s.src.n_full + sl.n0 + c) * 977 + 5) % 4096);
        try std.testing.expectEqual(want, sc[c]);
    }
    var short: [7]u16 = undefined;
    try std.testing.expectError(error.BadLength, int4.packScales(s.src, sl, &short));
    var words: [5]u32 = undefined;
    try std.testing.expectError(error.BadLength, int4.packWords(s.src, sl, &words));
}

test "int4: a slice outside the dialect is refused by name" {
    const gpa = std.testing.allocator;
    const s = try Synth.init(gpa, 16, 512, 9);
    defer s.deinit(gpa);
    const ok: int4.Slice = .{ .n0 = 0, .n = 8, .k0 = 0, .k = 128, .gs = 128 };
    const out = try gpa.alloc(u32, int4.wordsOf(ok.n, ok.k));
    defer gpa.free(out);
    try int4.packWords(s.src, ok, out);
    for ([_]int4.Slice{
        .{ .n0 = 0, .n = 8, .k0 = 0, .k = 128, .gs = 64 }, // a group of 64 is a TP=2 rank's, refused here
        .{ .n0 = 0, .n = 7, .k0 = 0, .k = 128, .gs = 128 }, // a part n8 tile
        .{ .n0 = 0, .n = 8, .k0 = 0, .k = 96, .gs = 128 }, // k not whole groups
        .{ .n0 = 0, .n = 8, .k0 = 64, .k = 128, .gs = 128 }, // k0 inside a group
        .{ .n0 = 8, .n = 16, .k0 = 0, .k = 128, .gs = 128 }, // past the source's columns
        .{ .n0 = 0, .n = 8, .k0 = 384, .k = 256, .gs = 128 }, // past the source's inputs
        .{ .n0 = 0, .n = 0, .k0 = 0, .k = 128, .gs = 128 }, // no columns
    }) |bad| try std.testing.expectError(error.BadSlice, int4.checkSlice(s.src, bad));
    var odd = s.src;
    odd.k_full = 384;
    try std.testing.expectError(error.BadLength, int4.checkSlice(odd, ok));
    var qw = s.src;
    qw.qweight = s.qw[0 .. s.qw.len - 1];
    try std.testing.expectError(error.BadLength, int4.checkSlice(qw, ok));
    var sc = s.src;
    sc.scales = s.sc[0 .. s.sc.len - 1];
    try std.testing.expectError(error.BadLength, int4.checkSlice(sc, ok));
}

test "int4: stored codes, scales and the binary64 reference are exact" {
    const gpa = std.testing.allocator;
    const n = 8;
    const k = 256;
    const qw = try gpa.alloc(u8, k / 8 * n * 4);
    defer gpa.free(qw);
    const sc = try gpa.alloc(u8, k / 128 * n * 2);
    defer gpa.free(sc);
    @memset(qw, 0);
    var w: u32 = 0;
    for (0..8) |j| w |= @as(u32, @intCast(j)) << @as(u5, @intCast(4 * j));
    std.mem.writeInt(u32, qw[(0 * n + 0) * 4 ..][0..4], w, .little);
    std.mem.writeInt(u32, qw[(0 * n + 1) * 4 ..][0..4], (w & ~@as(u32, 0xF)) | 8, .little);
    std.mem.writeInt(u32, qw[(0 * n + 3) * 4 ..][0..4], 9, .little);
    // group 0: 1, 0.5, 0.1's fp16 bits, fp16's least subnormal, -2, +inf, 1024, 2^-14
    const s0 = [_]u16{ 0x3C00, 0x3800, 0x2E66, 0x0001, 0xC000, 0x7C00, 0x6400, 0x0400 };
    // group 1: the same scales a column over
    for (0..n) |c| {
        std.mem.writeInt(u16, sc[(0 * n + c) * 2 ..][0..2], s0[c], .little);
        std.mem.writeInt(u16, sc[(1 * n + c) * 2 ..][0..2], s0[n - 1 - c], .little);
    }
    const src: int4.Source = .{ .qweight = qw, .scales = sc, .n_full = n, .k_full = k };
    // codes 0..7 at column 0 under a scale of 1: the weight is the code minus 8, exactly
    for (0..8) |j| {
        try std.testing.expectEqual(@as(u4, @intCast(j)), int4.code(src, 0, j));
        try std.testing.expectEqual(@as(f32, 1.0), int4.scale(src, 0, j));
        try std.testing.expectEqual(@as(f32, @floatFromInt(@as(i32, @intCast(j)) - 8)), int4.weight(src, 0, j));
        try std.testing.expectEqual(@as(f64, @floatFromInt(@as(i32, @intCast(j)) - 8)), int4.weight64(src, 0, j));
    }
    // the zero point itself: code 8 at column 1, and code 9 at column 3, are 0 and 1 times the scale
    try std.testing.expectEqual(@as(u4, 8), int4.code(src, 1, 0));
    try std.testing.expectEqual(@as(f32, 0.0), int4.weight(src, 1, 0));
    try std.testing.expectEqual(@as(f64, 0.0), int4.weight64(src, 1, 0));
    try std.testing.expectEqual(@as(u4, 9), int4.code(src, 3, 0));
    try std.testing.expectEqual(0x1p-24, int4.weight64(src, 3, 0));
    // 0.1 is fp16 0x2E66: the binary64 product is that widening times -8, and the f32 form rounds once
    const tenth: f64 = @floatCast(@as(f16, @bitCast(@as(u16, 0x2E66))));
    try std.testing.expectEqual(@as(u4, 0), int4.code(src, 2, 7));
    try std.testing.expectEqual(@as(f32, @floatCast(tenth)), int4.scale(src, 2, 7));
    try std.testing.expectEqual(-8.0 * tenth, int4.weight64(src, 2, 7));
    try std.testing.expectEqual(@as(f32, @floatCast(-8.0 * tenth)), int4.weight(src, 2, 7));
    // a negative scale, fp16's least subnormal, an infinite scale and 1024 all survive exactly
    try std.testing.expectEqual(@as(u4, 0), int4.code(src, 4, 0));
    try std.testing.expectEqual(@as(f32, -2.0), int4.scale(src, 4, 0));
    try std.testing.expectEqual(-8.0 * -2.0, int4.weight64(src, 4, 0));
    try std.testing.expectEqual(-8.0 * 0x1p-24, int4.weight64(src, 3, 1));
    try std.testing.expect(std.math.isInf(int4.weight(src, 5, 7)));
    try std.testing.expectEqual(-8.0 * 1024.0, int4.weight64(src, 6, 0));
    // the second group's scales are its own: column 7's group-1 scale is group 0's column 0, 1.0
    try std.testing.expectEqual(@as(f32, 1.0), int4.scale(src, 7, 128));
    try std.testing.expectEqual(-8.0, int4.weight64(src, 7, 255));
}

test "int4: a packed matrix of the checkpoint's shapes is sized as the kernel reads it" {
    try std.testing.expectEqual(@as(usize, 204800), int4.wordsOf(checkpoint.width, checkpoint.dims));
    try std.testing.expectEqual(@as(usize, 12800), int4.scalesOf(checkpoint.width, checkpoint.dims));
    try std.testing.expectEqual(@as(usize, 204800), int4.wordsOf(checkpoint.dims, checkpoint.width));
    try std.testing.expectEqual(@as(usize, 12800), int4.scalesOf(checkpoint.dims, checkpoint.width));
    try std.testing.expectEqual(@as(usize, 0), checkpoint.vocab % 8); // the head's n8 tiles are whole
    try std.testing.expectEqual(@as(usize, 0), checkpoint.dims % 128); // and its groups too
    var z: [20]u8 = @splat(0x77);
    try std.testing.expect(int4.zerosAreSymmetric(&z));
    z[19] = 0x87;
    try std.testing.expect(!int4.zerosAreSymmetric(&z));
    z[19] = 0x77;
    z[0] = 0x70;
    try std.testing.expect(!int4.zerosAreSymmetric(&z));
}

test "int4: every symbol is the compiler's mangling of a kept instantiation" {
    // one per kernel shape, copied from a compiler's Itanium mangling of fn_int4.cu's two templates
    try std.testing.expectEqualStrings("_ZN10tf_fn_int411int4_kernelILi128ELi4ELi1ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_iPvifi", int4.symbols[0][2][0]);
    try std.testing.expectEqualStrings("_ZN10tf_fn_int411int4_kernelILi128ELi2ELi2ELi1ELi0ELi4EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_iPvifi", int4.symbols[1][1][1]);
    try std.testing.expectEqualStrings("_ZN10tf_fn_int411int4_kernelILi128ELi1ELi1ELi1ELi3ELi4EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_iPvifi", int4.symbols[2][0][0]);
    try std.testing.expectEqualStrings("_ZN10tf_fn_int418int4_prompt_kernelILi128ELi1ELi0ELi2ELi4ELi2ELi4ELi2EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_Pvifi", int4.prompt_symbols[0]);
    try std.testing.expectEqualStrings("_ZN10tf_fn_int418int4_prompt_kernelILi128ELi1ELi3ELi2ELi4ELi2ELi4ELi2EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_Pvifi", int4.prompt_symbols[1]);
    try std.testing.expectEqualStrings("_ZN10tf_fn_int418int4_prompt_kernelILi128ELi2ELi2ELi2ELi4ELi2ELi4ELi3EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_Pvifi", int4.prompt_gu_symbol);
    // every kept pairing present, distinct, and its own template arguments in it
    var seen: [int4.symbols.len * int4.symbols[0].len * int4.symbols[0][0].len + 3][:0]const u8 = undefined;
    var at: usize = 0;
    for (int4.symbols) |row| for (row) |col| for (col) |s| {
        seen[at] = s;
        at += 1;
    };
    for (int4.prompt_symbols) |s| {
        seen[at] = s;
        at += 1;
    }
    seen[at] = int4.prompt_gu_symbol;
    at += 1;
    try std.testing.expectEqual(@as(usize, 21), at);
    for (seen[0..at], 0..) |s, i| {
        for (seen[i + 1 .. at]) |t| try std.testing.expect(!std.mem.eql(u8, s, t));
        try std.testing.expect(std.mem.startsWith(u8, s, "_ZN10tf_fn_int41"));
        try std.testing.expect(std.mem.endsWith(u8, s, "Pvifi"));
    }
    try std.testing.expect(std.mem.indexOf(u8, int4.symbols[1][2][1], "ILi128ELi4ELi2ELi1ELi0ELi4") != null);
    try std.testing.expect(std.mem.indexOf(u8, int4.symbols[0][0][1], "ILi128ELi1ELi2ELi2ELi2ELi4") != null);
}

test "int4: prompt shared memory and the gate/up launch admit the checkpoint's shapes" {
    const sh = int4.prompt_gu;
    try std.testing.expectEqual(@as(usize, 256), sh.wr * sh.wc * 32);
    try std.testing.expectEqual(@as(usize, 64), int4.promptGuRows(sh));
    try std.testing.expectEqual(@as(usize, 16), int4.promptGuTiles(sh));
    try std.testing.expect(int4.promptGuSmem(sh) <= int4.sm121_smem_optin);
    try std.testing.expect(int4.promptGuFits(0, int4.sm121_smem_optin));
    try std.testing.expect(!int4.promptGuFits(int4.sm121_smem_optin, int4.sm121_smem_optin));
    // the exact bytes each prompt launch asks the driver for, as int4_records.zig declares them
    try std.testing.expectEqual(@as(u32, 49664), int4.promptSmem(2));
    try std.testing.expectEqual(@as(u32, 99840), int4.promptGuSmem(sh));
    try std.testing.expectEqual(int4.promptSmemOf(1, 64, 16, 2), int4.promptSmem(2));
    const ex: int4.Experts = .{ .up = 4096, .up_s = 2048, .down = 4096, .down_s = 2048, .width = checkpoint.width, .dims = checkpoint.dims };
    try std.testing.expect(int4.gateUpPromptFits(ex, 65536, checkpoint.dims, 8192));
    for ([_]int4.Experts{
        .{ .up = 4096, .up_s = 2048, .down = 4096, .down_s = 2048, .width = 648, .dims = checkpoint.dims },
        .{ .up = 4096, .up_s = 2048, .down = 4096, .down_s = 2048, .width = checkpoint.width, .dims = 2561 },
        .{ .up = 8, .up_s = 2048, .down = 4096, .down_s = 2048, .width = checkpoint.width, .dims = checkpoint.dims },
        .{ .up = 4096, .up_s = 2, .down = 4096, .down_s = 2048, .width = checkpoint.width, .dims = checkpoint.dims },
    }) |bad| try std.testing.expect(!int4.gateUpPromptFits(bad, 65536, checkpoint.dims, 8192));
    try std.testing.expect(!int4.gateUpPromptFits(ex, 8, checkpoint.dims, 8192));
    try std.testing.expect(!int4.gateUpPromptFits(ex, 65536, 7, 8192));
    try std.testing.expect(!int4.gateUpPromptFits(ex, 65536, checkpoint.dims, 4));
}

test "int4: the module's constants are the checkpoint's dialect" {
    try std.testing.expectEqual(@as(usize, 128), int4.checkpoint_group);
    try std.testing.expectEqual(@as(usize, 16), int4.plan_tile);
    try std.testing.expectEqual(@as(usize, 64), int4.prompt_tile);
    try std.testing.expectEqual(@as(usize, 4), int4.warps);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 4 }, &int4.nts);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, &int4.mts);
}
