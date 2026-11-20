//! EXL3's decode-row kernels: weights decoded bit for bit, a row's bits in any window, the rotation by its fp32 steps.
const std = @import("std");
const core = @import("core");
const rig = @import("rig.zig");
const exl3 = @import("../launches/exl3.zig");
const fmt = core.exl3_format;
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

const had: f32 = 0.08838834764831845;

/// The kernels' unscaled 128-point butterfly in fp32: stages 1 to 64 apart, a + b low and a - b high.
fn fwht(v: *[128]f32) void {
    var d: usize = 1;
    while (d < 128) : (d *= 2) {
        var base: usize = 0;
        while (base < 128) : (base += 2 * d) for (0..d) |j| {
            const a = v[base + j];
            const b = v[base + j + d];
            v[base + j] = a + b;
            v[base + j + d] = a - b;
        };
    }
}

/// Random trellis words in the checkpoint's layout, [K/16, N/16, tile words].
fn randomTrellis(rng: *rig.Rng, w: fmt.Width, k: usize, n: usize) ![]i16 {
    const tr = try gpa.alloc(i16, (k / 16) * (n / 16) * w.tileWords());
    for (tr) |*v| v.* = @bitCast(@as(u16, @truncate(rng.next())));
    return tr;
}

/// A search for a tile's stream whose 256 values all decode above zero, or all below: a value's bits at a time.
const Search = struct {
    rng: *rig.Rng,
    neg: bool,
    /// Stream bits a value adds (a whole width's half-bits), and the ring's length.
    g: usize,
    ring: usize,
    bits: [128 * 16]u1 = undefined,
    budget: usize = 0,

    /// The 16-bit window ending at `end` decodes to a value of the wanted sign.
    fn signed(s: *const Search, end: usize) bool {
        var state: u16 = 0;
        for (0..16) |b| state = (state << 1) | s.bits[(end + s.ring - 16 + b) % s.ring];
        const v = fmt.codebookValue(.mul1, state);
        return if (s.neg) v < 0 else v > 0;
    }

    fn from(s: *Search, p: usize) bool {
        if (s.budget == 0) return false;
        s.budget -= 1;
        // the windows that wrap past the stream's start are whole once every bit is set
        if (p == 256) {
            var q: usize = 0;
            while ((q + 1) * s.g < 16) : (q += 1) if (!s.signed((q + 1) * s.g)) return false;
            return true;
        }
        const options = @as(usize, 1) << @intCast(s.g);
        const first: usize = @intCast(s.rng.next() % options);
        for (0..options) |i| {
            const v = (first + i) % options;
            for (0..s.g) |b| s.bits[p * s.g + b] = @intCast((v >> @intCast(s.g - 1 - b)) & 1);
            const end = (p + 1) * s.g;
            if ((end < 16 or s.signed(end)) and s.from(p + 1)) return true;
        }
        return false;
    }
};

/// A tile's words (checkpoint order: MSB-first bits in little-endian int16 pairs) whose values all have one sign.
fn signedTile(rng: *rig.Rng, w: fmt.Width, neg: bool, words: []i16) void {
    var s: Search = .{ .rng = rng, .neg = neg, .g = w.halves / 2, .ring = 128 * @as(usize, w.halves) };
    while (true) {
        s.budget = 100_000;
        if (s.from(0)) break;
    }
    var word: [64]u32 = @splat(0);
    for (0..s.ring) |i| word[i / 32] |= @as(u32, s.bits[i]) << @intCast(31 - i % 32);
    for (0..s.ring / 32) |j| {
        words[2 * j] = @bitCast(@as(u16, @truncate(word[j])));
        words[2 * j + 1] = @bitCast(@as(u16, @truncate(word[j] >> 16)));
    }
}

/// [K/16, N/16, tile words] from a pool of 16 one-signed tiles, a tile picked by its position.
fn signedTrellis(rng: *rig.Rng, w: fmt.Width, neg: bool, k: usize, n: usize) ![]i16 {
    const words = w.tileWords();
    var pool: [16 * 128]i16 = undefined;
    for (0..16) |i| signedTile(rng, w, neg, pool[i * words ..][0..words]);
    const tr = try gpa.alloc(i16, (k / 16) * (n / 16) * words);
    for (0..k / 16) |kt| for (0..n / 16) |nt| {
        const i = (kt * 7 + nt * 13) % 16;
        @memcpy(tr[(kt * (n / 16) + nt) * words ..][0..words], pool[i * words ..][0..words]);
    };
    return tr;
}

/// The same tiles as 128-column strips: strip nb holds its k steps of eight tiles each.
fn strips(w: fmt.Width, stored: []const i16, k: usize, n: usize) ![]i16 {
    const words = w.tileWords();
    const out = try gpa.alloc(i16, stored.len);
    for (0..n / 128) |nb| for (0..k / 16) |kt| for (0..8) |t| {
        const from = (kt * (n / 16) + nb * 8 + t) * words;
        const to = ((nb * (k / 16) + kt) * 8 + t) * words;
        @memcpy(out[to..][0..words], stored[from..][0..words]);
    };
    return out;
}

/// The checkpoint layout's strides in 32-bit words, or the strips layout's.
fn layout(w: fmt.Width, k: usize, n: usize, as_strips: bool) [2]u64 {
    const tw: u64 = w.tileWords() / 2;
    return if (as_strips) .{ 8 * tw, (k / 16) * 8 * tw } else .{ (n / 16) * tw, 8 * tw };
}

fn randomHalves(rng: *rig.Rng, n: usize) ![]u16 {
    const out = try gpa.alloc(u16, n);
    for (out) |*v| v.* = rig.f16Bits(rng.unit());
    return out;
}

// one-signed tiles, so each zero of the identity enters dot2 as +0: x +0 for positive weights, -0 for negative ones
test "the linear decodes every width's weights as exl3_format does, bit for bit, at 4 and 8 waves and both layouts" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xE3D1 };
    // (k, n): 4 waves on the checkpoint layout, 8 waves on strips
    const sizes = [_][3]usize{ .{ 256, 256, 0 }, .{ 1024, 8192, 1 } };
    for (sizes) |sz| {
        const k = sz[0];
        const n = sz[1];
        try std.testing.expectEqual(@as(u32, if (sz[2] == 1) 8 else 4), exl3.waves(k, n));
        // xh = I: row r is 1.0 at k = r, so y's row r is W's row r, rotated by 128 columns
        const xh = try gpa.alloc(u16, k * k);
        defer gpa.free(xh);
        const ones = try gpa.alloc(u16, n);
        defer gpa.free(ones);
        @memset(ones, 0x3c00);
        var dx = try t.alloc(k * k * 2);
        defer dx.free();
        var dsvh = try t.upload(ones);
        defer dsvh.free();
        var dy = try t.alloc(k * n * 4);
        defer dy.free();
        const wq = try gpa.alloc(f16, k * n);
        defer gpa.free(wq);
        for (exl3.widths) |halves| for ([_]bool{ false, true }) |neg| {
            const w = fmt.Width{ .halves = @intCast(halves) };
            const stored = try signedTrellis(&rng, w, neg, k, n);
            defer gpa.free(stored);
            fmt.unpack(w, .mul1, stored, k / 16, n / 16, wq);
            for (wq) |v| if ((v < 0) != neg) return error.TestUnexpectedResult;
            @memset(xh, if (neg) 0x8000 else 0);
            for (0..k) |r| xh[r * k + r] = 0x3c00;
            try dx.upload(0, std.mem.sliceAsBytes(xh));
            const words = if (sz[2] == 1) try strips(w, stored, k, n) else try gpa.dupe(i16, stored);
            defer gpa.free(words);
            var dt = try t.upload(words);
            defer dt.free();
            const st = layout(w, k, n, sz[2] == 1);
            const tr: exl3.Trellis = .{ .words = at(dt), .halves = halves, .stride_k = st[0], .stride_nb = st[1] };
            try exl3.linear(&t.on, at(dx), tr, at(dsvh), at(dy), 0, k, k, n, t.stream.handle);
            try t.stream.synchronize();
            const got = try rig.download(f32, dy);
            defer gpa.free(got);
            for (0..k) |r| for (0..n / 128) |b| {
                var v: [128]f32 = undefined;
                for (&v, 0..) |*e, q| e.* = wq[r * n + b * 128 + q];
                fwht(&v);
                for (v, 0..) |e, q| {
                    const want: u32 = @bitCast(e * had * 1.0);
                    const have: u32 = @bitCast(got[r * n + b * 128 + q]);
                    if (want != have) {
                        const at_ = .{ halves / 2, if (neg) "-" else "+", k, n, r, b * 128 + q, have, want };
                        std.debug.print("{d}-bit {s} k {d} n {d}: row {d} column {d} is {x} where {x}\n", at_);
                        return error.TestUnexpectedResult;
                    }
                }
            };
        };
    }
}

test "a row's outputs are the same bits alone, in any window and in every pass of 16 rows" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xE3D2 };
    const rows = 64;
    for ([_][3]usize{ .{ 512, 256, 6 }, .{ 1024, 8192, 10 } }) |sz| {
        const k = sz[0];
        const n = sz[1];
        const w = fmt.Width{ .halves = @intCast(sz[2]) };
        const words = try randomTrellis(&rng, w, k, n);
        defer gpa.free(words);
        const xh = try randomHalves(&rng, rows * k);
        defer gpa.free(xh);
        const svh = try randomHalves(&rng, n);
        defer gpa.free(svh);
        var dt = try t.upload(words);
        defer dt.free();
        var dx = try t.upload(xh);
        defer dx.free();
        var dsvh = try t.upload(svh);
        defer dsvh.free();
        var dall = try t.alloc(rows * n * 4);
        defer dall.free();
        var dw = try t.alloc(rows * n * 4);
        defer dw.free();
        const st = layout(w, k, n, false);
        const tr: exl3.Trellis = .{ .words = at(dt), .halves = w.halves, .stride_k = st[0], .stride_nb = st[1] };
        try exl3.linear(&t.on, at(dx), tr, at(dsvh), at(dall), 0, rows, k, n, t.stream.handle);
        try t.stream.synchronize();
        const all = try rig.download(f32, dall);
        defer gpa.free(all);
        // (first row, rows): alone, the rows path at 2 to 8, its two parts at 9 to 16, and passes past 16
        const windows = [_][2]usize{
            .{ 0, 1 }, .{ 5, 1 }, .{ 63, 1 }, .{ 0, 2 },   .{ 3, 7 },
            .{ 8, 8 }, .{ 9, 9 }, .{ 0, 16 }, .{ 17, 17 }, .{ 40, 24 },
        };
        for (windows) |win| {
            try exl3.linear(&t.on, at(dx) + win[0] * k * 2, tr, at(dsvh), at(dw), 0, win[1], k, n, t.stream.handle);
            try t.stream.synchronize();
            const got = try rig.download(f32, dw);
            defer gpa.free(got);
            const want = std.mem.sliceAsBytes(all[win[0] * n ..][0 .. win[1] * n]);
            if (std.mem.indexOfDiff(u8, want, std.mem.sliceAsBytes(got[0 .. win[1] * n]))) |i| {
                const at_ = .{ k, n, win[0], win[1], i };
                std.debug.print("k {d} n {d}: rows {d}+{d} differ from the 64-row run at byte {d}\n", at_);
                return error.TestUnexpectedResult;
            }
        }
    }
}

test "rot_in rounds the fp32 butterfly of x * suh to fp16 once, from fp32, fp16 and bf16 rows" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xE3D3 };
    // 391,680 roundings over the three kinds, so the rare fp16 output that a 1-ulp fp32 error moves shows up
    const k = 2048 + 128;
    const rows = 60;
    const suh = try randomHalves(&rng, k);
    defer gpa.free(suh);
    var dsuh = try t.upload(suh);
    defer dsuh.free();
    var dxh = try t.alloc(rows * k * 2);
    defer dxh.free();
    const xs = try gpa.alloc(f32, rows * k);
    defer gpa.free(xs);
    for (xs) |*v| v.* = rng.unit() * 3;
    for ([_]c_int{ 0, 1, 2 }) |kind| {
        const in = try gpa.alloc(u32, rows * k);
        defer gpa.free(in);
        const vals = try gpa.alloc(f32, rows * k);
        defer gpa.free(vals);
        for (xs, in, vals) |v, *b, *e| {
            b.* = switch (kind) {
                0 => @bitCast(v),
                1 => rig.f16Bits(v),
                else => rig.bf16Bits(v),
            };
            e.* = switch (kind) {
                0 => v,
                1 => @floatCast(rig.f16Value(@intCast(b.*))),
                else => @floatCast(rig.bf16Value(@intCast(b.*))),
            };
        }
        var dx = if (kind == 0) try t.upload(in) else blk: {
            const narrow = try gpa.alloc(u16, rows * k);
            defer gpa.free(narrow);
            for (narrow, in) |*a, b| a.* = @intCast(b);
            break :blk try t.upload(narrow);
        };
        defer dx.free();
        try exl3.rotIn(&t.on, at(dx), kind, at(dsuh), at(dxh), rows, k, t.stream.handle);
        try t.stream.synchronize();
        const got = try rig.download(u16, dxh);
        defer gpa.free(got);
        for (0..rows) |r| for (0..k / 128) |b| {
            var v: [128]f32 = undefined;
            for (&v, 0..) |*e, q| {
                const sc: f32 = @floatCast(rig.f16Value(suh[b * 128 + q]));
                e.* = vals[r * k + b * 128 + q] * sc;
            }
            fwht(&v);
            for (v, 0..) |e, q| {
                const want: u16 = @bitCast(@as(f16, @floatCast(e * had)));
                try std.testing.expectEqual(want, got[r * k + b * 128 + q]);
            }
        };
    }
}
