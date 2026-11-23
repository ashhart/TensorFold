//! The blocked row scan and the mask-word draw against the scalar walk they replace, over rows built to break them.
const std = @import("std");
const row = @import("sampling_row.zig");
const sampling = @import("sampling.zig");
const grammar = @import("grammar.zig");
const Sampling = sampling.Sampling;
const value = row.value;
const vocab = 248320;

/// The scalar first pass the scan replaces: every word checked finite, then the first strictly larger allowed value.
fn walkScan(logits: []const u16, mask: []const u32) !row.Scan {
    var best: ?usize = null;
    var allowed: usize = 0;
    for (logits, 0..) |word, id| {
        const x = value(word);
        if (!std.math.isFinite(x)) return error.NonfiniteLogits;
        if (mask.len > 0 and !grammar.allows(mask, id)) continue;
        allowed += 1;
        if (best == null or x > value(logits[best.?])) best = id;
    }
    return .{ .best = best, .allowed = allowed };
}

/// The whole scalar draw the blocked one replaces, kept as the oracle.
fn walk(gpa: std.mem.Allocator, logits: []const u16, settings: ?Sampling, position: u64, mask: []const u32) !u32 {
    const first = try walkScan(logits, mask);
    const best = first.best orelse return error.NoAllowedToken;
    const allowed = first.allowed;
    const s = settings orelse return @intCast(best);
    if (s.temperature <= 0) return @intCast(best);
    const k = if (s.top_k == 0) allowed else @min(allowed, s.top_k);
    const values = try gpa.alloc(f64, k);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, k);
    defer gpa.free(ids);
    var filled: usize = 0;
    for (logits, 0..) |word, id| {
        if (mask.len > 0 and !grammar.allows(mask, id)) continue;
        const x = value(word);
        if (k == allowed) {
            values[filled] = x;
            ids[filled] = id;
            filled += 1;
            continue;
        }
        if (filled == k and x <= values[k - 1]) continue;
        var at = @min(filled, k - 1);
        while (at > 0 and x > values[at - 1]) : (at -= 1) {
            values[at] = values[at - 1];
            ids[at] = ids[at - 1];
        }
        values[at] = x;
        ids[at] = id;
        filled = @min(filled + 1, k);
    }
    return @intCast(try sampling.choose(gpa, values, ids, position, s));
}

fn expectScan(logits: []const u16, mask: []const u32) !void {
    const want = walkScan(logits, mask);
    const got = row.scan(logits, mask);
    if (want) |w| {
        const g = try got;
        try std.testing.expectEqual(w.best, g.best);
        try std.testing.expectEqual(w.allowed, g.allowed);
    } else |err| try std.testing.expectError(err, got);
}

fn expectDraw(gpa: std.mem.Allocator, logits: []const u16, s: ?Sampling, position: u64, mask: []const u32) !void {
    const want = walk(gpa, logits, s, position, mask);
    const got = row.draw(gpa, logits, s, position, mask);
    if (want) |w| try std.testing.expectEqual(w, try got) else |err| try std.testing.expectError(err, got);
}

/// A finite word of one of several shapes: wide, a few tied values, zeros and subnormals, or near one.
fn randomWord(r: std.Random, style: usize) u16 {
    const w: u16 = switch (style) {
        0 => r.int(u16),
        1 => 0x4000 | r.uintLessThan(u16, 3),
        2 => switch (r.uintLessThan(u8, 4)) {
            0 => 0,
            1 => 0x8000,
            2 => r.uintLessThan(u16, 0x80),
            else => 0x8000 | r.uintLessThan(u16, 0x80),
        },
        else => 0x3f80 + r.uintLessThan(u16, 8),
    };
    return if (w & 0x7f80 == 0x7f80) w ^ 0x0080 else w;
}

/// A mask of one of several shapes, possibly shorter or longer than the row's words, with bits past the row set.
fn fillMask(r: std.Random, mask: []u32, n: usize, shape: usize) []const u32 {
    const words = (n + 31) / 32;
    @memset(mask, 0);
    switch (shape) {
        0 => return &.{},
        1 => return mask[0..words], // nothing allowed
        2 => @memset(mask[0..words], std.math.maxInt(u32)),
        3 => {
            const id = r.uintLessThan(usize, n);
            mask[id / 32] = @as(u32, 1) << @intCast(id % 32);
        },
        4 => mask[r.uintLessThan(usize, words)] = 0x80000000, // bit 31 alone, maybe past the row
        5 => for (mask[0..words]) |*m| {
            m.* = r.int(u32);
        },
        6 => for (0..1 + r.uintLessThan(usize, 24)) |_| {
            const id = r.uintLessThan(usize, n);
            mask[id / 32] |= @as(u32, 1) << @intCast(id % 32);
        },
        7 => {
            for (mask[0..words]) |*m| m.* = r.int(u32);
            return mask[0 .. 1 + r.uintLessThan(usize, words)]; // short: ids past it are not allowed
        },
        else => {
            for (mask[0..words]) |*m| m.* = if (r.boolean()) 0 else r.int(u32);
            for (mask[words..]) |*m| m.* = std.math.maxInt(u32); // long: words past the row change nothing
            return mask;
        },
    }
    if (n % 32 != 0 and r.boolean()) mask[words - 1] |= ~@as(u32, 0) << @intCast(n % 32); // bits past the row
    return mask[0..words];
}

/// Plants a largest value at several ids (often tied, often masked) and sometimes a non-finite word.
fn plant(r: std.Random, logits: []u16, case: usize) void {
    const n = logits.len;
    for (0..r.uintLessThan(usize, 4)) |_| logits[r.uintLessThan(usize, n)] = if (r.boolean()) 0x4700 else 0x46ff;
    if (case % 3 == 0) logits[n - 1] = 0x4700; // the tail's last id
    if (case % 7 == 2) {
        const nonfinite = [_]u16{ 0x7f80, 0xff80, 0x7fc0, 0xffc1, 0x7f81 };
        logits[r.uintLessThan(usize, n)] = nonfinite[r.uintLessThan(usize, nonfinite.len)];
    }
}

test "the blocked row scan equals the scalar walk over sizes, ties, zeros, masks, tails and non-finite words" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x2832);
    const r = prng.random();
    const logits = try gpa.alloc(u16, vocab + 97);
    defer gpa.free(logits);
    const mask = try gpa.alloc(u32, (vocab + 97 + 31) / 32 + 3);
    defer gpa.free(mask);
    const sizes = [_]usize{ 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 4099, 4096, vocab - 1, vocab, vocab + 97 };
    for (sizes) |n| {
        const cases: usize = if (n > 5000) 36 else 360;
        for (0..cases) |case| {
            const style = case % 4;
            for (logits[0..n]) |*w| w.* = randomWord(r, style);
            plant(r, logits[0..n], case);
            const m = fillMask(r, mask[0 .. (n + 31) / 32 + 3], n, (case / 4) % 9);
            try expectScan(logits[0..n], m);
            try expectDraw(gpa, logits[0..n], null, 0, m);
            try expectDraw(gpa, logits[0..n], .{ .seed = 3, .temperature = 0 }, 5, m);
        }
    }
}

test "the mask-word draw keeps the walk's candidates and sampled tokens" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x2832_5a);
    const r = prng.random();
    const logits = try gpa.alloc(u16, vocab);
    defer gpa.free(logits);
    const mask = try gpa.alloc(u32, (vocab + 31) / 32 + 3);
    defer gpa.free(mask);
    for ([_]usize{ 1, 33, 64, 65, 4099, vocab }) |n| {
        for (0..if (n > 5000) 27 else 81) |case| {
            for (logits[0..n]) |*w| w.* = randomWord(r, case % 4);
            plant(r, logits[0..n], case);
            const shape = (case / 3) % 9;
            const m = fillMask(r, mask[0 .. (n + 31) / 32 + 3], n, shape);
            // a wide row samples all allowed ids (top_k 0) only under a sparse mask: the fp64 sampler stays short
            const ks = [_]u32{ 0, 1, 7, 20, 300 };
            const sparse = shape == 1 or shape == 3 or shape == 4 or shape == 6;
            for (if (n > 5000 and !sparse) ks[1..] else &ks) |k| {
                const s: Sampling = .{ .seed = 41 + case, .top_k = k, .temperature = 0.8, .top_p = 0.9, .min_p = 0.02 };
                try expectDraw(gpa, logits[0..n], s, 1000 + case, m);
            }
        }
    }
}

test "the blocked row scan keeps the first of two equal or adjacent words in a block, the next block or the tail" {
    var logits: [136]u16 = @splat(0xff7f);
    for (0..65536) |bits| {
        const a: u16 = @intCast(bits);
        if (a & 0x7f80 == 0x7f80) continue;
        for ([_]u16{ a, a ^ 0x8000, a +% 1 }) |b| {
            if (b & 0x7f80 == 0x7f80) continue;
            for ([_]usize{ 40, 100, 130 }) |at| {
                logits[3] = a;
                logits[at] = b;
                try expectScan(&logits, &.{});
                logits[3] = 0xff7f;
                logits[at] = 0xff7f;
            }
        }
    }
}

test "the blocked row scan refuses every non-finite encoding in a block nothing is allowed from" {
    var logits: [200]u16 = @splat(0x3f80);
    var mask: [7]u32 = @splat(0);
    mask[6] = 1; // id 192 alone is allowed; ids 0..191 are three blocks without an allowed id
    for (0..65536) |bits| {
        const w: u16 = @intCast(bits);
        if (w & 0x7f80 != 0x7f80) continue;
        for ([_]usize{ 0, 63, 64, 130, 199 }) |at| {
            logits[at] = w;
            try std.testing.expectError(error.NonfiniteLogits, row.scan(&logits, &mask));
            try std.testing.expectError(error.NonfiniteLogits, row.scan(&logits, mask[0..1]));
            logits[at] = 0x3f80;
        }
    }
    try std.testing.expectEqual(@as(?usize, 192), (try row.scan(&logits, &mask)).best);
    try std.testing.expectEqual(@as(?usize, null), (try row.scan(&logits, mask[0..6])).best);
    try std.testing.expectError(error.NoAllowedToken, row.draw(std.testing.allocator, &logits, null, 0, mask[0..6]));
    try std.testing.expectError(error.NoAllowedToken, row.draw(std.testing.allocator, &.{}, null, 0, &.{}));
}
