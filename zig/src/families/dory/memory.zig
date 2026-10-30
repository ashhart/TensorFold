//! What Dory holds on the GPU: weight bytes as stored, KV bytes by loops, context and streams (per_loop caches).
const std = @import("std");
const config = @import("config.zig");
const schedule = @import("schedule.zig");

const Config = config.Config;

/// A projection's bytes as stored: NVFP4 codes, scales and the fp32 scale, MXFP8 codes and u8 scales, or bf16.
pub fn projBytes(c: Config, n: usize, k: usize) usize {
    return switch (c.quant) {
        .nvfp4 => n * k / 2 + n * k / 16 + 4,
        .mxfp8 => n * k + n * k / 32,
        .bf16 => n * k * 2,
    };
}

/// Every tensor's bytes, as the checkpoint stores them (the loader's packs pad rows to 128 on top).
pub fn weightBytes(c: Config) usize {
    const d = c.hidden;
    var total: usize = 2 * (2 * c.vocab * d) + 2 * c.vocab; // embeddings, head, logit_scale
    for (c.kinds[0..c.sublayers]) |k| total += 2 * d + if (k == .mlp)
        2 * projBytes(c, c.intermediate, d) + projBytes(c, d, c.intermediate) + 2 * 2 * c.intermediate
    else
        2 * projBytes(c, c.qDim(), d) + 2 * projBytes(c, c.kvDim(), d) + projBytes(c, d, c.qDim()) + 2 * 2 * c.qDim();
    return total;
}

/// Parameters, counting each NVFP4 or bf16 weight once (scales not counted).
pub fn parameters(c: Config) usize {
    const d = c.hidden;
    var total: usize = 2 * c.vocab * d + c.vocab;
    for (c.kinds[0..c.sublayers]) |k| total += d + if (k == .mlp) 3 * c.intermediate * d + 2 * c.intermediate else 2 * c.qDim() * d + 2 * c.kvDim() * d + d * c.qDim() + 2 * c.qDim();
    return total;
}

/// bf16 K and V of one token in one cache.
pub fn tokenBytes(c: Config) usize {
    return 2 * c.kvDim() * 2;
}

/// One stream's KV at `c.loops`: full caches hold `context` positions, sliding caches a ring of `ring` keys.
pub fn streamBytes(c: Config, context: usize, ring: usize) usize {
    const s = schedule.build(c);
    return (s.cacheCount(.full) * context + s.cacheCount(.sliding) * ring) * tokenBytes(c);
}

const testing = std.testing;

fn pinned() !Config {
    const text = try config.pinnedText(testing.allocator);
    defer testing.allocator.free(text);
    var why: config.Why = .{};
    return config.parse(testing.allocator, text, &why);
}

test "3.97B parameters; the tensors' bytes equal each pinned checkpoint's data (file size less its header)" {
    const c = try pinned();
    for ([_]config.Quant{ .nvfp4, .mxfp8, .bf16 }) |quant| {
        var q = c;
        q.quant = quant;
        try testing.expectEqual(@as(usize, 3_975_229_440), parameters(q));
    }
    // model.safetensors at 41f02335: 8 bytes of length, a 121,192-byte header, then the tensors
    try testing.expectEqual(@as(usize, 3_202_498_032 - 8 - 121_192), weightBytes(c));
    // the published MXFP8 model.safetensors: 8 bytes of length, a 91,616-byte header, then the tensors
    var mx = c;
    mx.quant = .mxfp8;
    try testing.expectEqual(@as(usize, 4_750_755_304 - 8 - 91_616), weightBytes(mx));
    // the published BF16 model.safetensors: 8 bytes of length, a 56,584-byte header, then the tensors
    var bf = c;
    bf.quant = .bf16;
    try testing.expectEqual(@as(usize, 7_950_515_472 - 8 - 56_584), weightBytes(bf));
}

test "KV a token: 36, 56, 96 KiB at 1, 2, 4 loops; 7 GiB at 128K and two loops, plus fixed sliding rings" {
    const c = try pinned();
    try testing.expectEqual(@as(usize, 4096), tokenBytes(c));
    for ([_][2]usize{ .{ 1, 36 }, .{ 2, 56 }, .{ 4, 96 } }) |row| {
        const cl = try c.withLoops(row[0]);
        try testing.expectEqual(row[1] * 1024, streamBytes(cl, 1, 0));
    }
    // 14 full caches over 128K keys, 42 sliding rings of 1,024 keys: 7 GiB + 168 MiB
    try testing.expectEqual((@as(usize, 7) << 30) + (168 << 20), streamBytes(try c.withLoops(2), 128 << 10, 1024));
    try testing.expectEqual((@as(usize, 24) << 30) + (288 << 20), streamBytes(try c.withLoops(4), 256 << 10, 1024));
}
