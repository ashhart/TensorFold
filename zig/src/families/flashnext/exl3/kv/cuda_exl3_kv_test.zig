//! Host tests of the KV placement: policy bytes and kernel smem.

const std = @import("std");
const hub = @import("cuda_exl3_kv.zig");
const int8k = @import("cuda_exl3_kv8.zig");

const kvRowBytes = hub.kvRowBytes;
const seqBytes = hub.seqBytes;
const chunkSmem = int8k.chunkSmem;

test "exl3: kv the default policy is upstream's seqBytes, split" {
    const prompt = @import("../../cuda_prompt.zig");
    for ([_]usize{ 1, 5, 4096, 200_016, 262_144 }) |cap| {
        const s = seqBytes(cap, .{});
        try std.testing.expectEqual(@as(u64, prompt.seqBytes(cap)), s.device());
        try std.testing.expectEqual(@as(u64, 0), s.host);
    }
}

test "exl3: kv int8 is 34 bytes a 32-group, int4 18; host placement moves K/V and the indexer keys off the device" {
    try std.testing.expectEqual(@as(usize, 2048), kvRowBytes(.bf16, 2));
    try std.testing.expectEqual(@as(usize, 1088), kvRowBytes(.int8, 2));
    try std.testing.expectEqual(@as(usize, 576), kvRowBytes(.int4, 2));
    const cap = 200_000;
    const a = seqBytes(cap, .{ .kv = .int8 });
    const h = seqBytes(cap, .{ .kv = .int8, .kv_host = true, .ikc_host = true });
    try std.testing.expectEqual(a.kv + a.ikc, h.host);
    try std.testing.expectEqual(a.device() - a.kv - a.ikc, h.device());
}

test "exl3: kv chunk smem fits the 99 KiB opt-in on sm_120" {
    try std.testing.expect(chunkSmem(2) <= 99 * 1024);
    try std.testing.expectEqual(@as(u32, 60928), chunkSmem(1));
    try std.testing.expectEqual(@as(u32, 78336), chunkSmem(2));
}
