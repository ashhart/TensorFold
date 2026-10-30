//! Host tests of the KV accounting: the pieces of a sequence's bytes.

const std = @import("std");
const hub = @import("cuda_exl3_kv.zig");

const seqBytes = hub.seqBytes;

test "exl3: kv the pieces of a sequence sum to upstream's seqBytes" {
    const prompt = @import("../../cuda_prompt.zig");
    for ([_]usize{ 1, 5, 4096, 200_016, 262_144 }) |cap| {
        const s = seqBytes(cap);
        try std.testing.expectEqual(@as(u64, prompt.seqBytes(cap)), s.device());
    }
}

test "exl3: kv a token's K and V are upstream's row twice" {
    const prompt = @import("../../cuda_prompt.zig");
    try std.testing.expectEqual(@as(usize, 2048), hub.kvRowBytes());
    try std.testing.expectEqual(2 * prompt.kv_row_bytes, hub.kvRowBytes());
}
