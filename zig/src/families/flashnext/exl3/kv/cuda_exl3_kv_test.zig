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

test "exl3: kv a prompt cut is the same busy or solo, with or without TF_FLASHNEXT_CHUNK" {
    const Rows = struct { chunk: usize, busy_chunk: usize };
    var unset = Rows{ .chunk = 1536, .busy_chunk = 1536 };
    hub.cut(&unset, false);
    try std.testing.expectEqual(hub.exl3_chunk, unset.chunk);
    try std.testing.expectEqual(unset.chunk, unset.busy_chunk);
    var set = Rows{ .chunk = 4096, .busy_chunk = 4096 };
    hub.cut(&set, true);
    try std.testing.expectEqual(@as(usize, 4096), set.chunk);
    try std.testing.expectEqual(set.chunk, set.busy_chunk);
}

test "exl3: kv TF_FLASHNEXT_CHUNK_SOLO is refused for an EXL3 pack only" {
    try std.testing.expectError(error.Exl3ChunkSoloUnsupported, hub.refuseSolo(true, true));
    try hub.refuseSolo(true, false);
    try hub.refuseSolo(false, true);
}
