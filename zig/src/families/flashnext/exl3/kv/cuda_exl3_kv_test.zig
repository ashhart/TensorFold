//! Host tests of the KV accounting: the pieces of a sequence's bytes.

const std = @import("std");
const hub = @import("cuda_exl3_kv.zig");

const seqBytes = hub.seqBytes;

test "exl3: kv the pieces of a sequence sum to upstream's seqBytes" {
    const prompt = @import("../../cuda_prompt.zig");
    for ([_]usize{ 1, 5, 4096, 200_016, 262_144 }) |cap| {
        const s = seqBytes(cap);
        try std.testing.expectEqual(@as(u64, prompt.seqBytes(cap, .full)), s.device());
    }
}

test "exl3: kv a prompt cut is the same busy or solo, with or without TF_FLASHNEXT_CHUNK" {
    const Rows = struct { chunk: usize, busy_chunk: usize };
    var unset = Rows{ .chunk = 1536, .busy_chunk = 1536 };
    hub.cut(&unset, false);
    try std.testing.expectEqual(hub.exl3_chunk, unset.chunk);
    try std.testing.expectEqual(unset.chunk, unset.busy_chunk);
    var set = Rows{ .chunk = 8192, .busy_chunk = 4096 };
    hub.cut(&set, true);
    try std.testing.expectEqual(@as(usize, 4096), set.chunk);
    try std.testing.expectEqual(set.chunk, set.busy_chunk);
}

test "exl3: kv TF_FLASHNEXT_CHUNK_SOLO is refused for an EXL3 pack only" {
    try std.testing.expectError(error.Exl3ChunkSoloUnsupported, hub.refuseSolo(true, true));
    try hub.refuseSolo(true, false);
    try hub.refuseSolo(false, true);
}

test "exl3: kv a non-EXL3 scratch keeps main's head_local rows below 32 pass rows, EXL3 lanes their cap" {
    const prompt = @import("../../cuda_prompt.zig");
    const state = @import("../../cuda_batch_state.zig");
    var coll: @import("../../cuda_collective.zig").Collective = undefined;
    var e: prompt.Engine = undefined;
    e.batch_k.native_lane.clusters = false;
    e.exl3_lanes = false;
    e.head_local_rows = std.math.maxInt(usize);
    for ([_]bool{ false, true }) |split| {
        e.collective = if (split) &coll else null;
        e.scratch_format = .classic;
        for ([_]usize{ 1, 4, 16, 31, 32, 64, 1024 }) |rows| {
            // main's reserve: never below head_rows_one for the one-GPU draft head's lanes, a split head a pass's rows
            const main: usize = if (split) @max(rows, state.head_rows_one) else state.head_rows_one;
            try std.testing.expectEqual(main, e.scratchSizes(rows).head_rows);
        }
    }
    // an EXL3 decoder, with an EXL3 or a classic draft, keeps the stack's cap: a pass's rows, then head_local_rows
    e.exl3_lanes = true;
    for ([_]@TypeOf(e.scratch_format){ .exl3, .classic }) |format| {
        e.scratch_format = format;
        e.collective = null;
        e.head_local_rows = hub.headLocalRows(true);
        try std.testing.expectEqual(@as(usize, 16), e.scratchSizes(16).head_rows);
        try std.testing.expectEqual(state.head_rows_one, e.scratchSizes(1024).head_rows);
        e.collective = &coll;
        e.head_local_rows = hub.headLocalRows(false);
        try std.testing.expectEqual(@as(usize, 16), e.scratchSizes(16).head_rows);
        try std.testing.expectEqual(hub.head_local_serve_rows, e.scratchSizes(1024).head_rows);
    }
}
