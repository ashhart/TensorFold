//! TF_Q35_DFLASH_ABSORB_BATCH's choice: one pass only with the switch, two or more streams, every absorb live.
const std = @import("std");
const lanes = @import("lanes");
const absorbs = @import("cuda_draft_absorb.zig");
const t = std.testing;

fn item(s: *lanes.Stream, n: u32, prompt: bool) lanes.drafter.Absorb {
    return .{ .stream = s, .features = .{ .buffer = 0x1000, .offset = 40, .rows = n, .row_bytes = 64, .space = .device, .dtype = .bf16 }, .start = 40, .follow = &.{}, .prompt = prompt };
}

test "batched absorb: only with the switch, two or more distinct streams, every one live on device bf16 rows" {
    var streams: [16]lanes.Stream = undefined;
    var many: [16]lanes.drafter.Absorb = undefined;
    for (&many, 0..) |*it, i| it.* = item(&streams[i], @intCast(i % 17), false); // 0 to 15 kept rows a stream
    for ([_]usize{ 4, 8, 16 }) |n| {
        try t.expect(absorbs.batches(true, many[0..n]));
        try t.expect(!absorbs.batches(false, many[0..n])); // the switch off: each stream's own absorb, as today
    }
    try t.expect(!absorbs.batches(true, many[0..1])); // one stream: its own absorb
    var mixed = many[0..4].*;
    mixed[2].prompt = true; // a prompt chunk keeps the prompt GEMM past 16 rows: not batched
    try t.expect(!absorbs.batches(true, &mixed));
    var twice = many[0..4].*;
    twice[3].stream = twice[1].stream; // a stream's rows twice: its ring writes stay in order
    try t.expect(!absorbs.batches(true, &twice));
    var host = many[0..4].*;
    host[0].features.space = .host;
    try t.expect(!absorbs.batches(true, &host));
    var wide = many[0..4].*;
    wide[1].features.rows = 200; // past one pass's rows
    try t.expect(!absorbs.batches(true, &wide));
}
