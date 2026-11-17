//! Message-start cuts in Nemotron's prompt pass: a resume cuts at the next start, not the grid step.
const std = @import("std");
const pass = @import("cuda_prompt.zig");
const t = std.testing;

/// Chunk ends a pass walks, as serialChunks does: each end is a cut where a state is kept.
fn walk(out: []usize, from: usize, len: usize, step: usize, chunks: []const u32, marks: []const u32) usize {
    var n: usize = 0;
    var at = from;
    while (at < len) {
        const end = pass.chunkEnd(at, len, step, chunks, marks);
        std.debug.assert(end > at);
        out[n] = end;
        n += 1;
        at = end;
    }
    return n;
}

test "nemotron: a fresh pass cuts at every message start, not at the grid step" {
    // A 4096 grid would run all 1000 rows in one chunk; the starts split it.
    const chunks = [_]u32{ 128, 300, 640 };
    var out: [8]usize = undefined;
    const n = walk(&out, 0, 1000, 4096, &chunks, &.{});
    try t.expectEqualSlices(usize, &.{ 128, 300, 640, 1000 }, out[0..n]);
}

test "nemotron: a resumed pass from a kept start cuts at the next start (the resume point)" {
    // Turn N's state was kept at 300 and the next turn resumes there. A fixed-grid resume would
    // run 300..1000 once, never computing the state at 640 a later turn resumes from.
    const chunks = [_]u32{ 128, 300, 640 };
    const marks = [_]u32{640};
    var out: [8]usize = undefined;
    const n = walk(&out, 300, 1000, 4096, &chunks, &marks);
    try t.expectEqualSlices(usize, &.{ 640, 1000 }, out[0..n]);
    // every cut before the prompt's end is a message start, so a state is kept there
    for (out[0 .. n - 1]) |end| try t.expect(std.mem.indexOfScalar(u32, &chunks, @intCast(end)) != null);
}

test "nemotron: a store mark off the grid ends a chunk on resume" {
    const marks = [_]u32{ 1000, 2200 };
    var out: [8]usize = undefined;
    const n = walk(&out, 0, 5000, 2048, &.{}, &marks);
    try t.expectEqualSlices(usize, &.{ 1000, 2200, 4248, 5000 }, out[0..n]);
}

test "nemotron: with no message starts the grid step is unchanged" {
    var out: [8]usize = undefined;
    const n = walk(&out, 0, 5000, 2048, &.{}, &.{});
    try t.expectEqualSlices(usize, &.{ 2048, 4096, 5000 }, out[0..n]);
}
