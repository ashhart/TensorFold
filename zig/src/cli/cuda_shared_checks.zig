//! Shared windows against serial decoding: several streams' continuations replayed in one forward a round.

const std = @import("std");
const lanes = @import("lanes");
const nemotron = @import("nemotron");
const lanes_cli = @import("cuda_lanes.zig");

const max_streams = nemotron.engine.max_streams;
const max_rows = nemotron.state.max_rows;

/// Compare serial streams with shared windows of varying splits and kept-row lengths.
pub fn widths(gpa: std.mem.Allocator, io: std.Io, e: *nemotron.Engine, path: []const u8, streams: usize, count: usize, sampling: ?lanes.Sampling) !u8 {
    if (streams < 2 or streams > max_streams) return error.BadStreams;
    var prompts = try lanes_cli.readPrompts(gpa, io, path);
    defer prompts.deinit();
    const ids = prompts.value.map.values();
    var seqs: [max_streams]*nemotron.state.Seq = undefined;
    var made: usize = 0;
    defer for (seqs[0..made]) |s| e.freeSeq(s);
    var refs: [max_streams]std.ArrayList(u32) = @splat(.empty);
    defer for (&refs) |*r| r.deinit(gpa);
    for (0..streams) |k| {
        seqs[k] = try e.newSeq();
        made += 1;
        e.bind(seqs[k]);
        try e.setSampling(sampling);
        const p = ids[k % ids.len];
        try refs[k].append(gpa, try e.prefill(p, null, null));
        while (refs[k].items.len < count) try refs[k].append(gpa, try e.step(refs[k].items[refs[k].items.len - 1], null));
        if (try e.prefill(p, null, null) != refs[k].items[0]) return error.FirstTokenDiffers;
    }
    var at: [max_streams]usize = @splat(0);
    var seen: [max_rows + 1]usize = @splat(0);
    var bad: [max_rows + 1]usize = @splat(0);
    var windows: usize = 0;
    var round: usize = 0;
    var bufs: [max_streams][max_rows]u32 = undefined;
    var keep: [max_streams]usize = undefined;
    var live: [max_streams]usize = undefined;
    while (true) : (round += 1) {
        var parts: [max_streams]nemotron.engine.Shared = undefined;
        var n: usize = 0;
        var room: usize = max_rows;
        var waiting: usize = 0;
        for (0..streams) |k| waiting += @intFromBool(at[k] + 1 < refs[k].items.len);
        for (0..streams) |k| {
            if (at[k] + 1 >= refs[k].items.len) continue;
            waiting -= 1;
            const cap = room - waiting; // at least a row for each stream after this one
            const width = 1 + (round * 7 + k * 5 + round / 3) % cap; // uneven splits come round in turn
            const rows = @min(width, refs[k].items.len - 1 - at[k]);
            @memcpy(bufs[k][0..rows], refs[k].items[at[k]..][0..rows]);
            keep[k] = rows;
            if ((round + k) % 3 == 2 and rows > 2) { // a wrong draft: the rows from it on read a token serial never fed
                keep[k] = rows / 2;
                bufs[k][keep[k]] = @intCast((bufs[k][keep[k]] + 1) % e.c.vocab);
            }
            parts[n] = .{ .seq = seqs[k], .ids = bufs[k][0..rows], .rows = rows };
            live[n] = k;
            n += 1;
            room -= rows;
        }
        if (n == 0) break;
        try e.verifyShared(parts[0..n]);
        windows += 1;
        for (parts[0..n], live[0..n], 0..) |p, k, j| {
            const drawn = try e.sharedTokens(j, p.rows);
            for (0..keep[k]) |r| {
                seen[p.rows] += 1;
                if (drawn[r] != refs[k].items[at[k] + r + 1]) bad[p.rows] += 1;
            }
            e.bind(seqs[k]);
            try e.commit(keep[k]);
            at[k] += keep[k];
        }
    }
    var total_bad: usize = 0;
    var total: usize = 0;
    for (1..max_rows + 1) |w| {
        total_bad += bad[w];
        total += seen[w];
        if (seen[w] > 0) std.debug.print("rows {d} a stream: {d} kept rows, {d} differ\n", .{ w, seen[w], bad[w] });
    }
    std.debug.print("{s} shared widths, {d} streams: {d} of {d} kept rows equal serial ({d} tokens a stream, {d} shared windows)\n", .{ if (total_bad == 0) "PASS" else "FAIL", streams, total - total_bad, total, count, windows });
    return if (total_bad == 0) 0 else 1;
}
