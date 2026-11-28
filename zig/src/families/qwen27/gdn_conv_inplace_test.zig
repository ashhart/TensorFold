//! Both conv commits, modelled over every small parent tree: same bytes, and no thread reads another's write.
const std = @import("std");

const channels = 3;
const Keep = struct { first: u32, rows: u32, slot: u32 };

/// `windows` as validatePlan requires them: old taps below `taps - 1`, then projected rows.
fn fillWindows(windows: []u32, parents: []const i32, taps: usize) void {
    for (parents, 0..) |_, row| for (0..taps) |tap| {
        var ancestor: i32 = @intCast(row);
        var hops: usize = 0;
        while (hops < taps - 1 - tap and ancestor >= 0) : (hops += 1) ancestor = parents[@intCast(ancestor)];
        windows[row * taps + tap] = @intCast(if (ancestor >= 0) taps - 1 + @as(usize, @intCast(ancestor)) else taps - 1 - (taps - 1 - tap - hops + 1));
    };
}

/// tf_delta_gdn_conv_commit then tf_delta_gdn_publish_conv.
fn commitTwice(conv: []u16, next: []u16, projection: []const u16, kept_rows: []const u32, keeps: []const Keep, windows: []const u32, taps: usize) void {
    const kept = taps - 1;
    const width = kept * channels;
    for (keeps) |keep| for (0..width) |item| {
        next[keep.slot * width + item] = if (keep.rows == 0) conv[keep.slot * width + item] else blk: {
            const row = kept_rows[keep.first + keep.rows - 1];
            const source = windows[row * taps + item / channels + 1];
            break :blk if (source < kept) conv[(keep.slot * kept + source) * channels + item % channels] else projection[(source - kept) * channels + item % channels];
        };
    };
    for (keeps) |keep| @memcpy(conv[keep.slot * width ..][0..width], next[keep.slot * width ..][0..width]);
}

/// tf_delta_gdn_conv_commit_in_place, one thread a channel and keep; returns false if a thread touches another's word.
fn commitInPlace(conv: []u16, projection: []const u16, kept_rows: []const u32, keeps: []const Keep, windows: []const u32, taps: usize) bool {
    const kept = taps - 1;
    for (keeps) |keep| for (0..channels) |channel| {
        if (keep.rows == 0) continue;
        const row = kept_rows[keep.first + keep.rows - 1];
        const base = keep.slot * kept;
        var words: [7]u16 = undefined;
        for (0..kept) |t| {
            const s = windows[row * taps + t + 1];
            if (s < kept) {
                const at = (base + s) * channels + channel;
                if (at % channels != channel or at / channels / kept != keep.slot) return false;
                words[t] = conv[at];
            } else words[t] = projection[(s - kept) * channels + channel];
        }
        for (0..kept) |t| conv[(base + t) * channels + channel] = words[t];
    };
    return true;
}

test "qwen27 speed: in-place conv commit stores the two-pass bytes for every parent tree, path and tap count" {
    var prng = std.Random.DefaultPrng.init(0x2715);
    const random = prng.random();
    var cases: usize = 0;
    for (2..9) |taps| {
        const kept = taps - 1;
        const slots = 3;
        var conv: [slots * 7 * channels]u16 = undefined;
        var twice: [slots * 7 * channels]u16 = undefined;
        var next: [slots * 7 * channels]u16 = undefined;
        var projection: [5 * channels]u16 = undefined;
        for (1..6) |rows| {
            var parents: [5]i32 = undefined;
            var windows: [5 * 8]u32 = undefined;
            var code: usize = 0;
            var total: usize = 1;
            for (1..rows) |r| total *= r + 1;
            while (code < total) : (code += 1) {
                parents[0] = -1;
                var rest = code;
                for (1..rows) |r| {
                    parents[r] = @as(i32, @intCast(rest % (r + 1))) - 1;
                    if (parents[r] < 0) parents[r] = 0;
                    rest /= r + 1;
                }
                fillWindows(&windows, parents[0..rows], taps);
                for (0..rows) |last| {
                    var path: [5]u32 = undefined;
                    var depth: usize = 0;
                    var at: i32 = @intCast(last);
                    while (at >= 0) : (at = parents[@intCast(at)]) {
                        path[depth] = @intCast(at);
                        depth += 1;
                    }
                    std.mem.reverse(u32, path[0..depth]);
                    const keeps = [_]Keep{ .{ .first = 0, .rows = @intCast(depth), .slot = 0 }, .{ .first = 0, .rows = 0, .slot = 2 } };
                    for (&conv) |*word| word.* = random.int(u16); // every bit pattern, NaNs included
                    for (&projection) |*word| word.* = random.int(u16);
                    twice = conv;
                    commitTwice(&twice, &next, &projection, path[0..depth], &keeps, &windows, taps);
                    try std.testing.expect(commitInPlace(&conv, &projection, path[0..depth], &keeps, &windows, taps));
                    try std.testing.expectEqualSlices(u16, twice[0 .. slots * kept * channels], conv[0 .. slots * kept * channels]);
                    cases += 1;
                }
            }
        }
    }
    try std.testing.expect(cases > 1000);
}
