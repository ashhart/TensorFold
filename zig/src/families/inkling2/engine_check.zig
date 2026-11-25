//! tf-inkling2-engine's host parts: its options, image placeholder widening and a prompt pass's last logits row.
const std = @import("std");
const image = @import("image.zig");
const engine = @import("engine.zig");

pub const Options = struct {
    graphs: bool,
    dir: []const u8,
    ids: []const u8,
    out: []const u8,
    context: usize = 32768,
    new: usize = 64,
    streams: usize = 1, // greedy: texts decoded together, each on its own slot
    image: ?[]const u8 = null, // the picture for every placeholder run
    virtual: bool = false, // greedy at two ranks on one GPU
};

/// The tool's words after its name; graphs runs one stream of text at a time on real ranks only.
pub fn parse(args: []const []const u8) !Options {
    if (args.len < 4) return error.Usage;
    const graphs = std.mem.eql(u8, args[0], "graphs");
    if (!graphs and !std.mem.eql(u8, args[0], "greedy")) return error.Usage;
    var o: Options = .{ .graphs = graphs, .dir = args[1], .ids = args[2], .out = args[3] };
    var plain: usize = 0;
    var i: usize = 4;
    while (i < args.len) : (i += 1) {
        const w = args[i];
        const more = i + 1 < args.len;
        if (std.mem.eql(u8, w, "--virtual")) {
            o.virtual = true;
        } else if (std.mem.eql(u8, w, "--streams") and more) {
            i += 1;
            o.streams = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, w, "--image") and more) {
            i += 1;
            o.image = args[i];
        } else if (std.mem.startsWith(u8, w, "--") or plain == 2) {
            return error.Usage;
        } else {
            const v = try std.fmt.parseInt(usize, w, 10);
            if (plain == 0) o.context = v else o.new = v;
            plain += 1;
        }
    }
    if (o.streams == 0 or o.streams > engine.max_shared) return error.Usage;
    if (graphs and (o.virtual or o.image != null or o.streams != 1)) return error.Usage;
    return o;
}

/// Each run of image placeholders widened to `n` of them, as the server expands a picture's one placeholder.
pub fn widen(a: std.mem.Allocator, ids: []const u32, n: u32) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    var k: usize = 0;
    while (k < ids.len) : (k += 1) {
        if (ids[k] != image.image_token) {
            try out.append(a, ids[k]);
            continue;
        }
        while (k + 1 < ids.len and ids[k + 1] == image.image_token) k += 1;
        try out.appendNTimes(a, image.image_token, n);
    }
    return out.toOwnedSlice(a);
}

/// The row of a prompt pass's logits prefillChunk draws from: its last chunk runs in calls of `most` rows.
pub fn lastRow(len: usize, most: usize) usize {
    const start = (len - 1) / engine.prompt_rows * engine.prompt_rows;
    return (len - start - 1) % most;
}

test "engine check: options take numbers in order and refuse what graphs or greedy does not run" {
    const o = try parse(&.{ "greedy", "m", "ids.json", "out", "131072", "8", "--streams", "4", "--virtual" });
    try std.testing.expectEqual(@as(usize, 131072), o.context);
    try std.testing.expectEqual(@as(usize, 8), o.new);
    try std.testing.expectEqual(@as(usize, 4), o.streams);
    try std.testing.expect(o.virtual and !o.graphs and o.image == null);
    const img = try parse(&.{ "greedy", "m", "ids.json", "out", "--image", "cat.png" });
    try std.testing.expectEqualStrings("cat.png", img.image.?);
    try std.testing.expectEqual(@as(usize, 32768), img.context);
    const refused = [_][]const []const u8{
        &.{ "graphs", "m", "i", "o", "--virtual" },
        &.{ "graphs", "m", "i", "o", "--streams", "2" },
        &.{ "graphs", "m", "i", "o", "--image", "x.png" },
        &.{ "greedy", "m", "i", "o", "--streams", "0" },
        &.{ "greedy", "m", "i", "o", "--streams", "17" },
        &.{ "greedy", "m", "i", "o", "--virtual-ranks" },
        &.{ "greedy", "m", "i", "o", "1", "2", "3" },
        &.{ "greedy", "m", "i" },
        &.{ "serve", "m", "i", "o" },
    };
    for (refused) |r| try std.testing.expectError(error.Usage, parse(r));
}

test "engine check: a picture's placeholder run widens to its visual tokens, other ids stay" {
    const t = image.image_token;
    const got = try widen(std.testing.allocator, &.{ 5, t, t, 6, t, 7 }, 3);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualSlices(u32, &.{ 5, t, t, t, 6, t, t, t, 7 }, got);
}

test "engine check: the last row is the last call's last row, as prefillChunk walks the last chunk" {
    for ([_]usize{ 1, 2047, 2048, 2049, 4096, 5000 }) |len| for ([_]usize{ 1, 7, 512, 2048 }) |most| {
        const start = (len - 1) / engine.prompt_rows * engine.prompt_rows;
        var at = start;
        var n: usize = 0;
        while (at < len) : (at += n) n = @min(len - at, most); // prefillChunk's loop over the last chunk
        try std.testing.expectEqual(n - 1, lastRow(len, most));
    };
}
