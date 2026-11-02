//! Clef's record encoder on a byte tokenizer: render as json.dumps, and the spans of the rendered pieces.
const std = @import("std");
const json = @import("json");
const enc = @import("clef_encode.zig");

/// One id per byte, and each special token whole: ids read back as the text they came from.
const Bytes = struct {
    const specials = [_][]const u8{ "<|im_start|>", "<|im_end|>" };
    pub fn ids(_: Bytes, a: std.mem.Allocator, text: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        outer: while (i < text.len) {
            for (specials, 0..) |s, k| if (std.mem.startsWith(u8, text[i..], s)) {
                try out.append(a, @intCast(256 + k));
                i += s.len;
                continue :outer;
            };
            try out.append(a, text[i]);
            i += 1;
        }
        return out.items;
    }
};

fn spelled(a: std.mem.Allocator, ids: []const u32) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (ids) |id| if (id >= 256) try out.appendSlice(a, Bytes.specials[id - 256]) else try out.append(a, @intCast(id));
    return out.items;
}

test "Clef: render is json.dumps with sorted keys, compact separators and text as itself" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = (try json.parseText(a, "{\"b\":[1,2.5,1e20,-0.0,true,null],\"a\":\"caf\\u00e9 \\\"q\\\"\\n\",\"\\u00e9\":{\"z\":1,\"y\":0.1}}")).ok;
    try std.testing.expectEqualStrings("{\"a\":\"café \\\"q\\\"\\n\",\"b\":[1,2.5,1e+20,-0.0,true,null],\"é\":{\"y\":0.1,\"z\":1}}", try enc.render(a, parsed));
    try std.testing.expectEqualStrings("plain text", try enc.render(a, .{ .string = "plain text" }));
}

test "Clef: a record's spans cover its rendered instruction and option semantics" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = "{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"choice\",\"instructions\":\"Pick\",\"criteria\":{\"b\":\"B\",\"a\":null}}}}";
    const r = try enc.encode(a, Bytes{}, (try json.parseText(a, body)).ok, 16384);
    const q = r.questions[0];
    try std.testing.expectEqualStrings("Pick", try spelled(a, r.ids[q.span[0]..q.span[1]]));
    try std.testing.expectEqualStrings("{\"option_id\":\"a\"}", try spelled(a, r.ids[q.options[0][0]..q.options[0][1]]));
    try std.testing.expectEqualStrings("{\"description\":\"B\",\"option_id\":\"b\"}", try spelled(a, r.ids[q.options[1][0]..q.options[1][1]]));
    try std.testing.expectEqual(@as(u32, 1), r.state_rows);
}
