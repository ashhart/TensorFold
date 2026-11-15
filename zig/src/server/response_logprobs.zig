//! Logprob entries attached to the Responses items whose text they spell, after the entries those items already carry.
const std = @import("std");
const json = @import("json");
const V = json.Value;
const A = std.mem.Allocator;

/// New entries for one item: `owner` holds them, the item's text part or, for a call, the item itself.
pub const Added = struct { item: *json.Object, owner: *json.Object, rows: []V };

/// Reasoning and prose entries to the items of their kind in order, each filled up to its text before the next.
pub fn attach(a: A, items: []const *json.Object, rows: V, skip_reasoning: bool, skip_content: bool, out: *std.ArrayList(Added)) A.Error!void {
    for ([_]struct { []const u8, []const u8, bool }{ .{ "reasoning", "reasoning_content", skip_reasoning }, .{ "message", "content", skip_content } }) |group| {
        if (group[2]) continue;
        const all = (rows.get(group[1]) orelse continue).items() orelse continue;
        var last: ?usize = null;
        for (items, 0..) |item, i| if (std.mem.eql(u8, item.get("type").?.string, group[0])) {
            last = i;
        };
        var at: usize = 0;
        for (items, 0..) |item, i| {
            if (at == all.len) break;
            if (!std.mem.eql(u8, item.get("type").?.string, group[0])) continue;
            const part = item.get("content").?.array[0].object;
            var used = spelled(if (part.get("logprobs")) |v| v.array else &.{});
            const size = part.get("text").?.string.len;
            const start = at;
            // the last item takes any entries past the text, so none is lost
            while (at < all.len and (used < size or i == last.?)) : (at += 1) used += spelled(all[at .. at + 1]);
            if (at > start) try out.append(a, .{ .item = item, .owner = part, .rows = all[start..at] });
        }
    }
}

fn spelled(entries: []const V) usize {
    var n: usize = 0;
    for (entries) |e| n += (e.get("bytes") orelse continue).array.len;
    return n;
}
