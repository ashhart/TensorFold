//! The chat encoding against the release's own: each request's text and request spans, as its renderer writes them.
const std = @import("std");
const prompt = @import("prompt.zig");

const t = std.testing;
pub const prompt_cases = @embedFile("fixtures/prompt_cases.json");
pub const mini_cases = @embedFile("fixtures/mini_cases.json");

pub const Chat = struct {
    user: []const u8,
    system: ?[]const u8 = null,
    thinking: bool = true,
    effort: ?[]const u8 = null,
    text: []const u8,
    spans: []const [2]usize,
    ids: []const u32,
};
pub const Raw = struct { text: []const u8, ids: []const u32 };
pub const Cases = struct { reference: std.json.Value, chat: []const Chat, raw: []const Raw };

pub fn cases(a: std.mem.Allocator, json: []const u8) !Cases {
    const doc = try std.json.parseFromSliceLeaky(Cases, a, json, .{});
    try t.expect(doc.chat.len > 0 and doc.raw.len > 0);
    return doc;
}

pub fn request(c: Chat) !prompt.Request {
    const effort = if (c.effort) |e| std.meta.stringToEnum(prompt.Effort, e) orelse return error.BadFixture else .high;
    return .{ .user = c.user, .system = c.system, .thinking = c.thinking, .effort = effort };
}

test "each request renders to the release encoding's text, with its request spans" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ prompt_cases, mini_cases }) |json| {
        for ((try cases(a, json)).chat) |c| {
            const r = try prompt.render(a, try request(c));
            try t.expectEqualStrings(c.text, r.text);
            try t.expectEqual(c.spans.len, r.spans.len);
            for (c.spans, r.spans) |want, got| try t.expectEqual(want, got);
        }
    }
}
