//! Re-encode all release golden prompts with an explicitly supplied local tokenizer; no weights are opened.
const std = @import("std");
const json = @import("json");
const mt = @import("model_text.zig");
const prompt = @import("kimi_prompt.zig");
const KimiText = @import("kimi_text.zig").KimiText;

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 2) {
        std.debug.print("usage: tf-k3-text-golden LOCAL_TOKENIZER_DIRECTORY\n", .{});
        std.process.exit(2);
    }
    const fixture = (try json.parse(a, @embedFile("fixtures/kimi/golden.json"))).ok;
    for ([_][]const u8{ "tiktoken.model", "tokenizer_config.json" }) |name| {
        const path = try std.fs.path.join(a, &.{ args[1], name });
        const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(1 << 30));
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
        const hex = std.fmt.bytesToHex(hash, .lower);
        if (!std.mem.eql(u8, &hex, fixture.get("files").?.strField(name).?)) return error.ReleaseTokenizerHashMismatch;
    }
    const tok = try KimiText.load(init.gpa, init.io, args[1]);
    defer tok.deinit();
    const cases = fixture.get("cases").?.array;
    for (cases) |c| {
        const kw = c.get("kwargs").?;
        const segments = try prompt.render(a, c.get("messages").?, .{
            .thinking = if (kw.get("thinking")) |v| v.bool else true,
            .add_generation_prompt = if (kw.get("add_generation_prompt")) |v| v.bool else true,
            .thinking_effort = kw.strField("thinking_effort"),
            .tools = kw.get("tools") orelse .null,
            .tool_choice = kw.get("tool_choice") orelse .null,
            .response_format = kw.get("response_format") orelse .null,
        });
        var text: std.ArrayList(u8) = .empty;
        var spans: std.ArrayList(mt.Span) = .empty;
        var cuts: std.ArrayList(mt.Segment) = .empty;
        for (segments) |s| {
            const start = text.items.len;
            try text.appendSlice(a, s.text);
            if (s.request) try spans.append(a, .{ start, text.items.len });
            try cuts.append(a, .{ .end = text.items.len, .allow_special = s.allow_special });
        }
        try std.testing.expectEqualStrings(c.strField("joined").?, text.items);
        const rendered: mt.Rendered = .{ .text = text.items, .spans = spans.items, .segments = cuts.items };
        var problem: []const u8 = "";
        const ids = try rendered.encode(tok.text(), a, &problem);
        const expected = c.get("ids").?.array;
        var first: ?usize = null;
        for (ids[0..@min(ids.len, expected.len)], 0..) |id, i| if (id != expected[i].int64().?) {
            first = i;
            break;
        };
        if (first != null or ids.len != expected.len) {
            std.debug.print("KIMI_RELEASE_TOKEN_IDS FAIL case={s} first={d} got={d} expected={d}\n", .{ c.strField("id").?, first orelse @min(ids.len, expected.len), ids.len, expected.len });
            return error.ReleasePromptTokenMismatch;
        }
    }
    std.debug.print("KIMI_RELEASE_TOKEN_IDS PASS cases={d}; tokenizer only, no weights or model forward\n", .{cases.len});
}
