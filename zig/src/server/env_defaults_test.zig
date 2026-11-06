//! The documented TF_GLM_CLEAR_THINKING default the server's text reader takes from the environment.
const std = @import("std");
const hf_text = @import("hf_text.zig");
const env = @import("sys").env;

fn set(value: ?[:0]const u8) !void {
    const rc = if (value) |v| env.setenv("TF_GLM_CLEAR_THINKING", v, 1) else env.unsetenv("TF_GLM_CLEAR_THINKING");
    try std.testing.expectEqual(@as(c_int, 0), rc);
}

test "env defaults: TF_GLM_CLEAR_THINKING=1 clears the template's thinking and a word is refused by name" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const saved = if (std.c.getenv("TF_GLM_CLEAR_THINKING")) |v| try gpa.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer {
        if (saved) |v| {
            _ = env.setenv("TF_GLM_CLEAR_THINKING", v, 1);
            gpa.free(v);
        } else {
            _ = env.unsetenv("TF_GLM_CLEAR_THINKING");
        }
    }
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data =
        \\{"model":{"type":"BPE","vocab":{"H":0},"merges":[]},"added_tokens":[]}
    });
    // the template's first revision (the TR3 checkpoint's): reasoning kept only when clear_thinking is false
    try tmp.dir.writeFile(io, .{ .sub_path = "chat_template.jinja", .data =
        \\{%- for m in messages -%}{%- if (clear_thinking is defined and not clear_thinking) and m.reasoning_content is defined -%}[{{ m.reasoning_content }}]{%- endif -%}{{ m.content }}{%- endfor -%}
    });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"glm5_next\"}" });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var problem: []const u8 = "";
    try set("1");
    const glm = try hf_text.HfText.load(gpa, io, dir, null, a, &problem);
    defer glm.deinit();
    try std.testing.expectEqual(@as(?bool, true), glm.clear_thinking);
    try set("2");
    try std.testing.expectError(error.Load, hf_text.HfText.load(gpa, io, dir, null, a, &problem));
    try std.testing.expectEqualStrings("TF_GLM_CLEAR_THINKING=2: expected 0 or 1", problem);
}
