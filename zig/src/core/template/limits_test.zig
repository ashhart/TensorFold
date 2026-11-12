//! The limits a chat template is held to: nesting when it is parsed, then steps, recursion and memory when it renders.
const std = @import("std");
const template = @import("template.zig");

/// `head`, then `open` and `close` each `n` times around `middle`, then `tail`.
fn nested(a: std.mem.Allocator, head: []const u8, open: []const u8, n: usize, middle: []const u8, close: []const u8, tail: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, head);
    for (0..n) |_| try out.appendSlice(a, open);
    try out.appendSlice(a, middle);
    for (0..n) |_| try out.appendSlice(a, close);
    try out.appendSlice(a, tail);
    return out.items;
}

/// Compiling `source` fails with a message that holds `part`.
fn refused(source: []const u8, part: []const u8) !void {
    const a = std.testing.allocator;
    var diag = template.Diag{};
    defer if (diag.msg.len > 0) a.free(diag.msg);
    var t = template.compile(a, source, &diag) catch |err| {
        try std.testing.expectEqual(error.TemplateFailed, err);
        return std.testing.expect(std.mem.indexOf(u8, diag.msg, part) != null);
    };
    t.deinit();
    return error.TestExpectedError;
}

/// `source` rendered with no variables under `limits`: its text, or a failure whose message holds one of `part`.
fn rendered(source: []const u8, limits: ?template.Limits, want: []const u8, part: ?[]const u8) !void {
    const a = std.testing.allocator;
    var diag = template.Diag{};
    defer if (diag.msg.len > 0) a.free(diag.msg);
    var t = try template.compile(a, source, &diag);
    defer t.deinit();
    if (limits) |l| t.limits = l;
    const out = template.render(a, &t, .null, .null, .null, true, &diag) catch |err| {
        try std.testing.expectEqual(error.TemplateFailed, err);
        var parts = std.mem.splitScalar(u8, part orelse return error.TestUnexpectedError, '|');
        while (parts.next()) |one| if (std.mem.indexOf(u8, diag.msg, one) != null) return;
        std.debug.print("failed with: {s}\n", .{diag.msg});
        return error.TestUnexpectedError;
    };
    defer a.free(out);
    try std.testing.expect(part == null);
    try std.testing.expectEqualStrings(want, out);
}

test "security: a template nested past the parser's depth is refused before anything recurses through it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const n = 100_000;
    // each shape recursed once a level in the parser, or built a tree that tall for the scope pass and the renderer
    const shapes = [_][5][]const u8{
        .{ "{{ ", "(", "1", ")", " }}" },           .{ "{{ ", "[", "1", "]", " }}" },
        .{ "{{ ", "not ", "x", "", " }}" },         .{ "{{ ", "- ", "1", "", " }}" },
        .{ "{{ 1", "", "", " if x else 1", " }}" }, .{ "", "{% if x %}", "", "{% endif %}", "" },
        .{ "{{ 1", "", "", " + 1", " }}" },         .{ "{{ x", "", "", ".a", " }}" },
        .{ "{{ x", "", "", "|string", " }}" },      .{ "{{ x", "", "", " or x", " }}" },
        .{ "{% set ", "(", "y", ")", " = 1 %}" },   .{ "{{ f", "", "", "(1)", " }}" },
    };
    for (shapes) |s| try refused(try nested(a, s[0], s[1], n, s[2], s[3], s[4]), "nested more than");
    // what a real template nests is far inside the limit and renders as before
    try rendered(try nested(a, "{{ ", "(", 100, "1", ")", " }}"), null, "1", null);
    try rendered(try nested(a, "{{ 1", "", 199, "", " + 1", " }}"), null, "200", null);
    try rendered(try nested(a, "", "{% if true %}", 40, "deep", "{% endif %}", ""), null, "deep", null);
}

test "security: a render stops at its steps, its recursion and its memory, and says which" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const loops = "{% for a in range(100000) %}{% for b in range(100000) %}{% endfor %}{% endfor %}done";
    try rendered(loops, .{ .steps = 100_000 }, "", "ran more than 100000 steps");
    try rendered("{% for a in range(1000) %}{% endfor %}done", .{ .steps = 100_000 }, "done", null);
    // a macro that calls itself stops at the macro limit, or sooner at the stack's when its frames are large
    try rendered("{% macro f(n) %}{{ f(n + 1) }}{% endmacro %}{{ f(0) }}", null, "", "recursion depth exceeded|of stack deep");
    // whatever the call is nested in, the stack limit stops it: each shape's frames are a different size
    for ([_][2][]const u8{ .{ "not ", "" }, .{ "", " | string" }, .{ "[", "]" }, .{ "namespace(a=", ")" }, .{ "- ", "" } }) |shape| {
        const source = try nested(a, "{% macro f(n) %}{{ ", shape[0], 40, "f(n + 1)", shape[1], " }}{% endmacro %}{{ f(0) }}");
        try rendered(source, null, "", "recursion depth exceeded|of stack deep");
        try rendered(source, .{ .stack = 256 << 10 }, "", "recursed more than 256 KiB of stack deep");
    }
    const doubling = "{% set ns = namespace(s='0123456789abcdef') %}{% for i in range(40) %}{% set ns.s = ns.s ~ ns.s %}{% endfor %}{{ ns.s | length }}";
    try rendered(doubling, .{ .bytes = 8 << 20 }, "", "needs more than 8 MiB");
    try rendered("{% set s = '0123456789abcdef' * 1000 %}{{ s | length }}", .{ .bytes = 8 << 20 }, "16000", null);
    // an empty sequence repeated is empty at once, whatever the count
    try rendered("{{ ('' * 1000000000000000000) | length }}|{{ ([] * 1000000000000000000) | length }}", null, "0|0", null);
}
