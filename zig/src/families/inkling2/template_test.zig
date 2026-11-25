//! The checkpoint's chat template through TensorFold's template core, against the reference's rendered prompts.
const std = @import("std");
const template = @import("template");
const prompt = @import("prompt.zig");

const t = std.testing;
pub const jinja = @embedFile("fixtures/chat_template.jinja");
pub const cases_json = @embedFile("fixtures/prompt_cases.json");

/// The effort a case passed the reference, as the text prompt.zig takes: a name, or a number written out.
pub fn effortText(a: std.mem.Allocator, v: ?std.json.Value) !?[]const u8 {
    const e = v orelse return null;
    return switch (e) {
        .string => |s| s,
        .integer => |i| try std.fmt.allocPrint(a, "{d}", .{i}),
        .float => |f| try std.fmt.allocPrint(a, "{d}", .{f}),
        else => error.BadCase,
    };
}

fn field(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return v.string;
}

test "every reference prompt is rendered byte for byte, thinking effort names and numbers included" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, cases_json, .{});
    var compiled = try template.compile(t.allocator, jinja, null);
    defer compiled.deinit();
    const cases = doc.object.get("chat").?.array.items;
    try t.expect(cases.len >= 20);
    for (cases) |c| {
        const o = c.object;
        const req: prompt.Request = .{ .user = field(o, "user").?, .system = field(o, "system"), .effort = try effortText(a, o.get("effort")) };
        const got = try prompt.render(t.allocator, &compiled, req, null);
        defer t.allocator.free(got);
        try t.expectEqualStrings(field(o, "prompt").?, got);
    }
}

test "the template raises for an unknown effort, one out of range and text that is no plain decimal, as the reference does" {
    var compiled = try template.compile(t.allocator, jinja, null);
    defer compiled.deinit();
    for ([_][]const u8{ "turbo", "1.5", "-0.1", "nan", "inf", "-infinity", "0x1p-1", "0.5_5" }) |bad| {
        var diag: template.Diag = .{};
        defer t.allocator.free(diag.msg);
        try t.expectError(error.TemplateFailed, prompt.render(t.allocator, &compiled, .{ .user = "hi", .effort = bad }, &diag));
        try t.expect(diag.raised);
    }
}
