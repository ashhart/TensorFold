//! Synthetic chat wrappers and a byte vocabulary; these fixtures load the production tokenizer and template engine.
const std = @import("std");
const json = @import("json");
pub const names = [_][]const u8{ "chatml", "glm", "deepseek" };
pub const Fixture = struct {
    tmp: std.testing.TmpDir,
    path: [:0]u8,
    a: std.mem.Allocator,
    pub fn init(a: std.mem.Allocator, index: usize) !Fixture {
        const io = std.testing.io;
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const token_data = if (index == 2) try std.Io.Dir.cwd().readFileAlloc(io, "zig/src/families/deepseek_v41/fixtures/serve/mini_tokenizer.json", a, .limited(1 << 20)) else @embedFile("fixtures/control_tokens/mini_tokenizer.json");
        defer if (index == 2) a.free(token_data);
        try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = token_data });
        const templates = [_][]const u8{ @embedFile("fixtures/control_tokens/chatml.jinja"), @embedFile("fixtures/control_tokens/glm.jinja"), @embedFile("fixtures/control_tokens/deepseek.jinja") };
        try tmp.dir.writeFile(io, .{ .sub_path = "chat_template.jinja", .data = templates[index] });
        return .{ .tmp = tmp, .path = try tmp.dir.realPathFileAlloc(io, ".", a), .a = a };
    }
    pub fn deinit(f: *Fixture) void {
        f.a.free(f.path);
        f.tmp.cleanup();
    }
};
pub fn messages(a: std.mem.Allocator, content: []const u8) !json.Value {
    const obj = try json.newObject(a);
    try obj.put(a, "role", .{ .string = "user" });
    try obj.put(a, "content", .{ .string = content });
    const out = try a.alloc(json.Value, 1);
    out[0] = .{ .object = obj };
    return .{ .array = out };
}
