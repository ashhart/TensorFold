//! Nemotron's disk and learned keys take their resolved policies where the keys are made, read from the source.
const std = @import("std");

/// The block of the function `head` opens, brace to matching brace, with no whitespace or line comments.
fn body(a: std.mem.Allocator, source: []const u8, head: []const u8) ![]const u8 {
    const at = std.mem.indexOf(u8, source, head) orelse return error.FunctionMissing;
    const open = at + (std.mem.indexOf(u8, source[at..], "{\n") orelse return error.FunctionMissing);
    var out: std.ArrayList(u8) = .empty;
    var depth: usize = 0;
    var lines = std.mem.splitScalar(u8, source[open..], '\n');
    while (lines.next()) |whole| {
        const line = if (std.mem.indexOf(u8, whole, "//")) |c| whole[0..c] else whole;
        for (line) |ch| {
            if (!std.ascii.isWhitespace(ch)) try out.append(a, ch);
            if (ch == '{') depth += 1;
            if (ch == '}') {
                depth -= 1;
                if (depth == 0) return out.items;
            }
        }
    }
    return error.FunctionUnclosed;
}

fn expectCall(source: []const u8, head: []const u8, call: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const b = try body(arena.allocator(), source, head);
    if (std.mem.indexOf(u8, b, call) == null) std.debug.print("`{s}` no longer has `{s}`\n", .{ head, call });
    try std.testing.expect(std.mem.indexOf(u8, b, call) != null);
}

test "persistence identity: Nemotron CUDA's disk key extends its identity with the policy the engine resolved" {
    try expectCall(@embedFile("src/native/cuda_disk.zig"), "pub fn identity(", "return.{.id=withPolicy(id,t.e),");
}

test "persistence identity: Nemotron Metal's learned key extends its identity with the backend's policy" {
    try expectCall(@embedFile("src/native/metal_identity.zig"), "pub fn learned(", "return.{.id=withPolicy(h.final(),b),");
}

test "persistence identity: Nemotron Metal's backend keeps TF_COOP_SPLIT as its forward pass resolves it" {
    const source = @embedFile("src/families/nemotron/backend.zig");
    try std.testing.expect(std.mem.indexOf(u8, source, "\nconst fwd = @import(\"forward.zig\");\n") != null);
    try expectCall(source, "pub fn init(gpa: std.mem.Allocator, m: *Model,", ".coop_split=fwd.splitOn(),");
}
