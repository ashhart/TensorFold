//! A text prompt as Inkling 2 Small's chat template renders it, and its ids and text through the tokenizer.
const std = @import("std");
const template = @import("template");
const tokenizer = @import("tokenizer");

/// One user turn, an optional system turn before it, and the thinking effort the template writes.
pub const Request = struct {
    user: []const u8,
    system: ?[]const u8 = null,
    /// A name (none, minimal, low, medium, high, xhigh, max) or a number from 0 to 0.99; null: the template's own 0.9.
    effort: ?[]const u8 = null,
};

/// Whether effort text is a plain decimal (digits, sign, point, exponent) rather than a name or Zig-only number syntax.
fn plainNumber(e: []const u8) bool {
    if (e.len == 0) return false;
    for (e) |ch| if (!std.ascii.isDigit(ch) and std.mem.indexOfScalar(u8, ".eE+-", ch) == null) return false;
    return true;
}

/// The rendered prompt: the template's text for `req`, owned by `gpa`; the template's own refusal lands in `diag`.
pub fn render(gpa: std.mem.Allocator, compiled: *const template.Compiled, req: Request, diag: ?*template.Diag) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const Msg = struct { role: []const u8, content: []const u8 };
    var msgs: [2]Msg = undefined;
    var n: usize = 0;
    if (req.system) |s| {
        msgs[n] = .{ .role = "system", .content = s };
        n += 1;
    }
    msgs[n] = .{ .role = "user", .content = req.user };
    n += 1;
    const messages = try std.json.parseFromSliceLeaky(std.json.Value, a, try std.json.Stringify.valueAlloc(a, msgs[0..n], .{}), .{});
    var context: std.json.ObjectMap = .empty;
    if (req.effort) |e| {
        const parsed: ?f64 = if (plainNumber(e)) (std.fmt.parseFloat(f64, e) catch null) else null;
        const v: std.json.Value = if (parsed) |x| (if (std.math.isFinite(x)) .{ .float = x } else .{ .string = e }) else .{ .string = e };
        try context.put(a, "reasoning_effort", v);
    }
    return template.render(gpa, compiled, messages, .null, .{ .object = context }, true, diag);
}

/// Token ids of `text`: the tokenizer applied to a rendered prompt as the reference does, no post-processor tokens.
pub fn encode(a: std.mem.Allocator, tok: *const tokenizer.Tokenizer, text: []const u8) ![]u32 {
    return tok.encode(a, text);
}

/// The text of `ids`, special tokens spelled out.
pub fn decode(a: std.mem.Allocator, tok: *const tokenizer.Tokenizer, ids: []const u32) ![]u8 {
    return tok.decode(a, ids, false);
}
