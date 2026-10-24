//! Byte tokenizer for storage tests; no inference or lookup dependencies.
const std = @import("std");
const json = @import("json");
const model_text = @import("model_text.zig");
/// Token ids are UTF-8 bytes, as the fixture's tokenizer; a prompt renders as "[role]content" lines.
pub const Bytes = struct {
    forgot: u32 = 0, // how often a store asked it to drop what it keeps (it keeps nothing)
    encodes_since_forget: std.atomic.Value(usize) = .init(0),

    pub fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .forget = forget } };
    }

    fn forget(ctx: *anyopaque) void {
        const t: *Bytes = @ptrCast(@alignCast(ctx));
        t.forgot += 1;
        t.encodes_since_forget.store(0, .monotonic);
    }

    fn encode(ctx: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const t: *Bytes = @ptrCast(@alignCast(ctx));
        _ = t.encodes_since_forget.fetchAdd(1, .monotonic);
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (ids, out) |id, *byte| byte.* = @intCast(id);
        return out;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: std.mem.Allocator, messages: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        var out: std.ArrayList(u8) = .empty;
        for (messages.items() orelse &.{}) |m| try out.print(a, "[{s}]{s}\n", .{ json.strOr(m.get("role")), json.strOr(m.get("content")) });
        try out.appendSlice(a, "[assistant]");
        return .{ .text = out.items };
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};
