//! Supplied recall prompts stay outside every lesson role under the drift gate.
const std = @import("std");
const api = @import("engine_api");
const asks = @import("slide_ask.zig");
const input = @import("slide_recall_input.zig");
const Server = @import("server.zig").Server;
const Cx = @import("errors.zig").Cx;
const roles = .{ "train", "think", "held", "near", "keep", "asked", "unasked", "kept" };

/// The gate compares the rendered token sequence, including the tokenizer's own normalization.
pub fn contains(ids: []const u32, head: []const u32) bool {
    return std.mem.indexOf(u32, ids, head) != null;
}

pub fn contaminated(full: api.LearnRequest, head: []const u32) bool {
    inline for (roles) |role| {
        for (@field(full, role)) |ex| if (contains(ex.ids, head)) return true;
    }
    return false;
}

pub const Reserved = struct {
    heads: []const []const u32 = &.{},

    pub fn init(srv: *Server, cx: *Cx, recall: []const input.Item) !Reserved {
        if (!srv.config.slide_drift_gate or recall.len == 0) return .{};
        const heads = try cx.a.alloc([]const u32, recall.len);
        for (recall, heads) |r, *head| head.* = (try asks.promptOnly(srv, cx, cx.a, null, r.question)).ids;
        return .{ .heads = heads };
    }

    pub fn reads(r: Reserved, ids: []const u32) bool {
        for (r.heads) |head| if (contains(ids, head)) return true;
        return false;
    }

    pub fn question(r: Reserved, srv: *Server, cx: *Cx, q: []const u8) !bool {
        if (r.heads.len == 0) return false;
        return r.reads((try asks.promptOnly(srv, cx, cx.a, null, q)).ids);
    }

    /// Also cover router, keep and contextual examples, using the gate's complete-role scan.
    pub fn clean(r: Reserved, a: std.mem.Allocator, full: api.LearnRequest) !api.LearnRequest {
        if (r.heads.len == 0) return full;
        var out = full;
        inline for (roles) |role| {
            var xs: std.ArrayList(api.Example) = .empty;
            for (@field(full, role)) |ex| if (!r.reads(ex.ids)) try xs.append(a, ex);
            @field(out, role) = try xs.toOwnedSlice(a);
        }
        return out;
    }
};

pub const refusal = "the plan could not avoid the supplied recall wordings";

test "Slide metrics: reserved cleaning covers exactly the gate's example roles" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const head = &[_]u32{ 1, 2, 3 };
    const reserved: Reserved = .{ .heads = &.{head} };
    const clean: api.Example = .{ .ids = &.{ 1, 3, 2 }, .start = 1, .decide = 0 };
    const dirty: api.Example = .{ .ids = &.{ 9, 1, 2, 3, 8 }, .start = 1, .decide = 0 };
    inline for (roles) |role| {
        var full: api.LearnRequest = .{};
        @field(full, role) = &.{ clean, dirty };
        try std.testing.expect(contaminated(full, head));
        const filtered = try reserved.clean(a, full);
        try std.testing.expect(!contaminated(filtered, head));
        try std.testing.expectEqual(@as(usize, 1), @field(filtered, role).len);
        try std.testing.expectEqualSlices(u32, clean.ids, @field(filtered, role)[0].ids);
        const unchanged = try (Reserved{}).clean(a, full);
        try std.testing.expect(@field(unchanged, role).ptr == @field(full, role).ptr);
    }
}
