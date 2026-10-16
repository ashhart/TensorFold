//! Replies a lesson's writer and checks got, kept with their tokens.
const std = @import("std");
const chat = @import("chat.zig");

/// A kept reply: its prompt's tokens, its own (an end token included), and its text.
pub const Entry = struct {
    prompt: []const u32,
    tokens: []const u32,
    content: []const u8,
    finish: []const u8,

    /// The kept reply as `ask` returns it.
    pub fn asReply(e: Entry) chat.Reply {
        return .{ .content = e.content, .stop_sequence = null, .reasoning = null, .tool_calls_streamed = false, .finish_reason = e.finish, .prompt_tokens = e.prompt.len, .cached_tokens = 0, .completion_tokens = e.tokens.len, .reasoning_tokens = 0, .runtime = .null, .speculative = .null, .prompt_ids = e.prompt, .tokens = e.tokens };
    }
};

pub const Known = struct {
    arena: std.heap.ArenaAllocator,
    mutex: std.Io.Mutex = .init,
    map: std.AutoHashMapUnmanaged(u64, Entry) = .empty,
    // counts over the server's life: kept replies scored, found unchanged, generated anyway
    scored: usize = 0,
    same: usize = 0,

    pub fn init(gpa: std.mem.Allocator) Known {
        return .{ .arena = .init(gpa) };
    }

    pub fn deinit(k: *Known) void {
        k.map.deinit(k.arena.child_allocator);
        k.arena.deinit();
    }

    /// The reply kept at `at`, if any (its slices live as long as the store).
    pub fn get(k: *Known, io: std.Io, at: u64) ?Entry {
        k.mutex.lockUncancelable(io);
        defer k.mutex.unlock(io);
        return k.map.get(at);
    }

    /// Keep `reply` at `at` (a key): a greedy reply that ended at its end token or its length, with its tokens.
    pub fn remember(k: *Known, io: std.Io, at: u64, reply: chat.Reply) void {
        if (reply.prompt_ids.len == 0 or reply.tokens.len == 0 or reply.stop_sequence != null or reply.thinking) return;
        const stop = std.mem.eql(u8, reply.finish_reason, "stop");
        if (!stop and !std.mem.eql(u8, reply.finish_reason, "length")) return;
        k.mutex.lockUncancelable(io);
        defer k.mutex.unlock(io);
        const a = k.arena.allocator();
        const e: Entry = .{
            .prompt = a.dupe(u32, reply.prompt_ids) catch return,
            .tokens = a.dupe(u32, reply.tokens) catch return,
            .content = a.dupe(u8, reply.content) catch return,
            .finish = if (stop) "stop" else "length",
        };
        k.map.put(k.arena.child_allocator, at, e) catch {};
    }
};

/// A greedy question without thinking as one number: its system note, the conversation before it, it, its length.
pub fn key(system: ?[]const u8, before: []const [2][]const u8, question: []const u8, max_tokens: i64) u64 {
    var h = std.hash.Wyhash.init(0x51de);
    h.update(if (system) |s| s else "\x00none");
    h.update("\x00");
    for (before) |t| {
        h.update(t[0]);
        h.update("\x00");
        h.update(t[1]);
        h.update("\x00");
    }
    h.update(question);
    h.update(std.mem.asBytes(&max_tokens));
    return h.final();
}

test "a question's key holds its system note, its conversation and its length" {
    const q = "What is my name?";
    try std.testing.expect(key(null, &.{}, q, 32) != key(null, &.{}, q, 48));
    try std.testing.expect(key(null, &.{}, q, 32) != key("note", &.{}, q, 32));
    try std.testing.expect(key(null, &.{.{ "hi", "hello" }}, q, 32) != key(null, &.{}, q, 32));
    try std.testing.expectEqual(key(null, &.{}, q, 32), key(null, &.{}, q, 32));
}
