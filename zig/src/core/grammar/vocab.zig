//! The grammar's view of a vocabulary: tokenizer.json's strings by id, added tokens and padding empty unless kept.

const std = @import("std");

pub const Vocab = struct {
    arena: std.heap.ArenaAllocator,
    /// `size` entries, id order
    tokens: [][]const u8,
    think_open: ?u32,
    think_end: ?u32,

    pub fn deinit(v: *Vocab) void {
        v.arena.deinit();
    }
};

pub const Error = error{ InvalidTokenizer, VocabTooSmall, OutOfMemory };

/// The vocabulary of `json_text` (a tokenizer.json) at `size` columns, added tokens blanked but `keep`.
pub fn build(gpa: std.mem.Allocator, json_text: []const u8, size: u32, keep: []const []const u8) Error!Vocab {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    var scratch = std.heap.ArenaAllocator.init(gpa);
    defer scratch.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, scratch.allocator(), json_text, .{}) catch return error.InvalidTokenizer;
    if (doc != .object) return error.InvalidTokenizer;
    const model = doc.object.get("model") orelse return error.InvalidTokenizer;
    if (model != .object) return error.InvalidTokenizer;
    const vocab = model.object.get("vocab") orelse return error.InvalidTokenizer;
    if (vocab != .object) return error.InvalidTokenizer;
    const added: []const std.json.Value = if (doc.object.get("added_tokens")) |x| (if (x == .array) x.array.items else &.{}) else &.{};

    // get_vocab(with_added_tokens=True): the model's map, then each added token's content -> its id
    var map: std.StringArrayHashMapUnmanaged(u32) = .empty;
    defer map.deinit(gpa);
    try map.ensureTotalCapacity(gpa, vocab.object.count() + added.len);
    var it = vocab.object.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.* != .integer) return error.InvalidTokenizer;
        map.putAssumeCapacity(e.key_ptr.*, @intCast(e.value_ptr.integer));
    }
    for (added) |t| {
        const id = idOf(t) orelse return error.InvalidTokenizer;
        map.putAssumeCapacity(contentOf(t) orelse return error.InvalidTokenizer, id);
    }
    const tokens = try a.alloc([]const u8, size);
    @memset(tokens, "");
    for (map.keys(), map.values()) |s, id| {
        if (id >= size) return error.VocabTooSmall;
        tokens[id] = try a.dupe(u8, s);
    }
    for (added) |t| {
        const s = contentOf(t).?;
        const id = idOf(t).?;
        const kept = for (keep) |k| {
            if (std.mem.eql(u8, k, s)) break true;
        } else false;
        if (!kept and id < size) tokens[id] = "";
    }
    return .{ .arena = arena, .tokens = tokens, .think_open = map.get("<think>"), .think_end = map.get("</think>") };
}

fn idOf(t: std.json.Value) ?u32 {
    if (t != .object) return null;
    const id = t.object.get("id") orelse return null;
    return if (id == .integer and id.integer >= 0) @intCast(id.integer) else null;
}

fn contentOf(t: std.json.Value) ?[]const u8 {
    if (t != .object) return null;
    const s = t.object.get("content") orelse return null;
    return if (s == .string) s.string else null;
}
