//! Validated checkpoint EOS ids shared by family runners and the native HTTP tokenizer.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub fn appendId(a: Allocator, out: *std.ArrayList(u32), id: u32, vocab: usize) !void {
    if (id >= vocab) return error.BadEos;
    if (std.mem.indexOfScalar(u32, out.items, id) == null) try out.append(a, id);
}

fn addValue(a: Allocator, out: *std.ArrayList(u32), value: Value, vocab: usize) !void {
    switch (value) {
        .null => {},
        .integer => |n| try appendId(a, out, std.math.cast(u32, n) orelse return error.BadEos, vocab),
        .array => |items| {
            var seen: std.AutoHashMapUnmanaged(u32, void) = .empty;
            defer seen.deinit(a);
            for (out.items) |id| try seen.put(a, id, {});
            for (items.items) |item| {
                if (item == .null) continue;
                if (item != .integer) return error.BadEos;
                const id = std.math.cast(u32, item.integer) orelse return error.BadEos;
                if (id >= vocab) return error.BadEos;
                if (seen.contains(id)) continue;
                try seen.put(a, id, {});
                try out.append(a, id);
            }
        },
        else => return error.BadEos,
    }
}

fn field(a: Allocator, out: *std.ArrayList(u32), doc: Value, vocab: usize) !void {
    if (doc == .null) return;
    if (doc != .object) return error.BadEos;
    if (doc.object.get("eos_token_id")) |value| try addValue(a, out, value, vocab);
}

fn validVocab(vocab: usize) !void {
    if (vocab == 0 or vocab - 1 > std.math.maxInt(u32)) return error.BadEos;
}

/// The text model's declared vocabulary also bounds ids, even when a tokenizer adds higher numbered tokens.
pub fn vocabulary(model: Value, tokenizer_vocab: usize) !usize {
    try validVocab(tokenizer_vocab);
    if (model == .null) return tokenizer_vocab;
    if (model != .object) return error.BadEos;
    const text = if (model.object.get("text_config")) |v| (if (v == .object) v else model) else model;
    const value = text.object.get("vocab_size") orelse model.object.get("vocab_size") orelse return tokenizer_vocab;
    if (value != .integer) return error.BadEos;
    const declared = std.math.cast(usize, value.integer) orelse return error.BadEos;
    try validVocab(declared);
    return @min(tokenizer_vocab, declared);
}

/// Model ids, with the existing top-level-before-text fallback, then every declared generation EOS.
pub fn resolve(a: Allocator, model: Value, generation: Value, vocab: usize) ![]u32 {
    const bound = try vocabulary(model, vocab);
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    try field(a, &out, model, bound);
    if (out.items.len == 0 and model == .object) if (model.object.get("text_config")) |text|
        try field(a, &out, text, bound);
    try field(a, &out, generation, bound);
    return out.toOwnedSlice(a);
}

/// Add generation EOS to an already parsed family's ids with the same validation and deduplication.
pub fn merge(a: Allocator, initial: []const u32, generation: Value, vocab: usize) ![]u32 {
    try validVocab(vocab);
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    for (initial) |id| try appendId(a, &out, id, vocab);
    try field(a, &out, generation, vocab);
    return out.toOwnedSlice(a);
}

test "EOS policy: checkpoint EOS merges declared pad EOS but never infers an undeclared pad" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const model = try std.json.parseFromSliceLeaky(Value, a, "{\"eos_token_id\":127906,\"pad_token_id\":127901}", .{});
    const generation = try std.json.parseFromSliceLeaky(Value, a, "{\"eos_token_id\":[127906,null,127901]}", .{});
    const direct = try resolve(a, model, generation, 128000);
    const initial = try resolve(a, model, .null, 128000);
    try std.testing.expectEqualSlices(u32, &.{127906}, initial);
    try std.testing.expectEqualSlices(u32, &.{ 127906, 127901 }, direct);
    try std.testing.expectEqualSlices(u32, direct, try merge(a, initial, generation, 128000));
}

test "EOS policy: checkpoint EOS retains nested fallback and rejects invalid explicit ids before casts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const nested = try std.json.parseFromSliceLeaky(Value, a, "{\"text_config\":{\"eos_token_id\":7}}", .{});
    try std.testing.expectEqualSlices(u32, &.{7}, try resolve(a, nested, .null, 8));
    for ([_][]const u8{ "-1", "8", "4294967296", "true", "1.0", "\"7\"", "[7,false]" }) |bad| {
        const doc = try std.json.parseFromSliceLeaky(Value, a, try std.fmt.allocPrint(a, "{{\"eos_token_id\":{s}}}", .{bad}), .{});
        try std.testing.expectError(error.BadEos, resolve(a, .null, doc, 8));
    }
    try std.testing.expectError(error.BadEos, resolve(a, .null, .{ .bool = true }, 8));
    const narrow = try std.json.parseFromSliceLeaky(Value, a, "{\"vocab_size\":4,\"eos_token_id\":7}", .{});
    try std.testing.expectError(error.BadEos, resolve(a, narrow, .null, 8));
}
