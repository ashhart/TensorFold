//! Validated checkpoint EOS ids shared by family runners and the native HTTP tokenizer.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub fn appendId(a: Allocator, out: *std.ArrayList(u32), id: u32, vocab: usize) !void {
    if (id >= vocab) return error.BadEos;
    if (std.mem.indexOfScalar(u32, out.items, id) == null) try out.append(a, id);
}

/// Where a family takes config.json's ids from; generation_config.json's are added after them in every order.
pub const Order = enum {
    /// The root's ids; text_config's only when the root gave none.
    text_if_root_empty,
    /// The root's, text_config's and generation_config.json's, all of them.
    every_source,
    /// The root's key whenever it exists (a root `null` or `[]` hides text_config's), else text_config's.
    root_key_else_text,
};

/// One family's EOS rule: the order, whether nested arrays hold more ids, the most ids, and whether none is refused.
pub const Policy = struct {
    order: Order,
    nested: bool,
    max: ?usize,
    required: bool,
};

/// `resolve`'s rule: the served stop set (server/hf_text.zig), dots_mocr, dory and kolibri1.
pub const served: Policy = .{ .order = .text_if_root_empty, .nested = false, .max = null, .required = false };
/// Qwen3.5's HIP loader (families/qwen3_5/config_checks.zig): every source, at most 8 ids.
pub const qwen35_hip: Policy = .{ .order = .every_source, .nested = true, .max = 8, .required = true };
/// qwen27 on Metal (families/qwen27/config.zig): root key else text_config, then generation, at most 8 ids.
pub const qwen27_metal: Policy = .{ .order = .root_key_else_text, .nested = true, .max = 8, .required = true };

/// BadEosValue: not a null, an id or an id array (a family names it); BadEos: an id at or past the vocabulary.
pub const BufferError = error{ BadEos, BadEosValue, TooManyEos, MissingEos };
pub const Error = BufferError || error{OutOfMemory};
/// `resolve` and `merge` name every refusal BadEos: `served` sets no most ids and allows none.
pub const ServedError = error{ BadEos, OutOfMemory };

fn servedError(e: Error) ServedError {
    return switch (e) {
        error.BadEos, error.BadEosValue => error.BadEos,
        error.OutOfMemory => error.OutOfMemory,
        error.TooManyEos, error.MissingEos => unreachable,
    };
}

const Ids = struct {
    list: std.ArrayList(u32),
    /// null: the caller's fixed buffer.
    a: ?Allocator,
    policy: Policy,
    vocab: usize,

    fn id(s: *Ids, n: i64) Error!void {
        const v = std.math.cast(u32, n) orelse return error.BadEosValue;
        if (v >= s.vocab) return error.BadEos;
        if (std.mem.indexOfScalar(u32, s.list.items, v) != null) return;
        if (s.policy.max) |m| if (s.list.items.len == m) return error.TooManyEos;
        if (s.a) |a| try s.list.append(a, v) else s.list.appendBounded(v) catch return error.TooManyEos;
    }

    fn add(s: *Ids, v: Value, in_array: bool) Error!void {
        switch (v) {
            .null => {},
            .integer => |n| try s.id(n),
            .array => |items| {
                if (in_array and !s.policy.nested) return error.BadEosValue;
                for (items.items) |item| try s.add(item, true);
            },
            else => return error.BadEosValue,
        }
    }

    /// A document's `eos_token_id`; a null document is none, any other non-object is refused.
    fn field(s: *Ids, doc: Value) Error!void {
        if (doc == .null) return;
        if (doc != .object) return error.BadEosValue;
        if (doc.object.get("eos_token_id")) |v| try s.add(v, false);
    }

    fn gather(s: *Ids, root: Value, text: Value, generation: Value) Error!void {
        switch (s.policy.order) {
            .text_if_root_empty => {
                try s.field(root);
                if (s.list.items.len == 0) try s.field(text);
            },
            .every_source => {
                try s.field(root);
                try s.field(text);
            },
            .root_key_else_text => if (root == .object and root.object.contains("eos_token_id"))
                try s.field(root)
            else
                try s.field(text),
        }
        try s.field(generation);
        if (s.policy.required and s.list.items.len == 0) return error.MissingEos;
    }
};

/// `policy`'s ids below `vocab` into `out`, each once; `out` holds at least `policy.max` ids.
pub fn collect(out: []u32, policy: Policy, root: Value, text: Value, generation: Value, vocab: usize) BufferError![]u32 {
    std.debug.assert(out.len >= policy.max orelse out.len);
    var s: Ids = .{ .list = .initBuffer(out), .a = null, .policy = policy, .vocab = vocab };
    s.gather(root, text, generation) catch |e| return switch (e) {
        error.OutOfMemory => unreachable, // a fixed buffer refuses past its end with TooManyEos
        else => |x| x,
    };
    return s.list.items;
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

/// Model ids by the `served` policy, the root's before text_config's, then every declared generation EOS.
pub fn resolve(a: Allocator, model: Value, generation: Value, vocab: usize) ServedError![]u32 {
    const bound = try vocabulary(model, vocab);
    const text = if (model == .object) model.object.get("text_config") orelse .null else .null;
    var s: Ids = .{ .list = .empty, .a = a, .policy = served, .vocab = bound };
    errdefer s.list.deinit(a);
    s.gather(model, text, generation) catch |e| return servedError(e);
    return s.list.toOwnedSlice(a);
}

/// Add generation EOS to an already parsed family's ids with the same validation and deduplication.
pub fn merge(a: Allocator, initial: []const u32, generation: Value, vocab: usize) ServedError![]u32 {
    try validVocab(vocab);
    var s: Ids = .{ .list = .empty, .a = a, .policy = served, .vocab = vocab };
    errdefer s.list.deinit(a);
    for (initial) |id| try appendId(a, &s.list, id, vocab);
    s.field(generation) catch |e| return servedError(e);
    return s.list.toOwnedSlice(a);
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

test {
    _ = @import("eos_policy_test.zig");
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
