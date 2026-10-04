//! Copy proposals from a stream's own context (Python SuffixLookupProposer) behind a small proposer interface.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// What the round loop asks of a proposer: a copied continuation, its evidence, and how it fared.
pub const Proposer = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        propose: *const fn (ptr: *anyopaque, context: []const u32, max_draft: i64) anyerror![]const u32,
        last_match: *const fn (ptr: *anyopaque) i64,
        observe: ?*const fn (ptr: *anyopaque, proposed: i64, accepted: i64) void = null,
    };

    /// Tokens valid until the next `propose`.
    pub fn propose(p: Proposer, context: []const u32, max_draft: i64) ![]const u32 {
        return p.vtable.propose(p.ptr, context, max_draft);
    }

    pub fn lastMatch(p: Proposer) i64 {
        return p.vtable.last_match(p.ptr);
    }

    pub fn observe(p: Proposer, proposed: i64, accepted: i64) void {
        if (p.vtable.observe) |f| f(p.ptr, proposed, accepted);
    }
};

pub const max_ngram = 8;
const Key = [max_ngram]u32;

pub const Params = struct {
    ngram: usize = 3,
    min_match: i64 = 6,
    max_extension: i64 = 64,
    silence_rounds: i64 = 16,
    window: usize = 4,
    confident_match: i64 = 24,
};

/// Continuations backed by enough matching context; silent for a while after a run of rejected proposals.
pub const SuffixLookup = struct {
    gpa: Allocator,
    params: Params,
    index: std.AutoHashMapUnmanaged(Key, std.ArrayList(u32)) = .empty,
    indexed: usize = 0,
    last_key: Key = undefined,
    recent: std.ArrayList(i64) = .empty,
    silent_for: i64 = 0,
    out: std.ArrayList(u32) = .empty,
    proposals: i64 = 0,
    proposed_tokens: i64 = 0,
    judged_tokens: i64 = 0,
    accepted_tokens: i64 = 0,
    silenced_rounds: i64 = 0,
    last_match: i64 = 0,
    last_confident: bool = false,

    pub fn init(gpa: Allocator, params: Params) !SuffixLookup {
        if (params.ngram < 1 or params.ngram > max_ngram) return error.BadNgram;
        if (params.min_match < @as(i64, @intCast(params.ngram))) return error.MinMatchBelowNgram;
        var p = params;
        p.window = @max(1, p.window);
        return .{ .gpa = gpa, .params = p };
    }

    pub fn deinit(s: *SuffixLookup) void {
        s.clearIndex();
        s.index.deinit(s.gpa);
        s.recent.deinit(s.gpa);
        s.out.deinit(s.gpa);
    }

    pub fn proposer(s: *SuffixLookup) Proposer {
        return .{ .ptr = s, .vtable = &vtable };
    }

    pub const vtable: Proposer.VTable = .{ .propose = proposeFn, .last_match = lastMatchFn, .observe = observeFn };

    fn proposeFn(ptr: *anyopaque, context: []const u32, max_draft: i64) anyerror![]const u32 {
        const s: *SuffixLookup = @ptrCast(@alignCast(ptr));
        return s.propose(context, max_draft);
    }

    fn lastMatchFn(ptr: *anyopaque) i64 {
        const s: *SuffixLookup = @ptrCast(@alignCast(ptr));
        return s.last_match;
    }

    fn observeFn(ptr: *anyopaque, proposed: i64, accepted: i64) void {
        const s: *SuffixLookup = @ptrCast(@alignCast(ptr));
        s.observe(proposed, accepted) catch {};
    }

    fn clearIndex(s: *SuffixLookup) void {
        var it = s.index.valueIterator();
        while (it.next()) |list| list.deinit(s.gpa);
        s.index.clearRetainingCapacity();
    }

    fn keyAt(s: *const SuffixLookup, context: []const u32, end: usize) Key {
        var key: Key = @splat(0xFFFF_FFFF);
        const n = s.params.ngram;
        @memcpy(key[0..n], context[end - n .. end]);
        return key;
    }

    fn extend(s: *SuffixLookup, context: []const u32) !void {
        const n = s.params.ngram;
        var position = @max(s.indexed, n - 1);
        while (position < context.len) : (position += 1) {
            const got = try s.index.getOrPut(s.gpa, s.keyAt(context, position + 1));
            if (!got.found_existing) got.value_ptr.* = .empty;
            try got.value_ptr.append(s.gpa, @intCast(position));
        }
        s.indexed = context.len;
    }

    fn matchLength(s: *const SuffixLookup, context: []const u32, end: usize) i64 {
        const len: i64 = @intCast(context.len);
        const e: i64 = @intCast(end);
        var length: i64 = 0;
        const limit = @min(s.params.max_extension, e);
        while (length < limit and context[@intCast(e - 1 - length)] == context[@intCast(len - 1 - length)]) {
            length += 1;
            if (len - 1 - length < 0) break;
        }
        return length;
    }

    /// Python `propose`: the copied continuation (owned by the proposer until the next call), or empty.
    pub fn propose(s: *SuffixLookup, context: []const u32, max_draft: i64) ![]const u32 {
        const n = s.params.ngram;
        s.out.clearRetainingCapacity();
        if (max_draft <= 0 or context.len < n + 1) return s.out.items;
        if (s.silent_for > 0) {
            s.silent_for -= 1;
            s.silenced_rounds += 1;
            return s.out.items;
        }
        // the context changed underneath the index (a new request): rebuild it
        const changed = s.indexed > context.len or (s.indexed != 0 and !std.mem.eql(u32, &s.keyAt(context, s.indexed), &s.last_key));
        if (changed) {
            s.clearIndex();
            s.indexed = 0;
        }
        try s.extend(context);
        s.last_key = s.keyAt(context, s.indexed);
        const positions = (s.index.get(s.keyAt(context, context.len)) orelse return s.out.items).items;
        if (positions.len == 0) return s.out.items;
        var best_end: i64 = -1;
        var best_len: i64 = 0;
        var i = positions.len;
        while (i > 0) {
            i -= 1;
            const end = positions[i] + 1;
            if (end >= context.len) continue;
            const length = s.matchLength(context, end);
            if (length > best_len) {
                best_len = length;
                best_end = end;
                if (length >= s.params.max_extension) break;
            }
        }
        s.last_confident = false;
        s.last_match = best_len;
        if (best_end < 0 or best_len < s.params.min_match) return s.out.items;
        s.last_confident = best_len >= s.params.confident_match;
        const from: usize = @intCast(best_end);
        const to = @min(context.len, from + @as(usize, @intCast(max_draft)));
        try s.out.appendSlice(s.gpa, context[from..to]);
        if (s.out.items.len > 0) {
            s.proposals += 1;
            s.proposed_tokens += @intCast(s.out.items.len);
        }
        return s.out.items;
    }

    pub fn observe(s: *SuffixLookup, proposed: i64, accepted: i64) !void {
        s.judged_tokens += proposed;
        s.accepted_tokens += accepted;
        if (proposed <= 0) return;
        try s.recent.append(s.gpa, accepted);
        if (s.recent.items.len > s.params.window) {
            const drop = s.recent.items.len - s.params.window;
            std.mem.copyForwards(i64, s.recent.items, s.recent.items[drop..]);
            s.recent.shrinkRetainingCapacity(s.params.window);
        }
        if (s.recent.items.len >= s.params.window and std.mem.max(i64, s.recent.items) == 0) {
            s.silent_for = s.params.silence_rounds;
            s.recent.clearRetainingCapacity();
        }
    }
};

test "suffix lookup copies the continuation after a repeated n-gram" {
    const gpa = std.testing.allocator;
    var s = try SuffixLookup.init(gpa, .{ .ngram = 2, .min_match = 2 });
    defer s.deinit();
    const ctx = [_]u32{ 1, 2, 3, 4, 5, 9, 1, 2, 3 };
    const got = try s.propose(&ctx, 3);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 9 }, got);
    try std.testing.expectEqual(@as(i64, 3), s.last_match);
}
