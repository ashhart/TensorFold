const std = @import("std");
const Proposal = @import("drafter.zig").Proposal;

pub const Lookup = struct {
    allocator: std.mem.Allocator,
    index: std.AutoHashMapUnmanaged([3]i32, std.ArrayList(usize)) = .empty,
    indexed: usize = 0,
    last_key: [3]i32 = @splat(0),
    min_match: usize = 4,
    max_extension: usize = 64,
    silent_for: usize = 0,
    recent: [4]usize = @splat(0),
    recent_count: usize = 0,
    last_confident: bool = false,
    last_match: usize = 0,
    proposals: usize = 0,
    proposed_tokens: usize = 0,
    judged_tokens: usize = 0,
    accepted_tokens: usize = 0,
    silenced_rounds: usize = 0,

    pub fn deinit(p: *Lookup) void {
        p.clear();
        p.index.deinit(p.allocator);
    }

    fn clear(p: *Lookup) void {
        var entries = p.index.valueIterator();
        while (entries.next()) |positions| positions.deinit(p.allocator);
        p.index.clearRetainingCapacity();
        p.indexed = 0;
    }

    pub fn propose(p: *Lookup, context: []const i32, max_draft: usize) !Proposal {
        var result = Proposal{};
        if (max_draft == 0 or context.len < 4) return result;
        if (p.silent_for > 0) {
            p.silent_for -= 1;
            p.silenced_rounds += 1;
            return result;
        }
        if (p.indexed > context.len or (p.indexed > 0 and !std.mem.eql(i32, context[p.indexed - 3 .. p.indexed], &p.last_key))) p.clear();
        for (@max(p.indexed, 2)..context.len) |position| {
            const key = context[position - 2 ..][0..3].*;
            const item = try p.index.getOrPut(p.allocator, key);
            if (!item.found_existing) item.value_ptr.* = .empty;
            try item.value_ptr.append(p.allocator, position);
        }
        p.indexed = context.len;
        p.last_key = context[context.len - 3 ..][0..3].*;
        const positions = p.index.get(p.last_key) orelse return result;
        var best_end: usize = 0;
        var best_len: usize = 0;
        var i = positions.items.len;
        while (i > 0) {
            i -= 1;
            const end = positions.items[i] + 1;
            if (end >= context.len) continue;
            var length: usize = 0;
            while (length < @min(p.max_extension, end) and context[end - 1 - length] == context[context.len - 1 - length]) : (length += 1) {}
            if (length > best_len) {
                best_len = length;
                best_end = end;
                if (length >= p.max_extension) break;
            }
        }
        p.last_confident = false;
        p.last_match = best_len;
        if (best_end == 0 or best_len < p.min_match) return result;
        p.last_confident = best_len >= 24;
        result.len = @min(@min(max_draft, result.tokens.len), context.len - best_end);
        for (0..result.len) |j| {
            result.tokens[j] = context[best_end + j];
            result.parents[j] = @as(i32, @intCast(j)) - 1;
            result.scores[j] = -0.06 * @as(f64, @floatFromInt(j + 1));
            result.probabilities[j] = @exp(result.scores[j]);
        }
        if (result.len > 0) {
            p.proposals += 1;
            p.proposed_tokens += result.len;
        }
        return result;
    }

    pub fn observe(p: *Lookup, proposed: usize, accepted: usize) void {
        p.judged_tokens += proposed;
        p.accepted_tokens += accepted;
        if (proposed == 0) return;
        if (p.recent_count == p.recent.len) {
            std.mem.copyForwards(usize, &p.recent, p.recent[1..]);
            p.recent_count -= 1;
        }
        p.recent[p.recent_count] = accepted;
        p.recent_count += 1;
        if (p.recent_count == p.recent.len and std.mem.allEqual(usize, &p.recent, 0)) {
            p.silent_for = 16;
            p.recent_count = 0;
        }
    }
};
