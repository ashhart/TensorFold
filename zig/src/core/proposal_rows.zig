//! Exact stored proposal rows, budgeted before allocation; target weights are never modified.
const std = @import("std");
const tree = @import("tree_round.zig");
const topk = @import("bf16_topk.zig");
const storage = @import("proposal_row_storage.zig");
pub const Storage = storage.Storage;
pub const Recipe = storage.Recipe;
pub const SideDtype = storage.SideDtype;
pub const Plane = storage.Plane;
pub const Source = storage.Views(Scores).Source;
pub const Owned = storage.Views(Scores).Owned;

/// An explicit per-view pitch; padding is never visited by projection-result selection.
pub const Scores = struct {
    dtype: enum { bf16, f32 } = .bf16,
    vocab: u32,
    logical: u32,
    stride: u32,
    ids: ?[]const u32 = null,
    pub fn check(v: Scores) !void {
        if (v.dtype != .bf16) return error.ProposalBasisUnsupported;
        if (v.logical == 0 or v.logical > v.vocab or v.stride < v.logical) return error.ProposalScoreShape;
        if (v.ids) |ids| {
            if (ids.len != v.logical) return error.ProposalScoreShape;
            for (ids, 0..) |id, i| if (id >= v.vocab or (i > 0 and id <= ids[i - 1])) return error.ProposalScoreShape;
        } else if (v.logical != v.vocab) return error.ProposalScoreShape;
    }
    pub fn global(v: Scores, col: u32) !u32 {
        if (col >= v.logical) return error.ProposalPaddingPick;
        return if (v.ids) |ids| ids[col] else col;
    }
    pub fn pick(v: Scores, row: []const u16) !u32 {
        try v.check();
        if (row.len != v.stride) return error.ProposalScoreShape;
        const p = tree.rowArgmax(row[0..v.logical]);
        if (p.nonfinite != 0 or p.token == tree.invalid_token) return error.ProposalNonfinite;
        return v.global(p.token);
    }
    pub fn best(v: Scores, row: []const u16, k: u32) !topk.Row {
        try v.check();
        if (row.len != v.stride) return error.ProposalScoreShape;
        var result = try topk.oracle(row[0..v.logical], k);
        if (result.nonfinite != 0) return error.ProposalNonfinite;
        for (result.entries[0..result.count]) |*entry| entry.token = try v.global(entry.token);
        return result;
    }
};

/// Projection rows may share a batch only when every immutable view, role and queue binding agrees.
pub const Key = struct {
    generation: [32]u8,
    binding: u64,
    logical: u32,
    stride: u32,
    hidden: u32,
    recipe: Recipe,
    queue: u64,
    rank: u32 = 0,
    full: bool = false,
    output_f32: bool = false,
};
pub const Group = struct { first: usize, count: usize };
pub const Partition = struct {
    gpa: std.mem.Allocator,
    order: []usize,
    groups: []Group,
    pub fn deinit(p: *Partition) void {
        p.gpa.free(p.order);
        p.gpa.free(p.groups);
        p.* = undefined;
    }
    pub fn scatter(p: Partition, comptime T: type, packed_rows: []const T, original: []T) !void {
        if (packed_rows.len != p.order.len or original.len != p.order.len) return error.ProposalRowShape;
        const a = @intFromPtr(packed_rows.ptr);
        const b = @intFromPtr(original.ptr);
        const distance = if (a < b) b - a else a - b;
        if (distance < try std.math.mul(usize, packed_rows.len, @sizeOf(T))) return error.ProposalRowAlias;
        for (packed_rows, p.order) |value, i| original[i] = value;
    }
};
pub fn partition(gpa: std.mem.Allocator, keys: []const Key) !Partition {
    const order = try gpa.alloc(usize, keys.len);
    errdefer gpa.free(order);
    var groups: std.ArrayList(Group) = .empty;
    defer groups.deinit(gpa);
    var used: usize = 0;
    for (keys, 0..) |key, i| {
        var seen = false;
        for (keys[0..i]) |before| if (std.meta.eql(key, before)) {
            seen = true;
            break;
        };
        if (seen) continue;
        const first = used;
        for (keys[i..], i..) |other, j| if (std.meta.eql(key, other)) {
            order[used] = j;
            used += 1;
        };
        try groups.append(gpa, .{ .first = first, .count = used - first });
    }
    return .{ .gpa = gpa, .order = order, .groups = try groups.toOwnedSlice(gpa) };
}
