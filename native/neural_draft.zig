const std = @import("std");
const mx = @import("mlx.zig");
const sampling = @import("sampling.zig");
const qwen = @import("model.zig");
pub const Drafter = @import("drafter.zig").Drafter;
const Proposal = @import("drafter.zig").Proposal;

pub const Options = struct {
    enabled: bool = false,
    directory: ?[]const u8 = null,
    bits: i32 = 8,
    max_draft: usize = 3,
    calibration: ?[]const u8 = null,

    pub fn validate(o: Options) !void {
        if (o.max_draft > 15) return error.InvalidDraftBudget;
        if (o.bits != 0 and o.bits != 4 and o.bits != 8) return error.InvalidDraftBits;
    }
};

pub fn enabled(m: anytype, d: ?*Drafter) bool {
    const M = @TypeOf(m.*);
    return if (M == qwen.Model) d != null else if (@hasField(M, "mtp")) m.mtp else m.has_mtp;
}

pub fn absorb(m: anytype, state: anytype, d: ?*Drafter, pass: anytype, tokens: []const i32, rows: []const i32) !void {
    const M = @TypeOf(m.*);
    if (!enabled(m, d)) return;
    if (M == qwen.Model) {
        var offset: usize = 0;
        while (offset < rows.len) {
            const end = @min(offset + 128, rows.len);
            try d.?.absorb(m, pass, rows[offset..end], tokens);
            offset = end;
        }
    } else {
        const on_commit = if (@hasDecl(M, "draftAbsorbsOnCommit")) m.draftAbsorbsOnCommit() else false;
        const hidden = if (@hasDecl(M, "draftHidden")) M.draftHidden(pass) else pass.hidden;
        if (comptime @hasDecl(M, "absorbDraftContext")) {
            if (rows.len == 0) return;
            const selected = try pass.scope.take(hidden, try pass.scope.ints(rows), 0);
            const n: i32 = @intCast(rows.len);
            const prefix = try pass.scope.slice(selected, 0, 0, n - 1);
            const previous = state.draft_hidden.ctx != null;
            const context = if (previous) try pass.scope.cat(&.{ state.draft_hidden, prefix }, 0) else prefix;
            const next = try mx.allocator.alloc(i32, rows.len);
            defer mx.allocator.free(next);
            for (rows, next) |row, *token| token.* = tokens[@intCast(row)];
            try m.absorbDraftContext(context, next[if (previous) 0 else 1..]);
            try mx.replace(&state.draft_hidden, try pass.scope.slice(selected, 0, n - 1, n));
            return;
        }
        for (rows) |row| {
            if (state.draft_hidden.ctx != null and !on_commit) {
                if (@hasDecl(M, "DraftCache")) {
                    _ = try m.draftStep(&pass.scope, state.draft_hidden, tokens[@intCast(row)], &state.head_cache);
                } else if (@hasDecl(M, "forwardMtp")) {
                    var head = try m.forwardMtp(state.draft_hidden, tokens[@intCast(row)..][0..1]);
                    defer head.deinit();
                    try m.commitMtp(&head, 1);
                }
            }
            try mx.replace(&state.draft_hidden, try pass.scope.slice(hidden, 0, row, row + 1));
        }
    }
}

pub fn propose(m: anytype, state: anytype, d: ?*Drafter, first: i32, budget: usize, settings: sampling.Sampling) !Proposal {
    const M = @TypeOf(m.*);
    if (budget == 0 or !enabled(m, d)) return .{};
    if (M == qwen.Model) return d.?.propose(m, first, budget, settings);
    var result = Proposal{};
    result.len = @min(15, if (@hasDecl(M, "maxDrafts")) @min(budget, m.maxDrafts()) else budget);
    if (result.len == 0) return result;
    var tokens: [16]i32 = undefined;
    tokens[0] = first;
    if (@hasDecl(M, "DraftCache")) {
        var scope = mx.Scope{};
        defer scope.deinit();
        var cache = try state.head_cache.clone();
        defer cache.deinit();
        var hidden = state.draft_hidden;
        for (0..result.len) |j| {
            hidden = try m.draftStep(&scope, hidden, tokens[j], &cache);
            const ids = try sampling.rowsMapped(&m.kernels, &scope, try m.draftHead(&scope, hidden), &.{m.position + @as(i32, @intCast(j)) + 1}, settings, m.weights.arrays.get("draft_ids"));
            defer mx.allocator.free(ids);
            tokens[j + 1] = ids[0];
        }
    } else try m.propose(state.draft_hidden, first, tokens[0 .. result.len + 1], settings);
    for (0..result.len) |j| {
        result.tokens[j] = tokens[j + 1];
        result.parents[j] = @as(i32, @intCast(j)) - 1;
        result.scores[j] = 0;
        result.probabilities[j] = 1;
    }
    return result;
}
