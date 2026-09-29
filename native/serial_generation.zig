const std = @import("std");
const mx = @import("mlx.zig");
const sampling = @import("sampling.zig");

pub const Result = struct {
    tokens: std.ArrayList(u32) = .empty,
    drafted: usize = 0,
    accepted: usize = 0,
    rounds: usize = 0,
    pub fn deinit(r: *Result) void {
        r.tokens.deinit(mx.allocator);
    }
};

pub fn generate(m: anytype, tokens: []const i32, max_tokens: usize, settings: sampling.Sampling, drafts: usize, dump: ?[]const u8) !Result {
    const M = @TypeOf(m.*);
    if (tokens.len == 0 or drafts > 15) return error.InvalidGeneration;
    if (drafts > 0 and !@hasDecl(M, "propose")) return error.UnsupportedDrafts;
    const draft_budget = if (@hasDecl(M, "maxDrafts")) @min(drafts, m.maxDrafts()) else drafts;
    const absorb_on_commit = if (@hasDecl(M, "draftAbsorbsOnCommit")) m.draftAbsorbsOnCommit() else false;
    var result = Result{};
    errdefer result.deinit();
    var hidden = mx.empty;
    defer mx.free(hidden);
    var pending: i32 = 0;
    var offset: usize = 0;
    while (offset < tokens.len) {
        const count = @min(16, tokens.len - offset);
        var pass = try m.forward(tokens[offset..][0..count]);
        defer pass.deinit();
        const draft_hidden = if (@hasDecl(M, "draftHidden")) M.draftHidden(&pass) else pass.hidden;
        if (comptime @hasDecl(M, "propose")) if (draft_budget > 0 and !absorb_on_commit) {
            if (offset > 0) try absorb(m, hidden, tokens[offset..][0..1]);
            if (count > 1) try absorb(m, try pass.scope.slice(draft_hidden, 0, 0, @intCast(count - 1)), tokens[offset + 1 ..][0 .. count - 1]);
        };
        const ids = try sampling.rows(&m.kernels, &pass.scope, try pass.scope.slice(pass.logits, 0, @intCast(count - 1), @intCast(count)), &.{@intCast(offset + count)}, settings);
        defer mx.allocator.free(ids);
        pending = ids[0];
        if (dump) |file| if (offset + count == tokens.len) {
            const z = try mx.allocator.dupeSentinel(u8, file, 0);
            defer mx.allocator.free(z);
            const logits = try pass.scope.cast(pass.logits, mx.f32t);
            try mx.eval(logits);
            try mx.check(mx.c.mlx_save(z, logits));
        };
        const next = try mx.retain(try pass.scope.slice(draft_hidden, 0, @intCast(count - 1), @intCast(count)));
        mx.free(hidden);
        hidden = next;
        try m.commit(&pass, count);
        offset += count;
    }
    while (result.tokens.items.len < max_tokens) {
        const count = @min(draft_budget + 1, max_tokens - result.tokens.items.len);
        var proposed: [16]i32 = undefined;
        proposed[0] = pending;
        if (comptime @hasDecl(M, "propose")) if (draft_budget > 0) try m.propose(hidden, pending, proposed[0..count], settings);
        var pass = try m.forward(proposed[0..count]);
        defer pass.deinit();
        const draft_hidden = if (@hasDecl(M, "draftHidden")) M.draftHidden(&pass) else pass.hidden;
        var positions: [16]i32 = undefined;
        for (positions[0..count], 0..) |*pos, j| pos.* = m.position + @as(i32, @intCast(j)) + 1;
        const targets = try sampling.rows(&m.kernels, &pass.scope, pass.logits, positions[0..count], settings);
        defer mx.allocator.free(targets);
        var keep: usize = 0;
        var stopped = false;
        while (keep < count) {
            if (keep > 0 and proposed[keep] != targets[keep - 1]) break;
            const id = proposed[keep];
            try result.tokens.append(mx.allocator, @intCast(id));
            keep += 1;
            stopped = if (@hasDecl(M, "isEos")) m.isEos(id) else M.eos(id);
            if (stopped) break;
        }
        if (comptime @hasDecl(M, "propose")) if (draft_budget > 0 and !absorb_on_commit) {
            const rows = if (keep == 1) hidden else try pass.scope.cat(&.{ hidden, try pass.scope.slice(draft_hidden, 0, 0, @intCast(keep - 1)) }, 0);
            try absorb(m, rows, proposed[0..keep]);
        };
        const next = try mx.retain(try pass.scope.slice(draft_hidden, 0, @intCast(keep - 1), @intCast(keep)));
        mx.free(hidden);
        hidden = next;
        try m.commit(&pass, keep);
        result.rounds += 1;
        result.drafted += count - 1;
        result.accepted += keep - 1;
        pending = targets[keep - 1];
        if (stopped) break;
    }
    return result;
}

fn absorb(m: anytype, hidden: mx.Array, tokens: []const i32) !void {
    if (comptime !@hasDecl(@TypeOf(m.*), "forwardMtp")) return error.UnsupportedDrafts;
    var pass = try m.forwardMtp(hidden, tokens);
    defer pass.deinit();
    try m.commitMtp(&pass, tokens.len);
}
