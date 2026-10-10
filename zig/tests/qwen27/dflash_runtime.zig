//! A generated five-layer target supplies accepted taps to the native draft forward and its bounded tree.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const target_graph = @import("synthetic_graph.zig");
const draft_graph = @import("dflash_graph.zig");
const Oracle = struct {
    a: std.mem.Allocator,
    runner: *q.decode_round.Runner,
    tokens: []const u32,
    prompt: usize,
    branch: bool = false,
    fn propose(ptr: *anyopaque, pending: u32, limit: u32) !q.dflash.selector.Tree {
        const oracle: *Oracle = @ptrCast(@alignCast(ptr));
        const at = oracle.runner.offsets[0] - oracle.prompt;
        if (at >= oracle.tokens.len or pending != oracle.tokens[at]) return error.OraclePosition;
        const children: usize = if (oracle.branch) 2 else 1;
        const count = @min(@as(usize, limit) / children, @min(7, oracle.tokens.len - at - 1));
        const ids = try oracle.a.alloc(u32, count * children);
        const parents = try oracle.a.alloc(i32, ids.len);
        const scores = try oracle.a.alloc(f64, ids.len);
        for (0..count) |depth| for (0..children) |child| {
            const i = depth * children + child;
            ids[i] = if (child + 1 == children) oracle.tokens[at + 1 + depth] else @intCast((oracle.tokens[at + 1 + depth] + 1) % oracle.runner.model.config.vocab);
            parents[i] = if (depth == 0) -1 else @intCast(depth * children - 1);
            scores[i] = 0;
        };
        return .{ .gpa = oracle.a, .tokens = ids, .parents = parents, .scores = scores };
    }
};
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const target = try target_graph.createLayers(a, 5);
    defer target.deinit();
    try @import("tree_commit.zig").run(target.device, target.queue);
    try @import("dflash_head.zig").run(a, target.device, target.queue);
    try @import("dflash_conv.zig").run(target.device, target.queue);
    var runner = try q.decode_round.Runner.init(a, target, 1, 256);
    defer runner.deinit();
    runner.capture_taps = true;
    runner.taps.ids = .{ 0, 1, 2, 3, 4 };
    const draft = try draft_graph.create(a, target);
    defer draft.deinit();
    const eager = try draft_graph.create(a, target);
    defer eager.deinit();
    const accepted = try target.device.buffer(128 * 5 * 512 * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer accepted.deinit();
    const kept = try target.device.buffer(128 * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer kept.deinit();
    const session = q.session.Session{ .runner = &runner };
    var prompt: [130]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast((i * 17 + 7) % 96);
    var first: usize = 0;
    while (first < prompt.len) {
        const rows = @min(prompt.len - first, 128);
        try session.promptChunk(prompt[first..][0..rows], first + rows == prompt.len);
        for (kept.slice(u32, rows), 0..) |*index, i| index.* = @intCast(i);
        const cb = target.queue.commandBuffer();
        const e = cb.compute(.serial);
        var path: [128]u32 = undefined;
        for (path[0..rows], 0..) |*index, i| index.* = @intCast(i);
        try runner.taps.accepted(target.glue, e, path[0..rows], .{ .buffer = kept }, .{ .buffer = accepted });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailure;
        try draft.absorb(.{ .buf = accepted }, @intCast(rows));
        if (draft.session.ready or draft.session.context.end != 0) return error.ContextWasNotDeferred;
        var sub: usize = 0;
        while (sub < rows) {
            const count = @min(rows - sub, 17);
            try eager.absorb(.{ .buf = accepted, .off = sub * draft.graph.config.tapWidth() * 2 }, @intCast(count));
            try eager.flush();
            sub += count;
        }
        first += rows;
    }
    const before = try session.fingerprint();
    var tree = try draft.propose(try session.greedy(), 15);
    defer tree.deinit();
    var eager_tree = try eager.propose(try session.greedy(), 15);
    defer eager_tree.deinit();
    if (!std.mem.eql(u32, tree.tokens, eager_tree.tokens) or !std.mem.eql(i32, tree.parents, eager_tree.parents) or !std.mem.eql(f64, tree.scores, eager_tree.scores)) return error.DeferredProposalDiffers;
    const stride = @as(usize, draft.graph.config.kvWidth()) * 2;
    for (draft.backend.caches, eager.backend.caches) |lazy_cache, eager_cache| for ([_]mtl.Buffer{ lazy_cache.keys, lazy_cache.values }, [_]mtl.Buffer{ eager_cache.keys, eager_cache.values }) |lazy_buffer, eager_buffer| {
        var position = draft.session.context.begin;
        while (position < draft.session.context.end) : (position += 1) {
            const slot: usize = @intCast(position % draft.graph.config.window);
            if (!std.mem.eql(u8, lazy_buffer.contents()[slot * stride ..][0..stride], eager_buffer.contents()[slot * stride ..][0..stride])) return error.DeferredContextDiffers;
        }
    };
    std.debug.print("deferred accepted context: wrapped/cropped130-row suffix, all visible per-layer KV and proposal token/parent/score bytes equal eager17-row flushes\n", .{});
    try draft.absorb(.{ .buf = accepted }, 128);
    var replay: usize = 0;
    while (replay < 128) {
        const count = @min(128 - replay, 17);
        try eager.absorb(.{ .buf = accepted, .off = replay * draft.graph.config.tapWidth() * 2 }, @intCast(count));
        try eager.flush();
        replay += count;
    }
    if (draft.session.context.end != 130 or draft.committed_end != 258) return error.DeferredPositionDiffers;
    var gap_tree = try draft.propose(try session.greedy(), 15);
    defer gap_tree.deinit();
    var replay_tree = try eager.propose(try session.greedy(), 15);
    defer replay_tree.deinit();
    if (!std.mem.eql(u32, gap_tree.tokens, replay_tree.tokens) or !std.mem.eql(i32, gap_tree.parents, replay_tree.parents) or !std.mem.eql(f64, gap_tree.scores, replay_tree.scores)) return error.DeferredGapProposalDiffers;
    if (draft.session.context.begin != 195 or draft.session.context.end != 258) return error.DeferredGapPosition;
    for (draft.backend.caches, eager.backend.caches) |lazy_cache, eager_cache| for ([_]mtl.Buffer{ lazy_cache.keys, lazy_cache.values }, [_]mtl.Buffer{ eager_cache.keys, eager_cache.values }) |lazy_buffer, eager_buffer| {
        for (195..258) |position| {
            const slot = position % draft.graph.config.window;
            if (!std.mem.eql(u8, lazy_buffer.contents()[slot * stride ..][0..stride], eager_buffer.contents()[slot * stride ..][0..stride])) return error.DeferredGapCacheDiffers;
        }
    };
    std.debug.print("deferred cache hole: materialized130/logical258 discards old prefix, retained195..258 and proposals equal eager\n", .{});
    const parents = try tree.verifyParents(a);
    defer a.free(parents);
    if (tree.tokens.len == 0 or parents.len != tree.tokens.len + 1) return error.EmptyDraftTree;
    for (tree.tokens) |id| if (id >= target.config.vocab) return error.BadDraftId;
    if (!std.mem.eql(u8, &before, &try session.fingerprint())) return error.DraftChangedTarget;
    if (draft.session.context.end != 258 or draft.session.context.begin != 195) return error.BadDraftContext;
    var plain: [16]u32 = undefined;
    var final_by_count: [16][32]u8 = undefined;
    for (&plain, 0..) |*id, i| {
        id.* = try session.greedy();
        final_by_count[i] = try session.fingerprint();
        if (i + 1 < plain.len) try session.step(id.*);
    }
    const plain_final = try session.fingerprint();
    try runner.reset(0);
    try draft.reset();
    var generation = try q.dflash.generation.Generation.init(a, &runner, draft);
    defer generation.deinit();
    try generation.prefill(&prompt, 16);
    if (!std.mem.eql(u8, &before, &try session.fingerprint())) return error.DraftPrefillTargetChanged;
    const result = try generation.run(16, true, 15);
    defer a.free(result.tokens);
    if (!std.mem.eql(u32, &plain, result.tokens)) return error.DraftedTokensDiffer;
    if (!std.mem.eql(u8, &plain_final, &try session.fingerprint())) return error.DraftedStateDiffer;
    std.debug.print("native host draft round: 16 tokens and final recurrent/KV/logits byte-equal plain, {d} rounds, {d} verified rows, {d} matched draft tokens\n", .{ result.rounds, result.verified, result.matched });
    try runner.reset(0);
    try draft.reset();
    var oracle = Oracle{ .a = a, .runner = &runner, .tokens = &plain, .prompt = prompt.len };
    generation.proposer = .{ .ptr = &oracle, .call = Oracle.propose };
    try generation.prefill(&prompt, 128);
    const perfect = try generation.run(16, true, 15);
    defer a.free(perfect.tokens);
    if (!std.mem.eql(u32, &plain, perfect.tokens) or !std.mem.eql(u8, &plain_final, &try session.fingerprint())) return error.PerfectTreeDiffers;
    if (perfect.matched < 7) return error.PerfectTreeNotConsumed;
    std.debug.print("perfect tree: all 16 tokens and final state/KV/logits equal plain, {d} matched draft tokens\n", .{perfect.matched});
    oracle.branch = true;
    try runner.reset(0);
    try draft.reset();
    try generation.prefill(&prompt, 16);
    const branching = try generation.run(16, true, 15);
    defer a.free(branching.tokens);
    if (!std.mem.eql(u32, &plain, branching.tokens) or !std.mem.eql(u8, &plain_final, &try session.fingerprint()) or branching.matched != perfect.matched) return error.BranchKeepDiffers;
    std.debug.print("GPU non-prefix keep: all16 tokens and final state/KV/logits equal plain, {d} matched draft tokens\n", .{branching.matched});
    for ([_]usize{ 1, 2, 3, 8 }) |budget| {
        try runner.reset(0);
        try draft.reset();
        try generation.prefill(&prompt, 32);
        const bounded = try generation.run(budget, true, 15);
        defer a.free(bounded.tokens);
        if (!std.mem.eql(u32, plain[0..budget], bounded.tokens) or !std.mem.eql(u8, &final_by_count[budget - 1], &try session.fingerprint())) return error.DraftBudgetDiffers;
    }
    const original_eos = target.config.eos[0];
    target.config.eos[0] = plain[2];
    defer target.config.eos[0] = original_eos;
    try runner.reset(0);
    try draft.reset();
    try generation.prefill(&prompt, 128);
    const stopped = try generation.run(16, false, 15);
    defer a.free(stopped.tokens);
    var stop_at: usize = 0;
    while (plain[stop_at] != plain[2]) : (stop_at += 1) {}
    if (!std.mem.eql(u32, plain[0 .. stop_at + 1], stopped.tokens) or !std.mem.eql(u8, &final_by_count[stop_at], &try session.fingerprint())) return error.DraftEosDiffers;
    target.config.eos[0] = original_eos;
    generation.proposer = null;
    for ([_]u32{ 1, 2, 8, 16 }) |block| {
        try runner.reset(0);
        try draft.reset();
        try draft.setTreeBlock(block);
        try generation.prefill(&prompt, 128);
        const bounded = try generation.run(16, true, 15);
        defer a.free(bounded.tokens);
        if (!std.mem.eql(u32, &plain, bounded.tokens) or !std.mem.eql(u8, &plain_final, &try session.fingerprint())) return error.RuntimeBlockDiffers;
    }
    try draft.backend.prepare();
    draft.preparation = .{ .mode = .prepared_q4_reference, .quantization_verified = true, .source_sha256 = @import("tensorfold").affine4.signature() };
    for ([_]u32{ 1, 2, 8, 16 }) |block| {
        try runner.reset(0);
        try draft.reset();
        try draft.setTreeBlock(block);
        try generation.prefill(&prompt, 128);
        const quantized = try generation.run(16, true, 15);
        defer a.free(quantized.tokens);
        if (!std.mem.eql(u32, &plain, quantized.tokens) or !std.mem.eql(u8, &plain_final, &try session.fingerprint())) return error.PreparedDraftDiffers;
    }
    for ([_]usize{ 1, 3, 7, 16 }) |batch_width| {
        try runner.reset(0);
        try draft.reset();
        try generation.prefill(&prompt, 32);
        var collected: std.ArrayList(u32) = .empty;
        defer collected.deinit(a);
        while (collected.items.len < plain.len) {
            const next = try generation.runBatch(@min(batch_width, plain.len - collected.items.len), 15, &.{}, collected.items.len != 0);
            defer a.free(next.tokens);
            if (next.tokens.len == 0 or try session.greedy() != next.tokens[next.tokens.len - 1]) return error.BatchPendingToken;
            try collected.appendSlice(a, next.tokens);
        }
        if (!std.mem.eql(u32, &plain, collected.items) or !std.mem.eql(u8, &plain_final, &try session.fingerprint())) return error.BatchedDraftDiffers;
    }
    std.debug.print("draft batches1/3/7/16: no repeated pending token, all target tokens and final state equal ordinary\n", .{});
    std.debug.print("GPU selection: branching budgets1/2/3/8, EOS, BF16/prepared-q4 runtime blocks1/2/8/16 target bytes equal ordinary\n", .{});
    std.debug.print("native draft synthetic: 81 BF16 tensors,47 primitive linears,initial130 target tap rows plus128 gap-fixture rows,checkpointblock8/runtimecap16,{d} draft nodes;target state/KV/logits unchanged\n", .{tree.tokens.len});
}
