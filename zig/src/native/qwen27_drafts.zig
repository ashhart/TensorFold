//! The 27B lane backend's drafts: each drafting slot's tree, up to eight in one drafter pass, and the held trees.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const be = lanes.backend;
const lb = @import("qwen27_lanes.zig");
const Metal = lb.Metal;
const Tree = q.dflash.selector.Tree;
const Proposal = q.dflash.runtime_model.Model.Proposal;

/// Each drafting slot proposes its next tree, up to eight in one drafter pass; a shared round's kept taps first.
pub fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
    const b = Metal.self(ptr);
    var batch: [8]Proposal = undefined;
    var owners: [8]u32 = undefined;
    var n: usize = 0;
    for (requests) |r| {
        const slot = try b.slotFor(r.stream);
        if (!b.draftsOwn(slot)) continue;
        const g = &b.draft.?.generation;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        if (b.count > 1) try absorbEarly(b, slot, r.rows orelse return error.NoKeptRows) else try b.settle();
        b.dropTreeOf(slot);
        const stands = b.runner.offsets[slot] + b.early[slot];
        if ((if (slot == 0) g.draft.committed_end else b.contexts[slot].?.committed_end) != stands) return error.BadDraftPosition;
        const room = b.runner.capacity - stands;
        if (r.depth == 0 or room < 2) continue;
        const pending = if (r.rows != null) r.follow[r.follow.len - 1] else try b.value(r.first orelse return error.NoFirstToken);
        const nodes: u32 = @intCast(@min(@min(r.depth, b.nodes), room - 1));
        if (slot == 0) if (g.proposer) |p| {
            b.trees[0] = try p.call(p.ptr, pending, nodes);
            continue;
        };
        batch[n] = .{ .context = if (slot == 0) null else b.contexts[slot].?, .pending = pending, .nodes = nodes };
        owners[n] = slot;
        n += 1;
        if (n == batch.len) {
            try proposeAll(b, batch[0..n], owners[0..n]);
            n = 0;
        }
    }
    if (n > 0) try proposeAll(b, batch[0..n], owners[0..n]);
    for (requests) |r| {
        const slot = b.slotOf(r.stream) orelse continue;
        if (b.trees[slot]) |*t| {
            const opening = if (r.rows == null) try b.value(r.first orelse return error.NoFirstToken) else null;
            try @import("qwen27_masks.zig").prune(b.gpa, r.stream.grammar, opening, t);
        }
    }
}

/// The batch's trees: a stream alone takes its own pass (slot 0's fused absorb and block); several share one.
fn proposeAll(b: *Metal, batch: []const Proposal, owners: []const u32) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = b.draft.?.generation.draft;
    if (batch.len == 1) {
        const p = batch[0];
        const x = p.context orelse {
            b.trees[owners[0]] = try m.propose(p.pending, p.nodes);
            return;
        };
        // the shared drafter takes the slot's context for one proposal, then hands it back
        try m.swap(x);
        const tree = m.propose(p.pending, p.nodes);
        m.swap(x) catch {
            m.backend.failed = true; // slot 0's context is parked in this slot's: recovery resets both
            return error.DraftContextSwap;
        };
        b.trees[owners[0]] = try tree;
        return;
    }
    var trees: [8]Tree = undefined;
    try m.proposeBatch(batch, trees[0..batch.len]);
    for (owners, trees[0..batch.len]) |slot, t| b.trees[slot] = t;
}

/// A slot's kept rows of a shared round's verify, gathered from its taps for its drafter before the keep.
fn absorbEarly(b: *Metal, slot: u32, path: []const u32) !void {
    if (b.early[slot] > 0) return error.AbsorbedTwice;
    const r = b.runner;
    const w = std.mem.indexOfScalar(u32, b.windows[0..b.count], slot) orelse return error.UnknownStream;
    const first = b.round.?.firsts[w];
    const rows = b.kept.slice(u32, path.len);
    for (rows, path) |*x, row| x.* = first + row;
    const cb = r.model.queue.commandBuffer();
    const e = cb.compute(.serial);
    try r.taps.accepted(r.model.glue, e, rows, .{ .buffer = b.kept }, .{ .buffer = b.gathered_taps.? });
    e.end();
    cb.commit();
    cb.wait();
    if (q.decode_round.gpuFailed(cb, "a taps gather")) return error.DraftTapGpuFailure;
    try b.ringOf(slot).?.absorb(b.gathered_taps.?, 0, @intCast(path.len));
    b.early[slot] = path.len;
}

/// Each held node's chance of landing: its own given its parent (e^own as the prior) times its parent's.
pub fn probabilitiesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
    const b = Metal.self(ptr);
    const slot = b.slotOf(s) orelse return false;
    const t = b.trees[slot] orelse return false;
    if (out.len > t.scores.len) return false;
    for (out, t.parents[0..out.len], 0..) |*p, parent, i| {
        p.* = b.landing.chance(s.sampling != null, Metal.local(t, i)) * (if (parent >= 0) out[@intCast(parent)] else 1);
    }
    return true;
}

/// A slot's held tree as host tokens; a slot that does not draft holds none (its windows verify copies or one row).
pub fn treeFn(ptr: *anyopaque, s: *lanes.Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
    const b = Metal.self(ptr);
    const slot = b.slotOf(s) orelse return null;
    const t = b.trees[slot] orelse return .{ .count = 0, .tokens = try gpa.alloc(u32, 0) };
    const tokens = try gpa.dupe(u32, t.tokens);
    errdefer gpa.free(tokens);
    return .{ .count = @intCast(t.tokens.len), .tokens = tokens, .parents = try gpa.dupe(i32, t.parents) };
}
