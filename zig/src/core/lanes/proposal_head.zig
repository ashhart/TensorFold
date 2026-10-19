//! Optional immutable proposal routing; committed context alone controls coverage, never target logits.
const std = @import("std");
const be = @import("backend.zig");
pub const Mode = enum { legacy, calibrated, full };
pub const Kind = enum { subset, full, paused };
pub const recent = 16;

/// Borrowed from a validated model-owned artifact; its generation outlives every queued use.
pub const Map = struct {
    ptr: *const anyopaque,
    ids: []const u32,
    fingerprint: [32]u8,
    column_fn: *const fn (*const anyopaque, u32) ?usize,
    global_fn: *const fn (*const anyopaque, usize) ?u32,
    pub fn column(m: Map, id: u32) ?usize {
        return m.column_fn(m.ptr, id);
    }
    pub fn global(m: Map, at: usize) ?u32 {
        return m.global_fn(m.ptr, at);
    }
    fn from(a: anytype) Map {
        const T = @TypeOf(a.*);
        const F = struct {
            fn column(ptr: *const anyopaque, id: u32) ?usize {
                const x: *const T = @ptrCast(@alignCast(ptr));
                return x.column(id);
            }
            fn global(ptr: *const anyopaque, at: usize) ?u32 {
                const x: *const T = @ptrCast(@alignCast(ptr));
                return x.global(at);
            }
        };
        return .{ .ptr = a, .ids = a.ids(), .fingerprint = a.fingerprint, .column_fn = F.column, .global_fn = F.global };
    }
};
pub const Plan = struct { kind: Kind, vocab: u32, map: ?Map = null };
pub const Capabilities = struct { subset: bool, full: bool, ranks: u32 = 1 };

/// Construct before publication, keep at a stable address, and drain queues before destroying its artifact.
pub const Service = struct {
    gpa: std.mem.Allocator,
    mode: Mode = .legacy,
    scorer: ?*const @import("proposal_score.zig").Scorer = null,
    compact: Plan = .{ .kind = .paused, .vocab = 0 },
    complete: Plan = .{ .kind = .paused, .vocab = 0 },
    paused: Plan = .{ .kind = .paused, .vocab = 0 },

    pub fn calibrated(gpa: std.mem.Allocator, artifact: anytype, actual: anytype, caps: Capabilities) !Service {
        if (!artifact.matches(actual)) return error.ProposalIdentityMismatch;
        if (!caps.subset) return error.ProposalSubsetUnsupported;
        if (caps.ranks != 1) return error.ProposalPlacementUnsupported;
        return .{ .gpa = gpa, .mode = .calibrated, .compact = .{ .kind = .subset, .vocab = actual.vocab, .map = Map.from(artifact) }, .complete = .{ .kind = if (caps.full) .full else .paused, .vocab = actual.vocab }, .paused = .{ .kind = .paused, .vocab = actual.vocab } };
    }
    pub fn full(gpa: std.mem.Allocator, vocab: u32, available: bool) !Service {
        if (!available or vocab == 0) return error.ProposalFullUnsupported;
        return .{ .gpa = gpa, .mode = .full, .complete = .{ .kind = .full, .vocab = vocab }, .paused = .{ .kind = .paused, .vocab = vocab } };
    }
    pub fn withScorer(s: Service, scorer: *const @import("proposal_score.zig").Scorer) !Service {
        if (s.mode == .legacy or scorer.descriptor.vocab != s.complete.vocab) return error.ProposalServiceUnsupported;
        if (s.compact.map) |m| {
            const generation = scorer.generation orelse return error.ProposalIdentityMismatch;
            if (!std.mem.eql(u8, &generation, &m.fingerprint)) return error.ProposalIdentityMismatch;
        }
        const logical: usize = if (s.compact.map) |m| m.ids.len else s.complete.vocab;
        switch (scorer.selection) {
            .topk, .coupled_topk => |k| if (k > logical) return error.ProposalScoreShape,
            else => {},
        }
        var bound = s;
        bound.scorer = scorer;
        return bound;
    }
    pub fn score(s: *const Service, plan: *const Plan, input: @import("proposal_score.zig").Normalized) !@import("proposal_score.zig").View {
        if (plan != &s.compact and plan != &s.complete and plan != &s.paused) return error.ProposalIdentityMismatch;
        const scorer = s.scorer orelse return error.ProposalServiceUnsupported;
        return scorer.score(plan, input);
    }
    fn choose(s: *const Service, r: be.DraftRequest) *const Plan {
        if (r.stream.finished or r.stream.paused) return &s.paused;
        if (s.mode == .full) return &s.complete;
        const m = s.compact.map.?;
        if (r.stream.grammar != null) return &s.complete;
        for (r.stream.force.items) |id| if (m.column(id) == null) return &s.complete;
        for (r.stream.eos) |id| if (m.column(id) == null) return &s.complete;
        if (r.first) |feed| switch (feed) {
            .handle => return &s.complete,
            .value => |id| if (m.column(id) == null) return &s.complete,
        };
        const context = r.stream.context.items;
        if (context.len < recent) return &s.complete;
        for (context[context.len - recent ..]) |id| if (m.column(id) == null) return &s.complete;
        if (r.follow.len > 0 and m.column(r.follow[r.follow.len - 1]) == null) return &s.complete;
        return &s.compact;
    }
};

/// Only opt-in producers may consume a plan; refusing an absent callback precedes absorption or scoring.
pub fn draft(service: *const Service, b: be.Backend, requests: []const be.DraftRequest) !void {
    const run = b.vtable.proposal_draft orelse return error.ProposalServiceUnsupported;
    const clear = b.vtable.clear_proposals orelse return error.ProposalServiceUnsupported;
    for (requests) |r| {
        r.stream.proposal_depth = 0;
        clear(b.ptr, r.stream);
    }
    errdefer for (requests) |r| {
        r.stream.proposal_depth = 0;
        clear(b.ptr, r.stream);
    };
    const routed = try service.gpa.dupe(be.DraftRequest, requests);
    defer service.gpa.free(routed);
    for (routed) |*r| {
        if (r.stream.isCancelled()) return error.Cancelled;
        r.plan = service.choose(r.*);
        if (r.plan.?.kind == .paused) {
            r.depth = 0;
            r.lanes = null;
            r.ranks = false;
        }
    }
    try run(b.ptr, routed);
    for (routed) |r| if (r.stream.isCancelled()) return error.Cancelled;
    for (routed) |r| r.stream.proposal_depth = r.depth;
}
