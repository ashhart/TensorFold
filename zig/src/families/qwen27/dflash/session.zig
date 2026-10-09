//! Only contiguous committed tap positions advance draft caches; the pending token is an input, not a committed tap.
const std = @import("std");
const cfg = @import("config.zig");
const op = @import("operators.zig");
const weights = @import("weights.zig");
const execution = @import("execution.zig");
const selector = @import("selector.zig");
pub const Batch = struct { values: op.View, positions: []const u64, layers: [5]u32, target_end: u64 };
pub const Session = struct {
    config: cfg.Config,
    context: op.Context,
    ready: bool = false,
    pub fn init(c: cfg.Config, cache: u64, first: u64) !Session {
        try c.check();
        if (cache == 0) return error.BadDraftContext;
        return .{ .config = c, .context = .{ .cache = cache, .begin = first, .end = first } };
    }
    pub fn admit(s: Session, batch: Batch) !op.Context {
        if (!std.mem.eql(u32, &batch.layers, &s.config.taps) or batch.positions.len == 0) return error.BadCommittedTaps;
        try batch.values.check(@intCast(batch.positions.len), s.config.tapWidth(), .bf16);
        const end = try std.math.add(u64, s.context.end, batch.positions.len);
        if (end != batch.target_end) return error.BadCommittedTaps;
        for (batch.positions, 0..) |position, i| if (position != s.context.end + i or position >= batch.target_end) return error.BadCommittedTaps;
        const begin = @max(s.context.begin, end -| (@as(u64, s.config.window) - 1));
        return .{ .cache = s.context.cache, .begin = begin, .end = end };
    }
    pub fn absorb(s: *Session, ops: op.Ops, graph: *const weights.Graph, preparation: op.Preparation, batch: Batch) !void {
        if (!std.meta.eql(s.config, graph.config)) return error.BadDraftContext;
        const next = try s.admit(batch);
        try execution.append(ops, graph, preparation, s.context, batch.values, batch.positions, next);
        s.context = next;
        s.ready = true;
    }
    /// absorb and treeBlock in one drafter frame.
    pub fn absorbTreeBlock(s: *Session, gpa: std.mem.Allocator, ops: op.Ops, graph: *const weights.Graph, preparation: op.Preparation, batch: Batch, pending: u32, block: u32, options: selector.Options) !selector.Tree {
        if (!std.meta.eql(s.config, graph.config) or !s.ready) return error.BadDraftContext;
        const next = try s.admit(batch);
        var lattice = try execution.appendBlock(ops, graph, preparation, s.context, batch.values, batch.positions, next, pending, block);
        defer lattice.deinit();
        s.context = next;
        return selector.propose(gpa, graph, lattice, pending, try std.math.add(u64, next.end, 1), options);
    }
    pub fn firstDraw(s: Session) !u64 {
        if (!s.ready) return error.DraftContextNotReady;
        return std.math.add(u64, s.context.end, 1);
    }
    pub fn propose(s: Session, ops: op.Ops, graph: *const weights.Graph, preparation: op.Preparation, pending: u32, first_draw: u64) !op.Lattice {
        if (!std.meta.eql(s.config, graph.config) or first_draw != try s.firstDraw()) return error.BadDraftPosition;
        return execution.forward(ops, graph, preparation, s.context, pending);
    }
    pub fn treeBlock(s: Session, gpa: std.mem.Allocator, ops: op.Ops, graph: *const weights.Graph, preparation: op.Preparation, pending: u32, first_draw: u64, block: u32, options: selector.Options) !selector.Tree {
        if (!std.meta.eql(s.config, graph.config) or first_draw != try s.firstDraw()) return error.BadDraftPosition;
        var lattice = try execution.forwardBlock(ops, graph, preparation, s.context, pending, block);
        defer lattice.deinit();
        return selector.propose(gpa, graph, lattice, pending, first_draw, options);
    }
    pub fn tree(s: Session, gpa: std.mem.Allocator, ops: op.Ops, graph: *const weights.Graph, preparation: op.Preparation, pending: u32, first_draw: u64, options: selector.Options) !selector.Tree {
        var lattice = try s.propose(ops, graph, preparation, pending, first_draw);
        defer lattice.deinit();
        return selector.propose(gpa, graph, lattice, pending, first_draw, options);
    }
};

test "committed tap admission excludes pending and future rows and shifts the draw once" {
    var s = try Session.init(.{}, 1, 10);
    const batch = Batch{ .values = .{ .handle = 2, .rows = 2, .width = 25600, .stride = 25600, .dtype = .bf16 }, .positions = &.{ 10, 11 }, .layers = cfg.tap_ids, .target_end = 12 };
    s.context = try s.admit(batch);
    s.ready = true;
    try std.testing.expectEqual(@as(u64, 13), try s.firstDraw());
    var bad = batch;
    bad.positions = &.{ 12, 14 };
    bad.target_end = 14;
    try std.testing.expectError(error.BadCommittedTaps, s.admit(bad));
}
test "cropping moves only the context beginning while absolute anchor stays committed end" {
    var c = cfg.Config{};
    c.window = 8;
    const s = try Session.init(c, 1, 20);
    const positions = [_]u64{ 20, 21, 22, 23, 24, 25, 26, 27 };
    const next = try s.admit(.{ .values = .{ .handle = 2, .rows = 8, .width = 25600, .stride = 25600, .dtype = .bf16 }, .positions = &positions, .layers = cfg.tap_ids, .target_end = 28 });
    try std.testing.expectEqual(@as(u64, 21), next.begin);
    try std.testing.expectEqual(@as(u64, 28), next.end);
}
test "wrong tap reading or dtype fails before a cache transaction" {
    const s = try Session.init(.{}, 1, 0);
    var b = Batch{ .values = .{ .handle = 2, .rows = 1, .width = 25600, .stride = 25600, .dtype = .bf16 }, .positions = &.{0}, .layers = cfg.tap_ids, .target_end = 1 };
    b.layers[0] = 4;
    try std.testing.expectError(error.BadCommittedTaps, s.admit(b));
    b.layers = cfg.tap_ids;
    b.values.dtype = .f32;
    try std.testing.expectError(error.DraftOperatorShape, s.admit(b));
}
