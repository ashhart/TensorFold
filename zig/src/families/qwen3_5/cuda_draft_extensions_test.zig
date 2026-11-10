//! Metadata overlays retain the original constructor's values while carrying draft extension capabilities.
const std = @import("std");
const extensions = @import("cuda_draft_extensions.zig");
const dr = @import("lanes").drafter;
const t = std.testing;
const Kind = enum { dflash2, dspine };
const Probe = struct {
    c: struct { kind: Kind, block: usize, window: usize = 2048, copy_match: i64 },
    tree_nodes: usize = 15,
    auxiliary_nodes: usize = 23,
    calls: usize = 0,
    fn base(p: *anyopaque) dr.Facts {
        const d: *Probe = @ptrCast(@alignCast(p));
        d.calls += 1;
        return .{ .depth = @intCast(d.c.block - 1), .step_ms = 0.375, .batched = true, .copy_match = 37, .stream_bytes = 8192, .tree_nodes = @intCast(if (d.c.block == 16) d.tree_nodes else 0) };
    }
};

test "draft extension facts: explicit copy overrides, history and tree support retain all constructor values" {
    for ([_]Kind{ .dflash2, .dspine }) |kind| for ([_]usize{ 8, 16 }) |block| for ([_]i64{ -1, 0, 3 }) |copy| {
        var d: Probe = .{ .c = .{ .kind = kind, .block = block, .copy_match = copy } };
        const actual = extensions.Facts(Probe, Probe.base).facts(&d);
        try t.expectEqual(@as(usize, 1), d.calls);
        try t.expectEqual(@as(u32, 2048), actual.history_rows);
        try t.expectEqual(@as(u32, @intCast(block - 1)), actual.depth);
        try t.expectEqual(@as(f64, 0.375), actual.step_ms);
        try t.expect(actual.batched);
        try t.expectEqual(@as(usize, 8192), actual.stream_bytes);
        try t.expectEqual(@as(i64, if (copy > 0) copy else 37), actual.copy_match);
        try t.expectEqual(@as(u32, if (kind == .dflash2 or block == 16) 15 else 0), actual.tree_nodes);
        try t.expectEqual(@as(usize, if (kind == .dflash2) 23 else if (block == 16) 15 else 0), extensions.nodeBudget(&d));
    };
}

test "draft extension graphs: clearing is conditional and retains the existing graph cache ownership" {
    const Cache = struct {
        clears: usize = 0,
        pub fn dropAll(self: *@This()) void {
            self.clears += 1;
        }
    };
    var d: struct { graphs: ?Cache = null } = .{};
    extensions.clearGraphs(&d);
    try t.expect(d.graphs == null);
    d.graphs = .{};
    extensions.clearGraphs(&d);
    try t.expectEqual(@as(usize, 1), d.graphs.?.clears);
}
