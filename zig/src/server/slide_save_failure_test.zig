//! Accepted lessons and failed save receipts share the actual server graph and generic learner.
const std = @import("std");
const living = @import("living");
const Graph = @import("slide_graph.zig").Graph;
const Observer = struct {
    graph: *Graph,
    id: u32 = 0,
    pub fn accepted(o: *@This()) void {
        o.id = o.graph.add("accepted lesson", "Save recovery:", 1);
        o.graph.mark(o.id, .learned);
        o.graph.saving(null);
    }
    pub fn failed(o: *@This(), stale: bool) !void {
        o.graph.saveEnded(null, 0.1, "injected persistence failure");
        try std.testing.expect(stale);
        try std.testing.expectEqual(.learned, o.graph.find(o.id).?.state);
        try std.testing.expect(o.graph.saved.state == .failed);
    }
};
test "Save recovery: actual graph save errors retain learned commitments matching the live generic learner" {
    var graph = Graph.init(std.testing.allocator, std.testing.io, null);
    defer graph.deinit();
    var observer: Observer = .{ .graph = &graph };
    inline for (.{ living.families.qwen4_exp, living.families.qwen3_5 }) |F| {
        try living.save_failure_fixture.observedSaveFailure(F, true, false, &observer);
        try living.save_failure_fixture.observedSaveFailure(F, false, false, &observer);
        try living.save_failure_fixture.observedSaveFailure(F, false, true, &observer);
    }
}
