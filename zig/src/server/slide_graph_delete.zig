//! A delete's records in the fact graph, each written to its file at once: when the file cannot be, nothing changes.
const std = @import("std");
const api = @import("engine_api");
const slide_graph = @import("slide_graph.zig");
const answers = @import("slide_answers.zig");
const Graph = slide_graph.Graph;
const Fact = slide_graph.Fact;
const Allocator = std.mem.Allocator;

/// Fact `id` in `state` (a delete's `deleting`, or `learned` again), written; on failure the graph is as it was.
pub fn record(g: *Graph, id: u32, state: slide_graph.State) !void {
    g.mutex.lockUncancelable(g.io);
    defer g.mutex.unlock(g.io);
    const f = g.find(id) orelse return error.NoSuchFact;
    const was = f.state;
    errdefer f.state = was;
    f.state = state;
    try g.persist();
}

/// A fact deleted on request at `at` (ms), and why: its neurons no longer fire. Written, or nothing changes.
pub fn remove(g: *Graph, id: u32, at: i64, why: []const u8) !void {
    g.mutex.lockUncancelable(g.io);
    defer g.mutex.unlock(g.io);
    const f = g.find(id) orelse return error.NoSuchFact;
    // what can fail before the fact changes is done first; a failed write then puts the whole fact back
    const message = try g.arena.allocator().dupe(u8, why);
    const was = f.*;
    errdefer f.* = was;
    f.state = .deleted;
    f.neurons = &.{};
    f.deleted = at;
    f.message = message;
    try g.persist();
}

/// Every fact as it stands, copied into `a`, for reading outside the lock.
pub fn snapshot(g: *Graph, a: Allocator) ![]Fact {
    g.mutex.lockUncancelable(g.io);
    defer g.mutex.unlock(g.io);
    const out = try a.alloc(Fact, g.facts.items.len);
    for (g.facts.items, out) |f, *o| {
        o.* = f;
        o.text = try a.dupe(u8, f.text);
        o.message = try a.dupe(u8, f.message);
        o.neurons = try a.dupe(api.Neuron, f.neurons);
        o.answers = try answers.dupe(a, f.answers);
        const qs = try a.alloc([]const u8, f.questions.len);
        for (f.questions, qs) |q, *c| c.* = try a.dupe(u8, q);
        o.questions = qs;
    }
    return out;
}

/// A lesson placed on a neuron a fact left `deleting` lists owns it: that listing drops it (under the lock).
pub fn takeOver(g: *Graph, f: *const Fact, taken: []const api.Neuron) void {
    for (g.facts.items) |*o| {
        // only a delete's neurons go free while the graph still lists them; a learned fact's never do
        if (o.id == f.id or o.state != .deleting or !overlaps(o.neurons, taken)) continue;
        // filtered where the graph's arena holds it: nothing to allocate, so no failure leaves a taken neuron listed
        const own = @constCast(o.neurons);
        var n: usize = 0;
        for (own) |x| if (!overlaps(&.{x}, taken)) {
            own[n] = x;
            n += 1;
        };
        o.neurons = own[0..n];
    }
}

/// Whether a neuron of `xs` sits where one of `ys` does (layer and index).
fn overlaps(xs: []const api.Neuron, ys: []const api.Neuron) bool {
    for (xs) |x| for (ys) |y| if (x.layer == y.layer and x.neuron == y.neuron) return true;
    return false;
}

test "a neuron a lesson takes from a deleting fact leaves its listing even when no memory is left" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var g = Graph.init(failing.allocator(), std.testing.io, null);
    defer g.deinit();
    const id = g.add("My sister is called Ana.", "chat", 100);
    g.mark(id, .learned);
    var many: [4096]api.Neuron = undefined; // a listing larger than an arena block has left
    for (&many, 0..) |*x, i| x.* = .{ .layer = @intCast(i / 64), .neuron = @intCast(i % 64), .key = .reply };
    g.placed(id, &many);
    try g.record(id, .deleting);
    const other = g.add("My dog is called Ginger.", "chat", 101);
    failing.fail_index = failing.alloc_index; // every allocation from here on fails
    const taken = [_]api.Neuron{.{ .layer = 0, .neuron = 0, .key = .reply }};
    takeOver(&g, g.find(other).?, &taken);
    failing.fail_index = std.math.maxInt(usize);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const f = (try g.snapshot(arena.allocator()))[0];
    try std.testing.expectEqual(@as(usize, many.len - 1), f.neurons.len);
    for (f.neurons) |x| try std.testing.expect(!(x.layer == 0 and x.neuron == 0));
}
