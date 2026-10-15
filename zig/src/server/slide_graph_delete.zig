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
        var keep: std.ArrayList(api.Neuron) = .empty;
        for (o.neurons) |n| if (!overlaps(&.{n}, taken)) keep.append(g.arena.allocator(), n) catch return;
        o.neurons = keep.items;
    }
}

/// Whether a neuron of `xs` sits where one of `ys` does (layer and index).
fn overlaps(xs: []const api.Neuron, ys: []const api.Neuron) bool {
    for (xs) |x| for (ys) |y| if (x.layer == y.layer and x.neuron == y.neuron) return true;
    return false;
}
