//! Fact usage and pinning live in the graph; one completed chat charges each firing fact at most once.
const std = @import("std");
const api = @import("engine_api");
const graph = @import("slide_graph.zig");
const json = @import("json");
const Graph = graph.Graph;

/// The audit supplies only neurons that contributed on the request's accepted path, once per external request.
pub fn used(g: *Graph, neurons: []const api.Neuron) !void {
    if (neurons.len == 0) return;
    var mask: api.pocket_usage.Masks = .{};
    for (neurons) |n| {
        if (n.layer >= mask.bits.len or n.neuron >= 64) return error.UsageGeometry;
        mask.bits[n.layer] |= @as(u64, 1) << @intCast(n.neuron);
    }
    var scope: @import("slide_usage_stream.zig").Scope = .{ .graph = g, .gpa = g.gpa, .enabled = true };
    defer scope.deinit();
    try scope.record(mask);
}

/// A pin changes metadata only; on a failed write the original pin stays.
pub fn pin(g: *Graph, id: u32, pinned: bool) !void {
    g.mutex.lockUncancelable(g.io);
    defer g.mutex.unlock(g.io);
    if (g.usage_invalid) return error.BadGraph;
    const f = g.find(id) orelse return error.NoSuchFact;
    if (f.state != .learned) return error.FactNotLearned;
    const old = f.pinned;
    errdefer f.pinned = old;
    f.pinned = pinned;
    try g.persist();
}

/// The oldest id breaks usage ties. Missing ownership and unfinished deletes cannot be eviction candidates.
pub fn least(facts: []const graph.Fact, except: u32) ?u32 {
    var chosen: ?graph.Fact = null;
    for (facts) |f| {
        if (f.id == except or f.pinned or f.state != .learned or f.neurons.len == 0) continue;
        if (chosen == null or f.uses < chosen.?.uses or (f.uses == chosen.?.uses and f.id < chosen.?.id)) chosen = f;
    }
    return if (chosen) |f| f.id else null;
}

/// Finish a pending usage write before any decision can discard a fact based on its count.
pub fn flush(g: *Graph) !void {
    g.mutex.lockUncancelable(g.io);
    defer g.mutex.unlock(g.io);
    if (g.usage_invalid) return error.BadGraph;
    if (g.usage_missing) return error.UsageIncomplete;
    if (g.usage_dirty) try g.persist();
}

/// A graph file that exists but does not load cannot supply trusted eviction counts.
pub fn loadFailed(g: *Graph, e: anyerror) void {
    g.usage_invalid = e != error.FileNotFound;
}

/// A clear leaves no unreadable or unwritten counts behind.
pub fn clear(g: *Graph) void {
    g.usage_invalid = false;
    g.usage_dirty = false;
}

/// Every observed use is in the graph file now, or there is no file to hold it.
pub fn written(g: *Graph) void {
    g.usage_dirty = false;
}

pub fn fields(a: std.mem.Allocator, o: *json.Object, f: graph.Fact) !void {
    if (f.uses > 0 or f.state == .deleted) try o.put(a, "uses", try json.intValue(a, f.uses));
    if (f.pinned) try o.put(a, "pinned", .{ .bool = true });
}

pub fn read(f: *graph.Fact, v: json.Value) !void {
    if (v.field("uses")) |n| {
        if (n != .int) return error.BadUsage;
        f.uses = std.fmt.parseInt(u64, n.int, 10) catch return error.BadUsage;
    }
    if (v.field("pinned")) |b| {
        if (b != .bool) return error.BadPin;
        f.pinned = b.bool;
    }
}

/// Missing firing metadata is persisted, so a restart cannot turn an unknown count into an eviction candidate.
pub fn root(g: *Graph, a: std.mem.Allocator, o: *json.Object) !void {
    if (g.usage_missing) try o.put(a, "usage_complete", .{ .bool = false });
}
pub fn readRoot(g: *Graph, v: json.Value) !void {
    if (v.field("usage_complete")) |x| {
        if (x != .bool) return error.BadUsage;
        g.usage_missing = !x.bool;
    }
}
