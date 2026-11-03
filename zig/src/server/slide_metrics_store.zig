//! Owned, persistent fixed-panel baselines, supplied recall sets and lesson receipts.
const std = @import("std");
const json = @import("json");
const Graph = @import("slide_graph.zig").Graph;
const answers = @import("slide_answers.zig");
const Allocator = std.mem.Allocator;

pub fn value(a: Allocator, data: anytype) !json.Value {
    const bytes = try std.json.Stringify.valueAlloc(a, data, .{});
    return (try json.parse(a, bytes)).ok;
}

pub fn copy(a: Allocator, v: json.Value) !json.Value {
    return (try json.parse(a, try json.stringify(a, v, .{}))).ok;
}

pub fn decode(comptime T: type, a: Allocator, v: json.Value) !T {
    return std.json.parseFromSliceLeaky(T, a, try json.stringify(a, v, .{}), .{});
}

pub fn supplied(g: *Graph, id: u32, xs: []const answers.Answer) !void {
    g.mutex.lockUncancelable(g.io);
    defer g.mutex.unlock(g.io);
    const f = g.find(id) orelse return error.NoSuchFact;
    const old = f.recall;
    f.recall = try answers.dupe(g.arena.allocator(), xs);
    errdefer f.recall = old;
    try g.persist();
}

pub fn record(g: *Graph, id: u32, data: anytype) !void {
    g.mutex.lockUncancelable(g.io);
    defer g.mutex.unlock(g.io);
    const f = g.find(id) orelse return error.NoSuchFact;
    const old = f.evaluation;
    f.evaluation = try value(g.arena.allocator(), data);
    errdefer f.evaluation = old;
    try g.persist();
}

pub fn baseline(g: *Graph, a: Allocator, data: ?json.Value) !?json.Value {
    g.mutex.lockUncancelable(g.io);
    defer g.mutex.unlock(g.io);
    if (g.drift_base == null) if (data) |v| {
        g.drift_base = try copy(g.arena.allocator(), v);
        errdefer g.drift_base = null;
        try g.persist();
    };
    return if (g.drift_base) |v| try copy(a, v) else null;
}

/// Earlier failures have no measured candidate; do not overwrite a measured rejection with its rolled-back state.
pub fn unmeasured(g: *Graph, id: u32, reason: []const u8) !void {
    g.mutex.lockUncancelable(g.io);
    defer g.mutex.unlock(g.io);
    const f = g.find(id) orelse return error.NoSuchFact;
    if (f.evaluation != null) return;
    f.evaluation = try value(g.arena.allocator(), .{ .status = "not_measured", .reason = reason });
    errdefer f.evaluation = null;
    try g.persist();
}
