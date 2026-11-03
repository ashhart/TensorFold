//! One external HTTP request may generate several replies; each owned fact is charged only once.
const std = @import("std");
const api = @import("engine_api");
const Graph = @import("slide_graph.zig").Graph;
const Server = @import("server.zig").Server;

pub const Scope = struct {
    graph: *Graph,
    gpa: std.mem.Allocator,
    charged: std.ArrayList(u32) = .empty,

    pub fn init(srv: *Server) Scope {
        return .{ .graph = &srv.slide, .gpa = srv.gpa };
    }
    pub fn deinit(s: *Scope) void {
        s.charged.deinit(s.gpa);
    }
    pub fn observer(s: *Scope) api.pocket_usage.Observer {
        return .{ .ctx = s, .done = done };
    }

    fn done(ctx: *anyopaque, result: ?api.pocket_usage.Masks) void {
        const s: *Scope = @ptrCast(@alignCast(ctx));
        s.record(result) catch |e| std.log.warn("slide: usage counts await a graph write ({s})", .{@errorName(e)});
    }
    pub fn record(s: *Scope, result: ?api.pocket_usage.Masks) !void {
        const g = s.graph;
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        if (g.usage_invalid) return error.BadGraph;
        if (result == null or !result.?.complete) {
            g.usage_missing = true;
            g.usage_dirty = true;
            try g.persist();
            return;
        }
        for (g.facts.items) |*f| {
            if (f.state != .learned or std.mem.indexOfScalar(u32, s.charged.items, f.id) != null) continue;
            for (f.neurons) |n| if (result.?.has(n.layer, n.neuron)) {
                s.charged.append(s.gpa, f.id) catch |e| {
                    g.usage_missing = true;
                    g.usage_dirty = true;
                    g.persist() catch {};
                    return e;
                };
                f.uses +|= 1;
                g.usage_dirty = true;
                break;
            };
        }
        if (g.usage_dirty) try g.persist();
    }
};
