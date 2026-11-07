//! Explicit read-only graph audit, serialized beside lessons and queued on the engine's owner thread.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const slide = @import("slide.zig");
const planning = @import("slide_audit_probes.zig");
const report = @import("slide_audit_report.zig");
const A = std.mem.Allocator;

pub const Options = struct { threshold: f64 = report.default_threshold };

/// Only audit=1 requests work; malformed values and duplicate audit options are refused.
pub fn options(path: []const u8) !?Options {
    const q = std.mem.indexOfScalar(u8, path, '?') orelse return null;
    var it = std.mem.splitScalar(u8, path[q + 1 ..], '&');
    var audit = false;
    var out: Options = .{};
    var threshold = false;
    while (it.next()) |part| {
        if (std.mem.startsWith(u8, part, "audit=")) {
            if (audit or !std.mem.eql(u8, part, "audit=1")) return error.BadAuditOptions;
            audit = true;
        } else if (std.mem.startsWith(u8, part, "threshold=")) {
            if (threshold) return error.BadAuditOptions;
            threshold = true;
            out.threshold = std.fmt.parseFloat(f64, part[10..]) catch return error.BadAuditOptions;
            if (!std.math.isFinite(out.threshold) or out.threshold < 0 or out.threshold > 1) return error.BadAuditOptions;
        }
    }
    if (threshold and !audit) return error.BadAuditOptions;
    return if (audit) out else null;
}

pub fn serve(srv: *Server, conn: *Conn, a: A, opts: Options) void {
    const body = run(srv, a, opts) catch |e| return slide.refuse(conn, a, 400, @errorName(e));
    conn.sendJson(200, body);
}

pub fn run(srv: *Server, a: A, opts: Options) ![]const u8 {
    srv.teacher.learning.lockUncancelable(srv.io);
    defer srv.teacher.learning.unlock(srv.io);
    // Probe support before touching the graph, rendering a prompt or creating the trainer.
    switch (slide.run(srv, a, &.{ .audit = &.{ .probe_only = true, .neurons = &.{}, .probes = &.{}, .peaks = &.{} } }, 0)) {
        .none => {},
        else => return error.AuditUnsupported,
    }
    const dir: ?[]const u8 = if (srv.config.model_dir.len == 0) null else srv.config.model_dir;
    const view = try srv.slide.auditView(a, dir);
    const plan = try planning.plan(a, view.facts);
    var ns: std.ArrayList(api.Neuron) = .empty;
    for (plan.facts) |f| try ns.appendSlice(a, f.source.neurons);
    if (ns.items.len > 1024) return error.AuditTooLarge;
    const ps = try a.alloc(api.pocket_audit.Probe, plan.probes.len);
    var cx: @import("errors.zig").Cx = .{ .a = a };
    for (plan.probes, ps) |p, *out| {
        const ex = try @import("slide_ask.zig").promptOnly(srv, &cx, a, null, p.text);
        out.* = .{ .ids = ex.ids };
    }
    const peaks = try a.alloc(api.pocket_audit.Peak, ns.items.len * ps.len);
    if (ns.items.len > 0) {
        var completed = false;
        const request: api.pocket_audit.Request = .{ .neurons = ns.items, .probes = ps, .peaks = peaks, .completed = &completed };
        switch (slide.run(srv, a, &.{ .audit = &request }, 0)) {
            .none => {},
            .failed => |m| return errorFor(m),
            else => return error.AuditUnavailable,
        }
        if (!completed) return error.AuditUnavailable;
    }
    const root = switch (try json.parse(a, view.document)) {
        .ok => |v| v,
        .err => return error.BadGraph,
    };
    try root.object.put(a, "audit", try report.value(a, plan, ns.items, peaks, opts.threshold));
    return json.stringify(a, root, .{ .compact = true });
}

fn errorFor(message: []const u8) anyerror {
    inline for (.{ error.AuditPromptTooLong, error.AuditMissingNeuron, error.AuditNonfinite, error.AuditUnresolved }) |e|
        if (std.mem.eql(u8, message, @errorName(e))) return e;
    return error.AuditUnavailable;
}

test "audit request: explicit activation, finite threshold and duplicate rejection" {
    try std.testing.expect((try options("/v1/slide/graph")) == null);
    try std.testing.expectEqual(@as(f64, 0.01), (try options("/v1/slide/graph?audit=1")).?.threshold);
    try std.testing.expectEqual(@as(f64, 0.2), (try options("/v1/slide/graph?audit=1&threshold=0.2")).?.threshold);
    for ([_][]const u8{ "/?audit=0", "/?audit=1&audit=1", "/?threshold=0.1", "/?audit=1&threshold=nan", "/?audit=1&threshold=2" }) |q|
        try std.testing.expectError(error.BadAuditOptions, options(q));
}
