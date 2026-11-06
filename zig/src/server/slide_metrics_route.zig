//! Gate-only request parsing and receipts around the existing lesson route.
const std = @import("std");
const json = @import("json");
const input = @import("slide_recall_input.zig");
const slide = @import("slide.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

pub fn read(srv: *Server, conn: *Conn, a: Allocator, body: json.Value) ?[]const input.Item {
    if (!srv.config.slide_drift_gate) return &.{};
    return input.read(a, body) catch {
        slide.refuse(conn, a, 400, input.message);
        return null;
    };
}

pub fn prepare(srv: *Server, out: *slide.Out, a: Allocator, id: u32, fact: []const u8, recall: []const input.Item) !bool {
    if (!srv.config.slide_drift_gate) return true;
    const external = try input.select(a, recall, fact);
    srv.slide.suppliedRecall(id, external) catch {
        slide.missed(srv, out, id, "the supplied recall set could not be saved");
        return false;
    };
    if (external.len != 0) return true;
    slide.missed(srv, out, id, "--slide-drift-gate requires supplied recall wordings for this exact fact");
    return false;
}

pub fn unmeasured(srv: *Server, id: u32, why: []const u8) void {
    if (srv.config.slide_drift_gate) @import("slide_metrics_store.zig").unmeasured(&srv.slide, id, why) catch {};
}
