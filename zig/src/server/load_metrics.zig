//! The load ledger in the scrape: this server's identity, parked state and activity, and each group a host fills.
const std = @import("std");
const api = @import("engine_api");
const ledger = api.load_ledger;

const prefix = "tensorfold:";

/// What `server_info` names: the served model as /health shows it, the backend and the release.
pub const Identity = struct { model: []const u8, backend: []const u8 = "", version: []const u8 = "" };

/// The ledger's families. The request counts and preemptions stay with metrics.zig, which takes the ledger's values.
pub fn render(w: *std.Io.Writer, s: ledger.Snapshot, id: Identity) !void {
    try family(w, "server_info", "gauge", "This server: its served model, backend, release and process instance; always 1.");
    try w.print("{s}server_info{{model=\"", .{prefix});
    try escaped(w, id.model);
    try w.writeAll("\",backend=\"");
    try escaped(w, id.backend);
    try w.writeAll("\",version=\"");
    try escaped(w, id.version);
    try w.print("\",instance=\"{s}\"}} 1\n", .{s.instance});
    try gauge(w, "process_start_time_seconds", "gauge", "Unix time the server process started.", s.start_s);
    try gauge(w, "engine_parked", "gauge", "1 while the idle unload has closed the engine: no engine value is shown then.", @intFromBool(s.parked));
    try gauge(w, "lanes", "gauge", "Lanes the engine serves at once.", s.lanes);
    try family(w, "http_requests_active", "gauge", "Model requests from the end of their headers to the end of their reply, by route group.");
    for (std.enums.values(ledger.RouteGroup)) |g|
        try w.print("{s}http_requests_active{{route=\"{s}\"}} {d}\n", .{ prefix, @tagName(g), s.active(g) });
    if (s.requests) |r| {
        try gauge(w, "waiting_prompt_tokens", "gauge", "Prompt tokens of the queued requests, before any kept prefix restores part of them.", r.waiting_prompt_tokens);
        if (r.filling_left) |left| try gauge(w, "filling_prompt_tokens_left", "gauge", "Prompt tokens still to prefill in the foreground passes filling now.", left);
        try gauge(w, "requests_started_total", "counter", "Requests the engine accepted into its queue.", r.started);
        try gauge(w, "requests_requeued_total", "counter", "Admitted requests that lost their lane for memory before a first token and queued again.", r.requeued);
        try gauge(w, "background_yields_total", "counter", "Background prompt passes that gave their lane up because foreground work arrived.", r.yields);
    }
    if (s.work) |k| {
        try family(w, "scheduler_tokens_total", "counter", "Tokens the engine computed, restored or committed, by mode, as the work happened.");
        try w.print("{s}scheduler_tokens_total{{mode=\"prefill_compute\"}} {d}\n", .{ prefix, k.prefill_compute });
        try w.print("{s}scheduler_tokens_total{{mode=\"prefill_cache\"}} {d}\n", .{ prefix, k.prefill_cache });
        try w.print("{s}scheduler_tokens_total{{mode=\"decode\"}} {d}\n", .{ prefix, k.decode });
        try family(w, "prefill_busy_seconds_total", "counter", "Engine seconds in prompt calls whose work was counted in prefill_compute.");
        try w.print("{s}prefill_busy_seconds_total {d}\n", .{ prefix, @as(f64, @floatFromInt(k.prefill_busy_ns)) / 1e9 });
    }
    if (s.capacity) |c| {
        try family(w, "kv_capacity_mode", "gauge", "How the engine holds KV state: a window a lane, one page pool, or both; always 1.");
        try w.print("{s}kv_capacity_mode{{mode=\"{s}\"}} 1\n", .{ prefix, @tagName(c.mode) });
        try gauge(w, "lanes_free", "gauge", "Lanes with no admitted request.", c.lanes_free);
        if (s.pool) |p| try pool(w, p);
    }
}

fn pool(w: *std.Io.Writer, p: ledger.Pool) !void {
    const rows = [_]struct { []const u8, []const u8, u64 }{
        .{ "kv_pool_capacity_tokens", "Tokens the page pool holds, at page granularity.", p.capacity },
        .{ "kv_pool_available_tokens", "Tokens of free pages not promised to a stream.", p.available },
        .{ "kv_pool_reclaimable_tokens", "Tokens of pages only the prompt store holds, which eviction can free.", p.reclaimable },
        .{ "kv_pool_promised_tokens", "Tokens of pages reserved for a stream and not yet taken.", p.promised },
    };
    for (rows) |row| {
        try family(w, row[0], "gauge", row[1]);
        try w.print("{s}{s}{{pool=\"main\"}} {d}\n", .{ prefix, row[0], row[2] });
    }
    if (p.run) |free| {
        try family(w, "kv_pool_run_available", "gauge", "1 when the family's required contiguous run of pages is free.");
        try w.print("{s}kv_pool_run_available{{pool=\"main\"}} {d}\n", .{ prefix, @intFromBool(free) });
    }
}

fn family(w: *std.Io.Writer, name: []const u8, kind: []const u8, help: []const u8) !void {
    try w.print("# HELP {s}{s} {s}\n# TYPE {s}{s} {s}\n", .{ prefix, name, help, prefix, name, kind });
}

fn gauge(w: *std.Io.Writer, name: []const u8, kind: []const u8, help: []const u8, value: anytype) !void {
    try family(w, name, kind, help);
    try w.print("{s}{s} {d}\n", .{ prefix, name, value });
}

/// A Prometheus label value: backslash, double quote and newline escaped.
fn escaped(w: *std.Io.Writer, text: []const u8) !void {
    for (text) |c| switch (c) {
        '\\' => try w.writeAll("\\\\"),
        '"' => try w.writeAll("\\\""),
        '\n' => try w.writeAll("\\n"),
        else => try w.writeByte(c),
    };
}

test {
    _ = @import("load_metrics_test.zig");
}
