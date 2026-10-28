//! The load ledger in the scrape: this server's identity, parked state and activity, and each group a host fills.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const ledger = api.load_ledger;
const Allocator = std.mem.Allocator;
const Value = json.Value;

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

/// GET /v1/load: the whole ledger as one object from one read; a group no host fills leaves its keys out.
pub fn get(srv: anytype, conn: anytype, a: Allocator) void {
    const snap = srv.load.read();
    const now_s = @as(f64, @floatFromInt(std.Io.Clock.real.now(srv.io).toNanoseconds())) / 1e9;
    const body = object(a, snap, now_s, srv.config.served_name) catch return;
    conn.quiet_log = true;
    @import("routes.zig").sendValue(conn, a, 200, body);
}

/// The `/v1/load` object; `now_s` is the read's wall time.
pub fn object(a: Allocator, s: ledger.Snapshot, now_s: f64, model: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "instance", .{ .string = try a.dupe(u8, &s.instance) });
    try o.put(a, "timestamp", .{ .float = now_s });
    try o.put(a, "model", .{ .string = model });
    try o.put(a, "parked", .{ .bool = s.parked });
    try o.put(a, "lanes", try json.intValue(a, s.lanes));
    const http = try json.newObject(a);
    for (std.enums.values(ledger.RouteGroup)) |g| try http.put(a, @tagName(g), try json.intValue(a, s.active(g)));
    try o.put(a, "http_active", .{ .object = http });
    if (s.requests) |r| {
        try o.put(a, "running", try json.intValue(a, r.running));
        try o.put(a, "waiting", try json.intValue(a, r.waiting));
        try o.put(a, "requests_started", try json.intValue(a, r.started));
        try o.put(a, "requests_requeued", try json.intValue(a, r.requeued));
        try o.put(a, "background_yields", try json.intValue(a, r.yields));
        try o.put(a, "waiting_prompt_tokens", try json.intValue(a, r.waiting_prompt_tokens));
        if (r.filling_left) |left| try o.put(a, "filling_prompt_tokens_left", try json.intValue(a, left));
        try o.put(a, "fill_unknown", .{ .bool = r.fill_unknown });
    }
    if (s.capacity) |c| {
        const cap = try json.newObject(a);
        try cap.put(a, "mode", .{ .string = @tagName(c.mode) });
        try cap.put(a, "lanes_free", try json.intValue(a, c.lanes_free));
        if (s.pool) |p| {
            const entry = try json.newObject(a);
            try entry.put(a, "name", .{ .string = "main" });
            try entry.put(a, "capacity_tokens", try json.intValue(a, p.capacity));
            try entry.put(a, "available_tokens", try json.intValue(a, p.available));
            try entry.put(a, "reclaimable_tokens", try json.intValue(a, p.reclaimable));
            try entry.put(a, "promised_tokens", try json.intValue(a, p.promised));
            if (p.run) |free| try entry.put(a, "run_available", .{ .bool = free });
            const pools = try a.alloc(Value, 1);
            pools[0] = .{ .object = entry };
            try cap.put(a, "pools", .{ .array = pools });
        }
        try o.put(a, "capacity", .{ .object = cap });
    }
    if (s.work) |k| {
        const tokens = try json.newObject(a);
        try tokens.put(a, "prefill_compute", try json.intValue(a, k.prefill_compute));
        try tokens.put(a, "prefill_cache", try json.intValue(a, k.prefill_cache));
        try tokens.put(a, "decode", try json.intValue(a, k.decode));
        try o.put(a, "tokens", .{ .object = tokens });
        const busy = try json.newObject(a);
        try busy.put(a, "prefill", try json.intValue(a, k.prefill_busy_ns / 1000));
        try o.put(a, "busy_us", .{ .object = busy });
    }
    return .{ .object = o };
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
