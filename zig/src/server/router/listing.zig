//! The router's own routes: /health, /v1/models with ``loaded``, and Ollama's /api/tags, /api/ps and /api/version.
const std = @import("std");
const json = @import("json");
const stream = @import("../benchmark_stream.zig");
const routes = @import("../routes.zig");
const pool_mod = @import("pool.zig");
const front = @import("front.zig");
const Conn = @import("../http_conn.zig").Conn;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// With keys on, health names no model, as a server's does; draining is a 503 that says so.
pub fn health(f: *front.Front, conn: *Conn, a: Allocator) void {
    if (f.draining.load(.acquire)) {
        const body = std.fmt.allocPrint(a, "{{\"status\": \"draining\", \"requests_in_progress\": {d}}}", .{f.pool.inFlight()}) catch return;
        return routes.sendDraining(conn, body);
    }
    if (f.keys) |k| if (k.enabled()) return conn.sendJson(200, "{\"status\": \"ok\"}");
    send(conn, a, openHealth(f, a) catch return);
}

fn openHealth(f: *front.Front, a: Allocator) !Value {
    const o = try json.newObject(a);
    try o.put(a, "status", .{ .string = "ok" });
    try o.put(a, "router", .{ .bool = true });
    try o.put(a, "models", try json.intValue(a, f.entries.len));
    try o.put(a, "max_loaded", try json.intValue(a, f.pool.options.max_loaded));
    const now = try f.pool.loaded(a);
    const list = try a.alloc(Value, now.len);
    for (now, list) |s, *slot| {
        const child = try json.newObject(a);
        try child.put(a, "id", .{ .string = f.entries[s.slot].id });
        try child.put(a, "state", .{ .string = @tagName(s.state) });
        try child.put(a, "requests_in_progress", try json.intValue(a, s.in_flight));
        slot.* = .{ .object = child };
    }
    try o.put(a, "loaded", .{ .array = list });
    if (f.version.len > 0) try o.put(a, "version", .{ .string = f.version });
    return .{ .object = o };
}

/// Every id and alias, as a server lists its own: a ready child's entries as it gives them, the rest by id alone.
pub fn models(f: *front.Front, conn: *Conn, a: Allocator) void {
    const now = f.pool.loaded(a) catch return;
    const created = std.Io.Clock.real.now(f.io).toSeconds();
    var data: std.ArrayList(Value) = .empty;
    for (f.entries, 0..) |e, i| {
        const ready = for (now) |s| {
            if (s.slot == i and s.state == .ready) break s;
        } else null;
        const listed = if (ready) |s| childModels(a, f.io, s.port, &s.key) else null;
        for ([_][]const []const u8{ &.{e.id}, e.aliases }) |names| for (names) |name| {
            const o = entry(a, listed, name, created, ready != null) catch return;
            data.append(a, .{ .object = o }) catch return;
        };
    }
    const o = json.newObject(a) catch return;
    o.put(a, "object", .{ .string = "list" }) catch return;
    o.put(a, "data", .{ .array = data.items }) catch return;
    send(conn, a, .{ .object = o });
}

/// The child's own entry for ``name`` when it gave one, else the OpenAI fields alone; ``loaded`` comes last.
fn entry(a: Allocator, listed: ?Value, name: []const u8, created: i64, loaded: bool) !*json.Object {
    const o = if (own(listed, name)) |x| try json.copyObject(a, x) else blk: {
        const m = try json.newObject(a);
        try m.put(a, "id", .{ .string = name });
        try m.put(a, "object", .{ .string = "model" });
        try m.put(a, "created", try json.intValue(a, created));
        try m.put(a, "owned_by", .{ .string = "tensorfold" });
        break :blk m;
    };
    try o.put(a, "loaded", .{ .bool = loaded });
    return o;
}

/// The object a child's /v1/models gives for ``name``.
fn own(listed: ?Value, name: []const u8) ?*json.Object {
    const data = (listed orelse return null).get("data") orelse return null;
    if (data != .array) return null;
    for (data.array) |x| if (x == .object and std.mem.eql(u8, x.strField("id") orelse "", name)) return x.object;
    return null;
}

/// A ready child's /v1/models, asked with its key; null when it does not answer in time.
fn childModels(a: Allocator, io: std.Io, port: u16, key: []const u8) ?Value {
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = io };
    client.connection_pool.free_size = 0;
    defer client.deinit();
    const url = std.fmt.allocPrint(a, "http://127.0.0.1:{d}/v1/models", .{port}) catch return null;
    const got = stream.fetch(a, io, &client, .{ .url = url, .limit = 1 << 20, .timeout_s = 2, .token = key });
    if (got.failed != .none or got.status != 200) return null;
    return switch (json.parse(a, got.body) catch return null) {
        .ok => |v| v,
        .err => null,
    };
}

/// Ollama's /api/tags: every id and alias, in the server's shape.
pub fn tags(f: *front.Front, conn: *Conn, a: Allocator) void {
    var list: std.ArrayList(Value) = .empty;
    for (f.entries) |e| for ([_][]const []const u8{ &.{e.id}, e.aliases }) |names| for (names) |name| {
        list.append(a, named(a, name) catch return) catch return;
    };
    sendModels(conn, a, list.items);
}

/// Ollama's /api/ps: the ready children, each by its id.
pub fn ps(f: *front.Front, conn: *Conn, a: Allocator) void {
    const now = f.pool.loaded(a) catch return;
    var list: std.ArrayList(Value) = .empty;
    for (now) |s| if (s.state == .ready) list.append(a, named(a, f.entries[s.slot].id) catch return) catch return;
    sendModels(conn, a, list.items);
}

/// Ollama's /api/version: the router's build.
pub fn version(f: *front.Front, conn: *Conn, a: Allocator) void {
    const o = json.newObject(a) catch return;
    o.put(a, "version", .{ .string = f.version }) catch return;
    send(conn, a, .{ .object = o });
}

fn named(a: Allocator, name: []const u8) !Value {
    const m = try json.newObject(a);
    try m.put(a, "name", .{ .string = name });
    try m.put(a, "model", .{ .string = name });
    return .{ .object = m };
}

fn sendModels(conn: *Conn, a: Allocator, list: []Value) void {
    const o = json.newObject(a) catch return;
    o.put(a, "models", .{ .array = list }) catch return;
    send(conn, a, .{ .object = o });
}

fn send(conn: *Conn, a: Allocator, v: Value) void {
    conn.sendJson(200, json.stringify(a, v, .{}) catch return);
}
