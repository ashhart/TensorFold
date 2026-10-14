//! GET and DELETE /v1/recall: Recall's record (not the prompt cache's learned states), its state and its emptying.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const log = @import("log.zig");
const routes = @import("routes.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Store = api.recall_store.Store;
const Allocator = std.mem.Allocator;

pub const path = "/v1/recall";

/// What the record holds and how its copies land.
pub fn get(srv: *Server, conn: *Conn, a: Allocator) void {
    const s = srv.recall orelse return off(conn, a);
    routes.sendValue(conn, a, 200, value(a, s) catch return);
}

/// Every id gone from the record and its files; a read-only server refuses.
pub fn empty(srv: *Server, conn: *Conn, a: Allocator) void {
    const s = srv.recall orelse return off(conn, a);
    s.empty() catch |e| return switch (e) {
        error.ReadOnly => refuse(conn, a, 409, read_only),
        error.Unlinked => refuse(conn, a, 500, std.fmt.allocPrint(a, unlinked, .{s.why()}) catch return),
    };
    log.line("recall: emptied {s}", .{s.dir});
    conn.sendJson(200, "{\"emptied\": true}");
}

const read_only = "this server reads the record only (--recall-read-only, or another server writes it): " ++
    "empty it from the server that writes it";
const unlinked = "the record was not emptied on disk: {s}. Its memory is empty, and the server adds nothing to it " ++
    "until a DELETE removes every chunk file";

fn off(conn: *Conn, a: Allocator) void {
    refuse(conn, a, 404, "Recall is off: serve with --recall");
}

fn refuse(conn: *Conn, a: Allocator, code: u16, message: []const u8) void {
    const quoted = json.quote(a, message, .{}) catch return;
    const body = std.fmt.allocPrint(a, "{{\"error\": {{\"message\": {s}}}}}", .{quoted}) catch return;
    conn.sendJson(code, body);
}

/// The record's state in a word: on, read-only, or one of the reasons it stopped adding.
pub fn state(s: *const Store, st: api.recall_store.Stats) []const u8 {
    if (s.read_only) return "read-only";
    if (!s.writes) return "read-only: another server writes this record";
    if (st.left) return "reading: an empty left a chunk file, nothing more is added";
    return if (st.writes) "on" else "reading: a write failed, nothing more is added";
}

fn value(a: Allocator, s: *Store) !json.Value {
    const st = s.stats();
    const o = try json.newObject(a);
    try o.put(a, "state", .{ .string = state(s, st) });
    try o.put(a, "directory", .{ .string = s.dir });
    const short = s.identity.short();
    try o.put(a, "tokenizer", .{ .string = try a.dupe(u8, &short) });
    try o.put(a, "vocab", try json.intValue(a, s.identity.vocab));
    try o.put(a, "first_written_by", .{ .string = s.name });
    const sizes = .{
        .{ "ids", st.ids },
        .{ "disk_bytes", st.disk_bytes },
        .{ "ram_bytes", st.ram_bytes },
        .{ "cap_bytes", st.cap_bytes },
        .{ "added_ids", st.added },
        .{ "already_held_ids", st.held },
    };
    inline for (sizes) |f| try o.put(a, f[0], try json.intValue(a, f[1]));
    const copies = try json.newObject(a);
    const counts = .{ .{ "proposals", st.proposals }, .{ "proposed", st.proposed }, .{ "accepted", st.accepted } };
    inline for (counts) |f| try copies.put(a, f[0], try json.intValue(a, f[1]));
    try o.put(a, "copies", .{ .object = copies });
    const bands = try a.alloc(json.Value, st.rates.len);
    for (bands, st.rates, api.recall_store.bands) |*b, rate, from| {
        const x = try json.newObject(a);
        try x.put(a, "matched_from", try json.intValue(a, from));
        try x.put(a, "landing_rate", .{ .float = rate });
        b.* = .{ .object = x };
    }
    try o.put(a, "bands", .{ .array = bands });
    try o.put(a, "probe_every", try json.intValue(a, st.probe_every));
    return .{ .object = o };
}

/// The startup line: the record's state, size and place.
pub fn line(s: *Store) void {
    const st = s.stats();
    const short = s.identity.short();
    const fmt = "recall: {s}, {d} ids ({d} MiB on disk of {d} MiB, {d} MiB in RAM) in {s}, tokenizer {s}";
    const mib = .{ st.disk_bytes >> 20, st.cap_bytes >> 20, st.ram_bytes >> 20 };
    log.line(fmt, .{ state(s, st), st.ids, mib[0], mib[1], mib[2], s.dir, &short });
}

test "the record's state and size as GET reports them, and its state in a word" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/recall", .{tmp.sub_path});
    var problem: []const u8 = "";
    const id: api.recall_store.Identity = .{ .digest = @splat(3), .vocab = 1000, .name = "m" };
    const o: api.recall_store.Options = .{ .dir = dir, .cap_bytes = 64 << 20 };
    const s = (try Store.open(std.testing.allocator, std.testing.io, id, o, &problem, a)).?;
    defer s.close();
    var prompt: [300]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast((i * 37) % 1000);
    s.readPrompt(&prompt, 0);
    const v = try value(a, s);
    try std.testing.expectEqualStrings("on", v.get("state").?.string);
    try std.testing.expectEqual(@as(i64, 301), v.get("ids").?.int64().?);
    try std.testing.expectEqual(@as(usize, 3), v.get("bands").?.array.len);
    try s.empty();
    try std.testing.expectEqual(@as(i64, 0), (try value(a, s)).get("ids").?.int64().?);
    const ro: api.recall_store.Options = .{ .dir = dir, .cap_bytes = 64 << 20, .read_only = true };
    const r = (try Store.open(std.testing.allocator, std.testing.io, id, ro, &problem, a)).?;
    defer r.close();
    try std.testing.expectEqualStrings("read-only", state(r, r.stats()));
}
