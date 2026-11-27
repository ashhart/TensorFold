//! Inline document payloads are redacted in request JSONL and journal start records.
const std = @import("std");
const builtin = @import("builtin");
const json = @import("json");
const request_log = @import("request_log.zig");
const journal_mod = @import("request_journal.zig");
const Trace = @import("request_trace.zig").Trace;
const Value = json.Value;
const t = std.testing;

const Payload = struct { name: []const u8, value: []const u8 };
const payloads = [_]Payload{
    .{ .name = "file_data", .value = "JVBERi1zZWNyZXQ=" },
    .{ .name = "data", .value = "JVBERi1kYXRh" },
    .{ .name = "url", .value = "data:application/pdf;base64,JVBERi11cmw=" },
    .{ .name = "file_url", .value = "https://invalid.test/private.pdf" },
};

fn logPath(a: std.mem.Allocator, tmp: anytype, name: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, name });
}

fn records(a: std.mem.Allocator, path: []const u8) ![]Value {
    const text = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 24));
    var out: std.ArrayList(Value) = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const parsed = try json.parse(a, line);
        if (parsed != .ok) return error.BadRecord;
        try out.append(a, parsed.ok);
    }
    return out.items;
}

fn recordFor(records_slice: []const Value, kind: []const u8, id: []const u8) !Value {
    for (records_slice) |r| {
        const actual = r.strField("type") orelse r.strField("kind") orelse "";
        if (std.mem.eql(u8, actual, kind) and std.mem.eql(u8, r.strField("id").?, id)) return r;
    }
    return error.MissingRecord;
}

fn checkPart(a: std.mem.Allocator, record: Value, payload: Payload, nested: bool) !void {
    const body = record.get("body").?;
    const part = body.get("messages").?.array[0].get("content").?.array[0];
    const container = if (nested) part.get("file").? else part;
    try t.expectEqualStrings("<redacted>", container.strField(payload.name).?);
    try t.expectEqualStrings("a.pdf", container.strField("filename").?);
    try t.expectEqualStrings("keep", container.strField("metadata").?);
    try t.expectEqualStrings("keep", part.strField("note").?);
    const encoded = try json.stringify(a, record, .{ .compact = true });
    try t.expect(std.mem.indexOf(u8, encoded, payload.value) == null);
}

test "request log: every accepted inline PDF alias and payload field is redacted in both logs" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // owner-only log files are refused on Windows
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const request_path = try logPath(a, tmp, "requests.jsonl");
    const journal_path = try logPath(a, tmp, "journal.jsonl");
    const journal = try journal_mod.Journal.open(t.allocator, t.io, journal_path);
    defer journal.close();
    var index: usize = 0;
    for ([_][]const u8{ "file", "input_file" }) |alias| {
        for ([_]bool{ false, true }) |nested| {
            for (payloads) |payload| {
                const id = try std.fmt.allocPrint(a, "document-{d}", .{index});
                index += 1;
                const part = try json.newObject(a);
                try part.put(a, "type", .{ .string = alias });
                try part.put(a, "note", .{ .string = "keep" });
                const container = try json.newObject(a);
                try container.put(a, "filename", .{ .string = "a.pdf" });
                try container.put(a, "metadata", .{ .string = "keep" });
                try container.put(a, payload.name, .{ .string = payload.value });
                if (nested) try part.put(a, "file", .{ .object = container });
                if (!nested) for (container.keys(), container.values()) |key, value| try part.put(a, key, value);
                const message = try json.newObject(a);
                try message.put(a, "role", .{ .string = "user" });
                try message.put(a, "content", .{ .array = try a.dupe(Value, &.{.{ .object = part }}) });
                const body = try json.newObject(a);
                try body.put(a, "messages", .{ .array = try a.dupe(Value, &.{.{ .object = message }}) });
                const value: Value = .{ .object = body };
                request_log.appendRequest(a, request_path, id, value, request_log.file_limit);
                var trace: Trace = .{ .journal = journal, .a = a, .id = id, .parent = undefined };
                try t.expect(!trace.begin(value, true, "test-key"));
            }
        }
    }
    const request_records = try records(a, request_path);
    const journal_records = try records(a, journal_path);
    for (0..16) |n| {
        const id = try std.fmt.allocPrint(a, "document-{d}", .{n});
        const payload = payloads[n % payloads.len];
        const nested = (n / payloads.len) % 2 == 1;
        const request = try recordFor(request_records, "request", id);
        const journal_record = try recordFor(journal_records, "start", id);
        try checkPart(a, request, payload, nested);
        try checkPart(a, journal_record, payload, nested);
    }
}

test "request log: an ordinary non-file body is unchanged by redaction" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = (try json.parse(a, "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"keep\"}]}],\"data\":\"ordinary\"}")).ok;
    const before = try json.stringify(a, raw, .{ .compact = true });
    const cleaned = try request_log.redact(a, raw);
    const after = try json.stringify(a, cleaned, .{ .compact = true });
    try t.expectEqualStrings(before, after);
}

fn saltLogCase(nested: bool) !void {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // owner-only log files are refused on Windows
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const request_path = try logPath(a, tmp, "requests.jsonl");
    const journal_path = try logPath(a, tmp, "journal.jsonl");
    const journal = try journal_mod.Journal.open(t.allocator, t.io, journal_path);
    defer journal.close();
    const part = try json.newObject(a);
    try part.put(a, "type", .{ .string = if (nested) "file" else "input_file" });
    try part.put(a, "note", .{ .string = "keep" });
    const container = try json.newObject(a);
    try container.put(a, "filename", .{ .string = "a.pdf" });
    try container.put(a, "metadata", .{ .string = "keep" });
    try container.put(a, "file_data", .{ .string = payloads[0].value });
    try container.put(a, "cache_salt", .{ .string = "tenant-secret" });
    if (nested) try part.put(a, "file", .{ .object = container });
    if (!nested) for (container.keys(), container.values()) |key, value| try part.put(a, key, value);
    const message = try json.newObject(a);
    try message.put(a, "role", .{ .string = "user" });
    try message.put(a, "content", .{ .array = try a.dupe(Value, &.{.{ .object = part }}) });
    const body = try json.newObject(a);
    try body.put(a, "messages", .{ .array = try a.dupe(Value, &.{.{ .object = message }}) });
    const value: Value = .{ .object = body };
    const id = if (nested) "nested-salt" else "top-level-salt";
    request_log.appendRequest(a, request_path, id, value, request_log.file_limit);
    var trace: Trace = .{ .journal = journal, .a = a, .id = id, .parent = undefined };
    try t.expect(!trace.begin(value, true, "test-key"));
    const request_records = try records(a, request_path);
    const journal_records = try records(a, journal_path);
    const request = try recordFor(request_records, "request", id);
    const journal_record = try recordFor(journal_records, "start", id);
    for ([_]Value{ request, journal_record }) |record| {
        try checkPart(a, record, payloads[0], nested);
        const record_part = record.get("body").?.get("messages").?.array[0].get("content").?.array[0];
        const record_container = if (nested) record_part.get("file").? else record_part;
        try t.expectEqualStrings("<redacted>", record_container.strField("cache_salt").?);
        const encoded = try json.stringify(a, record, .{ .compact = true });
        try t.expect(std.mem.indexOf(u8, encoded, "tenant-secret") == null);
    }
}

test "request log: a top-level input_file cache salt is redacted in both logs" {
    try saltLogCase(false);
}

test "request log: a nested file cache salt is redacted in both logs" {
    try saltLogCase(true);
}
