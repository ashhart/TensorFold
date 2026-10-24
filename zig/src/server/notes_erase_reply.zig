//! Allocate the erase reply before committing; its bounded diagnostics need no response allocations afterwards.
const std = @import("std");
const json = @import("json");
const places = @import("notes_erase_places.zig");
const erase = @import("notes/erase_storage.zig");
const Server = @import("server.zig").Server;

pub const Reply = struct {
    prefix: []const u8,
    buffer: []u8,

    pub fn prepare(a: std.mem.Allocator, id: u32, e: *const erase.Prepared) !Reply {
        const o = try json.newObject(a);
        try o.put(a, "erased", try json.intValue(a, id));
        const removed = try a.alloc(json.Value, e.removed.len);
        for (removed, e.removed) |*slot, source| slot.* = .{ .string = source };
        try o.put(a, "removed", .{ .array = removed });
        const moved = try a.alloc(json.Value, e.moved.len);
        for (moved, e.moved) |*slot, pair| {
            const row = try json.newObject(a);
            try row.put(a, "from", try json.intValue(a, pair[0]));
            try row.put(a, "to", try json.intValue(a, pair[1]));
            slot.* = .{ .object = row };
        }
        try o.put(a, "renumbered", .{ .array = moved });
        const encoded = try json.stringify(a, .{ .object = o }, .{});
        return .{ .prefix = encoded[0 .. encoded.len - 1], .buffer = try a.alloc(u8, encoded.len + 16384) };
    }

    pub fn finish(r: Reply, srv: *Server, a: std.mem.Allocator, e: *erase.Prepared) ![]const u8 {
        defer srv.text.forget();
        const forgotten = places.forgetSpans(srv, e.intent.spans, e.intent.short);
        var checked = places.check(srv, forgotten, .{
            .this_process = e.source_num >= e.storage.start_source,
            .source_num = e.source_num,
        });
        if (e.unsynced) checked.set(.journal, .may_hold);
        var retryable = e.unsynced or places.needsRetry(forgotten, checked);
        if (retryable) checked.set(.journal, .may_hold) else e.finish(a) catch {
            e.retire_failed = true;
            retryable = true;
            checked.set(.journal, .may_hold);
        };
        var b: Buffer = .{ .bytes = r.buffer };
        try b.put("{s},\"places\":{{", .{r.prefix});
        var pending: usize = 0;
        for (std.enums.values(places.Place), 0..) |p, i| {
            if (i > 0) try b.put(",", .{});
            try b.put("\"{s}\":\"{s}\"", .{ @tagName(p), @tagName(checked.get(p)) });
            pending += @intFromBool(checked.get(p) == .may_hold);
        }
        try b.put("}},\"cleanup_complete\":{s},\"retryable\":{s},\"complete\":{s},\"state\":\"{s}\",\"remaining_places\":[", .{
            if (retryable) "false" else "true",
            if (retryable) "true" else "false",
            if (pending == 0) "true" else "false",
            if (pending == 0) "complete" else if (retryable) "pending" else "unresolved",
        });
        var first = true;
        for (std.enums.values(places.Place)) |p| if (checked.get(p) == .may_hold) {
            if (!first) try b.put(",", .{});
            first = false;
            try b.put("\"{s}\"", .{@tagName(p)});
        };
        try b.put("],\"remaining\":[", .{});
        first = true;
        for (std.enums.values(places.Place)) |p| if (checked.get(p) == .may_hold) {
            if (!first) try b.put(",", .{});
            first = false;
            try b.put("\"{s}\"", .{p.why()});
        };
        const scope = if (srv.config.learn and (forgotten.out.disk or forgotten.out.conversation_disk))
            "current_configured_root_only"
        else
            "none";
        try b.put("],\"learned_folder_scope\":\"{s}\",\"learned_folders\":[", .{scope});
        for (forgotten.out.folders.items[0..forgotten.out.folders.len], 0..) |item, i| {
            if (i > 0) try b.put(",", .{});
            try b.put("{{\"identity\":\"{x:0>16}\",\"reason\":\"{s}\"}}", .{ item.identity, @tagName(item.reason) });
        }
        try b.put("],\"learned_folders_omitted\":{d},\"conversation_states_removed\":{d}", .{
            forgotten.out.folders.omitted, forgotten.out.conversations,
        });
        if (e.retire_failed) try b.put(",\"error\":{{\"message\":\"erase applied; cleanup record could not be retired\",\"type\":\"server_error\",\"code\":500}}", .{});
        if (e.unsynced) try b.put(",\"error\":{{\"message\":\"erase applied but journal directory sync failed\",\"type\":\"server_error\",\"code\":500}}", .{});
        try b.put("}}", .{});
        return b.bytes[0..b.len];
    }
};

const Buffer = struct {
    bytes: []u8,
    len: usize = 0,

    fn put(b: *Buffer, comptime fmt: []const u8, args: anytype) !void {
        const written = try std.fmt.bufPrint(b.bytes[b.len..], fmt, args);
        b.len += written.len;
    }
};
