//! Opt-in durable Responses state; one atomic record per scoped owner and response id.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const platform = @import("sys");
const translate = @import("responses_translate.zig");
const codec = @import("responses_durable_record.zig");
const Allocator = std.mem.Allocator;

pub const Principal = codec.Principal;
pub const Metadata = codec.Metadata;
pub const max_chain: usize = 1024;

pub const Store = struct {
    gpa: Allocator,
    io: std.Io,
    dir: []const u8,
    mutex: std.Io.Mutex = .init,
    max_record_bytes: usize,

    pub fn init(gpa: Allocator, io: std.Io, dir: []const u8, max_record_bytes: usize) !Store {
        if (max_record_bytes == 0) return error.BadConversationLimit;
        const owned = try gpa.dupe(u8, dir);
        errdefer gpa.free(owned);
        try std.Io.Dir.cwd().createDirPath(io, owned);
        return .{ .gpa = gpa, .io = io, .dir = owned, .max_record_bytes = max_record_bytes };
    }

    pub fn deinit(s: *Store) void {
        s.gpa.free(s.dir);
        s.* = undefined;
    }

    pub fn put(s: *Store, cx: *errors.Cx, response: json.Value, added: []const json.Value, metadata: Metadata) !void {
        const from_output = try translate.messages(cx, response.get("output").?.array);
        const all = try std.mem.concat(cx.a, json.Value, &.{ added, from_output });
        try s.insert(cx, response, all, true, metadata);
    }

    pub fn putWhole(s: *Store, cx: *errors.Cx, response: json.Value, messages: []json.Value, metadata: Metadata) !void {
        try s.insert(cx, response, messages, false, metadata);
    }

    fn insert(s: *Store, cx: *errors.Cx, response: json.Value, messages: []json.Value, chained: bool, metadata: Metadata) !void {
        const id = response.strField("id") orelse return error.BadConversationResponse;
        const parent = if (chained) response.strField("previous_response_id") else null;
        if (parent != null and std.mem.eql(u8, id, parent.?)) return error.ConversationCycle;
        const bytes = try codec.encode(cx.a, response, messages, parent, metadata);
        if (bytes.len > s.max_record_bytes) return error.ConversationRecordTooLarge;
        const file_path = try s.path(cx.a, metadata.principal, id);
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        const existing = try s.readFile(cx.a, metadata.principal, id);
        if (existing != null) {
            if (std.mem.eql(u8, existing.?, bytes)) return;
            return error.ConversationConflict;
        }
        if (parent) |rid| {
            var ancestor = try s.readLocked(cx.a, metadata, rid, 0) orelse return error.ConversationParentMissing;
            ancestor.deinit();
        }
        try platform.atomic_file.replacePrivate(s.gpa, file_path, &.{bytes});
    }

    pub fn get(s: *Store, a: Allocator, metadata: Metadata, id: []const u8, now: i64) !?json.Value {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        var record = try s.readLocked(a, metadata, id, now) orelse return null;
        defer record.deinit();
        return try json.deepCopy(a, record.response);
    }

    pub fn delete(s: *Store, a: Allocator, principal: Principal, id: []const u8) !bool {
        const file_path = try s.path(a, principal, id);
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        std.Io.Dir.cwd().deleteFile(s.io, file_path) catch |err| switch (err) {
            error.FileNotFound => return false,
            else => return err,
        };
        return true;
    }

    /// Exact committed items, oldest ancestor first; missing or expired ancestors invalidate the whole chain.
    pub fn history(s: *Store, a: Allocator, metadata: Metadata, id: []const u8, now: i64) !?[]json.Value {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        var chains: std.ArrayList([]json.Value) = .empty;
        var visited: std.StringHashMap(void) = .init(a);
        var at: ?[]const u8 = id;
        var count: usize = 0;
        while (at) |rid| {
            if (count == max_chain or visited.contains(rid)) return null;
            try visited.put(try a.dupe(u8, rid), {});
            var record = try s.readLocked(a, metadata, rid, now) orelse return null;
            defer record.deinit();
            try chains.append(a, (try json.deepCopy(a, .{ .array = record.messages })).array);
            at = if (record.parent) |parent| try a.dupe(u8, parent) else null;
            count += 1;
        }
        var out: std.ArrayList(json.Value) = .empty;
        var i = chains.items.len;
        while (i > 0) {
            i -= 1;
            for (chains.items[i]) |message| try out.append(a, message);
        }
        return out.items;
    }

    fn readLocked(s: *Store, a: Allocator, metadata: Metadata, id: []const u8, now: i64) !?codec.Record {
        const bytes = try s.readFile(a, metadata.principal, id) orelse return null;
        var record = try codec.decode(s.gpa, bytes);
        errdefer record.deinit();
        if (!codec.sameMetadataIgnoringExpiry(record.metadata, metadata)) return error.ConversationIdentityChanged;
        if (!std.mem.eql(u8, record.response.strField("id").?, id)) return error.ConversationIdentityChanged;
        if (codec.expired(record.metadata, now)) {
            record.deinit();
            return null;
        }
        return record;
    }

    fn readFile(s: *Store, a: Allocator, principal: Principal, id: []const u8) !?[]u8 {
        const file_path = try s.path(a, principal, id);
        return std.Io.Dir.cwd().readFileAlloc(s.io, file_path, a, .limited(s.max_record_bytes)) catch |err| switch (err) {
            error.FileNotFound => null,
            else => err,
        };
    }

    fn path(s: *Store, a: Allocator, principal: Principal, id: []const u8) ![]const u8 {
        if (id.len == 0 or id.len > 256 or principal.scope.len > 256) return error.BadConversationId;
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update("tensorfold-durable-response-v1");
        if (principal.key) |key| h.update(&key);
        lengthField(&h, principal.scope);
        lengthField(&h, id);
        const digest = h.finalResult();
        const name = try std.fmt.allocPrint(a, "{s}.json", .{std.fmt.bytesToHex(digest, .lower)});
        defer a.free(name);
        return std.fs.path.join(a, &.{ s.dir, name });
    }
};

fn lengthField(h: *std.crypto.hash.sha2.Sha256, value: []const u8) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, value.len, .little);
    h.update(&bytes);
    h.update(value);
}

test {
    _ = @import("responses_durable_test.zig");
}
